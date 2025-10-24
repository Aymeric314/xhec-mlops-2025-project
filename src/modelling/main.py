# This module is the training flow: it reads the data, preprocesses it, trains a model and saves it.

import argparse
from pathlib import Path

import mlflow
from predicting import evaluate_model, predict_rings
from prefect import flow
from preprocessing import extract_x_y, preprocessing, split_data
from training import train_model
from utils import get_data, pickle_object


@flow(name="training_flow")
def training_flow(trainset_path: Path) -> None:
    """Train a model using the data at the given path and save the model (pickle)."""

    mlflow.set_experiment("abalone_project")

    # Start a run
    with mlflow.start_run() as run:
        run_id = run.info.run_id

        # Set tags for the run
        mlflow.set_tags({"model_type": "random_forest", "framework": "sklearn"})

        # Load data
        train_df = get_data(trainset_path)

        # Split data first (before preprocessing to keep target column)
        X_train, X_test, y_train, y_test = split_data(train_df)

        # Preprocess training data
        X_train_processed = preprocessing(X_train.copy())
        X_test_processed = preprocessing(X_test.copy())

        # Extract features and create encoder for training data
        X_train_encoded, _, dv = extract_x_y(X_train_processed, with_target=False)

        # Use the same encoder for test data
        X_test_encoded, _, _ = extract_x_y(X_test_processed, dv=dv, with_target=False)

        # Pickle encoder for inference
        pickle_object(dv, "src/web_service/local_objects/encoder.pkl")

        # Train model
        model = train_model(X_train_encoded, y_train)

        # Evaluate model
        prediction = predict_rings(X_train_encoded, model)
        train_me = evaluate_model(y_train, prediction)
        mlflow.log_metric("train_rmse", train_me)

        # Evaluate model on test set
        y_pred_test = predict_rings(X_test_encoded, model)
        test_me = evaluate_model(y_test, y_pred_test)
        mlflow.log_metric("test_rmse", test_me)

        # Log your model
        mlflow.sklearn.log_model(model, "model")

        # Register your model in mlflow model registry
        mlflow.register_model(f"runs:/{run_id}/model", "abalone_rf_model")

        # Pickle model --> The model should be saved in pkl format the `src/web_service/local_objects` folder
        pickle_object(model, "src/web_service/local_objects/model.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model using the data at the given path."
    )
    parser.add_argument("trainset_path", type=str, help="Path to the training set")
    args = parser.parse_args()
    training_flow(args.trainset_path)
