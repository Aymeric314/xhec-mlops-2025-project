# This module is the training flow: it reads the data, preprocesses it, trains a model and saves it.

import argparse
from pathlib import Path

import mlflow
from predicting import evaluate_model, predict_rings
from preprocessing import preprocessing, split_data
from training import train_model
from utils import get_data


def main(trainset_path: Path) -> None:
    """Train a model using the data at the given path and save the model (pickle)."""

    mlflow.set_experiment("abalone_project")

    # Start a run
    with mlflow.start_run() as run:
        run_id = run.info.run_id

        # Set tags for the run
        mlflow.set_tags({"model_type": "random_forest", "framework": "sklearn"})

        # Load data
        train_df = get_data()

        # Encode categorical columns
        train_df = preprocessing(train_df)

        # Extract X and y
        X_train, X_test, y_train, y_test = split_data(train_df)

        # Train model
        model = train_model(X_train, y_train)

        # Evaluate model
        prediction = predict_rings(X_train, model)
        train_me = evaluate_model(y_train, prediction)
        mlflow.log_metric("train_rmse", train_me)

        # Evaluate model on test set
        y_pred_test = predict_rings(X_test, model)
        test_me = evaluate_model(y_test, y_pred_test)
        mlflow.log_metric("test_rmse", test_me)

        # Log your model
        mlflow.sklearn.log_model(model, "model")

        # Register your model in mlflow model registry
        mlflow.register_model(f"runs:/{run_id}/model", "abalone_rf_model")
    # Pickle model --> The model should be saved in pkl format the `src/web_service/local_objects` folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model using the data at the given path."
    )
    parser.add_argument("trainset_path", type=str, help="Path to the training set")
    args = parser.parse_args()
    main(args.trainset_path)
