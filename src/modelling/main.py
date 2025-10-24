# This module is the training flow: it reads the data, preprocesses it, trains a model and saves it.

import argparse
from pathlib import Path

import mlflow

try:
    # Try relative imports first (for when running in docker container)
    from .predicting import evaluate_model, predict_rings
    from .preprocessing import extract_x_y, preprocessing, split_data
    from .training import train_model
    from .utils import get_data, pickle_object
except ImportError:
    # Fall back to absolute imports (for when running directly)
    from predicting import evaluate_model, predict_rings
    from preprocessing import extract_x_y, preprocessing, split_data
    from training import train_model
    from utils import get_data, pickle_object

from prefect import flow


@flow(name="training_flow")
def training_flow(trainset_path: Path) -> None:
    """Train a model using the data at the given path and save the model and encoder(pickle)."""

    mlflow.set_experiment("abalone_project")

    # Start a run
    with mlflow.start_run() as run:
        run_id = run.info.run_id

        # Set tags for the run
        mlflow.set_tags({"model_type": "random_forest", "framework": "sklearn"})

        # Load data
        train_df = get_data(trainset_path)

        # Preprocess ALL data first to create consistent encoder
        train_df_processed = preprocessing(train_df.copy())

        # Create encoder using ALL data to ensure consistent vocabulary
        _, _, dv = extract_x_y(train_df_processed, with_target=True)

        # Now split the preprocessed data
        X_train, X_test, y_train, y_test = split_data(train_df_processed)

        # Extract features using the pre-fitted encoder
        X_train_encoded, _, _ = extract_x_y(X_train, dv=dv, with_target=False)
        X_test_encoded, _, _ = extract_x_y(X_test, dv=dv, with_target=False)

        # Pickle encoder for inference - use absolute path that works in both environments
        import os

        encoder_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "src",
            "web_service",
            "local_objects",
            "encoder.pkl",
        )
        pickle_object(dv, encoder_path)

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
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "src",
            "web_service",
            "local_objects",
            "model.pkl",
        )
        pickle_object(model, model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model using the data at the given path."
    )
    parser.add_argument("trainset_path", type=str, help="Path to the training set")
    args = parser.parse_args()
    training_flow(args.trainset_path)
