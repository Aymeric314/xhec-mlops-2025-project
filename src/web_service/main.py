import os
import subprocess
import sys

from fastapi import FastAPI, HTTPException
from lib.inference import predict_abalone
from lib.models import AbaloneInput, AbalonePrediction

app = FastAPI(title="Abalone Prediction API")


@app.get("/")
def home() -> dict:
    return {"health_check": "App up and running!"}


@app.post("/predict", response_model=AbalonePrediction)
def predict(payload: AbaloneInput):
    """Predict the number of rings (age) of an abalone"""
    input_data = payload.dict()
    predicted_rings = predict_abalone(input_data)
    return AbalonePrediction(predicted_rings=predicted_rings)


@app.post("/deploy")
def launch_deployment(data_path: str = "data/abalone.csv"):
    """
    Start the Prefect deployment for model retraining with specified data path

    ⚠️ WARNING: If you start multiple deployments simultaneously, they may conflict.
    Use Prefect UI to manage runs instead of calling this endpoint multiple times.
    """
    try:
        # Get the project root (go up from src/web_service to project root)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

        # Prepare the command
        cmd = [sys.executable, "-m", "src.modelling.deployment", data_path]

        # Change to project root directory
        os.chdir(project_root)

        # Start the deployment process in the background
        subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        return {
            "status": "success",
            "message": "Deployment started successfully",
            "data_path": data_path,
            "prefect_ui": "http://localhost:4201",
            "warning": "⚠️ If you start multiple deployments simultaneously, they may conflict. Use Prefect UI to manage runs.",
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to start deployment: {str(e)}"
        )
