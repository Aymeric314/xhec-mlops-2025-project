from fastapi import FastAPI
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
