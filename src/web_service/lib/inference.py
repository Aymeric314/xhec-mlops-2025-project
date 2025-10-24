import os
import sys

# Add the modelling module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "modelling"))

from utils import load_model_and_encoder


def predict_abalone(input_data):
    """Make prediction on abalone data using existing functions"""
    model, encoder = load_model_and_encoder()

    # Convert input to the format expected by DictVectorizer
    # Use exact feature names from training (with spaces)
    dict_input = {
        "Sex": input_data["Sex"],
        "Diameter": input_data["Diameter"],
        "Height": input_data["Height"],
        "Whole weight": input_data["Whole_weight"],
        "Shucked weight": input_data["Shucked_weight"],
        "Viscera weight": input_data["Viscera_weight"],
        "Shell weight": input_data["Shell_weight"],
    }

    # Transform using DictVectorizer
    X = encoder.transform([dict_input])

    # Make prediction
    prediction = model.predict(X)[0]

    return float(prediction)
