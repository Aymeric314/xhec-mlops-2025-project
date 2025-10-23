import os
import sys

import pandas as pd

# Add the modelling module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "modelling"))

from predicting import predict_rings
from preprocessing import extract_x_y, preprocessing

from web_service.utils import load_model_and_encoder


def predict_abalone(input_data):
    """Make prediction on abalone data using existing functions"""
    model, encoder = load_model_and_encoder()

    # Create DataFrame from input
    df = pd.DataFrame([input_data])

    # Use existing preprocessing function
    df_processed = preprocessing(df)

    # Use existing extract_x_y function
    x, _, _ = extract_x_y(df_processed, dv=encoder, with_target=False)

    # Use existing predict_rings function
    prediction = predict_rings(x, model)[0]

    return float(prediction)
