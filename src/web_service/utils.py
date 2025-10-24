import os
import pickle


def load_model_and_encoder():
    """Load the trained model and encoder"""
    model_path = os.path.join(os.path.dirname(__file__), "local_objects", "model.pkl")
    encoder_path = os.path.join(
        os.path.dirname(__file__), "local_objects", "encoder.pkl"
    )

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)

    return model, encoder
