# Pydantic models for the web service

from typing import Literal

from pydantic import BaseModel


class AbaloneInput(BaseModel):
    """Input model for abalone prediction"""

    Sex: Literal["M", "F", "I"]
    Diameter: float
    Height: float
    Whole_weight: float
    Shucked_weight: float
    Viscera_weight: float
    Shell_weight: float


class AbalonePrediction(BaseModel):
    """Output model for abalone prediction"""

    predicted_rings: float
