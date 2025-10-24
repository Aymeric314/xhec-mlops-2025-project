"""
Prefect deployment script for abalone model retraining.
This script creates a deployment that can be served immediately.
"""

try:
    # Try relative imports first (for when running as module)
    from .main import training_flow
except ImportError:
    # Fall back to absolute imports (for when running directly)
    from main import training_flow

import os
from pathlib import Path

from prefect import serve

# Create absolute path that works in both environments
data_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "abalone.csv"
)

training_flow_deployment = training_flow.to_deployment(
    name="abalone-model-retraining",
    version="1.0.0",
    tags=["ml", "retraining", "abalone", "hourly"],
    interval=3600,
    parameters={"trainset_path": Path(data_path)},
)

if __name__ == "__main__":
    print("Serving training flow deployment...")
    serve(training_flow_deployment)
