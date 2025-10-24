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

from pathlib import Path

from prefect import serve

training_flow_deployment = training_flow.to_deployment(
    name="abalone-model-retraining",
    version="1.0.0",
    tags=["ml", "retraining", "abalone", "daily"],
    interval=60,
    parameters={"trainset_path": Path("data/abalone.csv")},
)

if __name__ == "__main__":
    print("Serving training flow deployment...")
    serve(training_flow_deployment)
