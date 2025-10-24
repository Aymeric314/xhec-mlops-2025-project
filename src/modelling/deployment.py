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
import sys
from pathlib import Path

from prefect import serve

# Allow user to specify data path via command line argument or environment variable
if len(sys.argv) > 1:
    # User provided data path as command line argument
    data_path = sys.argv[1]
    # Get project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    # Convert to absolute path if it's relative
    if not os.path.isabs(data_path):
        # All relative paths are made relative to project root
        data_path = os.path.join(project_root, data_path)
else:
    # Default to the original path
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_path = os.path.join(project_root, "data", "abalone.csv")

print(f"Using data path: {data_path}")

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
