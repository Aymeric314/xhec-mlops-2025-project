#!/bin/bash

# Exit on any error
set -e

echo "🚀 Starting MLOps Services..."

# Start Prefect server in the background
echo "📊 Starting Prefect server on port 4201..."
prefect server start --host 0.0.0.0 --port 4201 &

# Start MLflow server in the background
echo "🔬 Starting MLflow server on port 5000..."
cd /app
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri file:///app/mlruns --default-artifact-root ./mlruns &

# Wait a moment for services to start
sleep 5

# Start FastAPI application
echo "🌐 Starting FastAPI application on port 8001..."
cd src/web_service
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
