# MLOps Project: Abalone Age Prediction - Complete Setup Guide

<div align="center">

[![Python Version](https://img.shields.io/badge/python-3.10%20or%203.11-blue.svg)]()
[![Docker](https://img.shields.io/badge/docker-enabled-blue.svg)]()
[![Prefect](https://img.shields.io/badge/prefect-workflow-green.svg)]()
[![MLflow](https://img.shields.io/badge/mlflow-tracking-orange.svg)]()

</div>

## 🎯 Project Overview

This MLOps project builds a complete machine learning system to predict the age of abalone (a type of sea snail) using physical measurements instead of the traditional time-consuming method of counting shell rings under a microscope.

**Mission**: Transform a simple ML model into a production-ready system with automated training, deployment, and prediction capabilities.

## 🚀 Quick Start with Docker (Recommended)

### Prerequisites
- Docker and Docker Compose installed
- Git (to clone the repository)

### 1. Clone and Setup
```bash
git clone <your-repo-url>
cd xhec-mlops-2025-project
```

### 2. Start All Services
```bash
docker compose up --build
```

### 3. Access the Services
- **FastAPI API**: http://localhost:8000
- **Prefect UI**: http://localhost:4200
- **MLflow UI**: http://localhost:5000

## 🐳 Docker Services Overview

### 🌐 **FastAPI Prediction API** (Port 8000)
- **Purpose**: REST API for making abalone age predictions
- **Endpoints**:
  - `GET /` - Health check
  - `POST /predict` - Make predictions
  - `POST /deploy` - Launch Prefect deployment

### 📊 **Prefect Workflow Management** (Port 4200)
- **Purpose**: Orchestrate and monitor ML training workflows
- **Features**: Deployment management, run monitoring, scheduling

### 🔬 **MLflow Experiment Tracking** (Port 5000)
- **Purpose**: Track experiments, log metrics, manage model versions
- **Features**: Experiment comparison, model registry, artifact storage

## 📝 Using the FastAPI API

### Health Check
```bash
curl http://localhost:8000/
```

### Make Predictions
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Sex": "M",
    "Diameter": 0.35,
    "Height": 0.105,
    "Whole_weight": 0.2255,
    "Shucked_weight": 0.0995,
    "Viscera_weight": 0.0485,
    "Shell_weight": 0.07
  }'
```

### Launch Model Retraining
```bash
curl -X POST "http://localhost:8000/deploy?data_path=data/abalone.csv"
```

## 🔄 Running Prefect Workflows Locally

### Prerequisites for Local Development
- Python 3.10 or 3.11
- Virtual environment setup

### 1. Initial Project Setup
This project is already configured with:
- ✅ **Dependencies**: All required packages in `pyproject.toml`
- ✅ **Code Quality**: Pre-commit hooks configured
- ✅ **CI/CD**: GitHub Actions workflow for automated checks

### 2. Setup Local Environment
```bash
# Create virtual environment
uv sync
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install pre-commit hooks
uv run pre-commit install
```

### 2. Start Prefect Server Locally
```bash
# Terminal 1: Start Prefect server
prefect server start --host 0.0.0.0 --port 4201
```

### 3. Access Prefect UI
- Open http://localhost:4201 in your browser

### 4. Run Training Workflow Locally

#### Data Path Options
The training workflow accepts data paths in three formats:

1. **No path specified** (uses default):
   ```bash
   uv run python -m src.modelling.deployment
   # Uses: data/abalone.csv (default)
   ```

2. **Relative path** (relative to project root):
   ```bash
   uv run python -m src.modelling.deployment data/abalone.csv
   uv run python -m src.modelling.deployment data/my_dataset.csv
   ```

3. **Absolute path** (full system path):
   ```bash
   uv run python -m src.modelling.deployment /home/user/datasets/abalone.csv
   uv run python -m src.modelling.deployment /Users/student/Downloads/data.csv
   ```

**⚠️ Important**: Absolute paths will **NOT work in Docker** because the container has its own filesystem. For Docker usage, always use relative paths or place data in the `data/` folder.

**💡 Recommendation**: Place your data files in the `data/` folder for easy access and consistency.

#### Option A: Direct Python Execution
```bash
# From project root
cd src/modelling
uv run python main.py ../../data/abalone.csv
```

#### Option B: Using Prefect Deployment
```bash
# From project root
uv run python -m src.modelling.deployment data/abalone.csv
```

### 5. Monitor Training Progress
- Check Prefect UI at http://localhost:4201
- View MLflow experiments at http://localhost:5000

## 🧪 Testing the Complete Pipeline

### 1. Test Prediction API
```bash
# Test with sample data
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Sex": "F",
    "Diameter": 0.45,
    "Height": 0.15,
    "Whole_weight": 0.5,
    "Shucked_weight": 0.25,
    "Viscera_weight": 0.1,
    "Shell_weight": 0.15
  }'
```

### 2. Test Model Retraining
```bash
# Trigger retraining via API
curl -X POST "http://localhost:8000/deploy?data_path=data/abalone.csv"

# Check Prefect UI for deployment status
# Check MLflow for new experiment runs
```

### 3. Verify Model Updates
- Check `src/web_service/local_objects/` for updated `model.pkl` and `encoder.pkl`
- Test predictions with new model

## 📊 Data Format

### Input Data Schema
```json
{
  "Sex": "M|F|I",           // Sex: Male, Female, or Infant
  "Diameter": 0.35,         // Shell diameter (mm)
  "Height": 0.105,          // Shell height (mm)
  "Whole_weight": 0.2255,   // Whole weight (grams)
  "Shucked_weight": 0.0995, // Shucked weight (grams)
  "Viscera_weight": 0.0485, // Viscera weight (grams)
  "Shell_weight": 0.07      // Shell weight (grams)
}
```

### Expected Output
```json
{
  "predicted_rings": 8.5    // Predicted number of rings (age)
}
```


## 📁 Project Structure

```
xhec-mlops-2025-project/
├── src/
│   ├── modelling/          # ML training pipeline
│   │   ├── main.py         # Prefect training flow
│   │   ├── deployment.py   # Prefect deployment script
│   │   ├── preprocessing.py
│   │   ├── training.py
│   │   ├── predicting.py
│   │   └── utils.py
│   └── web_service/        # FastAPI application
│       ├── main.py         # FastAPI app
│       ├── utils.py        # Utility functions
│       ├── lib/
│       │   ├── models.py   # Pydantic models
│       │   └── inference.py # Prediction logic
│       └── local_objects/  # Model storage
├── data/
│   └── abalone.csv        # Training data
├── docker-compose.yml     # Docker services
├── Dockerfile            # Container definition
└── bin/
    └── run_services.sh   # Service startup script
```

## 🎯 Key Features

- ✅ **Automated ML Pipeline**: Prefect-based workflow orchestration
- ✅ **REST API**: FastAPI for real-time predictions
- ✅ **Experiment Tracking**: MLflow integration
- ✅ **Containerized Deployment**: Docker-based deployment
- ✅ **Model Versioning**: Automatic model updates
- ✅ **Health Monitoring**: API health checks and monitoring

---

**Happy MLOps! 🎉**
