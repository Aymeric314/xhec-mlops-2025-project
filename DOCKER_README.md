# Docker Build and Run Instructions

## Quick Start

### Option 1: Using Docker Compose (Recommended)
```bash
# Build and start both services
docker-compose up --build

# Run in background
docker-compose up -d --build

# Stop services
docker-compose down
```

### Option 2: Using Docker directly
```bash
# Build the image
docker build -t mlops-app .

# Run the container
docker run -p 8001:8001 -p 4201:4201 -p 5000:5000 mlops-app
```

## Services

Once running, you can access:

- **FastAPI Application**: http://localhost:8001
  - Health check: http://localhost:8001/
  - API docs: http://localhost:8001/docs
  - Predict endpoint: http://localhost:8001/predict

- **Prefect Server**: http://localhost:4201
  - Prefect UI: http://localhost:4201

- **MLflow Server**: http://localhost:5000
  - MLflow UI: http://localhost:5000
  - Track experiments, models, and artifacts

## API Usage

### Health Check
```bash
curl http://localhost:8001/
```

### Make Predictions
```bash
curl -X POST "http://localhost:8001/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "Sex": "M",
       "Diameter": 0.35,
       "Height": 0.09,
       "Whole_weight": 0.2255,
       "Shucked_weight": 0.0995,
       "Viscera_weight": 0.0485,
       "Shell_weight": 0.07
     }'
```

## Development

### Rebuild after changes
```bash
docker-compose up --build
```

### View logs
```bash
docker-compose logs -f
```

### Access container shell
```bash
docker-compose exec mlops-app bash
```

## Volumes

The following directories are mounted as volumes:
- `./data` - Training data
- `./mlruns` - MLflow experiments
- `./src/web_service/local_objects` - Model artifacts

This ensures your data and models persist between container restarts.
