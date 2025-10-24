# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements-docker.txt requirements-dev.txt pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy the entire project
COPY . .

# Install the package in development mode
RUN pip install -e .

# Create necessary directories
RUN mkdir -p src/web_service/local_objects mlruns

# Make the run script executable
RUN chmod +x bin/run_services.sh

# Set Python path
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Expose ports for FastAPI (8001), Prefect (4201), and MLflow (5000)
EXPOSE 8001 4201 5000

# Use the run_services.sh script as entrypoint
CMD ["./bin/run_services.sh"]
