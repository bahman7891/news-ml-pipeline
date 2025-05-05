# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system packages
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
 && rm -rf /var/lib/apt/lists/*

# Copy all files into the container
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Set the command to run your pipeline
CMD ["python", "run_pipeline.py"]
