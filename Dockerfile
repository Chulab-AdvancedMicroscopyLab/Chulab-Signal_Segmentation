# Dockerfile for zarr file based analyzer
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3.10-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python package dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Set working directory
WORKDIR /workspace

# Copy application code and modules
COPY train.py ./
COPY inference.py ./
COPY converter.py ./
COPY analysis.py ./
COPY models ./models
COPY IO ./IO
COPY utils ./utils
