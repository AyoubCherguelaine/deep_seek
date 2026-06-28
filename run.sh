#!/usr/bin/env bash
set -e

APP_NAME="unlimited-ocr-api"
IMAGE_NAME="unlimited-ocr-api:latest"
CONTAINER_NAME="unlimited-ocr-api"

if [ ! -f ".env" ]; then
  echo "No .env found. Creating from .env.example..."
  cp .env.example .env
fi

echo "Checking NVIDIA GPU..."
nvidia-smi || {
  echo "ERROR: nvidia-smi failed. GPU or NVIDIA container runtime is not ready."
  exit 1
}

echo "Building Docker image..."
docker build -t "$IMAGE_NAME" .

echo "Stopping old container if it exists..."
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

echo "Starting container..."
docker run -d \
  --name "$CONTAINER_NAME" \
  --restart unless-stopped \
  --gpus all \
  --env-file .env \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -p 8000:8000 \
  -v unlimited_ocr_hf_cache:/root/.cache/huggingface \
  -v unlimited_ocr_tmp:/tmp/unlimited_ocr \
  "$IMAGE_NAME"

echo "Container started."
echo "Logs:"
docker logs -f "$CONTAINER_NAME"
