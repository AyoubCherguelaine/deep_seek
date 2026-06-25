# DeepSeek OCR API

FastAPI service for running DeepSeek-OCR on an NVIDIA GPU. The app accepts single image uploads or batches of images and returns OCR text per image.

## Features

- OCR for PNG, JPG, JPEG, WEBP, BMP, TIFF
- Optional bearer-token auth
- Eager attention by default for DeepSeek-OCR-2
- Local uvicorn runtime
- Health endpoint for readiness checks

## Requirements

- NVIDIA GPU
- NVIDIA driver with RTX 50 / CUDA 12.8-compatible container runtime
- Docker with GPU support, or a Python environment with PyTorch CUDA wheels installed

The default model is `deepseek-ai/DeepSeek-OCR-2`.

## Quick Start

1. Copy the example environment file:

```bash
cp .env.example .env
```

2. Adjust `.env` if needed:

- `AUTH_API_KEY` must be set for token auth
- `MODEL_NAME` controls the Hugging Face model
- `USE_FLASH_ATTENTION=true` tries `flash_attention_2` only when `flash-attn` is importable; otherwise the app uses eager attention
- `ALLOW_FLASH_ATTENTION_INSTALL=true` allows runtime install of `flash-attn==2.7.3`
- `BASE_SIZE`, `IMAGE_SIZE`, `CROP_MODE`, and `TEST_COMPRESS` tune OCR speed/quality

3. Start the app:

```bash
./run.sh
```

The service listens on `http://localhost:8000` by default.

## Local Development

If you want to run the app without Docker, make sure the Python dependencies and GPU-enabled PyTorch are installed, then start:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
```

The app loads the model on startup, so it will fail fast if CUDA is not available.

## Docker

Build and run with Docker directly:

```bash
docker build -t deepseek-ocr-api:latest .
docker run --gpus all --env-file .env -p 8000:8000 deepseek-ocr-api:latest
```

Or use docker compose:

```bash
docker compose up --build
```

Before loading the model on the server, you can verify the CUDA/PyTorch stack inside
the image without starting the API:

```bash
docker run --rm --gpus all deepseek-ocr-api:latest \
  python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_capability(0))"
```

For RTX 50 / Blackwell, expect a CUDA 12.8 PyTorch wheel and compute capability
around `(12, 0)`.

## Authentication

If `AUTH_API_KEY` is empty, `/ocr` is open for local development.

If `AUTH_API_KEY` is set:

1. Generate a bearer token locally with the helper script:

```bash
python3 ./generate_token.py
```

You can pass a different subject if you want:

```bash
python3 ./generate_token.py service-a
```

The script reads `AUTH_API_KEY` from `.env`, signs the same JWT the server would issue, and prints the token.

2. Use the returned token on OCR requests:

```bash
curl -X POST http://localhost:8000/ocr \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "files=@invoice.png"
```

## API

### `GET /health`

Returns model and GPU status.

Example:

```bash
curl http://localhost:8000/health
```

### `POST /auth/token`

Exchanges `AUTH_API_KEY` for a long-lived bearer token.

Form fields:

- `api_key` - required
- `subject` - optional, defaults to `api-client`

### `POST /ocr`

Uploads one image or a batch of images for OCR.

Form fields:

- `files` - one or more image uploads

OCR settings are read from `.env`; this endpoint does not accept per-request model options.

Supported inputs:

- Images: `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tiff`, `.tif`

Example:

```bash
curl -X POST http://localhost:8000/ocr \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "files=@image1.png" \
  -F "files=@image2.jpg"
```

## Environment Variables

All configuration comes from environment variables.

- `APP_NAME` - application name
- `HOST` - bind host, default `0.0.0.0`
- `PORT` - bind port, default `8000`
- `WORKERS` - uvicorn worker count, default `1`
- `LOG_LEVEL` - log level, default `INFO`
- `CUDA_VISIBLE_DEVICES` - GPU selection, default `0`
- `MODEL_NAME` - Hugging Face model ID, default `deepseek-ai/DeepSeek-OCR-2`
- `USE_FLASH_ATTENTION` - tries FlashAttention 2 when available; default `false` uses eager attention
- `ALLOW_FLASH_ATTENTION_INSTALL` - permits runtime `flash-attn==2.7.3` install, default `false`
- `BASE_SIZE` - OCR base size, default `640`
- `IMAGE_SIZE` - OCR image size, default `640`
- `CROP_MODE` - crop mode toggle, default `true`
- `TEST_COMPRESS` - compression test toggle, default `false`
- `SAVE_OCR_RESULTS` - writes model output files before reading them, default `false`
- `DEFAULT_PROMPT` - default OCR prompt
- `MAX_UPLOAD_MB` - upload limit in MB, default `40`
- `REQUEST_TIMEOUT_SECONDS` - per-page OCR timeout, default `180`
- `TEMP_DIR` - temporary working directory, default `/tmp/deepseek_ocr`
- `AUTH_API_KEY` - shared secret for token issuance
- `AUTH_TOKEN_TTL_DAYS` - bearer token lifetime, default `365`

## Notes

- The app defaults to eager attention for DeepSeek-OCR-2.
- For higher quality but slower OCR, use `BASE_SIZE=1024` and `IMAGE_SIZE=768`.
- `torch`, `torchvision`, and `torchaudio` are installed in the Dockerfile from the CUDA 12.8 PyTorch wheel index, not from `requirements.txt`.
- A single GPU request is processed at a time to reduce CUDA out-of-memory issues.
