# Unlimited OCR API

FastAPI service for running `baidu/Unlimited-OCR` on an NVIDIA GPU. It exposes a small JSON API for accurate single-image OCR and long document OCR with optional n-gram repeat control.

## Features

- `POST /ocr/base` for accurate image or PDF OCR using 1024 px non-crop mode
- `POST /ocr/long` for image or PDF OCR using 640 px crop mode
- Optional `use_ngram=true|false` on long OCR
- Optional bearer-token auth
- Docker deployment for RTX 50 / CUDA 12.8 GPUs
- One GPU inference at a time to reduce out-of-memory failures on 16 GB VRAM

## Requirements

- NVIDIA GPU, tested target: RTX 5070 Ti 16 GB
- NVIDIA driver with CUDA 12.8-compatible container runtime
- Docker with GPU support, or a Python environment with CUDA-enabled PyTorch

The default model is `baidu/Unlimited-OCR`.

## Quick Start

```bash
cp .env.example .env
./run.sh
```

The service listens on `http://localhost:8000`.

## Docker

Build and run directly:

```bash
docker build -t unlimited-ocr-api:latest .
docker run --gpus all --env-file .env -p 8000:8000 unlimited-ocr-api:latest
```

Or use Docker Compose:

```bash
docker compose up --build
```

Verify the CUDA/PyTorch stack inside the image:

```bash
docker run --rm --gpus all unlimited-ocr-api:latest \
  python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_capability(0))"
```

## Authentication

If `AUTH_API_KEY` is empty, OCR endpoints are open for local development.

If `AUTH_API_KEY` is set, generate a bearer token:

```bash
python3 ./generate_token.py
```

Then pass it on OCR requests:

```bash
curl -X POST http://localhost:8000/ocr/base \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@test/latex-sample-550723046.png"
```

You can also exchange the API key through the service:

```bash
curl -X POST http://localhost:8000/auth/token \
  -F "api_key=change-me-to-a-strong-secret" \
  -F "subject=api-client"
```

## API

### `GET /health`

Returns model and GPU status.

```bash
curl http://localhost:8000/health
```

### `POST /ocr/base`

Accurate OCR for one image or PDF. This matches the Space's Base mode.

Form fields:

- `file` - required image or PDF upload
- `prompt` - optional, defaults to the configured Quran-safe document parsing prompt

Example:

```bash
curl -X POST http://localhost:8000/ocr/base \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@test/latex-sample-550723046.png"
```

### `POST /ocr/long`

Long document OCR for one image or PDF. This matches the Space's Long mode, called `gundam` internally by the Space.

Form fields:

- `file` - required image or PDF upload
- `prompt` - optional, defaults to the configured Quran-safe document parsing prompt
- `use_ngram` - optional boolean, defaults to `true`

Examples:

```bash
curl -X POST http://localhost:8000/ocr/long \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@test/latex-sample-550723046.png" \
  -F "use_ngram=true"
```

```bash
curl -X POST http://localhost:8000/ocr/long \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@test/sample.pdf" \
  -F "use_ngram=false"
```

Response shape:

```json
{
  "ok": true,
  "filename": "page.png",
  "file_type": "image",
  "pages": 1,
  "elapsed_seconds": 0.0,
  "results": [
    {
      "page": 1,
      "text": "..."
    }
  ]
}
```

## Environment Variables

- `APP_NAME` - application name
- `HOST` - bind host, default `0.0.0.0`
- `PORT` - bind port, default `8000`
- `WORKERS` - uvicorn worker count, keep `1` for one model instance
- `LOG_LEVEL` - log level, default `INFO`
- `CUDA_VISIBLE_DEVICES` - GPU selection, default `0`
- `MODEL_NAME` - Hugging Face model ID, default `baidu/Unlimited-OCR`
- `DEFAULT_PROMPT` - default OCR prompt; the provided default avoids transcribing Quranic verses and asks for surah/ayah identification only
- `BASE_SIZE_BASE` - `/ocr/base` base size, default `1024`
- `IMAGE_SIZE_BASE` - `/ocr/base` image size, default `1024`
- `BASE_SIZE_LONG` - `/ocr/long` base size, default `1024`
- `IMAGE_SIZE_LONG` - `/ocr/long` image size, default `640`
- `MAX_LENGTH` - generation max length, default `8192`
- `NO_REPEAT_NGRAM_SIZE` - repeat control size, default `35`
- `NGRAM_WINDOW` - n-gram window, default `128`
- `PDF_DPI` - PDF rasterization DPI, default `200`
- `MAX_PDF_PAGES` - PDF page limit, default `200`
- `MAX_UPLOAD_MB` - upload limit in MB, default `80`
- `REQUEST_TIMEOUT_SECONDS` - per-page OCR timeout, default `300`
- `TEMP_DIR` - temporary working directory, default `/tmp/unlimited_ocr`
- `AUTH_API_KEY` - shared secret for token issuance
- `AUTH_TOKEN_TTL_DAYS` - bearer token lifetime, default `365`

## Notes

- The service loads the model at startup and fails fast if CUDA is unavailable.
- PDFs are converted to PNG pages with PyMuPDF before OCR.
- `/ocr/base` accepts images and PDFs.
- `/ocr/long` accepts images and PDFs.
- The default prompt asks the model not to transcribe Quranic verse text. It should return only the surah and ayah reference when detected.
- `torch`, `torchvision`, and `torchaudio` are installed in the Dockerfile from the CUDA 12.8 PyTorch wheel index, not from `requirements.txt`.
