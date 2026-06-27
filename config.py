import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_env_file(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip().strip('"').strip("'")
        if name:
            os.environ.setdefault(name, value)


_load_env_file()


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


@dataclass(frozen=True)
class Settings:
    # Server
    app_name: str = os.getenv("APP_NAME", "unlimited-ocr-api")
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = _get_int("PORT", 8000)
    workers: int = _get_int("WORKERS", 1)

    # GPU/model
    cuda_visible_devices: str = os.getenv("CUDA_VISIBLE_DEVICES", "0")
    model_name: str = os.getenv("MODEL_NAME", "baidu/Unlimited-OCR")

    # OCR modes
    base_size_base: int = _get_int("BASE_SIZE_BASE", 1024)
    image_size_base: int = _get_int("IMAGE_SIZE_BASE", 1024)
    base_size_long: int = _get_int("BASE_SIZE_LONG", 1024)
    image_size_long: int = _get_int("IMAGE_SIZE_LONG", 640)
    max_length: int = _get_int("MAX_LENGTH", 8192)
    no_repeat_ngram_size: int = _get_int("NO_REPEAT_NGRAM_SIZE", 35)
    ngram_window: int = _get_int("NGRAM_WINDOW", 128)
    default_prompt: str = os.getenv("DEFAULT_PROMPT", "document parsing.")

    # Upload/runtime
    max_upload_mb: int = _get_int("MAX_UPLOAD_MB", 80)
    max_pdf_pages: int = _get_int("MAX_PDF_PAGES", 200)
    pdf_dpi: int = _get_int("PDF_DPI", 200)
    request_timeout_seconds: int = _get_int("REQUEST_TIMEOUT_SECONDS", 300)
    temp_dir: str = os.getenv("TEMP_DIR", "/tmp/unlimited_ocr")
    image_extensions: set[str] = field(
        default_factory=lambda: {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
    )

    # Auth
    auth_api_key: str = os.getenv("AUTH_API_KEY", "")
    auth_token_ttl_days: int = _get_int("AUTH_TOKEN_TTL_DAYS", 365)


settings = Settings()
