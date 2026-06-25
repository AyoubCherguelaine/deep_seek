import os
from dataclasses import dataclass


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


@dataclass(frozen=True)
class Settings:
    # Server
    app_name: str = os.getenv("APP_NAME", "deepseek-ocr-api")
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = _get_int("PORT", 8000)
    workers: int = _get_int("WORKERS", 1)

    # Model
    model_name: str = os.getenv("MODEL_NAME", "deepseek-ai/DeepSeek-OCR-2")
    cuda_visible_devices: str = os.getenv("CUDA_VISIBLE_DEVICES", "0")
    use_flash_attention: bool = _get_bool("USE_FLASH_ATTENTION", False)
    allow_flash_attention_install: bool = _get_bool("ALLOW_FLASH_ATTENTION_INSTALL", False)

    # OCR mode
    # Mirrors the working notebook's OCR-2 crop-mode defaults for RTX 50.
    base_size: int = _get_int("BASE_SIZE", 1024)
    image_size: int = _get_int("IMAGE_SIZE", 768)
    crop_mode: bool = _get_bool("CROP_MODE", True)
    test_compress: bool = _get_bool("TEST_COMPRESS", False)

    # Prompts:
    # "<image>\nFree OCR. "
    # "<image>\n<|grounding|>Convert the document to markdown. "
    default_prompt: str = os.getenv(
        "DEFAULT_PROMPT",
        "<image>\n<|grounding|>Convert the document to markdown. "
    )

    # Upload limits
    max_upload_mb: int = _get_int("MAX_UPLOAD_MB", 40)
    max_pdf_pages: int = _get_int("MAX_PDF_PAGES", 10)

    # Runtime
    request_timeout_seconds: int = _get_int("REQUEST_TIMEOUT_SECONDS", 180)
    temp_dir: str = os.getenv("TEMP_DIR", "/tmp/deepseek_ocr")

    # Auth (1-year bearer tokens)
    auth_api_key: str = os.getenv("AUTH_API_KEY", "")
    auth_token_ttl_days: int = _get_int("AUTH_TOKEN_TTL_DAYS", 365)


settings = Settings()
