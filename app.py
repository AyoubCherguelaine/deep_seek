import asyncio
import hmac
import logging
import os
import shutil
import tempfile
import time
import traceback
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import jwt
import torch
from fastapi import FastAPI, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

from config import settings


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("deepseek-ocr-api")


# ---------------------------------------------------------
# Globals
# ---------------------------------------------------------

MODEL = None
TOKENIZER = None
MODEL_LOCK = asyncio.Lock()


# ---------------------------------------------------------
# Response schemas
# ---------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    cuda_available: bool
    device_name: Optional[str] = None
    gpu_memory_gb: Optional[float] = None


class TokenResponse(BaseModel):
    ok: bool
    access_token: str
    token_type: str
    expires_at: str


class OCRPageResult(BaseModel):
    page: int
    text: str


class OCRResponse(BaseModel):
    ok: bool
    filename: str
    file_type: str
    pages: int
    elapsed_seconds: float
    results: List[OCRPageResult]


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def ensure_temp_dir() -> None:
    Path(settings.temp_dir).mkdir(parents=True, exist_ok=True)


def get_gpu_info() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            "cuda_available": False,
            "device_name": None,
            "gpu_memory_gb": None,
        }

    props = torch.cuda.get_device_properties(0)
    return {
        "cuda_available": True,
        "device_name": props.name,
        "gpu_memory_gb": round(props.total_memory / 1024**3, 2),
    }


def clear_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def is_pdf(filename: str, content_type: Optional[str]) -> bool:
    return (
        filename.lower().endswith(".pdf")
        or content_type == "application/pdf"
    )


def is_image(filename: str, content_type: Optional[str]) -> bool:
    image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
    suffix = Path(filename.lower()).suffix
    return suffix in image_exts or bool(content_type and content_type.startswith("image/"))


async def save_upload_to_disk(upload: UploadFile, dst: Path) -> int:
    max_bytes = settings.max_upload_mb * 1024 * 1024
    written = 0

    with dst.open("wb") as f:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break

            written += len(chunk)
            if written > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Max allowed is {settings.max_upload_mb} MB.",
                )

            f.write(chunk)

    return written


def validate_image(path: Path) -> None:
    try:
        with Image.open(path) as img:
            img.verify()
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file.")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Image validation failed: {exc}")


def _coalesce_optional_int(value: Optional[int], default: int) -> int:
    return default if value is None else value


# ---------------------------------------------------------
# Auth (long-lived bearer tokens, ~1 year)
# ---------------------------------------------------------

_bearer = HTTPBearer(auto_error=False)


def _auth_enabled() -> bool:
    return bool(settings.auth_api_key.strip())


def _safe_compare(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def issue_token(subject: str, scopes: Optional[List[str]] = None) -> Dict[str, Any]:
    if not _auth_enabled():
        raise HTTPException(
            status_code=503,
            detail="Auth is not configured. Set AUTH_API_KEY in the environment.",
        )

    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(days=settings.auth_token_ttl_days)
    payload = {
        "sub": subject,
        "iat": int(now.timestamp()),
        "exp": int(expires_at.timestamp()),
        "scope": " ".join(scopes or ["ocr"]),
    }
    token = jwt.encode(payload, settings.auth_api_key, algorithm="HS256")
    return {
        "access_token": token,
        "token_type": "Bearer",
        "expires_at": expires_at.isoformat(),
    }


async def require_bearer(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> Dict[str, Any]:
    if not _auth_enabled():
        return {"sub": "anonymous", "scope": "ocr"}

    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="Missing bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.auth_api_key,
            algorithms=["HS256"],
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=401,
            detail="Token expired.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=401,
            detail=f"Invalid token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return payload


def convert_pdf_to_images(pdf_path: Path, out_dir: Path) -> List[Path]:
    """
    Converts PDF pages to PNG using PyMuPDF.
    Install dependency:
        pip install pymupdf
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="PDF support requires pymupdf. Install it with: pip install pymupdf",
        )

    image_paths: List[Path] = []

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not open PDF: {exc}")

    try:
        page_count = len(doc)
        if page_count == 0:
            raise HTTPException(status_code=400, detail="PDF has no pages.")

        if page_count > settings.max_pdf_pages:
            raise HTTPException(
                status_code=413,
                detail=f"PDF has {page_count} pages. Max allowed is {settings.max_pdf_pages}.",
            )

        # 2x zoom gives decent OCR quality without exploding memory.
        matrix = fitz.Matrix(2.0, 2.0)

        for idx in range(page_count):
            page = doc.load_page(idx)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            out_path = out_dir / f"page_{idx + 1}.png"
            pix.save(str(out_path))
            image_paths.append(out_path)

    finally:
        doc.close()

    return image_paths


def load_model() -> None:
    global MODEL, TOKENIZER

    if MODEL is not None and TOKENIZER is not None:
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = settings.cuda_visible_devices

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This app expects an NVIDIA GPU.")

    logger.info("Loading tokenizer: %s", settings.model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        settings.model_name,
        trust_remote_code=True,
    )

    logger.info("Loading model: %s", settings.model_name)

    model_kwargs = {
        "trust_remote_code": True,
        "use_safetensors": True,
    }

    if settings.use_flash_attention:
        model_kwargs["_attn_implementation"] = "flash_attention_2"

    try:
        model = AutoModel.from_pretrained(
            settings.model_name,
            **model_kwargs,
        )
    except Exception as exc:
        if settings.use_flash_attention:
            logger.warning(
                "Flash attention model load failed. Retrying without flash attention. Error: %s",
                exc,
            )
            model_kwargs.pop("_attn_implementation", None)
            model = AutoModel.from_pretrained(
                settings.model_name,
                **model_kwargs,
            )
        else:
            raise

    model = model.eval().cuda().to(torch.bfloat16)

    TOKENIZER = tokenizer
    MODEL = model

    gpu = get_gpu_info()
    logger.info(
        "Model loaded successfully on %s with %s GB VRAM",
        gpu["device_name"],
        gpu["gpu_memory_gb"],
    )


def run_deepseek_ocr(
    image_path: Path,
    output_dir: Path,
    prompt: str,
    base_size: int,
    image_size: int,
    crop_mode: bool,
    test_compress: bool,
) -> str:
    """
    Calls DeepSeek-OCR remote-code infer() method.
    """
    if MODEL is None or TOKENIZER is None:
        raise RuntimeError("Model is not loaded.")

    try:
        with torch.inference_mode():
            result = MODEL.infer(
                TOKENIZER,
                prompt=prompt,
                image_file=str(image_path),
                output_path=str(output_dir),
                base_size=base_size,
                image_size=image_size,
                crop_mode=crop_mode,
                test_compress=test_compress,
                save_results=False,
            )

        if result is None:
            return ""

        if isinstance(result, str):
            return result.strip()

        return str(result).strip()

    except torch.cuda.OutOfMemoryError:
        clear_cuda_cache()
        raise HTTPException(
            status_code=507,
            detail=(
                "GPU out of memory. Try lower BASE_SIZE/IMAGE_SIZE, "
                "for example BASE_SIZE=640 IMAGE_SIZE=640 CROP_MODE=false."
            ),
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.error("OCR inference failed: %s", exc)
        logger.debug(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"OCR inference failed: {exc}")


# ---------------------------------------------------------
# FastAPI lifecycle
# ---------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_temp_dir()

    try:
        load_model()
    except Exception as exc:
        logger.exception("Failed to load model during startup.")
        raise RuntimeError(f"Model startup failed: {exc}") from exc

    yield

    clear_cuda_cache()


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    gpu = get_gpu_info()

    return HealthResponse(
        status="ok" if MODEL is not None else "model_not_loaded",
        model_loaded=MODEL is not None,
        cuda_available=gpu["cuda_available"],
        device_name=gpu["device_name"],
        gpu_memory_gb=gpu["gpu_memory_gb"],
    )


@app.post("/auth/token", response_model=TokenResponse)
async def auth_token(
    api_key: str = Form(...),
    subject: str = Form("api-client"),
) -> TokenResponse:
    """
    Exchange the long-lived AUTH_API_KEY for a ~1-year bearer token.

    Example:
        curl -X POST http://localhost:8000/auth/token \\
          -F "api_key=YOUR_AUTH_API_KEY" \\
          -F "subject=service-a"
    """
    if not _auth_enabled():
        raise HTTPException(
            status_code=503,
            detail="Auth is not configured. Set AUTH_API_KEY in the environment.",
        )

    if not _safe_compare(api_key, settings.auth_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key.")

    issued = issue_token(subject=subject)
    return TokenResponse(
        ok=True,
        access_token=issued["access_token"],
        token_type=issued["token_type"],
        expires_at=issued["expires_at"],
    )


@app.post("/ocr", response_model=OCRResponse)
async def ocr(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
    base_size: Optional[int] = Form(None),
    image_size: Optional[int] = Form(None),
    crop_mode: Optional[bool] = Form(None),
    test_compress: Optional[bool] = Form(None),
    _claims: Dict[str, Any] = Depends(require_bearer),
) -> OCRResponse:
    """
    OCR endpoint.

    Supports:
    - PNG/JPG/JPEG/WEBP/BMP/TIFF
    - PDF, converted page-by-page to images

    Example:
        curl -X POST http://localhost:8000/ocr \
          -F "file=@invoice.pdf"
    """

    started = time.perf_counter()

    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    if not is_image(file.filename, file.content_type) and not is_pdf(file.filename, file.content_type):
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Upload an image or PDF.",
        )

    used_prompt = settings.default_prompt if prompt is None else prompt
    used_base_size = _coalesce_optional_int(base_size, settings.base_size)
    used_image_size = _coalesce_optional_int(image_size, settings.image_size)
    used_crop_mode = settings.crop_mode if crop_mode is None else crop_mode
    used_test_compress = settings.test_compress if test_compress is None else test_compress

    request_dir = Path(tempfile.mkdtemp(prefix="ocr_", dir=settings.temp_dir))
    input_path = request_dir / file.filename
    output_dir = request_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        await save_upload_to_disk(file, input_path)

        if is_pdf(file.filename, file.content_type):
            image_paths = convert_pdf_to_images(input_path, request_dir)
            file_type = "pdf"
        else:
            validate_image(input_path)
            image_paths = [input_path]
            file_type = "image"

        results: List[OCRPageResult] = []

        # DeepSeek-OCR on one 16GB GPU should run one request at a time.
        # This prevents CUDA OOM when multiple users hit the API.
        async with MODEL_LOCK:
            for idx, image_path in enumerate(image_paths, start=1):
                text = await asyncio.wait_for(
                    asyncio.to_thread(
                        run_deepseek_ocr,
                        image_path,
                        output_dir,
                        used_prompt,
                        used_base_size,
                        used_image_size,
                        used_crop_mode,
                        used_test_compress,
                    ),
                    timeout=settings.request_timeout_seconds,
                )

                results.append(OCRPageResult(page=idx, text=text))

        elapsed = round(time.perf_counter() - started, 3)

        return OCRResponse(
            ok=True,
            filename=file.filename,
            file_type=file_type,
            pages=len(results),
            elapsed_seconds=elapsed,
            results=results,
        )

    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"OCR timed out after {settings.request_timeout_seconds} seconds.",
        )

    finally:
        try:
            shutil.rmtree(request_dir, ignore_errors=True)
        except Exception:
            logger.warning("Failed to clean temp directory: %s", request_dir)


@app.exception_handler(HTTPException)
async def http_exception_handler(_, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "ok": False,
            "error": exc.detail,
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_, exc: Exception):
    logger.error("Unhandled error: %s", exc)
    logger.debug(traceback.format_exc())

    return JSONResponse(
        status_code=500,
        content={
            "ok": False,
            "error": "Internal server error.",
            "detail": str(exc),
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
    )
