import asyncio
import hmac
import logging
import os
import re
import shutil
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import jwt
import torch
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

from config import settings


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("unlimited-ocr-api")

MODEL = None
TOKENIZER = None
MODEL_LOCK = asyncio.Lock()
QURAN_SCRIPT_SKIPPED = "[QURAN_SCRIPT_SKIPPED]"
QURAN_REDACTION = "[آية قرآنية محذوفة - يرجى مراجعة المصدر المعتمد لتحديد السورة ورقم الآية]"


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


def request_id_from(request: Request) -> str:
    value = getattr(request.state, "request_id", None)
    return value or "unknown"


def error_code(status_code: int) -> str:
    codes = {
        400: "bad_request",
        401: "unauthorized",
        403: "forbidden",
        404: "not_found",
        405: "method_not_allowed",
        413: "payload_too_large",
        422: "validation_error",
        500: "internal_server_error",
        503: "service_unavailable",
        504: "timeout",
        507: "insufficient_storage",
    }
    return codes.get(status_code, "http_error")


def error_response(
    request: Request,
    status_code: int,
    message: str,
    *,
    details: Optional[Any] = None,
    headers: Optional[Dict[str, str]] = None,
) -> JSONResponse:
    request_id = request_id_from(request)
    payload: Dict[str, Any] = {
        "ok": False,
        "error": {
            "code": error_code(status_code),
            "message": message,
            "request_id": request_id,
        },
    }
    if details is not None:
        payload["error"]["details"] = details

    response_headers = {"X-Request-ID": request_id}
    if headers:
        response_headers.update(headers)

    return JSONResponse(
        status_code=status_code,
        content=jsonable_encoder(payload),
        headers=response_headers,
    )


def ensure_temp_dir() -> None:
    Path(settings.temp_dir).mkdir(parents=True, exist_ok=True)


def get_gpu_info() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {"cuda_available": False, "device_name": None, "gpu_memory_gb": None}

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


def is_image(filename: str, content_type: Optional[str]) -> bool:
    suffix = Path(filename.lower()).suffix
    return suffix in settings.image_extensions or bool(
        content_type and content_type.startswith("image/")
    )


def is_pdf(filename: str, content_type: Optional[str]) -> bool:
    suffix = Path(filename.lower()).suffix
    return suffix == ".pdf" or content_type == "application/pdf"


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
        with Image.open(path) as img:
            width, height = img.size
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file.")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Image validation failed: {exc}")

    if width < 16 or height < 16:
        raise HTTPException(
            status_code=400,
            detail="Image is too small for OCR. Minimum size is 16x16 pixels.",
        )


def pdf_to_images(pdf_path: Path, output_dir: Path) -> List[Path]:
    try:
        import fitz
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PyMuPDF is not available: {exc}")

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid PDF file: {exc}")

    try:
        if doc.page_count <= 0:
            raise HTTPException(status_code=400, detail="PDF contains no pages.")
        if doc.page_count > settings.max_pdf_pages:
            raise HTTPException(
                status_code=413,
                detail=f"PDF has {doc.page_count} pages. Max allowed is {settings.max_pdf_pages}.",
            )

        matrix = fitz.Matrix(settings.pdf_dpi / 72, settings.pdf_dpi / 72)
        paths: List[Path] = []
        for index, page in enumerate(doc):
            out_path = output_dir / f"page_{index + 1:04d}.png"
            page.get_pixmap(matrix=matrix).save(str(out_path))
            paths.append(out_path)
        return paths
    finally:
        doc.close()


def normalize_prompt() -> str:
    value = settings.default_prompt
    value = value.replace("\\n", "\n").strip()
    if not value:
        raise HTTPException(status_code=400, detail="OCR prompt cannot be empty.")
    return value


def collect_output(out_dir: Path) -> str:
    if not out_dir.exists():
        return ""

    preferred_suffixes = {".txt", ".md"}
    files = sorted(path for path in out_dir.rglob("*") if path.is_file())
    preferred = [path for path in files if path.suffix.lower() in preferred_suffixes]
    candidates = preferred or files

    chunks: List[str] = []
    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError:
            text = path.read_text(errors="ignore").strip()
        except Exception:
            logger.debug("Could not read OCR output file: %s", path)
            continue
        if text:
            chunks.append(text)

    return "\n\n".join(chunks).strip()


def redact_quranic_verses(text: str) -> str:
    if not text:
        return text

    quote_chars = '"“”«»'
    skip_marker = re.escape(QURAN_SCRIPT_SKIPPED)
    trigger = r"(?:قال\s+تعالى|قوله\s+تعالى|قال\s+الله\s+تعالى)"

    patterns = [
        rf"({trigger}\s*[:：]?\s*)[«“\"](.+?)[»”\"]",
        rf"({trigger}\s*[:：]?\s*)'(.+?)'",
        rf"({trigger}\s*[:：]?\s*)([^<\n\r]{{20,}})",
    ]

    redacted = text
    for pattern in patterns:
        redacted = re.sub(
            pattern,
            rf"\1{QURAN_REDACTION}",
            redacted,
            flags=re.DOTALL,
        )

    # Clean accidental leftovers from repeated OCR hallucinations after redaction.
    redacted = re.sub(
        rf"{re.escape(QURAN_REDACTION)}(?:\s*[{re.escape(quote_chars)}]?\s*[\(\[]?\d+[\)\]]?[{re.escape(quote_chars)}]?)+",
        QURAN_REDACTION,
        redacted,
    )

    # Normalize repeated skip markers and remove small hallucinated fragments around them.
    redacted = re.sub(
        rf"{skip_marker}(?:\s*{skip_marker})+",
        QURAN_SCRIPT_SKIPPED,
        redacted,
    )
    redacted = re.sub(
        rf"{skip_marker}(?:\s*[{re.escape(quote_chars)}]?\s*[\(\[]?\d+[\)\]]?[{re.escape(quote_chars)}]?)+",
        QURAN_SCRIPT_SKIPPED,
        redacted,
    )
    redacted = re.sub(
        rf"({skip_marker})(?:\s*(?:[\u06DD\u06DE]|\(\s*\d+\s*\)|\[\s*\d+\s*\]))+",
        QURAN_SCRIPT_SKIPPED,
        redacted,
    )
    return redacted


def load_model() -> None:
    global MODEL, TOKENIZER

    if MODEL is not None and TOKENIZER is not None:
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = settings.cuda_visible_devices
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This app expects an NVIDIA GPU.")

    logger.info("Loading tokenizer: %s", settings.model_name)
    tokenizer = AutoTokenizer.from_pretrained(settings.model_name, trust_remote_code=True)

    logger.info("Loading model: %s", settings.model_name)
    model = AutoModel.from_pretrained(
        settings.model_name,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
    ).eval().cuda()

    TOKENIZER = tokenizer
    MODEL = model

    gpu = get_gpu_info()
    logger.info(
        "Model loaded successfully on %s with %s GB VRAM",
        gpu["device_name"],
        gpu["gpu_memory_gb"],
    )


def run_model_infer(
    image_path: Path,
    prompt: str,
    base_size: int,
    image_size: int,
    crop_mode: bool,
    use_ngram: bool,
    max_length: int,
) -> str:
    if MODEL is None or TOKENIZER is None:
        raise RuntimeError("Model is not loaded.")

    try:
        with tempfile.TemporaryDirectory(prefix="ocr_out_", dir=settings.temp_dir) as out_dir_raw:
            out_dir = Path(out_dir_raw)
            infer_kwargs: Dict[str, Any] = {
                "prompt": f"<image>{prompt}",
                "image_file": str(image_path),
                "output_path": str(out_dir),
                "base_size": base_size,
                "image_size": image_size,
                "crop_mode": crop_mode,
                "max_length": max_length,
                "save_results": True,
            }

            if use_ngram:
                infer_kwargs["no_repeat_ngram_size"] = settings.no_repeat_ngram_size
                infer_kwargs["ngram_window"] = settings.ngram_window

            with torch.inference_mode():
                result = MODEL.infer(TOKENIZER, **infer_kwargs)

            text = collect_output(out_dir)
            if text:
                return redact_quranic_verses(text)
            return redact_quranic_verses("" if result is None else str(result).strip())
    except torch.cuda.OutOfMemoryError:
        clear_cuda_cache()
        raise HTTPException(
            status_code=507,
            detail="GPU out of memory. Try /ocr/long, lower IMAGE_SIZE_LONG, or process fewer pages.",
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("OCR inference failed.")
        raise HTTPException(
            status_code=500,
            detail="OCR inference failed. Check server logs with the response request_id.",
        )


async def ocr_pages(
    pages: List[Path],
    prompt: str,
    base_size: int,
    image_size: int,
    crop_mode: bool,
    use_ngram: bool,
    max_length: int,
) -> List[OCRPageResult]:
    results: List[OCRPageResult] = []

    async with MODEL_LOCK:
        for index, page_path in enumerate(pages, start=1):
            try:
                text = await asyncio.wait_for(
                    asyncio.to_thread(
                        run_model_infer,
                        page_path,
                        prompt,
                        base_size,
                        image_size,
                        crop_mode,
                        use_ngram,
                        max_length,
                    ),
                    timeout=settings.request_timeout_seconds,
                )
            except asyncio.TimeoutError:
                raise HTTPException(
                    status_code=504,
                    detail=f"OCR page {index} timed out after {settings.request_timeout_seconds} seconds.",
                )
            finally:
                clear_cuda_cache()

            results.append(OCRPageResult(page=index, text=text))

    return results


def cleanup_temp_root() -> None:
    temp_root = Path(settings.temp_dir)
    if temp_root.exists() and temp_root.is_dir():
        for child in temp_root.iterdir():
            if child.name.startswith(("ocr_req_", "ocr_out_")):
                shutil.rmtree(child, ignore_errors=True)


_bearer = HTTPBearer(auto_error=False)


def auth_enabled() -> bool:
    return bool(settings.auth_api_key.strip())


def safe_compare(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def issue_token(subject: str, scopes: Optional[List[str]] = None) -> Dict[str, Any]:
    if not auth_enabled():
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
    if not auth_enabled():
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


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_temp_dir()
    load_model()
    try:
        yield
    finally:
        clear_cuda_cache()
        cleanup_temp_root()


app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    request.state.request_id = request_id
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "Unhandled request error. method=%s path=%s request_id=%s",
            request.method,
            request.url.path,
            request_id,
        )
        raise
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    level = logging.ERROR if exc.status_code >= 500 else logging.WARNING
    logger.log(
        level,
        "HTTP error. status=%s method=%s path=%s request_id=%s detail=%s",
        exc.status_code,
        request.method,
        request.url.path,
        request_id_from(request),
        exc.detail,
    )
    return error_response(
        request,
        exc.status_code,
        str(exc.detail),
        headers=exc.headers,
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    logger.warning(
        "Request validation failed. method=%s path=%s request_id=%s errors=%s",
        request.method,
        request.url.path,
        request_id_from(request),
        exc.errors(),
    )
    return error_response(
        request,
        422,
        "Request validation failed.",
        details=exc.errors(),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(
        "Unhandled exception. method=%s path=%s request_id=%s",
        request.method,
        request.url.path,
        request_id_from(request),
    )
    return error_response(
        request,
        500,
        "Internal server error. Check server logs with the response request_id.",
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    gpu = get_gpu_info()
    return HealthResponse(
        status="ok" if MODEL is not None and TOKENIZER is not None else "loading",
        model_loaded=MODEL is not None and TOKENIZER is not None,
        cuda_available=gpu["cuda_available"],
        device_name=gpu["device_name"],
        gpu_memory_gb=gpu["gpu_memory_gb"],
    )


@app.post("/auth/token", response_model=TokenResponse)
async def auth_token(
    api_key: str = Form(...),
    subject: str = Form("api-client"),
) -> TokenResponse:
    if not auth_enabled():
        raise HTTPException(
            status_code=503,
            detail="Auth is not configured. Set AUTH_API_KEY in the environment.",
        )

    if not safe_compare(api_key, settings.auth_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key.")

    token = issue_token(subject)
    return TokenResponse(ok=True, **token)


@app.post("/ocr/base", response_model=OCRResponse)
async def ocr_base(
    file: UploadFile = File(...),
    _: Dict[str, Any] = Depends(require_bearer),
) -> OCRResponse:
    start = time.perf_counter()
    filename = file.filename or "upload"

    if not is_image(filename, file.content_type) and not is_pdf(filename, file.content_type):
        raise HTTPException(status_code=400, detail="/ocr/base accepts image or PDF files.")

    with tempfile.TemporaryDirectory(prefix="ocr_req_", dir=settings.temp_dir) as req_dir_raw:
        req_dir = Path(req_dir_raw)
        upload_path = req_dir / Path(filename).name
        await save_upload_to_disk(file, upload_path)

        if is_pdf(filename, file.content_type):
            pages_dir = req_dir / "pages"
            pages_dir.mkdir(parents=True, exist_ok=True)
            pages = pdf_to_images(upload_path, pages_dir)
            file_type = "pdf"
        else:
            validate_image(upload_path)
            pages = [upload_path]
            file_type = "image"

        results = await ocr_pages(
            pages,
            normalize_prompt(),
            settings.base_size_base,
            settings.image_size_base,
            crop_mode=False,
            use_ngram=True,
            max_length=settings.max_length_base,
        )

    return OCRResponse(
        ok=True,
        filename=filename,
        file_type=file_type,
        pages=len(results),
        elapsed_seconds=round(time.perf_counter() - start, 3),
        results=results,
    )


@app.post("/ocr/long", response_model=OCRResponse)
async def ocr_long(
    file: UploadFile = File(...),
    use_ngram: bool = Form(True),
    _: Dict[str, Any] = Depends(require_bearer),
) -> OCRResponse:
    start = time.perf_counter()
    filename = file.filename or "upload"

    if not is_image(filename, file.content_type) and not is_pdf(filename, file.content_type):
        raise HTTPException(status_code=400, detail="/ocr/long accepts image or PDF files.")

    with tempfile.TemporaryDirectory(prefix="ocr_req_", dir=settings.temp_dir) as req_dir_raw:
        req_dir = Path(req_dir_raw)
        upload_path = req_dir / Path(filename).name
        await save_upload_to_disk(file, upload_path)

        if is_pdf(filename, file.content_type):
            pages_dir = req_dir / "pages"
            pages_dir.mkdir(parents=True, exist_ok=True)
            pages = pdf_to_images(upload_path, pages_dir)
            file_type = "pdf"
        else:
            validate_image(upload_path)
            pages = [upload_path]
            file_type = "image"

        results = await ocr_pages(
            pages,
            normalize_prompt(),
            settings.base_size_long,
            settings.image_size_long,
            crop_mode=True,
            use_ngram=use_ngram,
            max_length=settings.max_length_long,
        )

    return OCRResponse(
        ok=True,
        filename=filename,
        file_type=file_type,
        pages=len(results),
        elapsed_seconds=round(time.perf_counter() - start, 3),
        results=results,
    )
