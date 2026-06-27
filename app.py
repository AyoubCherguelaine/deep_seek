import asyncio
import hmac
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import types
import warnings
from contextlib import asynccontextmanager
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import jwt
import torch
import torch.nn.functional as F
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

warnings.filterwarnings(
    "ignore",
    message=r".*seen_tokens.*deprecated.*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*get_max_cache\(\).*deprecated.*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*attention layers in this model are transitioning.*",
)


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


def get_cuda_capability() -> Optional[tuple[int, int]]:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_capability(0)


def _flash_attention_importable() -> bool:
    try:
        import flash_attn  # noqa: F401
        return True
    except Exception as exc:
        logger.info("flash-attn is not importable: %s", exc)
        return False


def _install_flash_attention() -> bool:
    logger.warning(
        "ALLOW_FLASH_ATTENTION_INSTALL=true. Attempting runtime install of flash-attn==2.7.3."
    )
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "flash-attn==2.7.3",
                "--no-build-isolation",
            ],
            check=True,
        )
    except Exception as exc:
        logger.warning("flash-attn runtime install failed: %s", exc)
        return False

    return _flash_attention_importable()


def select_attention_backend() -> str:
    capability = get_cuda_capability()
    if capability is None:
        raise RuntimeError("CUDA is not available. This app expects an NVIDIA GPU.")

    major, minor = capability
    cc = major + minor / 10

    logger.info(
        "Detected CUDA device: %s | compute capability: %s.%s | torch: %s | torch CUDA: %s",
        torch.cuda.get_device_name(0),
        major,
        minor,
        torch.__version__,
        torch.version.cuda,
    )

    if not settings.use_flash_attention:
        logger.info("USE_FLASH_ATTENTION=false. Using eager attention backend.")
        return "eager"

    if cc < 8.0:
        logger.info("Compute capability %.1f does not support flash_attention_2. Using eager.", cc)
        return "eager"

    if _flash_attention_importable():
        logger.info("flash-attn is importable. Using flash_attention_2.")
        return "flash_attention_2"

    if settings.allow_flash_attention_install and _install_flash_attention():
        logger.info("flash-attn installed and importable. Using flash_attention_2.")
        return "flash_attention_2"

    logger.warning("FlashAttention requested but unavailable. Using eager attention backend.")
    return "eager"


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


def _normalize_prompt(value: Optional[str]) -> str:
    prompt = settings.default_prompt if value is None else value
    prompt = prompt.replace("\\n", "\n").strip()

    if not prompt or prompt.lower() in {"none", "null"}:
        prompt = settings.default_prompt.replace("\\n", "\n").strip()

    if not prompt:
        raise HTTPException(status_code=400, detail="OCR prompt cannot be empty.")

    return prompt + " "


def _validate_positive_int(name: str, value: int) -> int:
    if value <= 0:
        raise HTTPException(status_code=400, detail=f"{name} must be greater than 0.")
    return value


def _patch_generation_defaults(model: Any, tokenizer: Any) -> None:
    if getattr(model, "_deepseek_ocr_api_generate_patched", False):
        return

    original_generate: Callable[..., Any] = model.generate

    def generate_with_defaults(*args: Any, **kwargs: Any) -> Any:
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]

        if "attention_mask" not in kwargs and input_ids is not None:
            kwargs["attention_mask"] = torch.ones_like(input_ids, dtype=torch.long)

        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        eos_token_id = kwargs.get("eos_token_id", getattr(tokenizer, "eos_token_id", None))

        if kwargs.get("pad_token_id") is None and pad_token_id is not None:
            kwargs["pad_token_id"] = pad_token_id

        if kwargs.get("eos_token_id") is None and eos_token_id is not None:
            kwargs["eos_token_id"] = eos_token_id

        do_sample = kwargs.setdefault("do_sample", False)
        if not do_sample and kwargs.get("temperature") == 0.0:
            kwargs.pop("temperature")

        return original_generate(*args, **kwargs)

    model.generate = generate_with_defaults
    model._deepseek_ocr_api_generate_patched = True


def _patch_dynamic_vision_queries(model: Any) -> None:
    if getattr(model, "_deepseek_ocr_api_vision_queries_patched", False):
        return

    patched = False
    for name, module in model.named_modules():
        if module.__class__.__name__ != "Qwen2Decoder2Encoder":
            continue

        def fixed_forward(self: Any, x: torch.Tensor) -> torch.Tensor:
            x = x.flatten(2).transpose(1, 2)
            batch_size, n_query, _ = x.shape

            if n_query == 144:
                param_img = self.query_768.weight
            elif n_query == 256:
                param_img = self.query_1024.weight
            else:
                weight_ref = self.query_1024.weight.permute(1, 0).unsqueeze(0).to(x.dtype)
                param_img = F.interpolate(
                    weight_ref,
                    size=n_query,
                    mode="linear",
                    align_corners=False,
                )
                param_img = param_img.squeeze(0).permute(1, 0)

            batch_query_imgs = param_img.unsqueeze(0).expand(batch_size, -1, -1)
            x_combined = torch.cat([x, batch_query_imgs], dim=1)
            token_type_ids = torch.cat(
                [
                    torch.zeros(batch_size, n_query, dtype=torch.long, device=x.device),
                    torch.ones(batch_size, n_query, dtype=torch.long, device=x.device),
                ],
                dim=1,
            )
            y = self.model(x_combined, token_type_ids)[0]
            return y[:, n_query:, :]

        module.forward = types.MethodType(fixed_forward, module)
        logger.info("Patched DeepSeek-OCR dynamic vision queries on module: %s", name)
        patched = True

    if not patched:
        logger.warning("DeepSeek-OCR dynamic vision query patch target was not found.")

    model._deepseek_ocr_api_vision_queries_patched = True


def _clean_ocr_text(text: str) -> str:
    text = re.sub(r"<\|ref\|>.*?<\|/ref\|>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|det\|>.*?<\|/det\|>", "", text, flags=re.DOTALL)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _read_saved_ocr_text(output_dir: Path) -> str:
    if not output_dir.exists():
        return ""

    text_files = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".md", ".txt", ".json"}
    )

    chunks: List[str] = []
    for path in text_files:
        try:
            content = path.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError:
            content = path.read_text(errors="ignore").strip()
        except Exception:
            logger.debug("Could not read OCR output file: %s", path)
            continue

        if content:
            chunks.append(content)

    return "\n\n".join(chunks).strip()


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

    attn_impl = select_attention_backend()
    logger.info("Loading model: %s with attention backend: %s", settings.model_name, attn_impl)

    model_kwargs = {
        "trust_remote_code": True,
        "use_safetensors": True,
        "attn_implementation": attn_impl,
        "torch_dtype": torch.bfloat16,
        "device_map": {"": "cuda:0"},
    }

    try:
        model = AutoModel.from_pretrained(
            settings.model_name,
            **model_kwargs,
        )
    except Exception as exc:
        if attn_impl == "eager":
            raise

        logger.warning(
            "Model load failed with attention backend %s. Retrying with eager. Error: %s",
            attn_impl,
            exc,
        )
        model_kwargs["attn_implementation"] = "eager"
        model = AutoModel.from_pretrained(
            settings.model_name,
            **model_kwargs,
        )

    model = model.eval()
    _patch_generation_defaults(model, tokenizer)
    _patch_dynamic_vision_queries(model)

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
        result = None
        captured_output = StringIO()
        crop_attempts = [crop_mode]
        fallback_crop_mode = not crop_mode
        if fallback_crop_mode not in crop_attempts:
            crop_attempts.append(fallback_crop_mode)

        for attempt_crop_mode in crop_attempts:
            try:
                with torch.inference_mode():
                    with redirect_stdout(captured_output):
                        result = MODEL.infer(
                            TOKENIZER,
                            prompt=prompt,
                            image_file=str(image_path),
                            output_path=str(output_dir),
                            base_size=base_size,
                            image_size=image_size,
                            crop_mode=attempt_crop_mode,
                            test_compress=test_compress,
                            save_results=settings.save_ocr_results,
                            eval_mode=True,
                        )
                break
            except UnboundLocalError as exc:
                if "param_img" not in str(exc) or attempt_crop_mode == crop_attempts[-1]:
                    raise

                logger.warning(
                    "DeepSeek-OCR failed with crop_mode=%s (%s). Retrying with crop_mode=%s.",
                    str(attempt_crop_mode).lower(),
                    exc,
                    str(crop_attempts[-1]).lower(),
                )

        text = ""
        if result is not None:
            text = result if isinstance(result, str) else str(result)

        if not text.strip():
            text = _read_saved_ocr_text(output_dir)

        if not text.strip():
            text = captured_output.getvalue()

        return _clean_ocr_text(text)

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


@app.get("/portal-resolver", include_in_schema=False)
async def portal_resolver() -> Dict[str, Any]:
    return {
        "ok": True,
        "service": settings.app_name,
        "health": "/health",
        "ocr": "/ocr",
    }


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
    files: Optional[List[UploadFile]] = File(None),
    file: Optional[UploadFile] = File(None),
    _claims: Dict[str, Any] = Depends(require_bearer),
) -> OCRResponse:
    """
    OCR endpoint.

    Supports:
    - PNG/JPG/JPEG/WEBP/BMP/TIFF

    Example:
        curl -X POST http://localhost:8000/ocr \
          -F "file=@image.png"
    """

    started = time.perf_counter()

    uploaded_files = list(files or [])
    if file is not None:
        uploaded_files.append(file)

    if not uploaded_files:
        raise HTTPException(status_code=400, detail="One image file is required.")

    if len(uploaded_files) > 1:
        raise HTTPException(status_code=400, detail="Only one image file is allowed per request.")

    used_prompt = _normalize_prompt(None)
    used_base_size = _validate_positive_int(
        "base_size",
        settings.base_size,
    )
    used_image_size = _validate_positive_int(
        "image_size",
        settings.image_size,
    )
    used_crop_mode = settings.crop_mode
    used_test_compress = settings.test_compress

    request_dir = Path(tempfile.mkdtemp(prefix="ocr_", dir=settings.temp_dir))
    output_dir = request_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        results: List[OCRPageResult] = []
        upload = uploaded_files[0]

        # DeepSeek-OCR on one 16GB GPU should run one request at a time.
        # This prevents CUDA OOM when multiple users hit the API.
        async with MODEL_LOCK:
            if not upload.filename:
                raise HTTPException(status_code=400, detail="Missing filename.")

            if not is_image(upload.filename, upload.content_type):
                raise HTTPException(
                    status_code=415,
                    detail="Unsupported file type. Upload images only.",
                )

            safe_filename = Path(upload.filename).name
            input_path = request_dir / safe_filename
            page_output_dir = output_dir / "image_1"
            page_output_dir.mkdir(parents=True, exist_ok=True)

            await save_upload_to_disk(upload, input_path)
            validate_image(input_path)

            text = await asyncio.wait_for(
                asyncio.to_thread(
                    run_deepseek_ocr,
                    input_path,
                    page_output_dir,
                    used_prompt,
                    used_base_size,
                    used_image_size,
                    used_crop_mode,
                    used_test_compress,
                ),
                timeout=settings.request_timeout_seconds,
            )

            results.append(OCRPageResult(page=1, text=text))

        elapsed = round(time.perf_counter() - started, 3)

        return OCRResponse(
            ok=True,
            filename=uploaded_files[0].filename or "",
            file_type="image",
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
