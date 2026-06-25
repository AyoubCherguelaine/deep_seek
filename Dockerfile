FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3-pip \
    git \
    curl \
    build-essential \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip setuptools wheel

COPY requirements.txt /app/requirements.txt

# RTX 50 / Blackwell: use the CUDA 12.8 PyTorch wheel stack.
# Keep torch out of requirements.txt so pip cannot replace these wheels later.
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

RUN pip install -r /app/requirements.txt

COPY app.py /app/app.py
COPY config.py /app/config.py
COPY .env.example /app/.env.example

RUN mkdir -p /tmp/deepseek_ocr /root/.cache/huggingface

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
