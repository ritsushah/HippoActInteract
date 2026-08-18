# HippoActInteract runtime for Docker Desktop on Apple Silicon.
#
# This image is linux/arm64. Metal (MPS) is a macOS API and cannot run inside
# Docker's Linux VM, so in-container compute uses CPU.
#
# Reclaim disk:
#   docker compose down --rmi all -v --remove-orphans

FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HIPPO_DEVICE=cpu \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install CPU torch from the official CPU index FIRST.
# PyPI linux/aarch64 torch>=2.13 pulls NVIDIA CUDA wheels (~2GB) that
# do not run in Docker Desktop and will exhaust disk on a 16GB M2.
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY src ./src
COPY tests ./tests
COPY pytest.ini ./pytest.ini

CMD ["pytest", "tests/test_device.py", "-v", "--log-cli-level=INFO"]
