# Central PulmoScan FL API (FastAPI + orchestrator).
# Python 3.11 is the newest line with wheels for torch 2.6.
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libgl1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install backend deps first for cache efficiency.
COPY backend/requirements.txt /tmp/backend-reqs.txt
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0 torchvision==0.21.0 \
    && pip install -r /tmp/backend-reqs.txt \
    && pip install opacus>=1.4.0

# Copy source. The orchestrator reaches into ../Model so we include both.
COPY backend/ /app/backend/
COPY Model/ /app/Model/

ENV PYTHONPATH=/app \
    FL_MIN_CLIENTS=3 \
    ALLOWED_ORIGINS=http://localhost:5173,http://127.0.0.1:5173

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
