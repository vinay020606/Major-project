# Edge hospital FL node image. Same base as the central API to keep the compose
# cache hot and avoid torch-wheel drift between services.
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

COPY Model/requirements.txt /tmp/model-reqs.txt
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0 torchvision==0.21.0 \
    && pip install -r /tmp/model-reqs.txt

COPY Model/ /app/Model/

ENV PYTHONPATH=/app \
    NODE_ID=node_A \
    API_SERVER=http://backend:8000

WORKDIR /app/Model

# The edge script reads a dataset from NODE_DATA_DIR if set, otherwise falls
# back to Model/data/hospital_<id>. docker-compose mounts synthetic data in.
# Using unbuffered python -u so logs stream in `docker compose logs -f`.
CMD ["sh", "-c", "python -u run_hospital_node.py --node_id ${NODE_ID} --api_server ${API_SERVER} --demo --dp ${EXTRA_ARGS:-}"]
