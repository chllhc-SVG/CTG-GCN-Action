FROM m.daocloud.io/docker.io/library/python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app/CTR-GCN

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
COPY scripts/service/requirements-docker.txt ./scripts/service/requirements-docker.txt

RUN python -m pip install --upgrade pip \
    && pip install -r scripts/service/requirements-docker.txt

COPY . .

EXPOSE 8000

CMD ["python", "scripts/service/realtime_inference_server.py", "--host", "0.0.0.0", "--port", "8000"]
