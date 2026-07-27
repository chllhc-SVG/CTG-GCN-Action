FROM python:3.10-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libgomp1; \
    rm -rf /var/lib/apt/lists/*

COPY requirements-digital-human.txt ./requirements-digital-human.txt
RUN python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && python -m pip install --timeout 120 --retries 10 torch==2.2.2+cpu --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --timeout 120 --retries 10 -r requirements-digital-human.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY scripts ./scripts
COPY work_dir/webcam_mediapipe_pose_hands_idle_waving_clapping_stop_tcn_pose_hands ./work_dir/webcam_mediapipe_pose_hands_idle_waving_clapping_stop_tcn_pose_hands

EXPOSE 8765

CMD ["python", "scripts/mediapipe_tcn_ws_server.py", "--weights", "work_dir/webcam_mediapipe_pose_hands_idle_waving_clapping_stop_tcn_pose_hands/best_model.pt", "--host", "0.0.0.0", "--port", "8765", "--feature-mode", "pose_hands", "--window-size", "64", "--stride", "4", "--model-complexity", "0", "--smooth", "5", "--confidence-threshold", "0.55", "--cpu"]
