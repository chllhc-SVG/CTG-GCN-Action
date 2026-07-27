@echo off
setlocal
cd /d "%~dp0\.."

echo Starting digital-human MediaPipe TCN WebSocket action-recognition service...
echo WebSocket: ws://127.0.0.1:8765
echo Labels: idle, waving, clapping, stop

python -c "import cv2, mediapipe, torch, numpy, websockets" >nul 2>nul
if errorlevel 1 (
    echo ERROR: Missing runtime dependencies.
    echo Please run: python -m pip install -r requirements-digital-human.txt
    pause
    exit /b 1
)

python scripts\mediapipe_tcn_ws_server.py --host 127.0.0.1 --port 8765 --weights work_dir\webcam_mediapipe_pose_hands_idle_waving_clapping_stop_tcn_pose_hands\best_model.pt --window-size 64 --stride 4 --model-complexity 0 --feature-mode auto --smooth 5 --confidence-threshold 0.55
pause
