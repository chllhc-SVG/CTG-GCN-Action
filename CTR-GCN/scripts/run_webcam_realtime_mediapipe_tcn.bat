@echo off
setlocal
cd /d "%~dp0\.."

python -c "import cv2, mediapipe, torch, numpy" >nul 2>nul
if errorlevel 1 (
    echo ERROR: Missing dependencies in this Python environment.
    echo Please activate the correct conda environment first, for example:
    echo   conda activate posec3d
    pause
    exit /b 1
)

python scripts\webcam_realtime_mediapipe_tcn.py --weights work_dir\webcam_mediapipe_tcn_pose_hands\best_model.pt --camera 0 --window-size 64 --stride 4 --width 640 --height 480 --display-width 960 --display-height 720 --model-complexity 0 --feature-mode auto --smooth 5 --confidence-threshold 0.55
pause
