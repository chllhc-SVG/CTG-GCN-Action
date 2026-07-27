@echo off
setlocal
cd /d "%~dp0\.."

python -c "import torch, numpy" >nul 2>nul
if errorlevel 1 (
    echo ERROR: Missing torch or numpy in this Python environment.
    echo Please activate the correct conda environment first, for example:
    echo   conda activate posec3d
    pause
    exit /b 1
)

python scripts\train_webcam_mediapipe_tcn.py --dataset-dir datasets\webcam_mediapipe_pose_hands --output-dir work_dir\webcam_mediapipe_tcn_pose_hands --labels idle,waving,stop,clapping --feature-mode pose_hands --window-size 64 --batch-size 32 --epochs 80 --lr 0.001 --weight-decay 0.0001
pause
