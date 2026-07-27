@echo off
setlocal
cd /d "%~dp0\.."

python -c "import cv2, yaml; import mediapipe as mp; assert hasattr(mp, 'solutions')" >nul 2>nul
if errorlevel 1 (
    echo Missing or incompatible dependencies. Installing compatible opencv-python mediapipe numpy pyyaml ...
    python -m pip install "numpy==1.26.4" "opencv-python==4.10.0.84" "mediapipe==0.10.14" pyyaml
)

python scripts\collect_webcam_mediapipe_dataset.py --output-dir datasets\webcam_mediapipe_pose_hands --labels idle,waving,stop,clapping --camera 0 --record-seconds 3 --width 640 --height 480 --display-width 960 --display-height 720 --model-complexity 0 --use-hands
pause
