@echo off
setlocal
cd /d "%~dp0\.."

echo ============================================
echo   CTR-GCN Training with MediaPipe Pose
echo ============================================
echo.

python -c "import torch, numpy" >nul 2>nul
if errorlevel 1 (
    echo ERROR: Missing torch or numpy.
    echo Please activate the correct Python environment first.
    pause
    exit /b 1
)

python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

echo.
echo Usage:
echo   run_train_mediapipe_pose.bat [dataset_dir]
echo.
echo Example:
echo   run_train_mediapipe_pose.bat datasets\mediapipe_pose
echo.

set DATASET_DIR=%1
if "%DATASET_DIR%"=="" set DATASET_DIR=datasets\webcam_mediapipe

echo Config: dataset=%DATASET_DIR%
echo.

python main.py --config config\mediapipe_pose\default.yaml --phase train --work-dir work_dir\mediapipe_pose\ctrgcn_three_actions --train-feeder-args data_path=%DATASET_DIR% --test-feeder-args data_path=%DATASET_DIR%

pause
