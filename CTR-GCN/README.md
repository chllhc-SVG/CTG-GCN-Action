# CTR-GCN Action Recognition

This repository trains a CTR-GCN model on MediaPipe Pose skeletons for three action classes:

- `idle`
- `waving`
- `clapping`

The current workflow is optimized for a clean skeleton-based pipeline:

```text
raw videos
  -> MediaPipe Pose extraction
  -> skeleton `.npz` dataset
  -> CTR-GCN training
  -> webcam inference
```

## What this project currently uses

- **Pose extractor**: MediaPipe Pose 33-point skeleton
- **Action model**: CTR-GCN
- **Training input**: `.npz` skeleton clips
- **Real-time input**: webcam frames -> MediaPipe -> CTR-GCN

---

## Repository layout

```text
CTR-GCN/
├── config/
│   └── mediapipe_pose/
│       └── default.yaml
├── datasets/
│   └── webcam_mediapipe/
│       ├── idle/
│       ├── waving/
│       ├── clapping/
│       ├── labels.txt
│       └── labels.json
├── feeders/
│   ├── feeder_mediapipe.py
│   └── bone_pairs_mediapipe.py
├── graph/
│   └── mediapipe_pose.py
├── model/
│   └── ctrgcn.py
├── scripts/
│   ├── dataset/
│   │   └── collect_webcam_mediapipe.py
│   ├── inference/
│   │   └── webcam_realtime_ctrgcn.py
│   └── pipelines/
│       └── extract_mediapipe_from_videos.py
├── work_dir/
│   └── webcam_mediapipe/
│       └── ctrgcn_three_actions/
├── main.py
└── requirements.txt
```

---

## Data format

Each training sample is stored as a compressed `.npz` file containing a full action clip.

Typical fields:

- `landmarks`: `[T, 33, 4]` 
- `valid_frames`: `[T]`
- `label`: action id
- `label_name`: action name
- `fps`: source video FPS
- `schema`: skeleton schema name

The training feeder converts this into CTR-GCN input format. The model keeps two person slots for compatibility with the CTR-GCN implementation; webcam samples use the first slot and leave the second slot as zeros.

---

## Recommended data rules

For best performance, each clip should satisfy:

- one person only
- full body in frame
- one clear action per clip
- stable camera
- clear hands and feet visibility when possible
- short clips, about `2~4s`

This matters a lot for skeleton action recognition.

---

## Workflow 1: collect new webcam data

Use this when you want to record your own action clips and save skeleton `.npz` samples.

```powershell
cd D:\B\python\xiaoke-project\ctrgcn-action-project\CTR-GCN
python scripts\dataset\collect_webcam_mediapipe.py
```

### Default labels

The collector uses these labels by default:

- `idle`
- `waving`
- `clapping`

### Controls

- `0 / 1 / 2` select a label
- `R`, `Space`, or `Enter` start recording
- `Q` or `Esc` quit

### Output

Samples are written to:

```text
datasets/webcam_mediapipe/<label>/<label>_0001.npz
```

The script also writes:

```text
datasets/webcam_mediapipe/labels.txt
datasets/webcam_mediapipe/labels.json
```

---

## Workflow 2: convert existing raw videos to skeletons

Use this when you already have raw `.mp4` clips under class folders.

Expected input structure:

```text
D:\B\python\xiaoke-project\dataset\
├── idle\
├── waving\
└── clapping\
```

Run:

```powershell
cd D:\B\python\xiaoke-project\ctrgcn-action-project\CTR-GCN
python scripts\pipelines\extract_mediapipe_from_videos.py ^
  --input-dir D:\B\python\xiaoke-project\dataset ^
  --output-dir D:\B\python\xiaoke-project\ctrgcn-action-project\CTR-GCN\datasets\webcam_mediapipe ^
  --labels idle,waving,clapping ^
  --save-visualization
```

### Output

The converter creates one `.npz` file per video:

```text
datasets/webcam_mediapipe/<label>/<video_name>.npz
```

If `--save-visualization` is enabled, it also stores a preview `.mp4` next to each sample.

---

## Workflow 3: train CTR-GCN

The default training config is already aligned to the three-class MediaPipe dataset.

Run:

```powershell
cd D:\B\python\xiaoke-project\ctrgcn-action-project\CTR-GCN
D:\B\Anaconda\envs\posec3d\python.exe main.py --config config\mediapipe_pose\default.yaml --phase train
```

### Training output

The default work directory is:

```text
work_dir/webcam_mediapipe/ctrgcn_three_actions
```

Training logs, checkpoints, and evaluation results will be written there.

---

## Workflow 4: run real-time webcam inference

After training, launch the webcam demo:

```powershell
cd D:\B\python\xiaoke-project\ctrgcn-action-project\CTR-GCN
python scripts\inference\webcam_realtime_ctrgcn.py ^
  --work-dir .\work_dir\webcam_mediapipe\ctrgcn_three_actions ^
  --labels .\datasets\webcam_mediapipe\labels.json ^
  --camera 0 ^
  --window-size 64 ^
  --stride 4 ^
  --device cpu
```

### What it does

- opens the webcam
- extracts MediaPipe Pose landmarks every frame
- keeps a sliding 64-frame window
- runs CTR-GCN on the window
- displays the predicted label on screen

For smoother realtime preview, the inference window is intentionally kept at a moderate display size and the MediaPipe model complexity is set low by default during collection.

### Exit keys

- `Q`
- `Esc`

---

## Label order

Keep the label order consistent everywhere:

```text
0 = idle
1 = waving
2 = clapping
```

The same order must be used by:

- `labels.txt`
- `labels.json`
- training config
- inference script

---

## Training config

Main configuration file:

```text
config/mediapipe_pose/default.yaml
```

Important settings:

- `num_class: 3`
- `num_point: 33`
- `window_size: 64`
- `normalization: True`
- `data_path: datasets/webcam_mediapipe`

---

## Notes on dataset quality

If the model feels inaccurate, the most common causes are:

- too few samples per class
- clips are too long or contain multiple actions
- pose is partially out of frame
- camera angle differs a lot from training data
- class balance is uneven
- raw videos are noisy or inconsistent

For this project, a small but clean dataset usually works better than a larger but messy one.

---

## Suggested next step

If you want the best results, collect more clips so each class has at least:

- `30` samples minimum
- `50+` samples is better

Keep each clip short and visually consistent.

---

## Step-by-step operation checklist

### 1) Activate the correct environment and enter the repo

```powershell
cd D:\B\python\xiaoke-project\ctrgcn-action-project\CTR-GCN
```

Make sure the environment has the required packages installed, especially:

- `torch`
- `numpy`
- `opencv-python`
- `mediapipe`
- `pyyaml`
- `tqdm`
- `tensorboardX`
- `scikit-learn`

For better real-time performance:

- use `--model-complexity 0` when collecting data
- use `--device cuda` with the verified `posec3d` environment on the RTX 3050
- keep the webcam window resolution around `1280x720` for capture and let the scripts fit the image to the screen while preserving aspect ratio
- use a fixed camera and simple background

---

### 2) Collect your own webcam dataset

Run:

```powershell
D:\B\Anaconda\envs\posec3d\python.exe scripts\dataset\collect_webcam_mediapipe.py --output-dir datasets\webcam_mediapipe --labels idle,waving,clapping --camera 0 --record-seconds 3 --width 1280 --height 720 --model-complexity 0 --save-video
```

During collection:

- `0` selects `idle`
- `1` selects `waving`
- `2` selects `clapping`
- `R` toggles continuous recording on/off
- `Q` or `Esc` exits

Recommended recording rules:

- one person only
- full body fully visible
- one clear action per clip
- stable camera
- 2~4 seconds per clip
- keep the action centered in frame

---

### 3) Check the collected dataset

After recording, confirm that the dataset folder contains:

```text
datasets/webcam_mediapipe/
├── idle/
├── waving/
└── clapping/
```

Also confirm that these files exist:

```text
datasets/webcam_mediapipe/labels.txt
datasets/webcam_mediapipe/labels.json
```

The label order must be:

```text
0 = idle
1 = waving
2 = clapping
```

---

### 4) Train CTR-GCN

Run:

```powershell
python main.py --config config\mediapipe_pose\default.yaml --phase train
```

Training uses:

- `config/mediapipe_pose/default.yaml`
- `datasets/webcam_mediapipe`
- `work_dir/webcam_mediapipe/ctrgcn_three_actions`

If training succeeds, the work directory should contain:

- `runs-*.pt`
- `log.txt`
- `config.yaml`

---

### 5) Run real-time webcam inference

After training, launch:

```powershell
D:\B\Anaconda\envs\posec3d\python.exe scripts\inference\webcam_realtime_ctrgcn.py --work-dir .\work_dir\webcam_mediapipe\ctrgcn_three_actions --labels .\datasets\webcam_mediapipe\labels.json --camera 0 --window-size 64 --stride 4 --display-width 1280 --display-height 720 --device cuda
```

During inference:

- the webcam is opened
- MediaPipe extracts pose landmarks frame by frame
- the last 64 frames are fed to CTR-GCN
- the predicted label is shown on screen

Exit with:

- `Q`
- `Esc`

---

### 6) If you already have raw `.mp4` videos

If you first recorded raw videos into class folders, use the extraction pipeline instead:

```powershell
python scripts\pipelines\extract_mediapipe_from_videos.py --input-dir D:\B\python\xiaoke-project\dataset --output-dir D:\B\python\xiaoke-project\ctrgcn-action-project\CTR-GCN\datasets\webcam_mediapipe --labels idle,waving,clapping --save-visualization
```

This converts raw clips into the same `.npz` format used by training.

---

### 7) If the accuracy is low

Usually the cause is one or more of these:

- too few samples per class
- clips contain multiple actions
- body is partially out of frame
- camera angle is inconsistent
- labels are imbalanced
- the pose detector misses too many frames

If this happens, collect more clean clips before changing the model.

---

## Requirements

Use the Python environment that already has:

- `torch`
- `numpy`
- `opencv-python`
- `mediapipe`
- `pyyaml`
- `tqdm`
- `tensorboardX`
- `scikit-learn`

---

## Quick start summary

```text
1. Collect or convert videos into datasets/webcam_mediapipe/
2. Train with main.py and config/mediapipe_pose/default.yaml
3. Run webcam_realtime_ctrgcn.py for live recognition
```
