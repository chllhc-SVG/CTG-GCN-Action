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
│   │   ├── webcam_realtime_ctrgcn.py
│   │   └── docker_realtime_server.py
│   └── service/
│       ├── host_camera_client.py
│       └── realtime_inference_server.py
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

---

## Workflow 5: Docker service + host camera client

This is the recommended way to run realtime inference when you want Docker for the model stack but still need the host machine to access the camera.

### What runs where

- **Docker container**: MediaPipe pose extraction, CTR-GCN inference, template matching
- **Host machine**: webcam capture, JPEG encoding, request loop, local preview

### Start the Docker service

```bash
cd CTR-GCN
docker compose up --build
```

The service listens on:

- `GET /health`
- `GET /config`
- `POST /infer-frame`

### Start the host camera client

```bash
cd CTR-GCN
python scripts/service/host_camera_client.py --server http://127.0.0.1:8000 --show-preview
```

### How it works

The host client:

1. opens the webcam directly on the host OS
2. encodes each frame as JPEG
3. sends the image to the Docker API
4. receives the prediction JSON
5. shows a local preview with the returned label

This avoids the usual Docker camera-access problems on macOS and Windows.

### Suggested container command

If you want to run the API server directly instead of Compose:

```bash
docker build -t ctrgcn-infer .
docker run --rm -p 8000:8000 \
  -e CTRGCN_WORK_DIR=/app/CTR-GCN/work_dir/webcam_mediapipe/ctrgcn_five_actions \
  -e CTRGCN_LABELS=/app/CTR-GCN/datasets/webcam_mediapipe/labels.json \
  -e CTRGCN_TEMPLATES_DIR=/app/CTR-GCN/templates \
  ctrgcn-infer \
  python scripts/inference/docker_realtime_server.py --host 0.0.0.0 --port 8000
```

### Suggested host client command

```bash
python scripts/service/host_camera_client.py --server http://127.0.0.1:8000 --camera 0 --show-preview
```

---

## Notes for Docker usage

- Make sure your `work_dir` contains a valid `runs-*.pt` checkpoint.
- Make sure `datasets/webcam_mediapipe/labels.json` exists or pass `--labels`.
- If you don't need template matching, you can leave `templates/` empty or omit it.
- On Windows, run Docker Desktop with WSL 2 backend enabled.
- On macOS, the camera stays on the host; the container only handles inference.
