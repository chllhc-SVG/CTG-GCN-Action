"""
HTTP inference service for host-captured webcam frames.

The host machine keeps control of camera access. It captures frames, encodes them
as JPEG, and sends them to this service running inside Docker. The service then
runs MediaPipe pose extraction, CTR-GCN recognition, and optional template-based
quality matching.

Endpoints:
- GET /health
- POST /infer-frame
- GET /state

Request body for /infer-frame:
{
  "image": "<base64-encoded jpeg>",
  "mirror": false
}

The response contains the current prediction and quality estimate. The service
keeps a short sliding window in memory so callers can stream frames one by one.
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import json
import sys
import time
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TORCHLIGHT_ROOT = ROOT / "torchlight"
if str(TORCHLIGHT_ROOT) not in sys.path:
    sys.path.insert(0, str(TORCHLIGHT_ROOT))

from model.ctrgcn import Model
from scripts.inference.template_matcher import TemplateMatcher

cv2.setUseOptimized(True)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

if not hasattr(mp, "solutions"):
    raise RuntimeError(
        "This service requires a MediaPipe version that exposes mp.solutions."
    )

NUM_LANDMARKS = 33
POSE_CONNECTIONS = tuple(mp.solutions.pose.POSE_CONNECTIONS)
POSE_LEFT_HIP = 23
POSE_RIGHT_HIP = 24
POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12

DEFAULT_WORK_DIR = ROOT / "work_dir" / "webcam_mediapipe" / "ctrgcn_five_actions"
DEFAULT_LABELS_CANDIDATES = [
    ROOT / "datasets" / "webcam_mediapipe" / "labels.json",
    ROOT / "datasets" / "mediapipe_pose" / "labels.json",
]


@dataclass
class InferenceState:
    predicted_label: str = "---"
    confidence: float = 0.0
    raw_label: str = "---"
    raw_confidence: float = 0.0
    quality_label: str | None = None
    quality_distance: float | None = None
    fps: float = 0.0
    buffer_fill: int = 0
    frame_count: int = 0
    last_update_ts: float = 0.0


def load_labels(labels_path: str | Path) -> dict[int, str]:
    with open(labels_path, "r", encoding="utf-8") as f:
        label_dict = json.load(f)
    return {int(k): v for k, v in label_dict.items()}


def find_best_weights(work_dir: str | Path) -> str | None:
    weights_dir = Path(work_dir)
    if not weights_dir.is_dir():
        return None
    weights = list(weights_dir.glob("runs-*.pt"))
    if not weights:
        return None

    def checkpoint_key(path: Path) -> tuple[int, int]:
        parts = path.stem.split("-")
        try:
            return int(parts[1]), int(parts[2])
        except (IndexError, ValueError):
            return -1, -1

    return str(sorted(weights, key=checkpoint_key)[-1])


def load_model(weights_path: str | Path, num_class: int, device: torch.device) -> torch.nn.Module:
    model = Model(
        num_class=num_class,
        num_point=33,
        num_person=2,
        graph="graph.mediapipe_pose.Graph",
        graph_args={"labeling_mode": "spatial"},
        in_channels=3,
    )
    weights = torch.load(weights_path, map_location=device)
    model.load_state_dict(weights)
    model = model.to(device)
    model.eval()
    return model


def normalize(sequence: np.ndarray) -> np.ndarray:
    seq = sequence.copy().astype(np.float32)
    left_hip = seq[:, POSE_LEFT_HIP, :]
    right_hip = seq[:, POSE_RIGHT_HIP, :]
    left_shoulder = seq[:, POSE_LEFT_SHOULDER, :]
    right_shoulder = seq[:, POSE_RIGHT_SHOULDER, :]
    hip_center = (left_hip + right_hip) / 2.0
    shoulder_center = (left_shoulder + right_shoulder) / 2.0
    torso = np.linalg.norm(shoulder_center - hip_center, axis=1)
    scale = float(np.nanmedian(torso))
    if not np.isfinite(scale) or scale < 1e-4:
        scale = 1.0
    seq = (seq - hip_center[:, None, :]) / scale
    seq[~np.isfinite(seq)] = 0.0
    return seq


def landmarks_to_model_input(landmarks_seq: list[np.ndarray], window_size: int = 64, p_interval: list[float] | None = None) -> np.ndarray:
    if p_interval is None:
        p_interval = [0.95]

    pose = np.stack(landmarks_seq, axis=0).astype(np.float32)
    pose = pose[:, :, :3]
    pose = normalize(pose)

    data = pose.transpose(2, 0, 1)
    data = data[:, :, :, np.newaxis]

    if data.shape[1] < window_size:
        pad = np.zeros((3, window_size, NUM_LANDMARKS, 1), dtype=np.float32)
        pad[:, :data.shape[1], :, :] = data
        data = pad
    elif data.shape[1] > window_size:
        begin = (data.shape[1] - window_size) // 2
        data = data[:, begin:begin + window_size, :, :]

    valid_frame_num = np.sum(data.sum(0).sum(-1).sum(-1) != 0)
    if valid_frame_num == 0:
        valid_frame_num = 1
    data = tools.valid_crop_resize(data, valid_frame_num, p_interval, window_size)

    if data.shape[3] == 1:
        data = np.concatenate([data, np.zeros_like(data)], axis=3)

    return data.astype(np.float32)


def motion_energy(landmarks_seq: list[np.ndarray], recent_frames: int = 12) -> float:
    if len(landmarks_seq) < 2:
        return 0.0

    seq = np.asarray(landmarks_seq[-max(2, recent_frames):], dtype=np.float32)[:, :, :3]
    seq = normalize(seq)
    diffs = np.linalg.norm(seq[1:] - seq[:-1], axis=2)
    if diffs.size == 0:
        return 0.0
    return float(np.median(np.mean(diffs, axis=1)))


def decode_frame_from_request(payload: dict[str, Any]) -> np.ndarray:
    encoded = payload.get("image")
    if not encoded:
        raise ValueError("Missing 'image' field")
    raw = base64.b64decode(encoded)
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Unable to decode image")
    return frame


def draw_pose(frame: np.ndarray, results: Any) -> None:
    if not results.pose_landmarks:
        return
    h, w = frame.shape[:2]
    points = []
    for lm in results.pose_landmarks.landmark:
        if lm.visibility < 0.35:
            points.append(None)
        else:
            points.append((int(lm.x * w), int(lm.y * h)))
    for start, end in POSE_CONNECTIONS:
        if points[start] is not None and points[end] is not None:
            cv2.line(frame, points[start], points[end], (64, 220, 255), 2, cv2.LINE_AA)
    for pt in points:
        if pt is not None:
            cv2.circle(frame, pt, 3, (0, 255, 80), -1, cv2.LINE_AA)


def _set_windows_hidpi_aware() -> None:
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def _load_state_from_args(args: argparse.Namespace) -> tuple[dict[int, str], torch.device, torch.nn.Module, TemplateMatcher, mp.solutions.pose.Pose, deque[np.ndarray], InferenceState]:
    labels_path = args.labels
    if labels_path is None:
        for candidate in DEFAULT_LABELS_CANDIDATES:
            if candidate.exists():
                labels_path = str(candidate)
                break
    if labels_path is None or not Path(labels_path).exists():
        raise FileNotFoundError("labels.json not found")

    weights_path = args.weights or find_best_weights(args.work_dir)
    if weights_path is None or not Path(weights_path).exists():
        raise FileNotFoundError("model weights not found")

    id_to_label = load_labels(labels_path)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = load_model(weights_path, len(id_to_label), device)
    template_matcher = TemplateMatcher(args.templates_dir, threshold=args.quality_threshold)
    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=args.model_complexity,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    buffer = deque(maxlen=args.window_size)
    state = InferenceState()
    return id_to_label, device, model, template_matcher, pose, buffer, state


def build_http_handler(args: argparse.Namespace):
    id_to_label, device, model, template_matcher, pose, landmarks_buffer, state = _load_state_from_args(args)
    label_to_id = {v: k for k, v in id_to_label.items()}
    fps_clock_start = time.perf_counter()
    fps_frames = 0
    stable_label = "---"
    stable_count = 0
    final_label_history = deque(maxlen=max(7, args.final_debounce))
    idle_hold_count = 0
    stable_non_idle_label = "---"
    stable_non_idle_count = 0
    frame_count = 0

    class Handler(BaseHTTPRequestHandler):
        server_version = "CTRGCNFrameServer/1.0"

        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._send_json({"ok": True})
                return
            if self.path == "/state":
                self._send_json(asdict(state))
                return
            self._send_json({"error": "not found"}, status=404)

        def do_POST(self) -> None:  # noqa: N802
            nonlocal fps_clock_start, fps_frames, stable_label, stable_count, idle_hold_count, stable_non_idle_label, stable_non_idle_count, frame_count
            if self.path != "/infer-frame":
                self._send_json({"error": "not found"}, status=404)
                return

            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0:
                self._send_json({"error": "empty body"}, status=400)
                return

            try:
                payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
                frame = decode_frame_from_request(payload)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)
                return

            if payload.get("mirror"):
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                lm = np.asarray([[l.x, l.y, l.z, l.visibility] for l in results.pose_landmarks.landmark], dtype=np.float32)
                landmarks_buffer.append(lm)
            else:
                landmarks_buffer.append(np.zeros((NUM_LANDMARKS, 4), dtype=np.float32))

            frame_count += 1
            raw_label = state.raw_label
            raw_confidence = state.raw_confidence
            quality_label = None
            quality_distance = None

            if len(landmarks_buffer) >= 16 and (frame_count % args.stride == 0):
                lm_seq = list(landmarks_buffer)
                data = landmarks_to_model_input(lm_seq, window_size=args.window_size)
                data_tensor = torch.from_numpy(data).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(data_tensor)
                    probs = torch.softmax(output, dim=1)

                probs_np = probs[0].cpu().numpy()
                top_idx = np.argsort(probs_np)[::-1]
                candidate_label = id_to_label[top_idx[0]]
                candidate_confidence = float(probs_np[top_idx[0]])
                candidate_top3 = [(id_to_label[top_idx[i]], float(probs_np[top_idx[i]])) for i in range(min(3, len(id_to_label)))]
                current_motion = motion_energy(lm_seq)
                raw_label = candidate_label
                raw_confidence = candidate_confidence
                idle_confidence = float(probs_np[label_to_id.get("idle", top_idx[0])]) if "idle" in label_to_id else 0.0
                predicted_label = raw_label
                confidence = raw_confidence

                if candidate_label != stable_label:
                    stable_label = candidate_label
                    stable_count = 1
                else:
                    stable_count += 1

                if current_motion < args.idle_motion_threshold:
                    predicted_label = "idle"
                    confidence = idle_confidence
                else:
                    accept_action = (
                        candidate_label == "idle"
                        or (
                            candidate_label != "idle"
                            and candidate_confidence >= args.action_confidence_threshold
                            and stable_count >= args.sustain_frames
                            and current_motion >= args.motion_threshold
                        )
                    )
                    predicted_label = candidate_label if accept_action else "idle"
                    confidence = candidate_confidence if accept_action else idle_confidence

                if predicted_label == "idle":
                    idle_hold_count += 1
                    if idle_hold_count < args.final_debounce and stable_non_idle_label != "---":
                        predicted_label = stable_non_idle_label
                        confidence = max(confidence, 0.01)
                else:
                    idle_hold_count = 0
                    if predicted_label != stable_non_idle_label:
                        stable_non_idle_label = predicted_label
                        stable_non_idle_count = 1
                    else:
                        stable_non_idle_count += 1

                if predicted_label != "idle" and stable_non_idle_count >= args.final_debounce:
                    final_label_history.append(predicted_label)
                    counts: dict[str, int] = {}
                    for label in final_label_history:
                        counts[label] = counts.get(label, 0) + 1
                    final_candidate = max(counts.items(), key=lambda item: item[1])[0]
                    if final_candidate != "idle":
                        predicted_label = final_candidate
                        confidence = max(confidence, 0.01)

                state.predicted_label = predicted_label
                state.confidence = confidence
                state.raw_label = raw_label
                state.raw_confidence = raw_confidence
                state.buffer_fill = len(landmarks_buffer)
                state.frame_count = frame_count
                state.last_update_ts = time.time()

                if (
                    predicted_label != "idle"
                    and predicted_label != "---"
                    and predicted_label in template_matcher.templates
                    and len(landmarks_buffer) >= 16
                    and confidence >= args.quality_min_confidence
                ):
                    lm_seq = list(landmarks_buffer)
                    data = landmarks_to_model_input(lm_seq, window_size=args.window_size)
                    quality_label, quality_distance = template_matcher.judge(predicted_label, data)

                state.quality_label = quality_label
                state.quality_distance = quality_distance

                fps_frames += 1
                now = time.perf_counter()
                elapsed = now - fps_clock_start
                if elapsed >= 1.0:
                    state.fps = fps_frames / elapsed
                    fps_clock_start = now
                    fps_frames = 0

                response = {
                    "ok": True,
                    "prediction": asdict(state),
                    "top3": candidate_top3,
                    "motion_energy": current_motion,
                }
            else:
                state.buffer_fill = len(landmarks_buffer)
                response = {
                    "ok": True,
                    "prediction": asdict(state),
                    "top3": [],
                    "motion_energy": motion_energy(list(landmarks_buffer)),
                }

            if args.preview:
                preview = frame.copy()
                draw_pose(preview, results)
                cv2.putText(preview, f"{state.predicted_label} {state.confidence * 100:.1f}%", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.imshow("CTR-GCN Docker Preview", preview)
                cv2.waitKey(1)

            self._send_json(response)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Docker-friendly CTR-GCN frame inference service")
    parser.add_argument("--weights", type=str, default=None, help="Path to model .pt file")
    parser.add_argument("--work-dir", type=str, default=str(DEFAULT_WORK_DIR), help="Training work directory containing runs-*.pt checkpoints")
    parser.add_argument("--labels", type=str, default=None, help="Path to labels.json")
    parser.add_argument("--window-size", type=int, default=64, help="Sliding window size")
    parser.add_argument("--stride", type=int, default=1, help="Run inference every N frames")
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--templates-dir", type=str, default=str(ROOT / "templates"), help="Root directory containing per-action template subfolders")
    parser.add_argument("--quality-threshold", type=float, default=0.55, help="Template matching threshold")
    parser.add_argument("--action-confidence-threshold", type=float, default=0.80, help="Minimum confidence required to accept action recognition")
    parser.add_argument("--motion-threshold", type=float, default=0.030, help="Minimum recent motion energy required to trust non-idle actions")
    parser.add_argument("--idle-motion-threshold", type=float, default=0.020, help="If motion stays below this value, prefer idle regardless of class")
    parser.add_argument("--sustain-frames", type=int, default=6, help="Number of consecutive frames needed to confirm a non-idle action")
    parser.add_argument("--quality-min-confidence", type=float, default=0.60, help="Only run template quality matching when action confidence is above this")
    parser.add_argument("--model-complexity", type=int, choices=(0, 1, 2), default=0, help="MediaPipe Pose model complexity")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--preview", action="store_true", help="Show a local preview window inside the container")
    return parser.parse_args()


def main() -> None:
    _set_windows_hidpi_aware()
    args = parse_args()
    handler = build_http_handler(args)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"CTR-GCN frame inference service listening on http://{args.host}:{args.port}")
    print("POST /infer-frame with base64 JPEG payload: {\"image\": \"...\"}")
    print("GET /health and GET /state are available for checks")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
