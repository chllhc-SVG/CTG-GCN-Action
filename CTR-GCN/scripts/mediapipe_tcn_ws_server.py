from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_webcam_mediapipe_tcn import MediaPipeTCN, normalize_sequence, resize_sequence

DEFAULT_WEIGHTS = ROOT / "work_dir" / "webcam_mediapipe_pose_hands_idle_waving_clapping_stop_tcn_pose_hands" / "best_model.pt"
DEFAULT_LABELS = ["idle", "waving", "clapping", "stop"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WebSocket action recognition server for digital-human-client")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--model-complexity", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--feature-mode", choices=("pose", "pose_hands", "pose_face", "pose_hands_face", "auto"), default="auto")
    parser.add_argument("--face-refine", action="store_true")
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--smooth", type=int, default=5)
    parser.add_argument("--confidence-threshold", type=float, default=0.55)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def load_labels(checkpoint: dict[str, Any], weights_path: Path) -> list[str]:
    config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    if isinstance(config, dict) and isinstance(config.get("labels"), list):
        return [str(label) for label in config["labels"]]
    labels_path = weights_path.parent / "labels.txt"
    if labels_path.exists():
        return [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return DEFAULT_LABELS


def load_model(weights_path: Path, device: torch.device, hidden_channels: int, dropout: float) -> tuple[MediaPipeTCN, list[str], str]:
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    checkpoint = torch.load(weights_path, map_location=device)
    labels = load_labels(checkpoint, weights_path)
    config = checkpoint.get("config") if isinstance(checkpoint, dict) else {}
    feature_mode = "pose"
    input_channels = 33 * 4
    if isinstance(config, dict):
        hidden_channels = int(config.get("hidden_channels", hidden_channels))
        dropout = float(config.get("dropout", dropout))
        feature_mode = str(config.get("feature_mode", feature_mode))
        input_channels = int(config.get("input_channels", input_channels))
    model = MediaPipeTCN(num_classes=len(labels), hidden_channels=hidden_channels, dropout=dropout, input_channels=input_channels).to(device)
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, labels, feature_mode


def landmark_array(results: Any) -> np.ndarray | None:
    if not results.pose_landmarks:
        return None
    return np.asarray([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark], dtype=np.float32)


def hand_arrays(results: Any) -> tuple[np.ndarray, np.ndarray]:
    left_hand = np.zeros((21, 4), dtype=np.float32)
    right_hand = np.zeros((21, 4), dtype=np.float32)
    landmarks_list = getattr(results, "multi_hand_landmarks", None)
    handedness_list = getattr(results, "multi_handedness", None)
    if not landmarks_list or not handedness_list:
        return left_hand, right_hand
    for hand_landmarks, handedness in zip(landmarks_list, handedness_list):
        label = handedness.classification[0].label.lower() if handedness.classification else ""
        arr = np.asarray([[lm.x, lm.y, lm.z, 1.0] for lm in hand_landmarks.landmark], dtype=np.float32)
        if label == "left":
            left_hand = arr
        elif label == "right":
            right_hand = arr
    return left_hand, right_hand


def face_array(results: Any) -> np.ndarray:
    face = np.zeros((478, 4), dtype=np.float32)
    landmarks_list = getattr(results, "multi_face_landmarks", None)
    if not landmarks_list:
        return face
    for i, lm in enumerate(landmarks_list[0].landmark[:478]):
        face[i] = [lm.x, lm.y, lm.z, 1.0]
    return face


def build_tensor(frames: deque[np.ndarray], window_size: int, device: torch.device) -> torch.Tensor:
    sequence = np.stack(list(frames), axis=0).astype(np.float32)
    sequence = resize_sequence(sequence, window_size)
    sequence = normalize_sequence(sequence)
    x = torch.from_numpy(sequence.reshape(window_size, -1).T).unsqueeze(0)
    return x.to(device)


def decode_jpeg(message: bytes) -> np.ndarray | None:
    data = np.frombuffer(message, dtype=np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return frame


class ActionRecognizer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda:0")
        self.model, self.labels, checkpoint_feature_mode = load_model(args.weights, self.device, args.hidden_channels, args.dropout)
        self.feature_mode = checkpoint_feature_mode if args.feature_mode == "auto" else args.feature_mode
        self.frames: deque[np.ndarray] = deque(maxlen=args.window_size)
        self.prob_history: deque[np.ndarray] = deque(maxlen=max(1, args.smooth))
        self.frame_count = 0
        self.last_probs: np.ndarray | None = None
        self.last_latency_ms = 0.0
        self.started_at = time.perf_counter()
        self.pose_frames = 0
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=args.model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.45,
            min_tracking_confidence=0.45,
        ) if "hands" in self.feature_mode else None
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=args.face_refine,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) if "face" in self.feature_mode else None
        print(f"Loaded {args.weights} on {self.device}; labels={self.labels}; feature_mode={self.feature_mode}")

    def close(self) -> None:
        self.pose.close()
        if self.hands is not None:
            self.hands.close()
        if self.face_mesh is not None:
            self.face_mesh.close()

    def process(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pose_results = self.pose.process(rgb)
        hand_results = self.hands.process(rgb) if self.hands is not None else None
        face_results = self.face_mesh.process(rgb) if self.face_mesh is not None else None
        pose_landmarks = landmark_array(pose_results)
        pose_ready = pose_landmarks is not None
        if pose_landmarks is not None:
            parts = [pose_landmarks]
            if "hands" in self.feature_mode:
                left_hand, right_hand = hand_arrays(hand_results)
                parts.extend([left_hand, right_hand])
            if "face" in self.feature_mode:
                parts.append(face_array(face_results))
            self.frames.append(np.concatenate(parts, axis=0).astype(np.float32))
            self.frame_count += 1
            self.pose_frames += 1

        should_infer = len(self.frames) >= self.args.window_size and self.frame_count % max(1, self.args.stride) == 0
        if should_infer:
            tensor = build_tensor(self.frames, self.args.window_size, self.device)
            start = time.perf_counter()
            with torch.no_grad():
                probs = torch.softmax(self.model(tensor), dim=1).squeeze(0).detach().cpu().numpy()
            self.last_latency_ms = (time.perf_counter() - start) * 1000.0
            self.prob_history.append(probs)
            self.last_probs = np.mean(np.stack(list(self.prob_history), axis=0), axis=0)

        probs = self.last_probs if self.last_probs is not None else np.zeros((len(self.labels),), dtype=np.float32)
        top_indices = probs.argsort()[-self.args.topk:][::-1] if probs.size else []
        top3 = [{"label": self.labels[int(i)], "state": self.labels[int(i)], "score": float(probs[int(i)])} for i in top_indices]
        if top3 and top3[0]["score"] >= self.args.confidence_threshold:
            action_state = str(top3[0]["state"])
            confidence = float(top3[0]["score"])
        else:
            action_state = "idle"
            confidence = float(top3[0]["score"]) if top3 else 0.0
        elapsed = max(1e-6, time.perf_counter() - self.started_at)
        return {
            "type": "action_recognition",
            "backend": "mediapipe_tcn",
            "timestamp": int(time.time() * 1000),
            "phase": "running",
            "action": {
                "rawLabel": action_state,
                "state": action_state,
                "confidence": confidence,
                "source": "mediapipe_tcn",
            },
            "top3": top3,
            "pose": {
                "ready": pose_ready,
                "bufferCurrent": len(self.frames),
                "bufferTarget": self.args.window_size,
            },
            "metrics": {
                "poseFps": self.pose_frames / elapsed,
                "actionLatencyMs": self.last_latency_ms,
            },
            "debug": f"feature={self.feature_mode} buffer={len(self.frames)}/{self.args.window_size}",
        }


async def main_async(args: argparse.Namespace) -> None:
    try:
        import websockets
    except ImportError as exc:
        raise RuntimeError("Missing dependency: websockets. Install with: python -m pip install websockets") from exc

    recognizer = ActionRecognizer(args)

    async def handler(websocket: Any) -> None:
        print("client connected")
        try:
            async for message in websocket:
                if isinstance(message, str):
                    continue
                frame = decode_jpeg(message)
                if frame is None:
                    await websocket.send(json.dumps({"phase": "error", "debug": "failed to decode jpeg"}))
                    continue
                result = recognizer.process(frame)
                await websocket.send(json.dumps(result, ensure_ascii=False))
        except Exception as exc:
            print(f"client error: {exc}")
        finally:
            print("client disconnected")

    print(f"MediaPipe TCN WebSocket server listening on ws://{args.host}:{args.port}")
    async with websockets.serve(handler, args.host, args.port, max_size=4_000_000):
        try:
            await asyncio.Future()
        finally:
            recognizer.close()


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
