from __future__ import annotations

import argparse
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

POSE_CONNECTIONS = tuple(mp.solutions.pose.POSE_CONNECTIONS)
HAND_CONNECTIONS = tuple(mp.solutions.hands.HAND_CONNECTIONS)
FACE_CONTOURS = tuple(mp.solutions.face_mesh.FACEMESH_CONTOURS)
DEFAULT_LABELS = ["idle", "waving", "stop", "clapping", "hands_together"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime webcam MediaPipe TCN action recognition")
    parser.add_argument("--weights", type=Path, default=ROOT / "work_dir" / "webcam_mediapipe_tcn" / "best_model.pt")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--display-width", type=int, default=960)
    parser.add_argument("--display-height", type=int, default=720)
    parser.add_argument("--model-complexity", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--feature-mode", choices=("pose", "pose_hands", "pose_face", "pose_hands_face", "auto"), default="auto")
    parser.add_argument("--face-refine", action="store_true", help="Enable refined FaceMesh landmarks around eyes/lips")
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--smooth", type=int, default=5, help="Average probabilities over the latest N predictions")
    parser.add_argument("--confidence-threshold", type=float, default=0.55)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--mirror", action="store_true", help="Mirror image before MediaPipe; use only if training used --mirror")
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


def draw_hands(frame: np.ndarray, results: Any) -> None:
    landmarks_list = getattr(results, "multi_hand_landmarks", None)
    if not landmarks_list:
        return
    h, w = frame.shape[:2]
    for hand_landmarks in landmarks_list:
        points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], (255, 180, 64), 2, cv2.LINE_AA)
        for point in points:
            cv2.circle(frame, point, 2, (255, 80, 80), -1, cv2.LINE_AA)


def draw_face(frame: np.ndarray, results: Any) -> None:
    landmarks_list = getattr(results, "multi_face_landmarks", None)
    if not landmarks_list:
        return
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks_list[0].landmark]
    for start, end in FACE_CONTOURS:
        if start < len(points) and end < len(points):
            cv2.line(frame, points[start], points[end], (180, 120, 255), 1, cv2.LINE_AA)
    for idx in (1, 33, 61, 133, 152, 263, 291, 362):
        if idx < len(points):
            cv2.circle(frame, points[idx], 2, (220, 120, 255), -1, cv2.LINE_AA)


def draw_pose(frame: np.ndarray, results: Any) -> None:
    if not results.pose_landmarks:
        return
    h, w = frame.shape[:2]
    points: list[tuple[int, int] | None] = []
    for landmark in results.pose_landmarks.landmark:
        if landmark.visibility < 0.35:
            points.append(None)
        else:
            points.append((int(landmark.x * w), int(landmark.y * h)))
    for start, end in POSE_CONNECTIONS:
        if points[start] is not None and points[end] is not None:
            cv2.line(frame, points[start], points[end], (64, 220, 255), 2, cv2.LINE_AA)
    for point in points:
        if point is not None:
            cv2.circle(frame, point, 3, (0, 255, 80), -1, cv2.LINE_AA)


def put_panel(frame: np.ndarray, lines: list[str]) -> None:
    x, y = 12, 12
    width = min(frame.shape[1] - 24, 900)
    height = 34 + 27 * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x + 12, y + 30 + i * 27), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (255, 255, 255), 2, cv2.LINE_AA)


def build_tensor(frames: deque[np.ndarray], window_size: int, device: torch.device) -> torch.Tensor:
    sequence = np.stack(list(frames), axis=0).astype(np.float32)
    sequence = resize_sequence(sequence, window_size)
    sequence = normalize_sequence(sequence)
    x = torch.from_numpy(sequence.reshape(window_size, -1).T).unsqueeze(0)
    return x.to(device)


def main() -> None:
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda:0")
    model, labels, checkpoint_feature_mode = load_model(args.weights, device, args.hidden_channels, args.dropout)
    feature_mode = checkpoint_feature_mode if args.feature_mode == "auto" else args.feature_mode
    print(f"Loaded {args.weights} on {device}, feature_mode={feature_mode}")
    print("Labels:", ", ".join(f"{i}:{label}" for i, label in enumerate(labels)))
    print("Press q or ESC in the camera window to quit.")

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera}")
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or args.width)
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or args.height)

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=args.model_complexity,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=0.45,
        min_tracking_confidence=0.45,
    ) if "hands" in feature_mode else None
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=args.face_refine,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) if "face" in feature_mode else None

    frames: deque[np.ndarray] = deque(maxlen=args.window_size)
    prob_history: deque[np.ndarray] = deque(maxlen=max(1, args.smooth))
    frame_count = 0
    predictions: list[tuple[str, float]] = []
    stable_label = "idle"
    stable_confidence = 0.0
    last_infer_ms = 0.0
    fps_clock_start = time.perf_counter()
    fps_frames = 0
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Warning: failed to read frame from camera")
                break

            if args.mirror:
                frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            hand_results = hands.process(rgb) if hands is not None else None
            face_results = face_mesh.process(rgb) if face_mesh is not None else None
            landmarks = landmark_array(results)
            if landmarks is not None:
                frame_parts = [landmarks]
                if "hands" in feature_mode:
                    left_hand, right_hand = hand_arrays(hand_results)
                    frame_parts.extend([left_hand, right_hand])
                if "face" in feature_mode:
                    frame_parts.append(face_array(face_results))
                frames.append(np.concatenate(frame_parts, axis=0).astype(np.float32))
                frame_count += 1
                draw_pose(display_frame, results)
                if face_results is not None:
                    draw_face(display_frame, face_results)
                if hand_results is not None:
                    draw_hands(display_frame, hand_results)

            fps_frames += 1
            now = time.perf_counter()
            elapsed_for_fps = now - fps_clock_start
            if elapsed_for_fps >= 1.0:
                fps = fps_frames / elapsed_for_fps
                fps_clock_start = now
                fps_frames = 0

            if len(frames) >= args.window_size and frame_count % max(1, args.stride) == 0:
                tensor = build_tensor(frames, args.window_size, device)
                start = time.perf_counter()
                with torch.no_grad():
                    probs = torch.softmax(model(tensor), dim=1).squeeze(0).detach().cpu().numpy()
                last_infer_ms = (time.perf_counter() - start) * 1000.0
                prob_history.append(probs)
                smooth_probs = np.mean(np.stack(list(prob_history), axis=0), axis=0)
                top_indices = smooth_probs.argsort()[-args.topk:][::-1]
                predictions = [(labels[i], float(smooth_probs[i])) for i in top_indices]
                best_idx = int(top_indices[0])
                best_conf = float(smooth_probs[best_idx])
                stable_label = labels[best_idx] if best_conf >= args.confidence_threshold else "idle"
                stable_confidence = best_conf

            lines = [
                f"MediaPipe TCN action | device={device} | FPS={fps:.1f} | {actual_width}x{actual_height}",
                f"window={len(frames)}/{args.window_size} stride={args.stride} infer={last_infer_ms:.1f}ms | q/ESC quit",
                f"Action: {stable_label}  {stable_confidence * 100:.1f}%",
            ]
            if predictions:
                lines.extend([f"Top {rank}: {label}  {prob * 100:.1f}%" for rank, (label, prob) in enumerate(predictions, 1)])
            else:
                lines.append("Collecting pose frames...")
            put_panel(display_frame, lines)

            if args.display_width > 0 and args.display_height > 0:
                display_frame = cv2.resize(display_frame, (args.display_width, args.display_height), interpolation=cv2.INTER_LINEAR)
            cv2.namedWindow("Webcam MediaPipe TCN Action Recognition", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Webcam MediaPipe TCN Action Recognition", max(320, args.display_width), max(240, args.display_height))
            cv2.imshow("Webcam MediaPipe TCN Action Recognition", display_frame)
            key = cv2.waitKeyEx(1)
            if key != -1:
                key = key & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
    finally:
        pose.close()
        if hands is not None:
            hands.close()
        if face_mesh is not None:
            face_mesh.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
