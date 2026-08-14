"""
Real-time webcam CTR-GCN action recognition.

Opens webcam -> extracts MediaPipe Pose landmarks per frame ->
accumulates a sliding window -> feeds to CTR-GCN model ->
displays the predicted action label on screen.

Use the same preprocessing and label order as the training dataset.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TORCHLIGHT_ROOT = ROOT / 'torchlight'
if str(TORCHLIGHT_ROOT) not in sys.path:
    sys.path.insert(0, str(TORCHLIGHT_ROOT))

from model.ctrgcn import Model
from feeders import tools
from scripts.inference.template_matcher import TemplateMatcher

cv2.setUseOptimized(True)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

if not hasattr(mp, 'solutions'):
    raise RuntimeError(
        'This script requires a MediaPipe version that exposes mp.solutions. '
        'Please run it with a compatible environment, for example:\n'
        '  D:\\B\\Anaconda\\envs\\posec3d\\python.exe scripts\\inference\\webcam_realtime_ctrgcn.py'
    )

NUM_LANDMARKS = 33
POSE_CONNECTIONS = tuple(mp.solutions.pose.POSE_CONNECTIONS)
POSE_LEFT_HIP = 23
POSE_RIGHT_HIP = 24
POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12

DEFAULT_WORK_DIR = ROOT / 'work_dir' / 'webcam_mediapipe' / 'ctrgcn_five_actions'
DEFAULT_LABELS_CANDIDATES = [
    ROOT / 'datasets' / 'webcam_mediapipe' / 'labels.json',
    ROOT / 'datasets' / 'mediapipe_pose' / 'labels.json',
]


def load_labels(labels_path):
    with open(labels_path, 'r', encoding='utf-8') as f:
        label_dict = json.load(f)
    id_to_label = {int(k): v for k, v in label_dict.items()}
    return id_to_label


def find_best_weights(work_dir):
    weights_dir = Path(work_dir)
    if not weights_dir.is_dir():
        return None
    weights = list(weights_dir.glob('runs-*.pt'))
    if not weights:
        return None

    def checkpoint_key(path):
        parts = path.stem.split('-')
        try:
            return int(parts[1]), int(parts[2])
        except (IndexError, ValueError):
            return -1, -1

    return str(sorted(weights, key=checkpoint_key)[-1])


def load_model(weights_path, num_class, device):
    model = Model(
        num_class=num_class,
        num_point=33,
        num_person=2,
        graph='graph.mediapipe_pose.Graph',
        graph_args={'labeling_mode': 'spatial'},
        in_channels=3,
    )
    weights = torch.load(weights_path, map_location=device)
    model.load_state_dict(weights)
    model = model.to(device)
    model.eval()
    return model


def normalize(sequence):
    """Center on hip midpoint, scale by torso length — same as feeder."""
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


def landmarks_to_model_input(landmarks_seq, window_size=64, p_interval=None):
    """
    landmarks_seq: list of [33, 4] arrays (x, y, z, visibility)
    Returns: [1, 3, T, 33, 1] tensor ready for model
    """
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


def motion_energy(landmarks_seq, recent_frames=12):
    """Robust recent motion score over the latest frames."""
    if len(landmarks_seq) < 2:
        return 0.0

    seq = np.asarray(landmarks_seq[-max(2, recent_frames):], dtype=np.float32)[:, :, :3]
    seq = normalize(seq)
    diffs = np.linalg.norm(seq[1:] - seq[:-1], axis=2)
    if diffs.size == 0:
        return 0.0
    return float(np.median(np.mean(diffs, axis=1)))


def draw_pose(frame, results):
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


def draw_prediction_panel(frame, predicted_label, confidence, top3, fps, buffer_fill, quality_label=None, quality_distance=None, raw_label=None, raw_confidence=None):
    """Draw a semi-transparent panel with prediction info."""
    h, w = frame.shape[:2]
    panel_w = min(w - 24, 760)
    panel_h = 250
    x, y = w - panel_w - 12, 12
    mid_x = x + panel_w // 2

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + panel_w, y + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)
    cv2.line(frame, (mid_x, y + 42), (mid_x, y + panel_h - 12), (90, 90, 90), 1, cv2.LINE_AA)

    cv2.putText(frame, 'CTR-GCN Yoga Recognition', (x + 12, y + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 255), 2, cv2.LINE_AA)

    # Header stats
    cv2.putText(frame, f'FPS: {fps:.1f}  Buffer: {buffer_fill} frames',
                (x + 12, y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    # Left zone: raw model output
    left_x = x + 12
    cv2.putText(frame, 'RAW MODEL', (left_x, y + 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 210, 90), 2, cv2.LINE_AA)
    if raw_label is not None and raw_confidence is not None:
        cv2.putText(frame, f'{raw_label} ({raw_confidence*100:.1f}%)',
                    (left_x, y + 108), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (150, 220, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, '---', (left_x, y + 108), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (150, 220, 255), 2, cv2.LINE_AA)

    for i, (label, prob) in enumerate(top3):
        bar_x = left_x
        bar_y = y + 128 + i * 24
        bar_w = int((panel_w * 0.42 - 24) * prob)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 14), (50, 180, 255), -1)
        cv2.putText(frame, f'{label} {prob*100:.0f}%',
                    (bar_x + bar_w + 6, bar_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Right zone: final decision
    right_x = mid_x + 12
    cv2.putText(frame, 'FINAL DECISION', (right_x, y + 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (120, 255, 120), 2, cv2.LINE_AA)
    final_color = (0, 255, 100) if confidence > 0.5 else (0, 200, 255)
    cv2.putText(frame, f'{predicted_label} ({confidence*100:.1f}%)',
                (right_x, y + 108), cv2.FONT_HERSHEY_SIMPLEX, 0.95, final_color, 2, cv2.LINE_AA)

    if quality_label is not None:
        quality_color = (0, 255, 100) if quality_label == 'standard' else (0, 0, 255)
        quality_text = quality_label if quality_distance is None else f'{quality_label}  d={quality_distance:.4f}'
        cv2.putText(frame, quality_text,
                    (right_x, y + 138), cv2.FONT_HERSHEY_SIMPLEX, 0.7, quality_color, 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, 'quality: n/a',
                    (right_x, y + 138), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (170, 170, 170), 1, cv2.LINE_AA)

    cv2.putText(frame, 'Press q/ESC to quit', (x + 12, y + panel_h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)


def _set_windows_hidpi_aware() -> None:
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def _fit_frame_to_screen(frame: np.ndarray) -> np.ndarray:
    try:
        user32 = ctypes.windll.user32
        screen_w = int(user32.GetSystemMetrics(0))
        screen_h = int(user32.GetSystemMetrics(1))
    except Exception:
        return frame

    h, w = frame.shape[:2]
    if w <= 0 or h <= 0 or screen_w <= 0 or screen_h <= 0:
        return frame

    scale = min(screen_w / w, screen_h / h, 1.0)
    if scale >= 0.999:
        return frame

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)


def main():
    _set_windows_hidpi_aware()
    parser = argparse.ArgumentParser(description='Real-time CTR-GCN Yoga Recognition')
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to model .pt file (auto-detect if omitted)')
    parser.add_argument('--work-dir', type=str,
                        default=str(DEFAULT_WORK_DIR),
                        help='Training work directory containing runs-*.pt checkpoints')
    parser.add_argument('--labels', type=str, default=None,
                        help='Path to labels.json (default: datasets/mediapipe_pose/labels.json)')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--window-size', type=int, default=32,
                        help='Sliding window size (frames)')
    parser.add_argument('--stride', type=int, default=1,
                        help='Run inference every N frames (reduces CPU load)')
    parser.add_argument('--final-debounce', type=int, default=5,
                        help='Number of consecutive final votes required before switching away from idle')
    parser.add_argument('--width', type=int, default=1280, help='Camera width')
    parser.add_argument('--height', type=int, default=720, help='Camera height')
    parser.add_argument('--display-width', type=int, default=1280)
    parser.add_argument('--display-height', type=int, default=720)
    parser.add_argument('--model-complexity', type=int, choices=(0, 1, 2), default=0,
                        help='MediaPipe Pose model complexity (0=fastest)')
    parser.add_argument('--mirror', action='store_true', default=False,
                        help='Mirror display horizontally before pose extraction')
    parser.add_argument('--device', type=str, default='auto', help='auto, cpu, or cuda')
    parser.add_argument('--templates-dir', type=str,
                        default=str(ROOT / 'templates'),
                        help='Root directory containing per-action template subfolders')
    parser.add_argument('--quality-threshold', type=float, default=0.55,
                        help='Distance threshold for standard/non_standard judgment')
    parser.add_argument('--action-confidence-threshold', type=float, default=0.80,
                        help='Minimum confidence required to accept action recognition')
    parser.add_argument('--motion-threshold', type=float, default=0.030,
                        help='Minimum recent motion energy required to trust non-idle actions')
    parser.add_argument('--idle-motion-threshold', type=float, default=0.020,
                        help='If motion stays below this value, prefer idle regardless of class')
    parser.add_argument('--sustain-frames', type=int, default=6,
                        help='Number of consecutive frames needed to confirm a non-idle action')
    parser.add_argument('--quality-min-confidence', type=float, default=0.60,
                        help='Only run template quality matching when action confidence is above this')
    args = parser.parse_args()

    weights_path = args.weights
    if weights_path is None:
        weights_path = find_best_weights(args.work_dir)
    if weights_path is None:
        print(f'ERROR: No checkpoint found in {args.work_dir}')
        print('Please specify --weights with a runs-*.pt file.')
        sys.exit(1)
    weights_path = str(Path(weights_path))
    if not Path(weights_path).exists():
        print(f'ERROR: Weights not found: {weights_path}')
        print('Please train first or specify --weights')
        sys.exit(1)

    labels_path = args.labels
    if labels_path is None:
        for candidate in DEFAULT_LABELS_CANDIDATES:
            if candidate.exists():
                labels_path = str(candidate)
                break
    if labels_path is None:
        print('ERROR: Labels not found in any default location:')
        for candidate in DEFAULT_LABELS_CANDIDATES:
            print(f'  - {candidate}')
        print('Please specify --labels explicitly')
        sys.exit(1)

    if not Path(labels_path).exists():
        print(f'ERROR: Labels not found: {labels_path}')
        sys.exit(1)

    id_to_label = load_labels(labels_path)
    label_to_id = {v: k for k, v in id_to_label.items()}
    num_class = len(id_to_label)

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f'Loading model from: {weights_path}')
    print(f'Device: {device}')
    print(f'Classes ({num_class}): {list(id_to_label.values())}')
    model = load_model(weights_path, num_class, device)
    template_matcher = TemplateMatcher(args.templates_dir, threshold=args.quality_threshold)
    print(f'Template root: {template_matcher.templates_dir}')
    print(f'Quality threshold: {template_matcher.threshold}')
    print(f'Action confidence threshold: {args.action_confidence_threshold}')
    print(f'Motion threshold: {args.motion_threshold}')
    print(f'Idle motion threshold: {args.idle_motion_threshold}')
    print(f'Sustain frames: {args.sustain_frames}')
    print(f'Quality min confidence: {args.quality_min_confidence}')
    print(f'Final debounce: {args.final_debounce}')
    print('Model loaded successfully!')

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print(f'ERROR: Cannot open camera {args.camera}')
        sys.exit(1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or args.width)
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or args.height)
    if actual_w <= 0:
        actual_w = args.width
    if actual_h <= 0:
        actual_h = args.height
    print(f'Camera: {actual_w}x{actual_h}')

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=args.model_complexity,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    landmarks_buffer = deque(maxlen=args.window_size)

    predicted_label = '---'
    confidence = 0.0
    top3 = [(l, 0.0) for l in list(id_to_label.values())[:3]]
    raw_label = '---'
    raw_confidence = 0.0
    stable_label = '---'
    stable_count = 0
    final_label_history = deque(maxlen=max(7, args.final_debounce))
    idle_hold_count = 0
    stable_non_idle_label = '---'
    stable_non_idle_count = 0

    fps_clock_start = time.perf_counter()
    fps_frames = 0
    fps = 0.0
    frame_count = 0

    print('Starting real-time recognition. Press q or ESC to quit.')

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print('Warning: failed to read frame')
                break

            if args.mirror:
                frame = cv2.flip(frame, 1)

            display = frame.copy()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            draw_pose(display, results)

            if results.pose_landmarks:
                lm = np.asarray(
                    [[l.x, l.y, l.z, l.visibility] for l in results.pose_landmarks.landmark],
                    dtype=np.float32,
                )
                landmarks_buffer.append(lm)
            else:
                landmarks_buffer.append(np.zeros((NUM_LANDMARKS, 4), dtype=np.float32))

            frame_count += 1

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
                candidate_top3 = [(id_to_label[top_idx[i]], float(probs_np[top_idx[i]])) for i in range(min(3, num_class))]
                current_motion = motion_energy(lm_seq)
                raw_label = candidate_label
                raw_confidence = candidate_confidence
                idle_confidence = float(probs_np[label_to_id.get('idle', top_idx[0])]) if 'idle' in label_to_id else 0.0
                predicted_label = raw_label
                confidence = raw_confidence

                if candidate_label != stable_label:
                    stable_label = candidate_label
                    stable_count = 1
                else:
                    stable_count += 1

                if current_motion < args.idle_motion_threshold:
                    predicted_label = 'idle'
                    confidence = idle_confidence
                else:
                    accept_action = (
                        candidate_label == 'idle'
                        or (
                            candidate_label != 'idle'
                            and candidate_confidence >= args.action_confidence_threshold
                            and stable_count >= args.sustain_frames
                            and current_motion >= args.motion_threshold
                        )
                    )
                    predicted_label = candidate_label if accept_action else 'idle'
                    confidence = candidate_confidence if accept_action else idle_confidence

                if predicted_label == 'idle':
                    idle_hold_count += 1
                    if idle_hold_count < args.final_debounce and stable_non_idle_label != '---':
                        predicted_label = stable_non_idle_label
                        confidence = max(confidence, 0.01)
                else:
                    idle_hold_count = 0
                    if predicted_label != stable_non_idle_label:
                        stable_non_idle_label = predicted_label
                        stable_non_idle_count = 1
                    else:
                        stable_non_idle_count += 1

                if predicted_label != 'idle' and stable_non_idle_count >= args.final_debounce:
                    final_label_history.append(predicted_label)
                    counts = {}
                    for label in final_label_history:
                        counts[label] = counts.get(label, 0) + 1
                    final_candidate = max(counts.items(), key=lambda item: item[1])[0]
                    if final_candidate != 'idle':
                        predicted_label = final_candidate
                        confidence = max(confidence, 0.01)

                top3 = candidate_top3

            quality_label = None
            quality_distance = None
            if (
                predicted_label != 'idle'
                and predicted_label != '---'
                and predicted_label in template_matcher.templates
                and len(landmarks_buffer) >= 16
                and confidence >= args.quality_min_confidence
            ):
                lm_seq = list(landmarks_buffer)
                data = landmarks_to_model_input(lm_seq, window_size=args.window_size)
                quality_label, quality_distance = template_matcher.judge(predicted_label, data)

            fps_frames += 1
            now = time.perf_counter()
            elapsed = now - fps_clock_start
            if elapsed >= 1.0:
                fps = fps_frames / elapsed
                fps_clock_start = now
                fps_frames = 0

            buffer_fill = len(landmarks_buffer)
            draw_prediction_panel(display, predicted_label, confidence, top3, fps, buffer_fill, quality_label, quality_distance, raw_label, raw_confidence)

            display = _fit_frame_to_screen(display)
            if args.display_width > 0 and args.display_height > 0:
                target_w = min(args.display_width, display.shape[1])
                target_h = min(args.display_height, display.shape[0])
                if target_w > 0 and target_h > 0:
                    display = cv2.resize(display, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

            cv2.namedWindow('CTR-GCN Yoga Recognition', cv2.WINDOW_NORMAL)
            cv2.setWindowProperty('CTR-GCN Yoga Recognition', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow('CTR-GCN Yoga Recognition', display)

            key = cv2.waitKeyEx(1)
            if key != -1:
                key = key & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break
    finally:
        pose.close()
        cap.release()
        cv2.destroyAllWindows()
        print('Done.')


if __name__ == '__main__':
    main()
