"""
HTTP inference service for CTR-GCN realtime pose recognition.

The service keeps the model and MediaPipe Pose pipeline inside Docker while the
camera remains on the host machine. The host sends JPEG frames to this service
and receives the predicted action back as JSON.
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import json
import sys
import threading
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

from feeders import tools
from model.ctrgcn import Model
from scripts.inference.template_matcher import TemplateMatcher

cv2.setUseOptimized(True)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

if not hasattr(mp, 'solutions'):
    raise RuntimeError(
        'This script requires a MediaPipe version that exposes mp.solutions.'
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
    return {int(k): v for k, v in label_dict.items()}


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
    if len(landmarks_seq) < 2:
        return 0.0

    seq = np.asarray(landmarks_seq[-max(2, recent_frames):], dtype=np.float32)[:, :, :3]
    seq = normalize(seq)
    diffs = np.linalg.norm(seq[1:] - seq[:-1], axis=2)
    if diffs.size == 0:
        return 0.0
    return float(np.median(np.mean(diffs, axis=1)))


class ActionEngine:
    def __init__(
        self,
        model,
        id_to_label,
        device,
        templates_dir,
        window_size=32,
        quality_threshold=0.55,
        action_confidence_threshold=0.80,
        motion_threshold=0.030,
        idle_motion_threshold=0.020,
        sustain_frames=6,
        quality_min_confidence=0.60,
        final_debounce=5,
        model_complexity=0,
    ):
        self.model = model
        self.id_to_label = id_to_label
        self.label_to_id = {v: k for k, v in id_to_label.items()}
        self.device = device
        self.window_size = window_size
        self.quality_threshold = quality_threshold
        self.action_confidence_threshold = action_confidence_threshold
        self.motion_threshold = motion_threshold
        self.idle_motion_threshold = idle_motion_threshold
        self.sustain_frames = sustain_frames
        self.quality_min_confidence = quality_min_confidence
        self.final_debounce = final_debounce
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.template_matcher = TemplateMatcher(templates_dir, threshold=quality_threshold)
        self.landmarks_buffer = deque(maxlen=window_size)
        self.stable_label = '---'
        self.stable_count = 0
        self.final_label_history = deque(maxlen=max(7, final_debounce))
        self.idle_hold_count = 0
        self.stable_non_idle_label = '---'
        self.stable_non_idle_count = 0
        self.frame_count = 0
        self.fps_clock_start = time.perf_counter()
        self.fps_frames = 0
        self.fps = 0.0

    def close(self):
        self.pose.close()

    def _predict_from_buffer(self):
        lm_seq = list(self.landmarks_buffer)
        data = landmarks_to_model_input(lm_seq, window_size=self.window_size)
        data_tensor = torch.from_numpy(data).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(data_tensor)
            probs = torch.softmax(output, dim=1)

        probs_np = probs[0].cpu().numpy()
        top_idx = np.argsort(probs_np)[::-1]
        candidate_label = self.id_to_label[top_idx[0]]
        candidate_confidence = float(probs_np[top_idx[0]])
        candidate_top3 = [
            (self.id_to_label[top_idx[i]], float(probs_np[top_idx[i]]))
            for i in range(min(3, len(top_idx)))
        ]
        current_motion = motion_energy(lm_seq)
        idle_confidence = float(probs_np[self.label_to_id.get('idle', top_idx[0])]) if 'idle' in self.label_to_id else 0.0
        predicted_label = candidate_label
        confidence = candidate_confidence

        if candidate_label != self.stable_label:
            self.stable_label = candidate_label
            self.stable_count = 1
        else:
            self.stable_count += 1

        if current_motion < self.idle_motion_threshold:
            predicted_label = 'idle'
            confidence = idle_confidence
        else:
            accept_action = (
                candidate_label == 'idle'
                or (
                    candidate_label != 'idle'
                    and candidate_confidence >= self.action_confidence_threshold
                    and self.stable_count >= self.sustain_frames
                    and current_motion >= self.motion_threshold
                )
            )
            predicted_label = candidate_label if accept_action else 'idle'
            confidence = candidate_confidence if accept_action else idle_confidence

        if predicted_label == 'idle':
            self.idle_hold_count += 1
            if self.idle_hold_count < self.final_debounce and self.stable_non_idle_label != '---':
                predicted_label = self.stable_non_idle_label
                confidence = max(confidence, 0.01)
        else:
            self.idle_hold_count = 0
            if predicted_label != self.stable_non_idle_label:
                self.stable_non_idle_label = predicted_label
                self.stable_non_idle_count = 1
            else:
                self.stable_non_idle_count += 1

        if predicted_label != 'idle' and self.stable_non_idle_count >= self.final_debounce:
            self.final_label_history.append(predicted_label)
            counts: dict[str, int] = {}
            for label in self.final_label_history:
                counts[label] = counts.get(label, 0) + 1
            final_candidate = max(counts.items(), key=lambda item: item[1])[0]
            if final_candidate != 'idle':
                predicted_label = final_candidate
                confidence = max(confidence, 0.01)

        quality_label = None
        quality_distance = None
        if (
            predicted_label != 'idle'
            and predicted_label != '---'
            and predicted_label in self.template_matcher.templates
            and len(self.landmarks_buffer) >= 16
            and confidence >= self.quality_min_confidence
        ):
            data = landmarks_to_model_input(lm_seq, window_size=self.window_size)
            quality_label, quality_distance = self.template_matcher.judge(predicted_label, data)

        return {
            'predicted_label': predicted_label,
            'confidence': confidence,
            'raw_label': candidate_label,
            'raw_confidence': candidate_confidence,
            'top3': candidate_top3,
            'quality_label': quality_label,
            'quality_distance': quality_distance,
            'buffer_fill': len(self.landmarks_buffer),
        }

    def process_frame(self, frame_bgr, mirror=False):
        if mirror:
            frame_bgr = cv2.flip(frame_bgr, 1)

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if results.pose_landmarks:
            lm = np.asarray(
                [[l.x, l.y, l.z, l.visibility] for l in results.pose_landmarks.landmark],
                dtype=np.float32,
            )
            self.landmarks_buffer.append(lm)
        else:
            self.landmarks_buffer.append(np.zeros((NUM_LANDMARKS, 4), dtype=np.float32))

        self.frame_count += 1

        prediction = {
            'predicted_label': '---',
            'confidence': 0.0,
            'raw_label': '---',
            'raw_confidence': 0.0,
            'top3': [],
            'quality_label': None,
            'quality_distance': None,
            'buffer_fill': len(self.landmarks_buffer),
        }

        if len(self.landmarks_buffer) >= 16:
            prediction = self._predict_from_buffer()

        self.fps_frames += 1
        now = time.perf_counter()
        elapsed = now - self.fps_clock_start
        if elapsed >= 1.0:
            self.fps = self.fps_frames / elapsed
            self.fps_clock_start = now
            self.fps_frames = 0

        prediction['fps'] = self.fps
        prediction['frame_count'] = self.frame_count
        prediction['has_pose'] = bool(results.pose_landmarks)
        return prediction


class InferenceHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address, RequestHandlerClass, engine):
        super().__init__(server_address, RequestHandlerClass)
        self.engine = engine
        self.engine_lock = threading.Lock()


class RequestHandler(BaseHTTPRequestHandler):
    server_version = 'CTRGCNInference/1.0'

    def _send_json(self, status_code, payload):
        body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ('/health', '/status'):
            self._send_json(200, {'status': 'ok'})
            return
        self._send_json(404, {'error': 'not_found'})

    def do_POST(self):
        if self.path != '/infer':
            self._send_json(404, {'error': 'not_found'})
            return

        content_length = int(self.headers.get('Content-Length', '0'))
        if content_length <= 0:
            self._send_json(400, {'error': 'empty_request'})
            return

        try:
            raw = self.rfile.read(content_length)
            payload = json.loads(raw.decode('utf-8'))
        except Exception as exc:
            self._send_json(400, {'error': 'invalid_json', 'detail': str(exc)})
            return

        frame_b64 = payload.get('frame_b64')
        if not frame_b64:
            self._send_json(400, {'error': 'missing_frame_b64'})
            return

        try:
            frame_bytes = base64.b64decode(frame_b64)
            frame_arr = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = cv2.imdecode(frame_arr, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError('failed to decode image')
        except Exception as exc:
            self._send_json(400, {'error': 'invalid_image', 'detail': str(exc)})
            return

        mirror = bool(payload.get('mirror', False))
        try:
            with self.server.engine_lock:
                result = self.server.engine.process_frame(frame, mirror=mirror)
        except Exception as exc:
            self._send_json(500, {'error': 'inference_failed', 'detail': str(exc)})
            return

        self._send_json(200, result)

    def log_message(self, format, *args):
        return


def build_engine(args):
    weights_path = args.weights
    if weights_path is None:
        weights_path = find_best_weights(args.work_dir)
    if weights_path is None:
        raise SystemExit(f'ERROR: No checkpoint found in {args.work_dir}')
    weights_path = str(Path(weights_path))
    if not Path(weights_path).exists():
        raise SystemExit(f'ERROR: Weights not found: {weights_path}')

    labels_path = args.labels
    if labels_path is None:
        for candidate in DEFAULT_LABELS_CANDIDATES:
            if candidate.exists():
                labels_path = str(candidate)
                break
    if labels_path is None or not Path(labels_path).exists():
        raise SystemExit('ERROR: Labels not found. Pass --labels explicitly.')

    id_to_label = load_labels(labels_path)
    num_class = len(id_to_label)

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f'Loading model from: {weights_path}')
    print(f'Device: {device}')
    print(f'Classes ({num_class}): {list(id_to_label.values())}')
    model = load_model(weights_path, num_class, device)
    print(f'Template root: {args.templates_dir}')
    engine = ActionEngine(
        model=model,
        id_to_label=id_to_label,
        device=device,
        templates_dir=args.templates_dir,
        window_size=args.window_size,
        quality_threshold=args.quality_threshold,
        action_confidence_threshold=args.action_confidence_threshold,
        motion_threshold=args.motion_threshold,
        idle_motion_threshold=args.idle_motion_threshold,
        sustain_frames=args.sustain_frames,
        quality_min_confidence=args.quality_min_confidence,
        final_debounce=args.final_debounce,
        model_complexity=args.model_complexity,
    )
    print('Model loaded successfully!')
    return engine


def parse_args():
    parser = argparse.ArgumentParser(description='CTR-GCN inference service for Docker-based camera workflows')
    parser.add_argument('--weights', type=str, default=None, help='Path to model .pt file (auto-detect if omitted)')
    parser.add_argument('--work-dir', type=str, default=str(DEFAULT_WORK_DIR), help='Training work directory containing runs-*.pt checkpoints')
    parser.add_argument('--labels', type=str, default=None, help='Path to labels.json')
    parser.add_argument('--templates-dir', type=str, default=str(ROOT / 'templates'), help='Root directory containing per-action template subfolders')
    parser.add_argument('--device', type=str, default='auto', help='auto, cpu, or cuda')
    parser.add_argument('--window-size', type=int, default=32, help='Sliding window size (frames)')
    parser.add_argument('--quality-threshold', type=float, default=0.55, help='Distance threshold for standard/non_standard judgment')
    parser.add_argument('--action-confidence-threshold', type=float, default=0.80, help='Minimum confidence required to accept action recognition')
    parser.add_argument('--motion-threshold', type=float, default=0.030, help='Minimum recent motion energy required to trust non-idle actions')
    parser.add_argument('--idle-motion-threshold', type=float, default=0.020, help='If motion stays below this value, prefer idle regardless of class')
    parser.add_argument('--sustain-frames', type=int, default=6, help='Number of consecutive frames needed to confirm a non-idle action')
    parser.add_argument('--quality-min-confidence', type=float, default=0.60, help='Only run template quality matching when action confidence is above this')
    parser.add_argument('--final-debounce', type=int, default=5, help='Number of consecutive final votes required before switching away from idle')
    parser.add_argument('--model-complexity', type=int, choices=(0, 1, 2), default=0, help='MediaPipe Pose model complexity (0=fastest)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Bind host')
    parser.add_argument('--port', type=int, default=8000, help='Bind port')
    return parser.parse_args()


def main():
    args = parse_args()
    engine = build_engine(args)
    server = InferenceHTTPServer((args.host, args.port), RequestHandler, engine)
    print(f'Serving on http://{args.host}:{args.port}')
    print('Health endpoint: GET /health')
    print('Inference endpoint: POST /infer with JSON {"frame_b64": "..."}')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('Stopping...')
    finally:
        engine.close()
        server.server_close()
        print('Done.')


if __name__ == '__main__':
    main()
