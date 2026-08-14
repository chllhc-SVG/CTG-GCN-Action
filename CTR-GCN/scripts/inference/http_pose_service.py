"""
HTTP inference service for Docker-based webcam workflows.

The host machine keeps the webcam and sends JPEG frames to this service.
The container performs MediaPipe pose extraction, CTR-GCN inference,
and optional template-based quality judgment.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TORCHLIGHT_ROOT = ROOT / 'torchlight'
if str(TORCHLIGHT_ROOT) not in sys.path:
    sys.path.insert(0, str(TORCHLIGHT_ROOT))

from feeders import tools
from model.ctrgcn import Model
from scripts.inference.template_matcher import TemplateMatcher

if not hasattr(mp, 'solutions'):
    raise RuntimeError(
        'This script requires a MediaPipe version that exposes mp.solutions. '
        'Please install a compatible mediapipe build.'
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


@dataclass
class SessionState:
    landmarks_buffer: deque = field(default_factory=lambda: deque(maxlen=32))
    final_label_history: deque = field(default_factory=lambda: deque(maxlen=7))
    predicted_label: str = '---'
    confidence: float = 0.0
    raw_label: str = '---'
    raw_confidence: float = 0.0
    stable_label: str = '---'
    stable_count: int = 0
    idle_hold_count: int = 0
    stable_non_idle_label: str = '---'
    stable_non_idle_count: int = 0
    frame_index: int = 0
    last_updated: float = field(default_factory=time.time)


class InferenceRuntime:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.id_to_label = self._load_labels(args.labels)
        self.label_to_id = {v: k for k, v in self.id_to_label.items()}
        self.num_class = len(self.id_to_label)

        if args.device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(args.device)

        self.weights_path = self._resolve_weights(args.weights, args.work_dir)
        self.model = self._load_model(self.weights_path, self.num_class, self.device)
        self.template_matcher = TemplateMatcher(args.templates_dir, threshold=args.quality_threshold)
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=args.model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.pose_lock = threading.Lock()
        self.session_lock = threading.Lock()
        self.sessions: dict[str, SessionState] = {}

        print(f'Loading model from: {self.weights_path}')
        print(f'Device: {self.device}')
        print(f'Classes ({self.num_class}): {list(self.id_to_label.values())}')
        print(f'Template root: {self.template_matcher.templates_dir}')
        print(f'Quality threshold: {self.template_matcher.threshold}')
        print(f'Action confidence threshold: {args.action_confidence_threshold}')
        print(f'Motion threshold: {args.motion_threshold}')
        print(f'Idle motion threshold: {args.idle_motion_threshold}')
        print(f'Sustain frames: {args.sustain_frames}')
        print(f'Quality min confidence: {args.quality_min_confidence}')
        print(f'Final debounce: {args.final_debounce}')
        print('Model loaded successfully!')

    @staticmethod
    def _load_labels(labels_path: str) -> dict[int, str]:
        with open(labels_path, 'r', encoding='utf-8') as f:
            label_dict = json.load(f)
        return {int(k): v for k, v in label_dict.items()}

    @staticmethod
    def _resolve_weights(weights_path: str | None, work_dir: str) -> str:
        if weights_path is not None:
            resolved = Path(weights_path)
            if not resolved.exists():
                raise FileNotFoundError(f'Weights not found: {weights_path}')
            return str(resolved)

        weights_dir = Path(work_dir)
        if not weights_dir.is_dir():
            raise FileNotFoundError(f'Work directory not found: {work_dir}')
        weights = list(weights_dir.glob('runs-*.pt'))
        if not weights:
            raise FileNotFoundError(
                f'No checkpoint found in {work_dir}. '\
                'Please train first or mount a directory containing runs-*.pt.'
            )

        def checkpoint_key(path: Path) -> tuple[int, int]:
            parts = path.stem.split('-')
            try:
                return int(parts[1]), int(parts[2])
            except (IndexError, ValueError):
                return -1, -1

        return str(sorted(weights, key=checkpoint_key)[-1])

    @staticmethod
    def _load_model(weights_path: str, num_class: int, device: torch.device) -> Model:
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

    def get_session(self, session_id: str) -> SessionState:
        with self.session_lock:
            state = self.sessions.get(session_id)
            if state is None:
                state = SessionState(
                    landmarks_buffer=deque(maxlen=self.args.window_size),
                    final_label_history=deque(maxlen=max(7, self.args.final_debounce)),
                )
                self.sessions[session_id] = state
            return state

    def reset_session(self, session_id: str) -> None:
        with self.session_lock:
            self.sessions[session_id] = SessionState(
                landmarks_buffer=deque(maxlen=self.args.window_size),
                final_label_history=deque(maxlen=max(7, self.args.final_debounce)),
            )

    @staticmethod
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

    def landmarks_to_model_input(self, landmarks_seq: list[np.ndarray]) -> np.ndarray:
        if not landmarks_seq:
            raise ValueError('Empty landmarks sequence')

        pose = np.stack(landmarks_seq, axis=0).astype(np.float32)
        pose = pose[:, :, :3]
        pose = self.normalize(pose)

        data = pose.transpose(2, 0, 1)
        data = data[:, :, :, np.newaxis]

        if data.shape[1] < self.args.window_size:
            pad = np.zeros((3, self.args.window_size, NUM_LANDMARKS, 1), dtype=np.float32)
            pad[:, :data.shape[1], :, :] = data
            data = pad
        elif data.shape[1] > self.args.window_size:
            begin = (data.shape[1] - self.args.window_size) // 2
            data = data[:, begin:begin + self.args.window_size, :, :]

        valid_frame_num = np.sum(data.sum(0).sum(-1).sum(-1) != 0)
        if valid_frame_num == 0:
            valid_frame_num = 1
        data = tools.valid_crop_resize(data, valid_frame_num, [0.95], self.args.window_size)

        if data.shape[3] == 1:
            data = np.concatenate([data, np.zeros_like(data)], axis=3)

        return data.astype(np.float32)

    @staticmethod
    def motion_energy(landmarks_seq: list[np.ndarray], recent_frames: int = 12) -> float:
        if len(landmarks_seq) < 2:
            return 0.0

        seq = np.asarray(landmarks_seq[-max(2, recent_frames):], dtype=np.float32)[:, :, :3]
        seq = InferenceRuntime.normalize(seq)
        diffs = np.linalg.norm(seq[1:] - seq[:-1], axis=2)
        if diffs.size == 0:
            return 0.0
        return float(np.median(np.mean(diffs, axis=1)))

    def process_frame(self, frame: np.ndarray, session_id: str) -> dict[str, Any]:
        if frame is None or frame.size == 0:
            raise ValueError('Empty frame received')

        state = self.get_session(session_id)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with self.pose_lock:
            results = self.pose.process(rgb)

        if results.pose_landmarks:
            lm = np.asarray(
                [[l.x, l.y, l.z, l.visibility] for l in results.pose_landmarks.landmark],
                dtype=np.float32,
            )
            state.landmarks_buffer.append(lm)
        else:
            state.landmarks_buffer.append(np.zeros((NUM_LANDMARKS, 4), dtype=np.float32))

        state.frame_index += 1
        state.last_updated = time.time()

        top3: list[tuple[str, float]] = [(label, 0.0) for label in list(self.id_to_label.values())[:3]]
        quality_label = None
        quality_distance = None

        if len(state.landmarks_buffer) >= 16:
            lm_seq = list(state.landmarks_buffer)
            data = self.landmarks_to_model_input(lm_seq)
            data_tensor = torch.from_numpy(data).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(data_tensor)
                probs = torch.softmax(output, dim=1)

            probs_np = probs[0].cpu().numpy()
            top_idx = np.argsort(probs_np)[::-1]
            candidate_label = self.id_to_label[top_idx[0]]
            candidate_confidence = float(probs_np[top_idx[0]])
            top3 = [(self.id_to_label[top_idx[i]], float(probs_np[top_idx[i]])) for i in range(min(3, self.num_class))]
            current_motion = self.motion_energy(lm_seq)
            state.raw_label = candidate_label
            state.raw_confidence = candidate_confidence
            idle_confidence = float(probs_np[self.label_to_id.get('idle', top_idx[0])]) if 'idle' in self.label_to_id else 0.0
            predicted_label = candidate_label
            confidence = candidate_confidence

            if candidate_label != state.stable_label:
                state.stable_label = candidate_label
                state.stable_count = 1
            else:
                state.stable_count += 1

            if current_motion < self.args.idle_motion_threshold:
                predicted_label = 'idle'
                confidence = idle_confidence
            else:
                accept_action = (
                    candidate_label == 'idle'
                    or (
                        candidate_label != 'idle'
                        and candidate_confidence >= self.args.action_confidence_threshold
                        and state.stable_count >= self.args.sustain_frames
                        and current_motion >= self.args.motion_threshold
                    )
                )
                predicted_label = candidate_label if accept_action else 'idle'
                confidence = candidate_confidence if accept_action else idle_confidence

            if predicted_label == 'idle':
                state.idle_hold_count += 1
                if state.idle_hold_count < self.args.final_debounce and state.stable_non_idle_label != '---':
                    predicted_label = state.stable_non_idle_label
                    confidence = max(confidence, 0.01)
            else:
                state.idle_hold_count = 0
                if predicted_label != state.stable_non_idle_label:
                    state.stable_non_idle_label = predicted_label
                    state.stable_non_idle_count = 1
                else:
                    state.stable_non_idle_count += 1

            if predicted_label != 'idle' and state.stable_non_idle_count >= self.args.final_debounce:
                state.final_label_history.append(predicted_label)
                counts: dict[str, int] = {}
                for label in state.final_label_history:
                    counts[label] = counts.get(label, 0) + 1
                final_candidate = max(counts.items(), key=lambda item: item[1])[0]
                if final_candidate != 'idle':
                    predicted_label = final_candidate
                    confidence = max(confidence, 0.01)

            state.predicted_label = predicted_label
            state.confidence = confidence

            if (
                predicted_label != 'idle'
                and predicted_label != '---'
                and predicted_label in self.template_matcher.templates
                and len(state.landmarks_buffer) >= 16
                and confidence >= self.args.quality_min_confidence
            ):
                quality_label, quality_distance = self.template_matcher.judge(predicted_label, data)

        return {
            'session_id': session_id,
            'label': state.predicted_label,
            'confidence': state.confidence,
            'raw_label': state.raw_label,
            'raw_confidence': state.raw_confidence,
            'quality_label': quality_label,
            'quality_distance': quality_distance,
            'buffer_fill': len(state.landmarks_buffer),
            'motion_energy': self.motion_energy(list(state.landmarks_buffer)) if len(state.landmarks_buffer) >= 2 else 0.0,
            'top3': [{'label': label, 'confidence': prob} for label, prob in top3],
            'frame_index': state.frame_index,
        }

    def close(self) -> None:
        try:
            self.pose.close()
        except Exception:
            pass


app = FastAPI(title='CTR-GCN Docker Inference Service', version='1.0.0')
runtime: InferenceRuntime | None = None


@app.get('/health')
def health() -> dict[str, str]:
    return {'status': 'ok'}


@app.post('/reset/{session_id}')
def reset_session(session_id: str) -> dict[str, str]:
    if runtime is None:
        raise HTTPException(status_code=503, detail='Runtime not ready')
    runtime.reset_session(session_id)
    return {'status': 'reset', 'session_id': session_id}


@app.post('/infer')
async def infer(
    frame: UploadFile = File(...),
    session_id: str = Form('default'),
) -> dict[str, Any]:
    if runtime is None:
        raise HTTPException(status_code=503, detail='Runtime not ready')

    raw = await frame.read()
    if not raw:
        raise HTTPException(status_code=400, detail='Empty frame payload')

    image = np.frombuffer(raw, dtype=np.uint8)
    decoded = cv2.imdecode(image, cv2.IMREAD_COLOR)
    if decoded is None:
        raise HTTPException(status_code=400, detail='Failed to decode frame')

    return runtime.process_frame(decoded, session_id=session_id)


@app.on_event('shutdown')
def _shutdown() -> None:
    if runtime is not None:
        runtime.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='CTR-GCN Docker inference service')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--work-dir', type=str, default=str(DEFAULT_WORK_DIR))
    parser.add_argument('--labels', type=str, default=None)
    parser.add_argument('--templates-dir', type=str, default=str(ROOT / 'templates'))
    parser.add_argument('--window-size', type=int, default=32)
    parser.add_argument('--device', type=str, default='auto', help='auto, cpu, or cuda')
    parser.add_argument('--model-complexity', type=int, choices=(0, 1, 2), default=0)
    parser.add_argument('--quality-threshold', type=float, default=0.55)
    parser.add_argument('--action-confidence-threshold', type=float, default=0.80)
    parser.add_argument('--motion-threshold', type=float, default=0.030)
    parser.add_argument('--idle-motion-threshold', type=float, default=0.020)
    parser.add_argument('--sustain-frames', type=int, default=6)
    parser.add_argument('--quality-min-confidence', type=float, default=0.60)
    parser.add_argument('--final-debounce', type=int, default=5)
    return parser


def pick_labels_path(labels_path: str | None) -> str:
    if labels_path is not None:
        resolved = Path(labels_path)
        if not resolved.exists():
            raise FileNotFoundError(f'Labels not found: {labels_path}')
        return str(resolved)

    for candidate in DEFAULT_LABELS_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        'Labels not found in any default location. Please mount datasets/webcam_mediapipe/labels.json '
        'or pass --labels explicitly.'
    )


def main() -> None:
    global runtime
    parser = build_parser()
    args = parser.parse_args()
    args.labels = pick_labels_path(args.labels)
    runtime = InferenceRuntime(args)
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level='info')
    finally:
        if runtime is not None:
            runtime.close()


if __name__ == '__main__':
    main()
