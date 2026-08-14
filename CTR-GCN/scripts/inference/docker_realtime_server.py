"""
HTTP inference service for Docker-based webcam capture.

The host machine keeps direct access to the camera. It captures frames locally
and posts JPEG images to this service. The service maintains the same sliding
window and decision logic as the native webcam demo, then returns JSON results.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path
from threading import Lock
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TORCHLIGHT_ROOT = ROOT / 'torchlight'
if str(TORCHLIGHT_ROOT) not in sys.path:
    sys.path.insert(0, str(TORCHLIGHT_ROOT))

from scripts.inference.template_matcher import TemplateMatcher
from scripts.inference.webcam_realtime_ctrgcn import (
    DEFAULT_LABELS_CANDIDATES,
    DEFAULT_WORK_DIR,
    NUM_LANDMARKS,
    find_best_weights,
    landmarks_to_model_input,
    load_labels,
    load_model,
    motion_energy,
)

if not hasattr(mp, 'solutions'):
    raise RuntimeError(
        'This script requires a MediaPipe version that exposes mp.solutions.'
    )

cv2.setUseOptimized(True)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

POSE_CONNECTIONS = tuple(mp.solutions.pose.POSE_CONNECTIONS)
POSE = mp.solutions.pose


class InferenceResponse(BaseModel):
    predicted_label: str
    confidence: float
    raw_label: str
    raw_confidence: float
    top3: list[dict[str, float | str]]
    quality_label: str | None = None
    quality_distance: float | None = None
    buffer_fill: int
    motion_energy: float
    pose_detected: bool
    frame_index: int
    processing_ms: float
    processing_fps: float


class RealtimeEngine:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = self._resolve_device(args.device)
        self.weights_path = self._resolve_weights(args.weights, args.work_dir)
        self.labels_path = self._resolve_labels(args.labels)
        self.templates_dir = Path(args.templates_dir)
        self.id_to_label = load_labels(self.labels_path)
        self.label_to_id = {v: k for k, v in self.id_to_label.items()}
        self.num_class = len(self.id_to_label)

        print(f'Loading model from: {self.weights_path}')
        print(f'Device: {self.device}')
        print(f'Classes ({self.num_class}): {list(self.id_to_label.values())}')
        self.model = load_model(self.weights_path, self.num_class, self.device)
        self.template_matcher = TemplateMatcher(self.templates_dir, threshold=args.quality_threshold)

        print(f'Template root: {self.template_matcher.templates_dir}')
        print(f'Quality threshold: {self.template_matcher.threshold}')
        print(f'Action confidence threshold: {args.action_confidence_threshold}')
        print(f'Motion threshold: {args.motion_threshold}')
        print(f'Idle motion threshold: {args.idle_motion_threshold}')
        print(f'Sustain frames: {args.sustain_frames}')
        print(f'Quality min confidence: {args.quality_min_confidence}')
        print(f'Final debounce: {args.final_debounce}')
        print('Model loaded successfully!')

        self.pose = POSE.Pose(
            static_image_mode=False,
            model_complexity=args.model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.lock = Lock()
        self.landmarks_buffer = deque(maxlen=args.window_size)
        self.predicted_label = '---'
        self.confidence = 0.0
        self.top3 = [(l, 0.0) for l in list(self.id_to_label.values())[:3]]
        self.raw_label = '---'
        self.raw_confidence = 0.0
        self.stable_label = '---'
        self.stable_count = 0
        self.final_label_history = deque(maxlen=max(7, args.final_debounce))
        self.idle_hold_count = 0
        self.stable_non_idle_label = '---'
        self.stable_non_idle_count = 0
        self.frame_index = 0
        self.fps_clock_start = time.perf_counter()
        self.fps_frames = 0
        self.processing_fps = 0.0

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)

    @staticmethod
    def _resolve_weights(weights: str | None, work_dir: str) -> str:
        if weights:
            path = Path(weights)
            if path.exists():
                return str(path)
            raise FileNotFoundError(f'Weights not found: {path}')

        best = find_best_weights(work_dir)
        if best is None:
            raise FileNotFoundError(
                f'No checkpoint found in {work_dir}. Provide --weights or mount a work_dir with runs-*.pt files.'
            )
        return best

    @staticmethod
    def _resolve_labels(labels: str | None) -> str:
        if labels:
            path = Path(labels)
            if path.exists():
                return str(path)
            raise FileNotFoundError(f'Labels not found: {path}')

        for candidate in DEFAULT_LABELS_CANDIDATES:
            if candidate.exists():
                return str(candidate)

        raise FileNotFoundError(
            'Labels not found in default locations. Provide --labels explicitly.'
        )

    @staticmethod
    def _decode_image(image_bytes: bytes) -> np.ndarray:
        buf = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError('Unable to decode JPEG frame')
        return frame

    @staticmethod
    def _extract_landmarks(results: Any) -> np.ndarray:
        if results.pose_landmarks:
            return np.asarray(
                [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark],
                dtype=np.float32,
            )
        return np.zeros((NUM_LANDMARKS, 4), dtype=np.float32)

    @staticmethod
    def _topk_from_probs(probs_np: np.ndarray, id_to_label: dict[int, str], k: int = 3) -> list[dict[str, float | str]]:
        top_idx = np.argsort(probs_np)[::-1][:k]
        return [
            {'label': id_to_label[int(idx)], 'prob': float(probs_np[int(idx)])}
            for idx in top_idx
        ]

    def _update_prediction(self, probs_np: np.ndarray, current_motion: float) -> tuple[str, float, list[dict[str, float | str]]]:
        top_idx = np.argsort(probs_np)[::-1]
        candidate_label = self.id_to_label[int(top_idx[0])]
        candidate_confidence = float(probs_np[int(top_idx[0])])
        candidate_top3 = self._topk_from_probs(probs_np, self.id_to_label, k=min(3, self.num_class))
        idle_confidence = float(probs_np[self.label_to_id.get('idle', int(top_idx[0]))]) if 'idle' in self.label_to_id else 0.0

        if candidate_label != self.stable_label:
            self.stable_label = candidate_label
            self.stable_count = 1
        else:
            self.stable_count += 1

        predicted_label = candidate_label
        confidence = candidate_confidence

        if current_motion < self.args.idle_motion_threshold:
            predicted_label = 'idle'
            confidence = idle_confidence
        else:
            accept_action = (
                candidate_label == 'idle'
                or (
                    candidate_label != 'idle'
                    and candidate_confidence >= self.args.action_confidence_threshold
                    and self.stable_count >= self.args.sustain_frames
                    and current_motion >= self.args.motion_threshold
                )
            )
            predicted_label = candidate_label if accept_action else 'idle'
            confidence = candidate_confidence if accept_action else idle_confidence

        if predicted_label == 'idle':
            self.idle_hold_count += 1
            if self.idle_hold_count < self.args.final_debounce and self.stable_non_idle_label != '---':
                predicted_label = self.stable_non_idle_label
                confidence = max(confidence, 0.01)
        else:
            self.idle_hold_count = 0
            if predicted_label != self.stable_non_idle_label:
                self.stable_non_idle_label = predicted_label
                self.stable_non_idle_count = 1
            else:
                self.stable_non_idle_count += 1

        if predicted_label != 'idle' and self.stable_non_idle_count >= self.args.final_debounce:
            self.final_label_history.append(predicted_label)
            counts: dict[str, int] = {}
            for label in self.final_label_history:
                counts[label] = counts.get(label, 0) + 1
            final_candidate = max(counts.items(), key=lambda item: item[1])[0]
            if final_candidate != 'idle':
                predicted_label = final_candidate
                confidence = max(confidence, 0.01)

        return predicted_label, confidence, candidate_top3

    def analyze_frame(self, image_bytes: bytes) -> dict[str, Any]:
        start = time.perf_counter()
        frame = self._decode_image(image_bytes)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        pose_detected = bool(results.pose_landmarks)

        landmarks = self._extract_landmarks(results)
        self.landmarks_buffer.append(landmarks)
        self.frame_index += 1

        predicted_label = self.predicted_label
        confidence = self.confidence
        top3 = self.top3
        raw_label = self.raw_label
        raw_confidence = self.raw_confidence
        quality_label = None
        quality_distance = None

        if len(self.landmarks_buffer) >= 16 and (self.frame_index % self.args.stride == 0):
            lm_seq = list(self.landmarks_buffer)
            data = landmarks_to_model_input(lm_seq, window_size=self.args.window_size)
            data_tensor = torch.from_numpy(data).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(data_tensor)
                probs = torch.softmax(output, dim=1)

            probs_np = probs[0].detach().cpu().numpy()
            current_motion = motion_energy(lm_seq)
            raw_top_idx = int(np.argmax(probs_np))
            raw_label = self.id_to_label[raw_top_idx]
            raw_confidence = float(probs_np[raw_top_idx])
            predicted_label, confidence, top3 = self._update_prediction(probs_np, current_motion)

            if (
                predicted_label != 'idle'
                and predicted_label != '---'
                and predicted_label in self.template_matcher.templates
                and len(self.landmarks_buffer) >= 16
                and confidence >= self.args.quality_min_confidence
            ):
                quality_label, quality_distance = self.template_matcher.judge(predicted_label, data)

            self.predicted_label = predicted_label
            self.confidence = confidence
            self.top3 = [(item['label'], float(item['prob'])) for item in top3]
            self.raw_label = raw_label
            self.raw_confidence = raw_confidence

        self.fps_frames += 1
        elapsed = time.perf_counter() - self.fps_clock_start
        if elapsed >= 1.0:
            self.processing_fps = self.fps_frames / elapsed
            self.fps_clock_start = time.perf_counter()
            self.fps_frames = 0

        processing_ms = (time.perf_counter() - start) * 1000.0
        return {
            'predicted_label': predicted_label,
            'confidence': float(confidence),
            'raw_label': raw_label,
            'raw_confidence': float(raw_confidence),
            'top3': top3,
            'quality_label': quality_label,
            'quality_distance': quality_distance,
            'buffer_fill': len(self.landmarks_buffer),
            'motion_energy': float(motion_energy(list(self.landmarks_buffer))) if len(self.landmarks_buffer) >= 2 else 0.0,
            'pose_detected': pose_detected,
            'frame_index': self.frame_index,
            'processing_ms': processing_ms,
            'processing_fps': self.processing_fps,
        }

    def close(self) -> None:
        self.pose.close()


class ServiceState:
    def __init__(self, engine: RealtimeEngine):
        self.engine = engine
        self.lock = Lock()



def build_app(engine: RealtimeEngine) -> FastAPI:
    app = FastAPI(title='CTR-GCN Docker Inference Service', version='1.0.0')
    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*'],
    )
    state = ServiceState(engine)

    @app.get('/health')
    def health() -> dict[str, str]:
        return {'status': 'ok'}

    @app.get('/config')
    def config() -> dict[str, Any]:
        return {
            'weights_path': engine.weights_path,
            'labels_path': engine.labels_path,
            'templates_dir': str(engine.templates_dir),
            'device': str(engine.device),
            'window_size': engine.args.window_size,
            'stride': engine.args.stride,
            'quality_threshold': engine.args.quality_threshold,
        }

    @app.post('/infer-frame', response_model=InferenceResponse)
    def infer_frame(frame: UploadFile = File(...)) -> InferenceResponse:
        if frame.content_type and not frame.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail='frame must be an image upload')

        image_bytes = frame.file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail='empty frame upload')

        with state.lock:
            result = engine.analyze_frame(image_bytes)

        return InferenceResponse(**result)

    @app.on_event('shutdown')
    def _shutdown() -> None:
        engine.close()

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Docker-friendly CTR-GCN inference API')
    parser.add_argument('--host', default=os.environ.get('CTRGCN_HOST', '0.0.0.0'))
    parser.add_argument('--port', type=int, default=int(os.environ.get('CTRGCN_PORT', '8000')))
    parser.add_argument('--weights', default=os.environ.get('CTRGCN_WEIGHTS'))
    parser.add_argument('--work-dir', default=os.environ.get('CTRGCN_WORK_DIR', str(DEFAULT_WORK_DIR)))
    parser.add_argument('--labels', default=os.environ.get('CTRGCN_LABELS'))
    parser.add_argument('--templates-dir', default=os.environ.get('CTRGCN_TEMPLATES_DIR', str(ROOT / 'templates')))
    parser.add_argument('--window-size', type=int, default=int(os.environ.get('CTRGCN_WINDOW_SIZE', '64')))
    parser.add_argument('--stride', type=int, default=int(os.environ.get('CTRGCN_STRIDE', '4')))
    parser.add_argument('--model-complexity', type=int, choices=(0, 1, 2), default=int(os.environ.get('CTRGCN_MODEL_COMPLEXITY', '0')))
    parser.add_argument('--device', default=os.environ.get('CTRGCN_DEVICE', 'auto'))
    parser.add_argument('--quality-threshold', type=float, default=float(os.environ.get('CTRGCN_QUALITY_THRESHOLD', '0.55')))
    parser.add_argument('--action-confidence-threshold', type=float, default=float(os.environ.get('CTRGCN_ACTION_CONFIDENCE_THRESHOLD', '0.80')))
    parser.add_argument('--motion-threshold', type=float, default=float(os.environ.get('CTRGCN_MOTION_THRESHOLD', '0.030')))
    parser.add_argument('--idle-motion-threshold', type=float, default=float(os.environ.get('CTRGCN_IDLE_MOTION_THRESHOLD', '0.020')))
    parser.add_argument('--sustain-frames', type=int, default=int(os.environ.get('CTRGCN_SUSTAIN_FRAMES', '6')))
    parser.add_argument('--quality-min-confidence', type=float, default=float(os.environ.get('CTRGCN_QUALITY_MIN_CONFIDENCE', '0.60')))
    parser.add_argument('--final-debounce', type=int, default=int(os.environ.get('CTRGCN_FINAL_DEBOUNCE', '5')))
    return parser.parse_args()


args = parse_args()
engine = RealtimeEngine(args)
app = build_app(engine)


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)
