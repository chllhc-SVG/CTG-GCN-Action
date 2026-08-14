"""
Host-side webcam client.

This script runs on the machine that has direct access to the camera. It captures
frames, encodes them as JPEG, and sends them to the Docker inference service.
It can optionally show the returned prediction in a local preview window.
"""

from __future__ import annotations

import argparse
import base64
import json
import time
from dataclasses import dataclass
from typing import Any

import cv2
import requests


@dataclass
class ClientState:
    fps: float = 0.0
    server_fps: float = 0.0
    last_prediction: str = "---"
    last_confidence: float = 0.0
    last_quality: str | None = None
    last_latency_ms: float = 0.0


def encode_frame(frame: Any, quality: int = 85) -> str:
    ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError('failed to encode frame')
    return base64.b64encode(buf.tobytes()).decode('utf-8')


def draw_overlay(frame, state: ClientState, server_payload: dict[str, Any]) -> None:
    prediction = server_payload.get('prediction', {})
    label = prediction.get('predicted_label', '---')
    confidence = float(prediction.get('confidence', 0.0))
    quality = prediction.get('quality_label')
    fps = float(prediction.get('fps', 0.0))
    latency = state.last_latency_ms

    lines = [
        f'local fps: {state.fps:.1f}',
        f'server fps: {fps:.1f}',
        f'latency: {latency:.0f} ms',
        f'label: {label} ({confidence*100:.1f}%)',
        f'quality: {quality or "n/a"}',
    ]
    x, y = 20, 30
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x, y + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description='Host camera client for Docker-based CTR-GCN service')
    parser.add_argument('--server', type=str, default='http://127.0.0.1:8000', help='Inference server base URL')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--width', type=int, default=1280, help='Camera width')
    parser.add_argument('--height', type=int, default=720, help='Camera height')
    parser.add_argument('--mirror', action='store_true', help='Mirror frames before sending')
    parser.add_argument('--show-preview', action='store_true', help='Show a local preview window')
    parser.add_argument('--jpeg-quality', type=int, default=85, help='JPEG encode quality')
    parser.add_argument('--send-every-n-frames', type=int, default=1, help='Only send every Nth frame')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit(f'Cannot open camera {args.camera}')

    state = ClientState()
    server_state: dict[str, Any] = {}
    frame_count = 0
    local_fps_clock = time.perf_counter()
    local_fps_frames = 0

    print(f'Connecting to {args.server}')
    print('Press q or ESC to quit')

    try:
        health = requests.get(f'{args.server}/health', timeout=3)
        health.raise_for_status()
    except Exception as exc:
        raise SystemExit(f'Cannot reach server: {exc}')

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print('Failed to read frame')
                break

            if args.mirror:
                frame = cv2.flip(frame, 1)

            frame_count += 1
            local_fps_frames += 1
            now = time.perf_counter()
            elapsed = now - local_fps_clock
            if elapsed >= 1.0:
                state.fps = local_fps_frames / elapsed
                local_fps_clock = now
                local_fps_frames = 0

            if frame_count % args.send_every_n_frames == 0:
                start = time.perf_counter()
                payload = {'image': encode_frame(frame, quality=args.jpeg_quality), 'mirror': False}
                resp = requests.post(f'{args.server}/infer-frame', json=payload, timeout=30)
                resp.raise_for_status()
                server_state = resp.json()
                state.last_latency_ms = (time.perf_counter() - start) * 1000.0

                prediction = server_state.get('prediction', {})
                state.last_prediction = prediction.get('predicted_label', '---')
                state.last_confidence = float(prediction.get('confidence', 0.0))
                state.last_quality = prediction.get('quality_label')
                state.server_fps = float(prediction.get('fps', 0.0))

            if args.show_preview:
                preview = frame.copy()
                draw_overlay(preview, state, server_state)
                cv2.imshow('CTR-GCN Host Client', preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q'), ord('Q')):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
