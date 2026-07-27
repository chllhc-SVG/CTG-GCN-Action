from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

POSE_CONNECTIONS = tuple(mp.solutions.pose.POSE_CONNECTIONS)
HAND_CONNECTIONS = tuple(mp.solutions.hands.HAND_CONNECTIONS)
FACE_CONTOURS = tuple(mp.solutions.face_mesh.FACEMESH_CONTOURS)
DEFAULT_LABELS = [
    "idle",
    "waving",
    "stop",
    "clapping",
    "hands_together",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect a webcam MediaPipe skeleton action dataset")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "datasets" / "webcam_mediapipe")
    parser.add_argument("--labels", type=str, default=",".join(DEFAULT_LABELS), help="Comma separated action labels")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--record-seconds", type=float, default=3.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--display-width", type=int, default=1280, help="Displayed window width; does not affect saved landmarks")
    parser.add_argument("--display-height", type=int, default=960, help="Displayed window height; does not affect saved landmarks")
    parser.add_argument("--model-complexity", type=int, choices=(0, 1, 2), default=0, help="MediaPipe Pose model complexity; 0 is fastest")
    parser.add_argument("--use-hands", action="store_true", help="Also collect MediaPipe Hands landmarks")
    parser.add_argument("--use-face", action="store_true", help="Also collect MediaPipe FaceMesh landmarks")
    parser.add_argument("--face-refine", action="store_true", help="Enable refined FaceMesh landmarks around eyes/lips")
    parser.add_argument("--min-pose-frames", type=int, default=8)
    parser.add_argument("--save-video", action="store_true", help="Also save the raw recorded clip as mp4 for checking labels")
    parser.add_argument("--mirror", action="store_true", help="Mirror the saved image stream horizontally before MediaPipe")
    return parser.parse_args()


def next_sample_index(label_dir: Path) -> int:
    label_dir.mkdir(parents=True, exist_ok=True)
    indices: list[int] = []
    for path in label_dir.glob("*.npz"):
        try:
            indices.append(int(path.stem.split("_")[-1]))
        except ValueError:
            continue
    return max(indices, default=0) + 1


def landmark_arrays(results: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not results.pose_landmarks:
        return None, None
    image_landmarks = np.asarray(
        [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark],
        dtype=np.float32,
    )
    world_landmarks = None
    if results.pose_world_landmarks:
        world_landmarks = np.asarray(
            [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_world_landmarks.landmark],
            dtype=np.float32,
        )
    return image_landmarks, world_landmarks


def hand_arrays(results: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left_hand = np.zeros((21, 4), dtype=np.float32)
    right_hand = np.zeros((21, 4), dtype=np.float32)
    left_valid = np.asarray(False, dtype=np.bool_)
    right_valid = np.asarray(False, dtype=np.bool_)
    landmarks_list = getattr(results, "multi_hand_landmarks", None)
    handedness_list = getattr(results, "multi_handedness", None)
    if not landmarks_list or not handedness_list:
        return left_hand, right_hand, left_valid, right_valid
    for hand_landmarks, handedness in zip(landmarks_list, handedness_list):
        label = handedness.classification[0].label.lower() if handedness.classification else ""
        arr = np.asarray([[lm.x, lm.y, lm.z, 1.0] for lm in hand_landmarks.landmark], dtype=np.float32)
        if label == "left":
            left_hand = arr
            left_valid = np.asarray(True, dtype=np.bool_)
        elif label == "right":
            right_hand = arr
            right_valid = np.asarray(True, dtype=np.bool_)
    return left_hand, right_hand, left_valid, right_valid


def face_array(results: Any) -> tuple[np.ndarray, np.ndarray]:
    face = np.zeros((478, 4), dtype=np.float32)
    valid = np.asarray(False, dtype=np.bool_)
    landmarks_list = getattr(results, "multi_face_landmarks", None)
    if not landmarks_list:
        return face, valid
    landmarks = landmarks_list[0].landmark
    for i, lm in enumerate(landmarks[:478]):
        face[i] = [lm.x, lm.y, lm.z, 1.0]
    return face, np.asarray(True, dtype=np.bool_)


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
    landmarks = landmarks_list[0].landmark
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
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
    width = min(frame.shape[1] - 24, 980)
    height = 34 + 27 * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x + 12, y + 30 + i * 27), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (255, 255, 255), 2, cv2.LINE_AA)


def save_labels(output_dir: Path, labels: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "labels.json").write_text(json.dumps({str(i): label for i, label in enumerate(labels)}, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")


def save_sample(
    output_dir: Path,
    label_id: int,
    label_name: str,
    image_seq: list[np.ndarray],
    world_seq: list[np.ndarray | None],
    left_hand_seq: list[np.ndarray],
    right_hand_seq: list[np.ndarray],
    left_hand_valid_seq: list[np.ndarray],
    right_hand_valid_seq: list[np.ndarray],
    face_seq: list[np.ndarray],
    face_valid_seq: list[np.ndarray],
    frames_bgr: list[np.ndarray],
    fps: float,
    actual_width: int,
    actual_height: int,
    args: argparse.Namespace,
) -> Path:
    label_dir = output_dir / label_name
    sample_idx = next_sample_index(label_dir)
    sample_path = label_dir / f"{label_name}_{sample_idx:04d}.npz"
    world_valid = [w is not None for w in world_seq]
    world_filled = [w if w is not None else np.zeros((33, 4), dtype=np.float32) for w in world_seq]
    np.savez_compressed(
        sample_path,
        landmarks=np.stack(image_seq, axis=0).astype(np.float32),
        world_landmarks=np.stack(world_filled, axis=0).astype(np.float32),
        world_valid=np.asarray(world_valid, dtype=np.bool_),
        left_hand_landmarks=np.stack(left_hand_seq, axis=0).astype(np.float32) if left_hand_seq else np.zeros((len(image_seq), 21, 4), dtype=np.float32),
        right_hand_landmarks=np.stack(right_hand_seq, axis=0).astype(np.float32) if right_hand_seq else np.zeros((len(image_seq), 21, 4), dtype=np.float32),
        left_hand_valid=np.asarray(left_hand_valid_seq, dtype=np.bool_) if left_hand_valid_seq else np.zeros((len(image_seq),), dtype=np.bool_),
        right_hand_valid=np.asarray(right_hand_valid_seq, dtype=np.bool_) if right_hand_valid_seq else np.zeros((len(image_seq),), dtype=np.bool_),
        face_landmarks=np.stack(face_seq, axis=0).astype(np.float32) if face_seq else np.zeros((len(image_seq), 478, 4), dtype=np.float32),
        face_valid=np.asarray(face_valid_seq, dtype=np.bool_) if face_valid_seq else np.zeros((len(image_seq),), dtype=np.bool_),
        label=np.asarray(label_id, dtype=np.int64),
        label_name=np.asarray(label_name),
        fps=np.asarray(fps, dtype=np.float32),
        record_seconds=np.asarray(args.record_seconds, dtype=np.float32),
        schema=np.asarray("mediapipe_pose33_hands42_face478_xyzw_v3" if args.use_hands and args.use_face else "mediapipe_pose33_hands42_xyzw_v2" if args.use_hands else "mediapipe_pose33_face478_xyzw_v2" if args.use_face else "mediapipe33_xyzw_v1"),
        camera=np.asarray(args.camera, dtype=np.int64),
        requested_width=np.asarray(args.width, dtype=np.int64),
        requested_height=np.asarray(args.height, dtype=np.int64),
        actual_width=np.asarray(actual_width, dtype=np.int64),
        actual_height=np.asarray(actual_height, dtype=np.int64),
        mirrored=np.asarray(args.mirror, dtype=np.bool_),
        created_at=np.asarray(time.time(), dtype=np.float64),
    )
    if args.save_video and frames_bgr:
        video_path = sample_path.with_suffix(".mp4")
        h, w = frames_bgr[0].shape[:2]
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, fps), (w, h))
        for frame in frames_bgr:
            writer.write(frame)
        writer.release()
    return sample_path


def main() -> None:
    args = parse_args()
    labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    if not labels:
        raise ValueError("No labels configured")
    save_labels(args.output_dir, labels)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera}")
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or args.width)
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or args.height)

    selected_label = 0
    is_recording = False
    record_started_at = 0.0
    image_seq: list[np.ndarray] = []
    world_seq: list[np.ndarray | None] = []
    left_hand_seq: list[np.ndarray] = []
    right_hand_seq: list[np.ndarray] = []
    left_hand_valid_seq: list[np.ndarray] = []
    right_hand_valid_seq: list[np.ndarray] = []
    face_seq: list[np.ndarray] = []
    face_valid_seq: list[np.ndarray] = []
    frames_bgr: list[np.ndarray] = []
    sample_counts = {label: len(list((args.output_dir / label).glob("*.npz"))) for label in labels}
    last_message = "Select label with number keys, press r/SPACE/ENTER to record"

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
    ) if args.use_hands else None
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=args.face_refine,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) if args.use_face else None

    print("Labels:")
    for i, label in enumerate(labels):
        print(f"  {i}: {label}")
    print(f"Camera {args.camera}: requested {args.width}x{args.height}, actual {actual_width}x{actual_height}, display {args.display_width}x{args.display_height}, model_complexity={args.model_complexity}, use_hands={args.use_hands}, use_face={args.use_face}")
    print("Keys: number=select label, r/SPACE/ENTER=record, q/ESC=quit")

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
            image_landmarks, world_landmarks = landmark_arrays(results)
            left_hand, right_hand, left_hand_valid, right_hand_valid = hand_arrays(hand_results) if hand_results is not None else (
                np.zeros((21, 4), dtype=np.float32),
                np.zeros((21, 4), dtype=np.float32),
                np.asarray(False, dtype=np.bool_),
                np.asarray(False, dtype=np.bool_),
            )
            face_landmarks, face_valid = face_array(face_results) if face_results is not None else (
                np.zeros((478, 4), dtype=np.float32),
                np.asarray(False, dtype=np.bool_),
            )
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

            if is_recording:
                elapsed = now - record_started_at
                if image_landmarks is not None:
                    image_seq.append(image_landmarks)
                    world_seq.append(world_landmarks)
                    left_hand_seq.append(left_hand)
                    right_hand_seq.append(right_hand)
                    left_hand_valid_seq.append(left_hand_valid)
                    right_hand_valid_seq.append(right_hand_valid)
                    face_seq.append(face_landmarks)
                    face_valid_seq.append(face_valid)
                    if args.save_video:
                        frames_bgr.append(frame.copy())
                remain = max(0.0, args.record_seconds - elapsed)
                cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], display_frame.shape[0]), (0, 0, 255), 8)
                cv2.putText(display_frame, f"REC {remain:.1f}s", (display_frame.shape[1] - 230, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
                if elapsed >= args.record_seconds:
                    is_recording = False
                    label_name = labels[selected_label]
                    if len(image_seq) >= args.min_pose_frames:
                        sample_path = save_sample(args.output_dir, selected_label, label_name, image_seq, world_seq, left_hand_seq, right_hand_seq, left_hand_valid_seq, right_hand_valid_seq, face_seq, face_valid_seq, frames_bgr, fps, actual_width, actual_height, args)
                        sample_counts[label_name] = sample_counts.get(label_name, 0) + 1
                        last_message = f"Saved {sample_path} pose_frames={len(image_seq)}"
                        print(last_message)
                    else:
                        last_message = f"Discarded: too few pose frames ({len(image_seq)}). Step back/show full body."
                        print(last_message)

            label_lines = [f"{i}:{label}({sample_counts.get(label, 0)})" for i, label in enumerate(labels)]
            lines = [
                f"Collect MediaPipe dataset | label={selected_label}:{labels[selected_label]} | FPS={fps:.1f} | {actual_width}x{actual_height}",
                "Keys: 0-9 select label | r/SPACE/ENTER record | q/ESC quit",
                "Labels: " + "  ".join(label_lines[:6]),
                last_message,
            ]
            if len(label_lines) > 6:
                lines.insert(3, "Labels: " + "  ".join(label_lines[6:12]))
            put_panel(display_frame, lines)
            if args.display_width > 0 and args.display_height > 0:
                display_frame = cv2.resize(display_frame, (args.display_width, args.display_height), interpolation=cv2.INTER_LINEAR)
            cv2.namedWindow("Collect Webcam MediaPipe Dataset", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Collect Webcam MediaPipe Dataset", max(320, args.display_width), max(240, args.display_height))
            cv2.imshow("Collect Webcam MediaPipe Dataset", display_frame)
            key = cv2.waitKeyEx(1)
            if key != -1:
                key = key & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if not is_recording and ord("0") <= key <= ord("9"):
                idx = key - ord("0")
                if idx < len(labels):
                    selected_label = idx
                    last_message = f"Selected label {selected_label}:{labels[selected_label]}"
            if not is_recording and key in (ord("r"), ord("R"), 13, 10, 32):
                image_seq = []
                world_seq = []
                left_hand_seq = []
                right_hand_seq = []
                left_hand_valid_seq = []
                right_hand_valid_seq = []
                face_seq = []
                face_valid_seq = []
                frames_bgr = []
                is_recording = True
                record_started_at = time.perf_counter()
                last_message = f"Recording {labels[selected_label]}... perform the action now"
                print(last_message)
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
