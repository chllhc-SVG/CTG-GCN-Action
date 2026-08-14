from __future__ import annotations

import argparse
import ctypes
import json
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABELS = ["idle", "waving", "clapping"]
POSE_CONNECTIONS = tuple(mp.solutions.pose.POSE_CONNECTIONS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect fixed-length webcam RGB videos and MediaPipe Pose skeletons"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "datasets" / "webcam_mediapipe",
    )
    parser.add_argument(
        "--labels",
        default=",".join(DEFAULT_LABELS),
        help="Comma-separated labels selected with number keys",
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--record-seconds", type=float, default=3.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--model-complexity", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--min-valid-ratio", type=float, default=0.9)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--keep-aspect", action="store_true", default=True)
    return parser.parse_args()


def save_labels(output_dir: Path, labels: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "labels.txt").write_text(
        "\n".join(labels) + "\n",
        encoding="utf-8",
    )
    (output_dir / "labels.json").write_text(
        json.dumps(
            {str(index): label for index, label in enumerate(labels)},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def next_sample_path(output_dir: Path, label: str) -> Path:
    label_dir = output_dir / label
    label_dir.mkdir(parents=True, exist_ok=True)
    existing = []
    for path in label_dir.glob(f"{label}_*.npz"):
        try:
            existing.append(int(path.stem.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return label_dir / f"{label}_{max(existing, default=0) + 1:04d}.npz"


def draw_pose(frame: np.ndarray, result: object) -> None:
    pose_landmarks = getattr(result, "pose_landmarks", None)
    if pose_landmarks is None:
        return

    height, width = frame.shape[:2]
    points: list[tuple[int, int] | None] = []
    for landmark in pose_landmarks.landmark:
        if landmark.visibility < 0.35:
            points.append(None)
        else:
            points.append((int(landmark.x * width), int(landmark.y * height)))

    for start, end in POSE_CONNECTIONS:
        if points[start] is not None and points[end] is not None:
            cv2.line(frame, points[start], points[end], (64, 220, 255), 2, cv2.LINE_AA)
    for point in points:
        if point is not None:
            cv2.circle(frame, point, 3, (0, 255, 80), -1, cv2.LINE_AA)


def draw_panel(frame: np.ndarray, lines: list[str]) -> None:
    x, y = 12, 12
    height = 18 + 27 * len(lines)
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x, y),
        (frame.shape[1] - 12, y + height),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x + 12, y + 25 + index * 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


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


def landmarks_from_result(result: object) -> np.ndarray:
    pose_landmarks = getattr(result, "pose_landmarks", None)
    if pose_landmarks is None:
        return np.zeros((33, 4), dtype=np.float32)
    return np.asarray(
        [
            [landmark.x, landmark.y, landmark.z, landmark.visibility]
            for landmark in pose_landmarks.landmark
        ],
        dtype=np.float32,
    )


def save_sample(
    output_dir: Path,
    label_id: int,
    label: str,
    landmarks: list[np.ndarray],
    valid_frames: list[bool],
    video_frames: list[np.ndarray],
    fps: float,
    args: argparse.Namespace,
) -> Path:
    sample_path = next_sample_path(output_dir, label)
    np.savez_compressed(
        sample_path,
        landmarks=np.stack(landmarks).astype(np.float32),
        valid_frames=np.asarray(valid_frames, dtype=np.bool_),
        label=np.asarray(label_id, dtype=np.int64),
        label_name=np.asarray(label),
        fps=np.asarray(fps, dtype=np.float32),
        record_seconds=np.asarray(args.record_seconds, dtype=np.float32),
        camera=np.asarray(args.camera, dtype=np.int64),
        width=np.asarray(args.width, dtype=np.int64),
        height=np.asarray(args.height, dtype=np.int64),
        mirrored=np.asarray(args.mirror, dtype=np.bool_),
        schema=np.asarray("mediapipe_pose33_xyzw_v1"),
    )

    if args.save_video and video_frames:
        video_path = sample_path.with_suffix(".mp4")
        height, width = video_frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            max(fps, 1.0),
            (width, height),
        )
        for frame in video_frames:
            writer.write(frame)
        writer.release()

    return sample_path


def main() -> None:
    _set_windows_hidpi_aware()
    args = parse_args()
    labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    if not labels:
        raise ValueError("At least one label is required")
    if args.record_seconds <= 0:
        raise ValueError("--record-seconds must be greater than zero")
    if not 0 < args.min_valid_ratio <= 1:
        raise ValueError("--min-valid-ratio must be in the range (0, 1]")

    save_labels(args.output_dir, labels)
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera}")

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or args.width)
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or args.height)
    if actual_width <= 0:
        actual_width = args.width
    if actual_height <= 0:
        actual_height = args.height
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=args.model_complexity,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    counts = {
        label: len(list((args.output_dir / label).glob(f"{label}_*.npz")))
        for label in labels
    }
    selected_label = 0
    recording_mode = False
    stop_after_current_clip = False
    clip_started = 0.0
    clip_open = False
    landmarks: list[np.ndarray] = []
    valid_frames: list[bool] = []
    video_frames: list[np.ndarray] = []
    message = "Select a label with 0-9, then press R to toggle continuous 3-second recording"

    print(f"Output: {args.output_dir}")
    print(f"Camera: {actual_width}x{actual_height}, target duration: {args.record_seconds:.1f}s")
    print("Labels:")
    for index, label in enumerate(labels):
        print(f"  {index}: {label}")
    print("Keys: 0-9 select label | R/SPACE record | Q/ESC quit")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Failed to read a frame from the camera")

            if args.mirror:
                frame = cv2.flip(frame, 1)
            result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            skeleton = landmarks_from_result(result)
            has_pose = getattr(result, "pose_landmarks", None) is not None
            display = frame.copy()
            draw_pose(display, result)

            now = time.perf_counter()
            if recording_mode:
                landmarks.append(skeleton)
                valid_frames.append(has_pose)
                if args.save_video:
                    video_frames.append(frame.copy())
                if not clip_open:
                    clip_started = now
                    clip_open = True
                remaining = max(0.0, args.record_seconds - (now - clip_started))
                cv2.putText(
                    display,
                    f"REC {remaining:.1f}s",
                    (display.shape[1] - 180, 42),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                if now - clip_started >= args.record_seconds:
                    valid_ratio = sum(valid_frames) / max(len(valid_frames), 1)
                    if valid_ratio >= args.min_valid_ratio:
                        sample_path = save_sample(
                            args.output_dir,
                            selected_label,
                            labels[selected_label],
                            landmarks,
                            valid_frames,
                            video_frames,
                            fps,
                            args,
                        )
                        counts[labels[selected_label]] += 1
                        message = (
                            f"Saved {sample_path.name} | frames={len(landmarks)} "
                            f"valid={valid_ratio:.1%}"
                        )
                        print(message)
                    else:
                        message = (
                            f"Discarded: pose valid ratio {valid_ratio:.1%} "
                            f"is below {args.min_valid_ratio:.1%}"
                        )
                        print(message)
                    landmarks = []
                    valid_frames = []
                    video_frames = []
                    clip_started = now
                    clip_open = False

            label_summary = " ".join(
                f"{index}:{label}={counts[label]}"
                for index, label in enumerate(labels)
            )
            draw_panel(
                display,
                [
                    f"MediaPipe Pose | label {selected_label}:{labels[selected_label]} | {actual_width}x{actual_height}",
                    f"Samples: {label_summary}",
                    message,
                ],
            )
            display = _fit_frame_to_screen(display)
            cv2.namedWindow("Collect MediaPipe RGB Skeleton Dataset", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("Collect MediaPipe RGB Skeleton Dataset", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("Collect MediaPipe RGB Skeleton Dataset", display)
            key = cv2.waitKeyEx(1)
            key = key & 0xFF if key != -1 else -1

            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("r"), ord("R")):
                recording_mode = not recording_mode
                if recording_mode:
                    landmarks = []
                    valid_frames = []
                    video_frames = []
                    clip_started = 0.0
                    clip_open = False
                    message = f"Recording {labels[selected_label]} clips every {args.record_seconds:.1f}s"
                else:
                    message = "Recording paused"
            if ord("0") <= key <= ord("9"):
                index = key - ord("0")
                if index < len(labels):
                    selected_label = index
                    message = f"Selected {selected_label}:{labels[selected_label]}"
    finally:
        pose.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
