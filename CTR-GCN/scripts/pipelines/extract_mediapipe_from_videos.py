from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABELS = ["idle", "waving", "clapping"]
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract MediaPipe Pose skeletons from action videos")
    parser.add_argument("--input-dir", type=Path, default=ROOT.parent / "dataset", help="Root directory containing idle/waving/clapping folders")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "datasets" / "mediapipe_pose", help="Output skeleton dataset directory")
    parser.add_argument("--labels", type=str, default=",".join(DEFAULT_LABELS), help="Comma-separated labels")
    parser.add_argument("--model-complexity", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--save-visualization", action="store_true", help="Also save visualized mp4 next to each npz")
    parser.add_argument("--min-valid-frames", type=int, default=8, help="Skip videos with too few valid frames")
    return parser.parse_args()


def write_labels(output_dir: Path, labels: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
    (output_dir / "labels.json").write_text(
        json.dumps({str(i): label for i, label in enumerate(labels)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def list_videos(label_dir: Path) -> list[Path]:
    return sorted(
        p for p in label_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


def extract_video(
    video_path: Path,
    output_path: Path,
    label_id: int,
    label_name: str,
    pose: mp.solutions.pose.Pose,
    save_visualization: bool,
) -> bool:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[skip] cannot open: {video_path}")
        return False

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    landmarks_seq: list[np.ndarray] = []
    valid_frames: list[bool] = []
    vis_frames: list[np.ndarray] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if result.pose_landmarks is None:
            landmarks_seq.append(np.zeros((33, 4), dtype=np.float32))
            valid_frames.append(False)
        else:
            landmarks = np.asarray(
                [[lm.x, lm.y, lm.z, lm.visibility] for lm in result.pose_landmarks.landmark],
                dtype=np.float32,
            )
            landmarks_seq.append(landmarks)
            valid_frames.append(True)

        if save_visualization:
            vis = frame.copy()
            if result.pose_landmarks is not None:
                mp.solutions.drawing_utils.draw_landmarks(
                    vis,
                    result.pose_landmarks,
                    mp.solutions.pose.POSE_CONNECTIONS,
                )
            vis_frames.append(vis)

    cap.release()

    if not landmarks_seq:
        print(f"[skip] empty video: {video_path}")
        return False

    valid_count = int(sum(valid_frames))
    if valid_count < min(8, len(landmarks_seq)):
        print(f"[skip] too few valid frames: {video_path} ({valid_count}/{len(landmarks_seq)})")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        landmarks=np.stack(landmarks_seq, axis=0).astype(np.float32),
        valid_frames=np.asarray(valid_frames, dtype=np.bool_),
        label=np.asarray(label_id, dtype=np.int64),
        label_name=np.asarray(label_name),
        fps=np.asarray(fps, dtype=np.float32),
        source_video=np.asarray(str(video_path)),
        schema=np.asarray("mediapipe_pose33_xyzw_v1"),
    )

    if save_visualization and vis_frames:
        h, w = vis_frames[0].shape[:2]
        out_video = output_path.with_suffix(".mp4")
        writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for frame in vis_frames:
            writer.write(frame)
        writer.release()

    print(f"[ok] {label_name}: {video_path.name} -> {output_path.name} | frames={len(landmarks_seq)} valid={valid_count}")
    return True


def main() -> None:
    args = parse_args()
    labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    if not labels:
        raise ValueError("No labels configured")

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    write_labels(args.output_dir, labels)

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=args.model_complexity,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    total = 0
    try:
        for label_id, label_name in enumerate(labels):
            input_label_dir = args.input_dir / label_name
            if not input_label_dir.is_dir():
                print(f"[warning] missing label dir: {input_label_dir}")
                continue

            videos = list_videos(input_label_dir)
            print(f"{label_name}: {len(videos)} videos")
            for video_path in videos:
                output_path = args.output_dir / label_name / f"{video_path.stem}.npz"
                if output_path.exists():
                    print(f"[skip] exists: {output_path}")
                    continue
                if extract_video(video_path, output_path, label_id, label_name, pose, args.save_visualization):
                    total += 1
    finally:
        pose.close()

    print(f"Finished. Converted {total} videos.")
    print(f"Output dataset: {args.output_dir}")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
