from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class MappingConfig:
    input_landmarks: int = 33
    output_landmarks: int = 25
    coordinate_dims: int = 3


CFG = MappingConfig()


# MediaPipe 33 -> NTU-style 25 mapping
# NTU joint semantics are approximated so the official CTR-GCN backbone can be reused.
# Output layout is [T, 25, 3].

def _mean(points: np.ndarray, indices: Iterable[int]) -> np.ndarray:
    idx = list(indices)
    if not idx:
        raise ValueError("indices must not be empty")
    return points[:, idx, :3].mean(axis=1)


def mediapipe33_to_ntu25(landmarks: np.ndarray) -> np.ndarray:
    mp = np.asarray(landmarks, dtype=np.float32)

    if mp.ndim != 3:
        raise ValueError(f"Expected [T, 33, C], got {mp.shape}")
    if mp.shape[1] != CFG.input_landmarks:
        raise ValueError(f"Expected 33 landmarks, got {mp.shape[1]}")
    if mp.shape[2] < 3:
        raise ValueError(f"Expected at least 3 coordinates, got {mp.shape[2]}")

    xyz = mp[:, :, :3]
    t = xyz.shape[0]
    out = np.zeros((t, CFG.output_landmarks, CFG.coordinate_dims), dtype=np.float32)

    shoulder_center = (xyz[:, 11] + xyz[:, 12]) / 2.0
    hip_center = (xyz[:, 23] + xyz[:, 24]) / 2.0
    spine_mid = (shoulder_center + hip_center) / 2.0
    head_center = xyz[:, 0]

    left_hand_center = _mean(xyz, [17, 19, 21])
    right_hand_center = _mean(xyz, [18, 20, 22])

    # Torso / head
    out[:, 0] = hip_center
    out[:, 1] = spine_mid
    out[:, 2] = shoulder_center
    out[:, 3] = head_center

    # Left arm
    out[:, 4] = xyz[:, 11]
    out[:, 5] = xyz[:, 13]
    out[:, 6] = xyz[:, 15]
    out[:, 7] = left_hand_center

    # Right arm
    out[:, 8] = xyz[:, 12]
    out[:, 9] = xyz[:, 14]
    out[:, 10] = xyz[:, 16]
    out[:, 11] = right_hand_center

    # Left leg
    out[:, 12] = xyz[:, 23]
    out[:, 13] = xyz[:, 25]
    out[:, 14] = xyz[:, 27]
    out[:, 15] = xyz[:, 31]

    # Right leg
    out[:, 16] = xyz[:, 24]
    out[:, 17] = xyz[:, 26]
    out[:, 18] = xyz[:, 28]
    out[:, 19] = xyz[:, 32]

    # Extra torso / hand points
    out[:, 20] = shoulder_center
    out[:, 21] = xyz[:, 19]
    out[:, 22] = xyz[:, 21]
    out[:, 23] = xyz[:, 20]
    out[:, 24] = xyz[:, 22]

    return out


def center_like_ntu(sequence: np.ndarray, valid_frames: np.ndarray | None = None) -> np.ndarray:
    data = np.asarray(sequence, dtype=np.float32).copy()

    if data.ndim != 3 or data.shape[1:] != (25, 3):
        raise ValueError(f"Expected [T, 25, 3], got {data.shape}")

    if valid_frames is None:
        valid_frames = np.abs(data).sum(axis=(1, 2)) > 0

    valid_indices = np.flatnonzero(valid_frames)
    if len(valid_indices) == 0:
        return np.zeros_like(data)

    first_valid = int(valid_indices[0])
    origin = data[first_valid, 1].copy()

    data[valid_frames] -= origin
    data[~valid_frames] = 0.0
    return data


def load_mediapipe_npz(path: str | Path) -> dict[str, np.ndarray]:
    npz = np.load(path, allow_pickle=True)
    required = {"landmarks"}
    missing = required.difference(npz.files)
    if missing:
        raise KeyError(f"Missing keys in {path}: {sorted(missing)}")
    return {k: npz[k] for k in npz.files}


def save_mapped_sample(
    output_path: str | Path,
    landmarks_25: np.ndarray,
    **metadata: object,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, landmarks=landmarks_25.astype(np.float32), **metadata)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Convert MediaPipe 33-keypoints to NTU-style 25-keypoints")
    parser.add_argument("input", type=str, help="Input .npz file with key 'landmarks' shaped [T,33,3/4]")
    parser.add_argument("output", type=str, help="Output .npz file")
    parser.add_argument("--center", action="store_true", help="Apply NTU-style centering")
    args = parser.parse_args()

    data = load_mediapipe_npz(args.input)
    landmarks = data["landmarks"]

    mapped = mediapipe33_to_ntu25(landmarks)
    if args.center:
        mapped = center_like_ntu(mapped)

    save_mapped_sample(args.output, mapped, source=str(args.input), coordinate_space="world", schema_version="mediapipe33_to_ntu25_v1")
    print(f"Saved: {args.output}")
    print(f"Shape: {mapped.shape}")


if __name__ == "__main__":
    main()
