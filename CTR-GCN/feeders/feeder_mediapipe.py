"""
Feeder for MediaPipe 33-point skeleton data.

Each sample is stored as an .npz file produced by collect_webcam_mediapipe.py
or extract_mediapipe_from_videos.py, containing at minimum:

    landmarks: [T, 33, 4]   (x, y, z, visibility) — image-space normalized

The feeder reads these scattered files directly (no pre-packing into a single .npz)
and produces tensors shaped [C, T, V, M] = [3, window_size, 33, 2].
"""

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from feeders import tools


class Feeder(Dataset):
    def __init__(
        self,
        data_path,
        label_path=None,
        split='train',
        val_ratio=0.2,
        seed=42,
        p_interval=1,
        random_choose=False,
        random_shift=False,
        random_move=False,
        random_rot=False,
        window_size=64,
        normalization=False,
        debug=False,
        use_mmap=False,
        bone=False,
        vel=False,
    ):
        self.data_path = Path(data_path)
        self.split = split
        self.val_ratio = val_ratio
        self.seed = seed
        self.p_interval = p_interval
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.random_rot = random_rot
        self.window_size = window_size
        self.normalization = normalization
        self.debug = debug
        self.bone = bone
        self.vel = vel
        self.num_point = 33
        self.num_person = 2

        self.samples = []      # list of (file_path, label_id)
        self.label_map = {}    # label_name -> label_id
        self.id_to_label = {}  # label_id -> label_name

        self._collect_samples()
        self._split_samples()

        if self.debug:
            self.samples = self.samples[:100]

    def _resolve_label_dir(self, label_name):
        candidates = [
            self.data_path / label_name,
            self.data_path / label_name.replace('_', ' '),
            self.data_path / label_name.replace(' ', '_'),
        ]
        seen = set()
        for candidate in candidates:
            candidate_key = str(candidate)
            if candidate_key in seen:
                continue
            seen.add(candidate_key)
            if candidate.is_dir():
                return candidate
        return None

    def _collect_samples(self):
        # Try to load labels.json or labels.txt from dataset root
        labels_json = self.data_path / 'labels.json'
        labels_txt = self.data_path / 'labels.txt'

        labels = []
        if labels_json.exists():
            with labels_json.open('r', encoding='utf-8') as f:
                label_dict = json.load(f)
            labels = [label_dict[k] for k in sorted(label_dict.keys(), key=int)]
        elif labels_txt.exists():
            with labels_txt.open('r', encoding='utf-8') as f:
                labels = [line.strip() for line in f if line.strip()]
        else:
            # Auto-discover label directories
            for d in sorted(self.data_path.iterdir()):
                if d.is_dir() and not d.name.startswith('.'):
                    labels.append(d.name)

        for label_id, label_name in enumerate(labels):
            self.label_map[label_name] = label_id
            self.id_to_label[label_id] = label_name

            label_dir = self._resolve_label_dir(label_name)
            if label_dir is None:
                continue
            for npz_file in sorted(label_dir.glob('*.npz')):
                self.samples.append((str(npz_file), label_id))

    def _split_samples(self):
        rng = random.Random(self.seed)
        by_label = {}
        for sample in self.samples:
            by_label.setdefault(sample[1], []).append(sample)

        train, test = [], []
        for label_id, label_samples in by_label.items():
            rng.shuffle(label_samples)
            val_count = max(1, int(round(len(label_samples) * self.val_ratio))) if len(label_samples) >= 2 else 0
            if self.split == 'train':
                train.extend(label_samples[val_count:])
            else:
                test.extend(label_samples[:val_count])

        rng.shuffle(train)
        rng.shuffle(test)
        self.samples = train if self.split == 'train' else test

    def __len__(self):
        return len(self.samples)

    def _load_landmarks(self, path):
        data = np.load(path, allow_pickle=True)
        if 'landmarks' in data.files:
            pose = np.asarray(data['landmarks'], dtype=np.float32)
        elif 'world_landmarks' in data.files:
            pose = np.asarray(data['world_landmarks'], dtype=np.float32)
        else:
            raise KeyError(f"No 'landmarks' or 'world_landmarks' key in {path}")

        # Expected: [T, 33, >=3]
        if pose.ndim != 3 or pose.shape[1] != self.num_point:
            raise ValueError(f"Expected [T, {self.num_point}, C] in {path}, got {pose.shape}")

        # Use only x, y, z (first 3 channels)
        return pose[:, :, :3]

    def _normalize(self, sequence):
        """Center on hip midpoint, scale by torso length."""
        seq = sequence.copy().astype(np.float32)
        left_hip = seq[:, 23, :]
        right_hip = seq[:, 24, :]
        left_shoulder = seq[:, 11, :]
        right_shoulder = seq[:, 12, :]
        hip_center = (left_hip + right_hip) / 2.0
        shoulder_center = (left_shoulder + right_shoulder) / 2.0
        torso = np.linalg.norm(shoulder_center - hip_center, axis=1)
        scale = float(np.nanmedian(torso))
        if not np.isfinite(scale) or scale < 1e-4:
            scale = 1.0
        seq = (seq - hip_center[:, None, :]) / scale
        seq[~np.isfinite(seq)] = 0.0
        return seq

    def __getitem__(self, index):
        path, label = self.samples[index]
        pose = self._load_landmarks(path)

        if self.normalization:
            pose = self._normalize(pose)

        # pose: [T, 33, 3] -> [C=3, T, V=33, M=1]
        T = pose.shape[0]
        data = pose.transpose(2, 0, 1)  # [3, T, 33]
        data = data[:, :, :, np.newaxis]  # [3, T, 33, 1]

        # Pad or crop to window_size
        if data.shape[1] < self.window_size:
            pad = np.zeros((3, self.window_size, self.num_point, 1), dtype=np.float32)
            pad[:, :data.shape[1], :, :] = data
            data = pad
        elif data.shape[1] > self.window_size:
            # Random crop or center crop
            if self.split == 'train' and self.random_choose:
                begin = random.randint(0, data.shape[1] - self.window_size)
            else:
                begin = (data.shape[1] - self.window_size) // 2
            data = data[:, begin:begin + self.window_size, :, :]

        # valid_crop_resize expects [C, T, V, M]
        valid_frame_num = np.sum(data.sum(0).sum(-1).sum(-1) != 0)
        if valid_frame_num == 0:
            valid_frame_num = 1
        data = tools.valid_crop_resize(data, valid_frame_num, self.p_interval, self.window_size)

        if self.random_rot:
            data = tools.random_rot(data)

        if self.bone:
            from .bone_pairs_mediapipe import mediapipe_pairs
            bone_data = np.zeros_like(data)
            for v1, v2 in mediapipe_pairs:
                bone_data[:, :, v1] = data[:, :, v1] - data[:, :, v2]
            data = bone_data

        if self.vel:
            data[:, :-1] = data[:, 1:] - data[:, :-1]
            data[:, -1] = 0

        # Ensure M=2 for CTR-GCN compatibility
        if data.shape[3] == 1:
            data = np.concatenate([data, np.zeros_like(data)], axis=3)

        return data.astype(np.float32), label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        labels = [s[1] for s in self.samples]
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(labels)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
