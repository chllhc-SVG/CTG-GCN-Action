"""
Match a live skeleton sequence against an action's standard templates.

This module is intentionally lightweight so it does not affect the original
CTR-GCN action recognition pipeline. If the template directory is missing,
matching simply stays disabled.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class TemplateMatcher:
    def __init__(self, templates_dir: str | Path, threshold: float = 0.55):
        self.templates_dir = Path(templates_dir)
        self.threshold = float(threshold)
        self.templates = self._load_templates()

    def _load_templates(self) -> dict[str, list[np.ndarray]]:
        templates: dict[str, list[np.ndarray]] = {}
        if not self.templates_dir.exists():
            return templates

        for action_dir in sorted(self.templates_dir.iterdir()):
            if not action_dir.is_dir():
                continue
            action_templates: list[np.ndarray] = []
            for npy_path in sorted(action_dir.glob("*.npy")):
                try:
                    feat = np.load(npy_path)
                    if feat.ndim == 3:
                        action_templates.append(feat.astype(np.float32))
                except Exception:
                    continue
            if action_templates:
                templates[action_dir.name] = action_templates
        return templates

    @staticmethod
    def _prepare_feature(data: np.ndarray) -> np.ndarray:
        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim == 5:
            arr = arr[0]
        if arr.ndim == 4:
            arr = arr[:, :, :, 0]
        if arr.ndim != 3:
            raise ValueError(f"Unsupported feature shape: {arr.shape}")
        return arr

    @staticmethod
    def _distance(a: np.ndarray, b: np.ndarray) -> float:
        if a.shape != b.shape:
            raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
        return float(np.mean((a - b) ** 2))

    def judge(self, action_name: str, current_feature: np.ndarray) -> tuple[str, float]:
        """Return (quality_label, distance)."""
        if action_name not in self.templates:
            return "unknown", float("inf")

        feature = self._prepare_feature(current_feature)
        best_distance = min(self._distance(feature, tmpl) for tmpl in self.templates[action_name])
        quality = "standard" if best_distance < self.threshold else "non_standard"
        return quality, best_distance
