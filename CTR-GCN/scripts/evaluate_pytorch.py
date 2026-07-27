from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from model.ctrgcn import Model


DATA_PATH = Path(r"D:\B\python\xiaoke-project\dataset\ntu_processed\ntu_skeleton_64.npz")
WEIGHT_PATH = Path("work_dir/ntu60/xsub/ctrgcn_joint/joint.pt")


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "net"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break

    if not isinstance(checkpoint, dict):
        raise TypeError("权重文件不是有效的 state_dict")

    result: dict[str, torch.Tensor] = {}
    for key, value in checkpoint.items():
        if not torch.is_tensor(value):
            continue
        if key.startswith("module."):
            key = key[len("module.") :]
        result[key] = value
    return result


def load_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(path)

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")

    return extract_state_dict(checkpoint)


class NumpySkeletonDataset(Dataset):
    def __init__(self, npz_path: Path) -> None:
        if not npz_path.exists():
            raise FileNotFoundError(npz_path)

        data = np.load(npz_path)
        self.x = data["X"].astype(np.float32)
        self.y = data["y"].astype(np.int64)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        sample = self.x[index]  # [T, V, C]
        sample = np.transpose(sample, (2, 0, 1))  # [C, T, V]
        sample = np.expand_dims(sample, axis=-1)  # [C, T, V, M=1]
        sample = np.repeat(sample, 2, axis=-1)  # [C, T, V, 2]
        sample = np.ascontiguousarray(sample, dtype=np.float32)
        label = int(self.y[index])
        return sample, label, index


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = NumpySkeletonDataset(DATA_PATH)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    num_class = int(np.max(dataset.y)) + 1

    model = Model(
        num_class=num_class,
        num_point=25,
        num_person=2,
        graph="graph.ntu_rgb_d.Graph",
        graph_args={"labeling_mode": "spatial"},
        in_channels=3,
    )

    state_dict = load_checkpoint(WEIGHT_PATH)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print("missing:", missing)
    print("unexpected:", unexpected)

    model.to(device)
    model.eval()

    total = 0
    top1_correct = 0
    top5_correct = 0

    with torch.no_grad():
        for data, label, _ in loader:
            x = torch.as_tensor(data, dtype=torch.float32, device=device)
            y = torch.as_tensor(label, dtype=torch.long, device=device)

            logits = model(x)
            if logits.shape[1] != num_class:
                raise RuntimeError(f"输出类别数错误: {logits.shape}")

            topk = min(5, num_class)
            top5 = logits.topk(k=topk, dim=1).indices
            top1_correct += (top5[:, 0] == y).sum().item()
            top5_correct += (top5 == y[:, None]).any(dim=1).sum().item()
            total += y.size(0)

    print(f"类别数: {num_class}")
    print(f"样本数: {total}")
    print(f"Top-1: {top1_correct / total:.4%}")
    print(f"Top-{min(5, num_class)}: {top5_correct / total:.4%}")


if __name__ == "__main__":
    main()
