from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_LABELS = ["idle", "waving", "stop", "clapping"]


@dataclass
class TrainConfig:
    dataset_dir: str
    output_dir: str
    labels: list[str]
    window_size: int
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    val_ratio: float
    seed: int
    input_key: str
    feature_mode: str
    input_channels: int
    num_workers: int
    hidden_channels: int
    dropout: float
    device: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a TCN action classifier on webcam MediaPipe skeleton data")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "datasets" / "webcam_mediapipe")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "work_dir" / "webcam_mediapipe_tcn")
    parser.add_argument("--labels", type=str, default=",".join(DEFAULT_LABELS))
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-key", choices=("landmarks", "world_landmarks"), default="landmarks")
    parser.add_argument("--feature-mode", choices=("pose", "pose_hands", "pose_face", "pose_hands_face"), default="pose")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_sequence(path: Path, input_key: str, feature_mode: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if input_key not in data:
            raise KeyError(f"{path} does not contain key {input_key}")
        pose = np.asarray(data[input_key], dtype=np.float32)
        if pose.ndim != 3 or pose.shape[1] != 33 or pose.shape[2] < 4:
            raise ValueError(f"Expected [T,33,4+] in {path}, got {pose.shape}")
        pose = pose[:, :, :4]
        if feature_mode == "pose":
            return pose
        t = pose.shape[0]
        parts = [pose]
        if "hands" in feature_mode:
            left_hand = np.asarray(data["left_hand_landmarks"], dtype=np.float32)[:, :, :4] if "left_hand_landmarks" in data else np.zeros((t, 21, 4), dtype=np.float32)
            right_hand = np.asarray(data["right_hand_landmarks"], dtype=np.float32)[:, :, :4] if "right_hand_landmarks" in data else np.zeros((t, 21, 4), dtype=np.float32)
            if left_hand.shape[0] != t or right_hand.shape[0] != t:
                raise ValueError(f"Hand sequence length mismatch in {path}")
            parts.extend([left_hand, right_hand])
        if "face" in feature_mode:
            face = np.asarray(data["face_landmarks"], dtype=np.float32)[:, :, :4] if "face_landmarks" in data else np.zeros((t, 478, 4), dtype=np.float32)
            if face.shape[0] != t:
                raise ValueError(f"Face sequence length mismatch in {path}")
            parts.append(face)
        return np.concatenate(parts, axis=1).astype(np.float32)


def resize_sequence(sequence: np.ndarray, window_size: int) -> np.ndarray:
    t = sequence.shape[0]
    num_points = sequence.shape[1]
    num_dims = sequence.shape[2]
    if t == window_size:
        return sequence.astype(np.float32)
    if t <= 0:
        return np.zeros((window_size, num_points, num_dims), dtype=np.float32)
    if t == 1:
        return np.repeat(sequence, window_size, axis=0).astype(np.float32)
    src = np.linspace(0.0, 1.0, t, dtype=np.float32)
    dst = np.linspace(0.0, 1.0, window_size, dtype=np.float32)
    flat = sequence.reshape(t, -1)
    out = np.empty((window_size, flat.shape[1]), dtype=np.float32)
    for i in range(flat.shape[1]):
        out[:, i] = np.interp(dst, src, flat[:, i])
    return out.reshape(window_size, num_points, num_dims).astype(np.float32)


def normalize_sequence(sequence: np.ndarray) -> np.ndarray:
    seq = sequence.copy().astype(np.float32)
    xyzw = seq[:, :, :4]
    left_hip = xyzw[:, 23, :3]
    right_hip = xyzw[:, 24, :3]
    left_shoulder = xyzw[:, 11, :3]
    right_shoulder = xyzw[:, 12, :3]
    hip_center = (left_hip + right_hip) / 2.0
    shoulder_center = (left_shoulder + right_shoulder) / 2.0
    torso = np.linalg.norm(shoulder_center - hip_center, axis=1)
    scale = float(np.nanmedian(torso))
    if not np.isfinite(scale) or scale < 1e-4:
        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder, axis=1)
        scale = float(np.nanmedian(shoulder_width))
    if not np.isfinite(scale) or scale < 1e-4:
        scale = 1.0
    xyzw[:, :, :3] = (xyzw[:, :, :3] - hip_center[:, None, :]) / scale
    xyzw[:, :, 3] = np.clip(xyzw[:, :, 3], 0.0, 1.0)
    xyzw[~np.isfinite(xyzw)] = 0.0
    return xyzw


class WebcamMediaPipeDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, samples: list[tuple[Path, int]], window_size: int, input_key: str, feature_mode: str) -> None:
        self.samples = samples
        self.window_size = window_size
        self.input_key = input_key
        self.feature_mode = feature_mode

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[index]
        sequence = load_sequence(path, self.input_key, self.feature_mode)
        sequence = resize_sequence(sequence, self.window_size)
        sequence = normalize_sequence(sequence)
        x = torch.from_numpy(sequence.reshape(self.window_size, -1).T)
        y = torch.tensor(label, dtype=torch.long)
        return x, y


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = ((kernel_size - 1) * dilation) // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.net(x) + self.downsample(x))


class MediaPipeTCN(nn.Module):
    def __init__(self, num_classes: int, hidden_channels: int = 128, dropout: float = 0.25, input_channels: int = 33 * 4) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            TemporalBlock(input_channels, hidden_channels, kernel_size=5, dilation=1, dropout=dropout),
            TemporalBlock(hidden_channels, hidden_channels, kernel_size=5, dilation=2, dropout=dropout),
            TemporalBlock(hidden_channels, hidden_channels, kernel_size=5, dilation=4, dropout=dropout),
            TemporalBlock(hidden_channels, hidden_channels, kernel_size=3, dilation=1, dropout=dropout),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def collect_samples(dataset_dir: Path, labels: list[str]) -> list[tuple[Path, int]]:
    samples: list[tuple[Path, int]] = []
    for label_id, label in enumerate(labels):
        label_dir = dataset_dir / label
        if not label_dir.exists():
            print(f"Warning: missing label directory: {label_dir}")
            continue
        paths = sorted(label_dir.glob("*.npz"))
        print(f"{label_id}:{label} -> {len(paths)} samples")
        samples.extend((path, label_id) for path in paths)
    return samples


def split_samples(samples: list[tuple[Path, int]], val_ratio: float, seed: int) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    rng = random.Random(seed)
    by_label: dict[int, list[tuple[Path, int]]] = {}
    for sample in samples:
        by_label.setdefault(sample[1], []).append(sample)
    train: list[tuple[Path, int]] = []
    val: list[tuple[Path, int]] = []
    for label, label_samples in by_label.items():
        rng.shuffle(label_samples)
        val_count = max(1, int(round(len(label_samples) * val_ratio))) if len(label_samples) >= 2 else 0
        val.extend(label_samples[:val_count])
        train.extend(label_samples[val_count:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return float((pred == target).float().mean().item())


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, num_classes: int) -> tuple[float, float, np.ndarray]:
    model.eval()
    losses: list[float] = []
    accs: list[float] = []
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            losses.append(float(loss.item()))
            accs.append(accuracy(logits, y))
            pred = logits.argmax(dim=1).detach().cpu().numpy()
            true = y.detach().cpu().numpy()
            for t, p in zip(true, pred):
                confusion[int(t), int(p)] += 1
    return float(np.mean(losses)) if losses else 0.0, float(np.mean(accs)) if accs else 0.0, confusion


def save_checkpoint(path: Path, model: nn.Module, config: TrainConfig, epoch: int, val_acc: float, confusion: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(config),
            "epoch": epoch,
            "val_acc": val_acc,
            "confusion": confusion,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda:0")
    input_points = 33
    if "hands" in args.feature_mode:
        input_points += 42
    if "face" in args.feature_mode:
        input_points += 478
    input_channels = input_points * 4
    config = TrainConfig(
        dataset_dir=str(args.dataset_dir),
        output_dir=str(args.output_dir),
        labels=labels,
        window_size=args.window_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_ratio=args.val_ratio,
        seed=args.seed,
        input_key=args.input_key,
        feature_mode=args.feature_mode,
        input_channels=input_channels,
        num_workers=args.num_workers,
        hidden_channels=args.hidden_channels,
        dropout=args.dropout,
        device=str(device),
    )

    samples = collect_samples(args.dataset_dir, labels)
    if len(samples) < len(labels) * 2:
        raise RuntimeError(f"Too few samples: {len(samples)}. Collect more data before training.")
    train_samples, val_samples = split_samples(samples, args.val_ratio, args.seed)
    print(f"Total samples: {len(samples)} | train: {len(train_samples)} | val: {len(val_samples)}")

    train_loader = DataLoader(
        WebcamMediaPipeDataset(train_samples, args.window_size, args.input_key, args.feature_mode),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        WebcamMediaPipeDataset(val_samples, args.window_size, args.input_key, args.feature_mode),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = MediaPipeTCN(num_classes=len(labels), hidden_channels=args.hidden_channels, dropout=args.dropout, input_channels=input_channels).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "labels.json").write_text(json.dumps({str(i): label for i, label in enumerate(labels)}, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")

    best_acc = -1.0
    best_epoch = 0
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: list[float] = []
        train_accs: list[float] = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
            train_accs.append(accuracy(logits.detach(), y))
        scheduler.step()

        val_loss, val_acc, confusion = evaluate(model, val_loader, criterion, device, len(labels))
        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        train_acc = float(np.mean(train_accs)) if train_accs else 0.0
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc * 100:.1f}% | "
            f"val_loss={val_loss:.4f} val_acc={val_acc * 100:.1f}%"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            save_checkpoint(args.output_dir / "best_model.pt", model, config, epoch, val_acc, confusion)
            np.savetxt(args.output_dir / "best_confusion_matrix.txt", confusion, fmt="%d")

    save_checkpoint(args.output_dir / "last_model.pt", model, config, args.epochs, val_acc, confusion)
    elapsed = time.time() - start_time
    print(f"Training finished in {elapsed:.1f}s. Best val_acc={best_acc * 100:.1f}% at epoch {best_epoch}.")
    print(f"Best model: {args.output_dir / 'best_model.pt'}")
    print("Best confusion matrix rows=true labels, cols=pred labels:")
    print(np.loadtxt(args.output_dir / "best_confusion_matrix.txt", dtype=np.int64))


if __name__ == "__main__":
    main()
