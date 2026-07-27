from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import onnxruntime as ort


DATA_PATH = Path(r"D:\B\python\xiaoke-project\dataset\ntu_processed\ntu_skeleton_64.npz")
ONNX_PATH = Path(r"D:\B\python\xiaoke-project\ctrgcn-action-project\CTR-GCN\artifacts\ctrgcn_ntu60_joint.onnx")


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    if not {"X", "y", "names"}.issubset(data.files):
        raise KeyError(f"{path} must contain X, y, names; got {data.files}")
    return data["X"], data["y"], data["names"]


def to_ctrgcn_input(sample: np.ndarray) -> np.ndarray:
    if sample.ndim != 3 or sample.shape != (64, 25, 3):
        raise ValueError(f"Expected sample shape (64, 25, 3), got {sample.shape}")

    x = sample.transpose(2, 0, 1)  # [C, T, V]
    x = x[..., None]               # [C, T, V, 1]
    x = np.concatenate([x, np.zeros_like(x)], axis=-1)  # [C, T, V, 2]
    return x[None, ...].astype(np.float32)               # [1, C, T, V, M]


def run_onnx(sess: ort.InferenceSession, sample: np.ndarray) -> np.ndarray:
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    logits = sess.run([output_name], {input_name: sample})[0]
    return np.asarray(logits, dtype=np.float32)


def topk_accuracy(logits: np.ndarray, labels: np.ndarray, k: int) -> float:
    k = min(k, logits.shape[1])
    topk = np.argpartition(-logits, kth=k - 1, axis=1)[:, :k]
    hits = (topk == labels[:, None]).any(axis=1)
    return float(hits.mean())


def evaluate(onnx_path: Path, data_path: Path, limit: int | None = None) -> None:
    x, y, names = load_dataset(data_path)

    if limit is not None:
        x = x[:limit]
        y = y[:limit]
        names = names[:limit]

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    preds = []
    for i, sample in enumerate(x):
        inp = to_ctrgcn_input(sample)
        logits = run_onnx(sess, inp)
        preds.append(logits[0])
        if (i + 1) % 1000 == 0 or i + 1 == len(x):
            print(f"Processed {i + 1}/{len(x)}")

    logits = np.stack(preds, axis=0)
    top1 = topk_accuracy(logits, y, 1)
    top5 = topk_accuracy(logits, y, 5)
    pred_label = logits.argmax(axis=1)
    acc = float((pred_label == y).mean())

    print(f"ONNX: {onnx_path}")
    print(f"Data: {data_path}")
    print(f"Samples: {len(x)}")
    print(f"Top-1: {top1:.4%}")
    print(f"Top-5: {top5:.4%}")
    print(f"Argmax accuracy: {acc:.4%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CTR-GCN ONNX accuracy")
    parser.add_argument("--onnx", type=Path, default=ONNX_PATH, help="Path to ONNX model")
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Path to dataset npz with X/y/names")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of samples to evaluate")
    args = parser.parse_args()

    evaluate(args.onnx, args.data, args.limit)


if __name__ == "__main__":
    main()
