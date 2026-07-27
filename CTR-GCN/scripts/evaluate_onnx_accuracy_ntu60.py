from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import onnxruntime as ort

from feeders.feeder_ntu import Feeder


DATA_PATH = Path(r"D:\B\python\xiaoke-project\ctrgcn-action-project\CTR-GCN\data\ntu\NTU60_CS.npz")
ONNX_PATH = Path(r"D:\B\python\xiaoke-project\ctrgcn-action-project\CTR-GCN\artifacts\ctrgcn_ntu60_joint.onnx")


def load_dataset(path: Path) -> Feeder:
    return Feeder(
        data_path=str(path),
        split="test",
        p_interval=[1],
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
    )


def to_ctrgcn_input(sample: np.ndarray) -> np.ndarray:
    if sample.ndim != 4:
        raise ValueError(f"Expected [C, T, V, M], got {sample.shape}")

    if sample.shape == (3, 64, 25, 2):
        return sample[None, ...].astype(np.float32)

    raise ValueError(f"Unsupported sample shape: {sample.shape}")


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
    dataset = load_dataset(data_path)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    preds = []
    labels = []
    total = len(dataset) if limit is None else min(limit, len(dataset))
    for i in range(total):
        sample, label, _ = dataset[i]
        inp = to_ctrgcn_input(sample)
        logits = run_onnx(sess, inp)
        preds.append(logits[0])
        labels.append(label)
        if (i + 1) % 1000 == 0 or i + 1 == total:
            print(f"Processed {i + 1}/{total}")

    logits = np.stack(preds, axis=0)
    y = np.asarray(labels, dtype=np.int64)
    top1 = topk_accuracy(logits, y, 1)
    top5 = topk_accuracy(logits, y, 5)
    pred_label = logits.argmax(axis=1)
    acc = float((pred_label == y).mean())

    print(f"ONNX: {onnx_path}")
    print(f"Data: {data_path}")
    print(f"Samples: {total}")
    print(f"Top-1: {top1:.4%}")
    print(f"Top-5: {top5:.4%}")
    print(f"Argmax accuracy: {acc:.4%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CTR-GCN ONNX accuracy on official NTU60 CS split")
    parser.add_argument("--onnx", type=Path, default=ONNX_PATH, help="Path to ONNX model")
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Path to official NTU60_CS.npz")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of samples to evaluate")
    args = parser.parse_args()

    evaluate(args.onnx, args.data, args.limit)


if __name__ == "__main__":
    main()
