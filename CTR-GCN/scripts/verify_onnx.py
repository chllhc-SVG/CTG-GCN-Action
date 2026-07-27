from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import onnxruntime as ort
import torch

from model.ctrgcn import Model


DATA_PATH = Path(r"D:\B\python\xiaoke-project\dataset\ntu_processed\ntu_skeleton_64.npz")
WEIGHT_PATH = Path("work_dir/ntu60/xsub/ctrgcn_joint/joint.pt")
ONNX_PATH = Path("artifacts/ctrgcn_ntu60_joint.onnx")


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


def load_sample() -> tuple[np.ndarray, int, str]:
    data = np.load(DATA_PATH)
    x = data["X"]
    y = data["y"]
    names = data["names"]

    if len(x) == 0:
        raise RuntimeError("数据集为空")

    sample = np.asarray(x[0], dtype=np.float32)
    label = int(y[0])
    name = str(names[0])
    return sample, label, name


def to_ctrgcn_input(sample: np.ndarray) -> np.ndarray:
    if sample.ndim != 3 or sample.shape != (64, 25, 3):
        raise ValueError(f"样本形状错误: {sample.shape}")

    # [T, V, C] -> [N, C, T, V, M]
    # 这里补成两个人的格式，第二个人全零，和官方 NTU 输入保持一致。
    x = sample.transpose(2, 0, 1)  # [C, T, V]
    x = x[..., None]  # [C, T, V, 1]
    x = np.concatenate([x, np.zeros_like(x)], axis=-1)  # [C, T, V, 2]
    return x[None, ...]


def main() -> None:
    sample, label, name = load_sample()
    x_numpy = to_ctrgcn_input(sample)

    num_class = int(np.max(np.load(DATA_PATH)["y"])) + 1

    model = Model(
        num_class=num_class,
        num_point=25,
        num_person=2,
        graph="graph.ntu_rgb_d.Graph",
        graph_args={"labeling_mode": "spatial"},
        in_channels=3,
    )

    try:
        checkpoint = torch.load(WEIGHT_PATH, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(WEIGHT_PATH, map_location="cpu")

    model.load_state_dict(extract_state_dict(checkpoint), strict=True)
    model.eval()

    with torch.no_grad():
        torch_output = model(torch.from_numpy(x_numpy)).cpu().numpy()

    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    onnx_output = session.run(["logits"], {"skeleton": x_numpy})[0]

    max_abs_error = np.max(np.abs(torch_output - onnx_output))
    torch_pred = int(torch_output.argmax(axis=1)[0])
    onnx_pred = int(onnx_output.argmax(axis=1)[0])

    print("样本:", name)
    print("真实标签:", label)
    print("PyTorch预测:", torch_pred)
    print("ONNX预测:", onnx_pred)
    print("最大绝对误差:", max_abs_error)

    np.testing.assert_allclose(torch_output, onnx_output, rtol=1e-3, atol=1e-4)
    assert torch_pred == onnx_pred

    print("PyTorch和ONNX输出一致")


if __name__ == "__main__":
    main()
