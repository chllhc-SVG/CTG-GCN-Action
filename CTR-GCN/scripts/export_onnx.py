from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import onnx
import torch

from model.ctrgcn import Model


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


def main() -> None:
    ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)

    model = Model(
        num_class=60,
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

    dummy_input = torch.randn(1, 3, 64, 25, 2, dtype=torch.float32)

    with torch.no_grad():
        output = model(dummy_input)

    print("PyTorch输入:", dummy_input.shape)
    print("PyTorch输出:", output.shape)

    torch.onnx.export(
        model,
        dummy_input,
        str(ONNX_PATH),
        input_names=["skeleton"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={"skeleton": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )

    onnx_model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(onnx_model)

    print("ONNX导出成功:", ONNX_PATH)


if __name__ == "__main__":
    main()
