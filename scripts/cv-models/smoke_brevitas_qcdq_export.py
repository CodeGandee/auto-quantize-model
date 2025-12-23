#!/usr/bin/env python
"""Smoke test: Brevitas QCDQ ONNX export under Torch 2.9 (rtx5090 env).

This script exists to validate the reusable compatibility shim in
`auto_quantize_model.brevitas_onnx_export_compat` without any YOLO-specific
complexity.

It:
1) Builds a tiny quantized Conv+ReLU module (W4A8-like).
2) Exports it to QCDQ ONNX via Brevitas (`dynamo=False`).
3) Validates the ONNX loads and can run one ORT inference (CUDA preferred).

Outputs are written under `tmp/` and are safe to delete.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, List

import numpy as np
import onnx
import onnxruntime as ort  # type: ignore[import-untyped]
import torch

from auto_quantize_model.brevitas_onnx_export_compat import (
    apply_brevitas_torch_onnx_compat,
    get_brevitas_onnx_compat_status,
)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test Brevitas QCDQ ONNX export on Torch 2.9.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tmp/brevitas_qcdq_smoke"),
        help="Output directory root (run-id subdir is created inside).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run id (defaults to timestamp).",
    )
    parser.add_argument(
        "--providers",
        type=str,
        nargs="+",
        default=["CUDAExecutionProvider", "CPUExecutionProvider"],
        help="ONNX Runtime providers in priority order.",
    )
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version for export.")
    return parser.parse_args(argv)


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def build_providers(requested: list[str]) -> list[str]:
    available = set(ort.get_available_providers())
    filtered = [p for p in requested if p in available]
    return filtered or ["CPUExecutionProvider"]


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    run_id = args.run_id or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_root = args.out_dir / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    onnx_path = run_root / "toy-w4a8-qcdq.onnx"

    apply_brevitas_torch_onnx_compat()

    import brevitas.nn as qnn  # type: ignore[import-untyped]
    from brevitas.export import export_onnx_qcdq  # type: ignore[import-untyped]
    from brevitas.quant.scaled_int import Int4WeightPerTensorFloatDecoupled, Int8ActPerTensorFloat  # type: ignore[import-untyped]

    class Toy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = qnn.QuantConv2d(
                3,
                8,
                3,
                padding=1,
                weight_quant=Int4WeightPerTensorFloatDecoupled,
                input_quant=Int8ActPerTensorFloat,
                return_quant_tensor=False,
            )
            self.act = torch.nn.ReLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.act(self.conv(x))

    model = Toy().eval().cpu()
    example = torch.randn(1, 3, 16, 16, dtype=torch.float32)

    export_onnx_qcdq(model, args=example, export_path=str(onnx_path), opset_version=int(args.opset), dynamo=False)
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    providers = build_providers(list(args.providers))
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    input_name = session.get_inputs()[0].name

    x_np = np.random.default_rng(0).random((1, 3, 16, 16), dtype=np.float32)
    start = time.perf_counter()
    outputs = session.run(None, {input_name: x_np})
    runtime_ms = (time.perf_counter() - start) * 1000.0

    summary = {
        "onnx_path": str(onnx_path),
        "providers": providers,
        "runtime_ms": float(runtime_ms),
        "output0_shape": list(outputs[0].shape) if outputs else None,
        "compat": get_brevitas_onnx_compat_status(),
        "torch": torch.__version__,
    }
    json_dump(run_root / "smoke_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
