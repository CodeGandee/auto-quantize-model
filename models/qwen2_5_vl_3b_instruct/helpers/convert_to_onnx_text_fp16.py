#!/usr/bin/env python
"""
Export the Qwen2.5-VL-3B-Instruct language model (text tower) to ONNX in FP16.

This exports a text-only wrapper that maps:
  (input_ids, attention_mask) -> logits

Output:
  models/qwen2_5_vl_3b_instruct/onnx/qwen2_5_vl_3b_text_fp16.onnx
  models/qwen2_5_vl_3b_instruct/onnx/qwen2_5_vl_3b_text_fp16.onnx_data  (aggregated external data)

Run from repo root:

  pixi run -e rtx5090 python models/qwen2_5_vl_3b_instruct/helpers/convert_to_onnx_text_fp16.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from contextlib import nullcontext

import torch
import onnx
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration
from optimum.onnx.utils import (
    _get_onnx_external_constants,
    _get_onnx_external_data_tensors,
    check_model_uses_external_data,
)


class Qwen25VLTextWrapper(torch.nn.Module):
    """Thin wrapper exposing a text-only ONNX-friendly interface."""

    def __init__(self, core_model: Qwen2_5_VLForConditionalGeneration) -> None:
        super().__init__()
        self.core_model = core_model

    def forward(  # type: ignore[override]
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.core_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        return outputs.logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export Qwen2.5-VL-3B-Instruct language model (text tower) to ONNX FP16."
        )
    )
    default_ckpt = (
        Path("models")
        / "qwen2_5_vl_3b_instruct"
        / "checkpoints"
        / "Qwen2.5-VL-3B-Instruct"
    )
    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        default=default_ckpt,
        help=f"Path to the HF checkpoint (default: {default_ckpt}).",
    )
    default_out = (
        Path("models")
        / "qwen2_5_vl_3b_instruct"
        / "onnx"
        / "qwen2_5_vl_3b_text_fp16.onnx"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=default_out,
        help=f"Output ONNX path (default: {default_out}).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for export (default: cpu).",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=256,
        help="Maximum sequence length to bake into the dummy input (default: 256).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.ckpt_dir.is_dir():
        print(
            f"Error: checkpoint directory not found at {args.ckpt_dir}. "
            "Run models/qwen2_5_vl_3b_instruct/bootstrap.sh first.",
        )
        return 1

    device = torch.device(args.device)

    print(f"[QWEN2.5-VL] Loading HF checkpoint from {args.ckpt_dir} ...")
    core_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(args.ckpt_dir),
        torch_dtype=torch.float16,
        device_map=None,
        attn_implementation="eager",
    )
    core_model.to(device)
    core_model.eval()
    core_model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(str(args.ckpt_dir))

    print("[QWEN2.5-VL] Building dummy text batch ...")
    dummy = tokenizer(
        "Hello, world",
        return_tensors="pt",
        max_length=args.max_seq_len,
        padding="max_length",
        truncation=True,
    )
    input_ids = dummy["input_ids"].to(device)
    attention_mask = dummy["attention_mask"].to(device)

    wrapper = Qwen25VLTextWrapper(core_model).to(device)

    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[QWEN2.5-VL] Exporting text LM ONNX to {out_path} ...")

    # Work around PyTorch export bug:
    # https://github.com/pytorch/pytorch/issues/163713
    # Disable fake tensor cache during torch.export-based ONNX export.
    try:
        import torch._dynamo as torch_dynamo  # type: ignore[import]

        dynamo_context = torch_dynamo.config.patch(fake_tensor_cache_enabled=False)  # type: ignore[attr-defined]
    except Exception:
        dynamo_context = nullcontext()

    with dynamo_context:
        torch.onnx.export(
            wrapper,
            (input_ids, attention_mask),
            str(out_path),
            opset_version=17,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "logits": {0: "batch", 1: "seq"},
            },
            do_constant_folding=True,
        )

    # Aggregate external data into a single file alongside the graph.
    print("[QWEN2.5-VL] Checking for external data tensors to aggregate ...")
    onnx_model = onnx.load(str(out_path), load_external_data=False)
    if check_model_uses_external_data(onnx_model):
        tensors_paths = _get_onnx_external_data_tensors(onnx_model)
        constant_paths = _get_onnx_external_constants(onnx_model)

        onnx_model_full = onnx.load(str(out_path), load_external_data=True)
        data_filename = out_path.name + "_data"
        onnx.save(
            onnx_model_full,
            str(out_path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_filename,
            convert_attribute=True,
            size_threshold=100,
        )

        for tensor in tensors_paths:
            tensor_path = out_path.parent / tensor
            if tensor_path.is_file():
                tensor_path.unlink()
        for tensor in constant_paths:
            tensor_path = out_path.parent / tensor
            if tensor_path.is_file():
                tensor_path.unlink()

        print(
            f"[QWEN2.5-VL] Aggregated external data into {data_filename} "
            f"and removed per-tensor external data files."
        )
    else:
        print("[QWEN2.5-VL] Model does not use external data; no aggregation needed.")

    print(f"[QWEN2.5-VL] Text LM ONNX export complete: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
