#!/usr/bin/env python
"""
Export the full Qwen2.5-VL-3B-Instruct model (vision + text towers)
to an FP16 ONNX graph with a fixed 640x640 image resolution.

This is intended for downstream ONNXRuntime / INC-based layer
quantization sensitivity analysis on the full VLM graph, not just the
text tower.

Usage (from repo root):

    pixi run python models/qwen2_5_vl_3b_instruct/helpers/convert_to_onnx_full_vlm_fp16_640.py

Optional flags:

    --ckpt-dir   Path to HF checkpoint (default:
                 models/qwen2_5_vl_3b_instruct/checkpoints/Qwen2.5-VL-3B-Instruct)
    --image-path Path to a sample RGB image to use for the dummy
                 forward pass (default: tries a COCO2017 val image;
                 falls back to a synthetic 640x640 image)
    --out        Output ONNX path (default:
                 models/qwen2_5_vl_3b_instruct/onnx/qwen2_5_vl_3b_full_vlm_640_fp16.onnx)

The export uses a single representative multimodal batch:

  - One image (H=W=640) + one text prompt.
  - Processor builds all required inputs (input_ids, attention_mask,
    pixel_values, image_grid_thw, etc.).
  - The wrapper module exposes only the inputs actually used by the
    model and returns `logits`, making the ONNX graph easier to consume.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration


class Qwen25VLMultiModalWrapper(torch.nn.Module):
    """Thin wrapper exposing a stable ONNX-friendly interface.

    We expose only the core multimodal inputs and return logits,
    which is sufficient for sensitivity analysis and avoids ONNX
    graphs with multiple unused outputs.
    """

    def __init__(self, core_model: Qwen2_5_VLForConditionalGeneration) -> None:
        super().__init__()
        self.core_model = core_model

    def forward(  # type: ignore[override]
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        output = self.core_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            use_cache=False,
        )
        return output.logits


def _load_or_create_image(image_path: Path) -> Image.Image:
    """Load an RGB image if available; otherwise create a 640x640 dummy."""
    if image_path.is_file():
        img = Image.open(image_path).convert("RGB")
    else:
        img = Image.new("RGB", (640, 640), color=(128, 128, 128))
    # Ensure the underlying resolution is exactly 640x640; the processor
    # may still resize internally but this keeps the dummy input simple.
    if img.size != (640, 640):
        img = img.resize((640, 640))
    return img


def _build_dummy_batch(
    ckpt_dir: Path,
    image_path: Path,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], AutoTokenizer]:
    """Build a single multimodal batch for tracing/export."""
    processor = AutoProcessor.from_pretrained(str(ckpt_dir))
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir))

    img = _load_or_create_image(image_path)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path.as_posix()},
                {"type": "text", "text": "Describe this image briefly."},
            ],
        }
    ]
    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, _ = process_vision_info(messages)

    inputs = processor(
        text=[chat_text],
        images=[image_inputs],
        videos=None,
        padding=True,
        return_tensors="pt",
    )
    # Some processors may not actually use the Pillow image object when
    # given structured image_inputs, but we keep img available so that
    # the message format is consistent with Qwen2.5-VL expectations.

    batch = {k: v.to(device) for k, v in inputs.items()}
    return batch, tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export Qwen2.5-VL-3B-Instruct (vision + text) to ONNX FP16 "
            "with a fixed 640x640 image input."
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
        help=(
            "Path to the Qwen2.5-VL-3B-Instruct HF snapshot "
            f"(default: {default_ckpt})."
        ),
    )
    parser.add_argument(
        "--image-path",
        type=Path,
        default=Path("datasets")
        / "coco2017"
        / "source-data"
        / "val2017"
        / "000000000139.jpg",
        help=(
            "Path to an example RGB image to use as dummy input. "
            "If missing, a synthetic 640x640 image is used."
        ),
    )
    default_out = (
        Path("models")
        / "qwen2_5_vl_3b_instruct"
        / "onnx"
        / "qwen2_5_vl_3b_full_vlm_640_fp16.onnx"
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
        default="cuda",
        help="Device to use for export (default: cuda).",
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

    print("[QWEN2.5-VL] Building dummy multimodal batch (640x640 image) ...")
    dummy_batch, _tokenizer = _build_dummy_batch(
        ckpt_dir=args.ckpt_dir,
        image_path=args.image_path,
        device=device,
    )

    required_keys = ["input_ids", "attention_mask", "pixel_values", "image_grid_thw"]
    missing = [k for k in required_keys if k not in dummy_batch]
    if missing:
        print(
            f"Error: processor did not produce required keys: {missing}. "
            "Check Qwen2.5-VL processor / qwen_vl_utils versions.",
        )
        return 1

    wrapper = Qwen25VLMultiModalWrapper(core_model).to(device)

    out_path: Path = args.out
    onnx_dir = out_path.parent
    onnx_dir.mkdir(parents=True, exist_ok=True)

    print(f"[QWEN2.5-VL] Exporting ONNX to {out_path} ...")
    input_names = required_keys
    example_inputs = tuple(dummy_batch[name] for name in input_names)

    torch.onnx.export(
        wrapper,
        example_inputs,
        str(out_path),
        input_names=input_names,
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
    )

    print(f"[QWEN2.5-VL] ONNX export complete: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
