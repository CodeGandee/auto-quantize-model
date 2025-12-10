#!/usr/bin/env python
"""
Export the Qwen2.5-VL-3B-Instruct language model (text tower) to ONNX in FP16,
with an additional input for pre-aligned image embeddings.

This exports a wrapper that maps:
  (input_ids, attention_mask, image_embeds) -> logits

Where:
- image_embeds: (batch, seq, hidden_size)
- At positions where input_ids == config.image_token_id, image_embeds entries
  are used instead of token embeddings. Everywhere else, token embeddings are used.

Output:
  models/qwen2_5_vl_3b_instruct/onnx/qwen2_5_vl_3b_text_with_image_fp16.onnx
  models/qwen2_5_vl_3b_instruct/onnx/qwen2_5_vl_3b_text_with_image_fp16.onnx_data
    (aggregated external data)

Run from repo root:

  pixi run -e rtx5090-vllm python models/qwen2_5_vl_3b_instruct/helpers/convert_to_onnx_text_with_image_fp16.py
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import onnx
import torch
from optimum.onnx.utils import (
    _get_onnx_external_constants,
    _get_onnx_external_data_tensors,
    check_model_uses_external_data,
)
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration


class Qwen25VLTextWithImageWrapper(torch.nn.Module):
    """Wrapper exposing a text+image ONNX-friendly interface.

    The wrapper:
    - Embeds input_ids via the model's token embedding matrix.
    - Replaces positions equal to image_token_id with precomputed image_embeds.
    - Forwards fused embeddings through the full Qwen2.5-VL model (text path only).
    """

    def __init__(self, core_model: Qwen2_5_VLForConditionalGeneration) -> None:
        super().__init__()
        self.core_model = core_model
        self.embed_tokens = core_model.get_input_embeddings()
        self.image_token_id = core_model.config.image_token_id

    def forward(  # type: ignore[override]
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image_embeds: torch.Tensor,
    ) -> torch.Tensor:
        token_embeds = self.embed_tokens(input_ids)
        # image_mask: (batch, seq, 1)
        image_mask = (input_ids == self.image_token_id).unsqueeze(-1)

        # Ensure image_embeds is same dtype/device as token_embeds
        image_embeds = image_embeds.to(dtype=token_embeds.dtype, device=token_embeds.device)

        fused_embeds = torch.where(image_mask, image_embeds, token_embeds)

        outputs = self.core_model(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=fused_embeds,
            use_cache=False,
        )
        return outputs.logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export Qwen2.5-VL-3B-Instruct language model (text tower) with "
            "pre-embedded image inputs to ONNX FP16."
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
        / "qwen2_5_vl_3b_text_with_image_fp16.onnx"
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

    print("[QWEN2.5-VL] Building dummy text batch and image embeddings ...")
    dummy = tokenizer(
        "Describe the image.",
        return_tensors="pt",
        max_length=args.max_seq_len,
        padding="max_length",
        truncation=True,
    )
    input_ids = dummy["input_ids"].to(device)
    attention_mask = dummy["attention_mask"].to(device)

    batch_size, seq_len = input_ids.shape
    hidden_size: int = core_model.config.hidden_size

    # Dummy image_embeds: all zeros, will be selectively used at image token positions.
    image_embeds = torch.zeros(
        batch_size,
        seq_len,
        hidden_size,
        dtype=core_model.dtype,
        device=device,
    )

    # Make at least a few positions act as image tokens so the ONNX graph contains the
    # image fusion path. The actual layout at runtime is defined by the host orchestrator.
    num_image_tokens = min(4, seq_len)
    image_token_id = core_model.config.image_token_id
    input_ids[:, :num_image_tokens] = image_token_id

    wrapper = Qwen25VLTextWithImageWrapper(core_model).to(device)

    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[QWEN2.5-VL] Exporting text+image LM ONNX to {out_path} ...")

    # Work around PyTorch export issues:
    # - Disable fake tensor cache during export (see PyTorch issue #163713).
    # - Force legacy TorchScript exporter with dynamo=False.
    try:
        import torch._dynamo as torch_dynamo  # type: ignore[import]

        dynamo_context = torch_dynamo.config.patch(fake_tensor_cache_enabled=False)  # type: ignore[attr-defined]
    except Exception:
        dynamo_context = nullcontext()

    with dynamo_context:
        torch.onnx.export(
            wrapper,
            (input_ids, attention_mask, image_embeds),
            str(out_path),
            opset_version=17,
            input_names=["input_ids", "attention_mask", "image_embeds"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "image_embeds": {0: "batch", 1: "seq"},
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

    print(f"[QWEN2.5-VL] Text+image LM ONNX export complete: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
