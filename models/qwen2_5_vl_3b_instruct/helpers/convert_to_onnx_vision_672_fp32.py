#!/usr/bin/env python
"""
Export the Qwen2.5-VL-3B-Instruct vision encoder to ONNX with a fixed 672x672 image input.

This follows the pattern used in public Qwen2.5-VL vision export examples:
- Manually build patches + grid_thw from (N, C, H, W) pixel_values.
- Call model.visual(fp, grid_thw) as the exported forward.

Output:
- models/qwen2_5_vl_3b_instruct/onnx/qwen2_5_vl_3b_vision_672_fp32.onnx

Run from repo root:

    pixi run -e rtx5090 python models/qwen2_5_vl_3b_instruct/helpers/convert_to_onnx_vision_672_fp32.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from contextlib import nullcontext

import torch
import onnx
from optimum.onnx.utils import (
    _get_onnx_external_constants,
    _get_onnx_external_data_tensors,
    check_model_uses_external_data,
)
from transformers import Qwen2_5_VLForConditionalGeneration


def build_patches_and_grid(
    pixel_values: torch.Tensor,
    temporal_patch_size: int,
    patch_size: int,
    merge_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct flattened patches and grid_thw for the vision encoder.

    pixel_values: (N, C, H, W)
    Returns:
      flatten_patches: (T*H*W, C * temporal_patch_size * patch_size * patch_size)
      grid_thw: (1, 3) tensor [T, H, W] with dtype int32
    """
    assert pixel_values.dim() == 4, "pixel_values must be (N, C, H, W)"
    n, c, h, w = pixel_values.shape
    if h % patch_size != 0 or w % patch_size != 0:
        raise ValueError(f"H({h}) and W({w}) must be divisible by patch_size({patch_size})")
    if (h // patch_size) % merge_size != 0 or (w // patch_size) % merge_size != 0:
        raise ValueError(
            f"(H/patch_size, W/patch_size)=({h//patch_size},{w//patch_size}) must be divisible by merge_size({merge_size})"
        )

    if n == 1:
        pixel_values = pixel_values.repeat(temporal_patch_size, 1, 1, 1)
    elif n % temporal_patch_size != 0:
        repeat_time = temporal_patch_size - (n % temporal_patch_size)
        repeat_image = pixel_values[-1:, ...].repeat(repeat_time, 1, 1, 1)
        pixel_values = torch.cat((pixel_values, repeat_image), dim=0)

    grid_t = pixel_values.shape[0] // temporal_patch_size
    grid_h = h // patch_size
    grid_w = w // patch_size

    patches = pixel_values.reshape(
        grid_t,
        temporal_patch_size,
        c,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flatten_patches = patches.reshape(
        grid_t * grid_h * grid_w,
        c * temporal_patch_size * patch_size * patch_size,
    )
    grid_thw = torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.int32, device=flatten_patches.device)
    return flatten_patches, grid_thw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export Qwen2.5-VL-3B-Instruct vision encoder to ONNX with fixed 672x672 image input."
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
        / "qwen2_5_vl_3b_vision_672_fp32.onnx"
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
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(args.ckpt_dir),
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).eval()
    model.to(device)

    vcfg = model.visual.config
    merge_size = int(vcfg.spatial_merge_size)
    patch_size = int(vcfg.patch_size)
    temporal_patch_size = int(vcfg.temporal_patch_size)

    print(
        f"[QWEN2.5-VL] Vision config: patch_size={patch_size}, "
        f"spatial_merge_size={merge_size}, temporal_patch_size={temporal_patch_size}"
    )

    n, c, h, w = 1, 3, 672, 672
    pixel_values = torch.randn(n, c, h, w, dtype=torch.float32, device=device)

    with torch.no_grad():
        fp, gthw = build_patches_and_grid(pixel_values, temporal_patch_size, patch_size, merge_size)
        vision_features = model.visual(fp, gthw)
        print(f"[QWEN2.5-VL] Vision features shape: {vision_features.shape}")

    def vision_forward(pixel_values_in: torch.Tensor) -> torch.Tensor:
        fp_local, gthw_local = build_patches_and_grid(
            pixel_values_in, temporal_patch_size, patch_size, merge_size
        )
        return model.visual(fp_local, gthw_local)

    model.forward = vision_forward  # type: ignore[assignment]

    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[QWEN2.5-VL] Exporting vision encoder ONNX to {out_path} ...")
    # Work around PyTorch export issues by disabling fake tensor cache during export
    # (see https://github.com/pytorch/pytorch/issues/163713 for similar failures).
    try:
        import torch._dynamo as torch_dynamo  # type: ignore[import]

        dynamo_context = torch_dynamo.config.patch(fake_tensor_cache_enabled=False)  # type: ignore[attr-defined]
    except Exception:
        dynamo_context = nullcontext()

    with dynamo_context:
        torch.onnx.export(
            model,
            (pixel_values,),
            str(out_path),
            opset_version=17,
            dynamo=False,
            input_names=["pixel_values"],
            output_names=["vision_features"],
            do_constant_folding=True,
        )

    # Aggregate external data into a single file alongside the graph.
    print("[QWEN2.5-VL] Checking for external data tensors to aggregate ...")
    onnx_model = onnx.load(str(out_path), load_external_data=False)
    if check_model_uses_external_data(onnx_model):
        tensors_paths = _get_onnx_external_data_tensors(onnx_model)
        constant_paths = _get_onnx_external_constants(onnx_model)

        # Load full external data, then re-save into a single external data file.
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

        # Delete the old per-tensor external data files to keep the directory compact.
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

    print(f"[QWEN2.5-VL] Vision ONNX export complete: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
