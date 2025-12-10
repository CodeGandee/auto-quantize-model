#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
from neural_compressor.config import (
    AccuracyCriterion,
    PostTrainingQuantConfig,
    TuningCriterion,
    options as nc_options,
)
from transformers import Qwen2_5_VLForConditionalGeneration

from auto_quantize_model.inc_pytorch_mse_patching import run_single_mse_v2_sensitivity_pass
from auto_quantize_model.qwen2_5_vl_inc_data import (
    QwenCalibConfig,
    build_qwen_calib_dataloader,
    build_qwen_eval_dataloader,
    make_qwen_eval_func,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sanity-check Intel Neural Compressor MSE_V2 per-op sensitivity "
            "extraction for Qwen2.5-VL-3B-Instruct using a tiny calibration set."
        )
    )
    default_model_dir = (
        Path("models")
        / "qwen2_5_vl_3b_instruct"
        / "checkpoints"
        / "Qwen2.5-VL-3B-Instruct"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=default_model_dir,
        help="Path to the Qwen2.5-VL-3B-Instruct HF snapshot.",
    )
    parser.add_argument(
        "--calib-captions",
        type=Path,
        default=QwenCalibConfig().captions_path,
        help=(
            "Path to newline-delimited COCO2017 captions used for calibration "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--max-calib-samples",
        type=int,
        default=3,
        help="Maximum number of calibration samples to use for the sanity check.",
    )
    parser.add_argument(
        "--calib-batch-size",
        type=int,
        default=1,
        help="Batch size for calibration dataloader.",
    )
    parser.add_argument(
        "--max-eval-prompts",
        type=int,
        default=8,
        help="Maximum number of evaluation prompts.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=2,
        help="Batch size for evaluation dataloader.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=128,
        help="Maximum sequence length (tokens) for calibration/eval.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of most-sensitive ops to print from each map.",
    )
    return parser.parse_args(argv)


def _print_top_k(
    title: str,
    mse_map: Dict[Tuple[str, str], float],
    top_k: int,
) -> None:
    if not mse_map:
        print(f"[INFO] {title}: no entries captured.")
        return

    print(f"[INFO] {title}: {len(mse_map)} ops captured.")
    # Larger MSE means more sensitivity; sort descending.
    sorted_items = sorted(
        mse_map.items(),
        key=lambda kv: kv[1],
        reverse=True,
    )
    limit = min(top_k, len(sorted_items))
    print(f"[INFO] Top {limit} most sensitive ops by MSE:")
    for rank, ((op_name, op_type), mse) in enumerate(sorted_items[:limit], start=1):
        print(
            f"  #{rank:02d} mse={mse:.6e} op_name='{op_name}' op_type='{op_type}'"
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    # Allow INC/PyTorch to use most CPU cores even in the
    # sanity run, while leaving a couple for the OS.
    cpu_count = os.cpu_count() or 1
    num_threads = max(1, cpu_count - 2)
    torch.set_num_threads(num_threads)
    try:
        torch.set_num_interop_threads(max(1, num_threads // 2))
    except AttributeError:
        pass

    # Ensure INC workspace (tuning history, deploy.yaml, etc.) lives under
    # the repository tmp/ tree instead of cluttering the repo root.
    repo_root = Path(__file__).resolve().parents[2]
    workspace_dir = repo_root / "tmp" / "nc_workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    nc_options.workspace = str(workspace_dir)

    if not args.model_dir.is_dir():
        raise FileNotFoundError(
            f"Model directory not found: {args.model_dir}. "
            "Bootstrap Qwen2.5-VL-3B-Instruct first."
        )

    # Build tiny calibration and evaluation dataloaders.
    calib_cfg = QwenCalibConfig(
        captions_path=args.calib_captions,
        max_samples=args.max_calib_samples,
        max_seq_len=args.max_seq_len,
        batch_size=args.calib_batch_size,
        shuffle=True,
    )
    calib_dataloader = build_qwen_calib_dataloader(
        model_dir=args.model_dir,
        config=calib_cfg,
    )

    eval_dataloader = build_qwen_eval_dataloader(
        model_dir=args.model_dir,
        num_prompts=args.max_eval_prompts,
        max_seq_len=args.max_seq_len,
        batch_size=args.eval_batch_size,
    )

    print(
        f"[INFO] Built calibration dataloader with "
        f"{len(calib_dataloader.dataset)} samples."
    )
    print(
        f"[INFO] Built evaluation dataloader with "
        f"{len(eval_dataloader.dataset)} samples."
    )

    # Load the HF model; INC will control execution backend.
    print(f"[INFO] Loading Qwen2.5-VL-3B-Instruct from {args.model_dir} ...")
    # Use float32 on CPU for compatibility with INC's PyTorch FX backend.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(args.model_dir),
        torch_dtype=torch.float32,
        device_map=None,
    )
    model.to("cpu")

    # Build eval_func wrapper expected by INC.
    eval_func = make_qwen_eval_func(
        model=model,
        dataloader=eval_dataloader,
        max_batches=2,
    )

    conf = PostTrainingQuantConfig(
        backend="default",
        device="cpu",
        approach="static",
        quant_level=1,
        tuning_criterion=TuningCriterion(
            max_trials=1,
            strategy="mse_v2",
            strategy_kwargs={"confidence_batches": 1},
        ),
        accuracy_criterion=AccuracyCriterion(
            criterion="relative",
            tolerable_loss=0.0,
            higher_is_better=True,
        ),
    )

    confidence_batches = conf.tuning_criterion.strategy_kwargs.get("confidence_batches", 1)

    print("[INFO] Running forced INC MSE_V2 sensitivity sanity pass ...")
    fp32_mse_map, int8_mse_map = run_single_mse_v2_sensitivity_pass(
        model=model,
        conf=conf,
        calib_dataloader=calib_dataloader,
        confidence_batches=confidence_batches,
    )
    print("[INFO] MSE_V2 sensitivity sanity pass completed.")

    if not fp32_mse_map and not int8_mse_map:
        print(
            "[ERROR] No per-op MSE values were captured by the forced sensitivity pass. "
            "Check INC version and MSE helper monkeypatching."
        )
        return 1

    _print_top_k("FP32-fallback sensitivity (get_mse_order_per_fp32)", fp32_mse_map, args.top_k)
    _print_top_k("INT8-requant sensitivity (get_mse_order_per_int8)", int8_mse_map, args.top_k)

    # Persist sanity-check sensitivity results under the model tree so they are
    # colocated with Qwen2.5-VL-3B assets.
    model_root = args.model_dir
    for parent in args.model_dir.parents:
        if parent.name == "qwen2_5_vl_3b_instruct":
            model_root = parent
            break
    else:
        # Fallback: place results next to the provided model dir.
        model_root = args.model_dir.parent

    out_dir = model_root / "inc_mse_v2_sanity"
    out_dir.mkdir(parents=True, exist_ok=True)

    fp32_items = [
        {"op_name": name, "op_type": op_type, "mse": mse}
        for (name, op_type), mse in fp32_mse_map.items()
    ]
    int8_items = [
        {"op_name": name, "op_type": op_type, "mse": mse}
        for (name, op_type), mse in int8_mse_map.items()
    ]
    fp32_items_sorted = sorted(fp32_items, key=lambda x: x["mse"])
    int8_items_sorted = sorted(int8_items, key=lambda x: x["mse"])

    json_path = out_dir / "op_sensitivity_mse_v2_sanity.json"
    with json_path.open("w", encoding="utf-8") as jf:
        json.dump(
            {
                "meta": {
                    "backend": "pytorch_fx",
                    "device": "cpu",
                    "strategy": "mse_v2",
                    "confidence_batches": 1,
                    "calib_samples": args.max_calib_samples,
                    "calib_batch_size": args.calib_batch_size,
                    "max_seq_len": args.max_seq_len,
                },
                "fp32_fallback_sensitivity": fp32_items_sorted,
                "int8_requant_sensitivity": int8_items_sorted,
            },
            jf,
            indent=2,
        )

    md_path = out_dir / "op_sensitivity_mse_v2_sanity.md"
    with md_path.open("w", encoding="utf-8") as mf:
        mf.write("# MSE_V2 Op Sensitivity Sanity Check (PyTorch FX, CPU)\n\n")
        mf.write(
            f"- Calib samples: {args.max_calib_samples}\n"
            f"- Calib batch size: {args.calib_batch_size}\n"
            f"- Max seq len: {args.max_seq_len}\n\n"
        )
        mf.write("## INT8-op fallback sensitivity (lower MSE = less sensitive)\n\n")
        mf.write("| Rank | Op Name | Op Type | MSE |\n")
        mf.write("| --- | --- | --- | --- |\n")
        for idx, item in enumerate(fp32_items_sorted, start=1):
            mf.write(
                f"| {idx} | `{item['op_name']}` | `{item['op_type']}` | {item['mse']:.6e} |\n"
            )
        if int8_items_sorted:
            mf.write("\n## FP32-op re-quant sensitivity (lower MSE = less sensitive)\n\n")
            mf.write("| Rank | Op Name | Op Type | MSE |\n")
            mf.write("| --- | --- | --- | --- |\n")
            for idx, item in enumerate(int8_items_sorted, start=1):
                mf.write(
                    f"| {idx} | `{item['op_name']}` | `{item['op_type']}` | {item['mse']:.6e} |\n"
                )

    print(f"[INFO] Saved sanity-check sensitivity JSON to {json_path}")
    print(f"[INFO] Saved sanity-check sensitivity Markdown to {md_path}")

    print("[INFO] Per-op sensitivity extraction appears to be functioning.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
