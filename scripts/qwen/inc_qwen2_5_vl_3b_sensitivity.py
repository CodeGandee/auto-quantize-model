#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Optional, Sequence

import torch
from neural_compressor.config import PostTrainingQuantConfig, TuningCriterion, options as nc_options
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
            "Run Intel Neural Compressor PTQ and optional sensitivity analysis "
            "for Qwen2.5-VL-3B-Instruct using COCO2017 captions."
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
        help="Maximum number of calibration samples to use.",
    )
    parser.add_argument(
        "--calib-batch-size",
        type=int,
        default=8,
        help="Batch size for calibration dataloader.",
    )
    parser.add_argument(
        "--max-eval-prompts",
        type=int,
        default=64,
        help="Maximum number of evaluation prompts.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=4,
        help="Batch size for evaluation dataloader.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=256,
        help="Maximum sequence length (tokens) for calibration/eval.",
    )
    parser.add_argument(
        "--max-mse-ops",
        type=int,
        default=32,
        help=(
            "Maximum number of ops to score in the INC MSE_V2 sensitivity pass. "
            "Implemented via the INC_MSE_MAX_OPS environment variable. "
            "Use 0 to allow all ops (slow)."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("tmp/qwen2_5_vl_3b_inc/q_model.pt"),
        help="Where to save the quantized model checkpoint (optional).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    # Allow INC/PyTorch to use most CPU cores for the heavy
    # CPU-only PTQ run, while leaving a couple for the OS.
    cpu_count = os.cpu_count() or 1
    num_threads = max(1, cpu_count - 2)
    torch.set_num_threads(num_threads)
    try:
        torch.set_num_interop_threads(max(1, num_threads // 2))
    except AttributeError:
        # Older PyTorch versions may not expose set_num_interop_threads.
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

    # Build calibration and evaluation dataloaders.
    calib_cfg = QwenCalibConfig(
        captions_path=args.calib_captions,
        max_samples=args.max_calib_samples,
        max_seq_len=args.max_seq_len,
        batch_size=args.calib_batch_size,
        shuffle=True,
    )
    calib_dataloader = build_qwen_calib_dataloader(
        model_dir=args.model_dir, config=calib_cfg
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

    # Load the HF model in bfloat16; INC will control execution backend.
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
        max_batches=8,
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
    )

    confidence_batches = conf.tuning_criterion.strategy_kwargs.get("confidence_batches", 1)

    # Limit the number of ops scored by the patched MSE helpers to keep
    # runtimes reasonable for large LLMs. This is especially important
    # because each scored op requires preparing and running an FX-quantized
    # copy of the model.
    if args.max_mse_ops is not None and args.max_mse_ops > 0:
        os.environ["INC_MSE_MAX_OPS"] = str(args.max_mse_ops)

    print("[INFO] Computing INC MSE_V2 op sensitivity via adaptor.calculate_op_sensitivity ...")
    fp32_mse_map, int8_mse_map = run_single_mse_v2_sensitivity_pass(
        model=model,
        conf=conf,
        calib_dataloader=calib_dataloader,
        confidence_batches=confidence_batches,
    )
    print("[INFO] INC MSE_V2 sensitivity pass completed.")

    # Persist sensitivity results if available.
    out_sens_dir = (
        args.output_path.parent
        if args.output_path is not None
        else Path("tmp/qwen2_5_vl_3b_inc")
    )
    out_sens_dir.mkdir(parents=True, exist_ok=True)
    if fp32_mse_map or int8_mse_map:
        import json

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

        json_path = out_sens_dir / "op_sensitivity_mse_v2_cpu.json"
        with json_path.open("w", encoding="utf-8") as jf:
            json.dump(
                {
                    "meta": {
                        "backend": "pytorch_fx",
                        "device": "cpu",
                        "strategy": "mse_v2",
                        "confidence_batches": confidence_batches,
                        "calib_samples": calib_cfg.max_samples,
                        "calib_batch_size": calib_cfg.batch_size,
                        "max_seq_len": calib_cfg.max_seq_len,
                    },
                    "fp32_fallback_sensitivity": fp32_items_sorted,
                    "int8_requant_sensitivity": int8_items_sorted,
                },
                jf,
                indent=2,
            )

        md_path = out_sens_dir / "op_sensitivity_mse_v2_cpu.md"
        with md_path.open("w", encoding="utf-8") as mf:
            mf.write("# MSE_V2 Op Sensitivity (PyTorch FX, CPU)\n\n")
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
        print(f"[INFO] Saved op sensitivity JSON to {json_path}")
        print(f"[INFO] Saved op sensitivity Markdown to {md_path}")

        # Also persist a stable copy under context/summaries/inc-kb for reuse.
        stable_root = repo_root / "context" / "summaries" / "inc-kb"
        stable_root.mkdir(parents=True, exist_ok=True)
        stable_json = stable_root / "qwen2_5_vl_3b_mse_sensitivity.json"
        stable_md = stable_root / "qwen2_5_vl_3b_mse_sensitivity.md"
        shutil.copy2(json_path, stable_json)
        shutil.copy2(md_path, stable_md)
        print(f"[INFO] Copied op sensitivity JSON to {stable_json}")
        print(f"[INFO] Copied op sensitivity Markdown to {stable_md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
