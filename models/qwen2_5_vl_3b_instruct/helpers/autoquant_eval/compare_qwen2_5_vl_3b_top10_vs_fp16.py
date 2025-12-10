#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Sequence

import torch
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

from text_eval_common import (
    EvalConfig,
    build_eval_dataloader,
    compute_logit_metrics,
    compute_perplexity_for_model,
    upcast_fp8_weights_to_dtype,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Qwen2.5-VL-3B FP16 vs a 10-percent-split FP8 quantized "
            "model using perplexity and logit-based metrics."
        )
    )
    base_dir = Path("models") / "qwen2_5_vl_3b_instruct"
    default_fp16 = base_dir / "checkpoints" / "Qwen2.5-VL-3B-Instruct"
    default_quant = (
        base_dir / "quantized" / "fp8_autoquant_all_layers_top10_coco2017"
    )
    default_captions = (
        Path("datasets") / "vlm-quantize-calib" / "coco2017_captions.txt"
    )
    default_out = Path("tmp") / "modelopt-autoquant-fp8" / "eval-all-layers-top10"

    parser.add_argument(
        "--fp16-model-dir",
        type=Path,
        default=default_fp16,
        help="Path to the FP16/BF16 base model checkpoint.",
    )
    parser.add_argument(
        "--quant-model-dir",
        type=Path,
        default=default_quant,
        help=(
            "Path to the 10-percent split quantized model checkpoint "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--captions-path",
        type=Path,
        default=default_captions,
        help="Path to a text file with one caption per line for evaluation.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1024,
        help="Maximum number of evaluation samples to use.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum sequence length for tokenization.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device to use (default: cuda).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=default_out,
        help=(
            "Output directory for metrics JSON and Markdown summary "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--max-logit-batches",
        type=int,
        default=64,
        help=(
            "Maximum number of batches to use for logit-based metrics "
            "(to keep runtime manageable)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if not args.fp16_model_dir.is_dir():
        raise FileNotFoundError(f"FP16 model directory not found: {args.fp16_model_dir}")
    if not args.quant_model_dir.is_dir():
        raise FileNotFoundError(
            f"Quantized model directory not found: {args.quant_model_dir}"
        )
    if not args.captions_path.is_file():
        raise FileNotFoundError(f"Captions file not found: {args.captions_path}")

    device = torch.device(args.device)

    print(f"[INFO] Loading tokenizer from {args.fp16_model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(args.fp16_model_dir))
    tokenizer.padding_side = "left"

    print(f"[INFO] Building evaluation dataloader from {args.captions_path}")
    dataloader = build_eval_dataloader(
        captions_path=args.captions_path,
        tokenizer=tokenizer,
        max_samples=args.max_samples,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    print(f"[INFO] Loading FP16/BF16 base model from {args.fp16_model_dir}")
    fp16_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(args.fp16_model_dir)
    ).to(device)

    print(f"[INFO] Loading 10-percent split quantized model from {args.quant_model_dir}")
    quant_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(args.quant_model_dir)
    )
    quant_model.to(device)
    upcast_fp8_weights_to_dtype(
        quant_model,
        target_dtype=next(fp16_model.parameters()).dtype,
    )

    eval_config = EvalConfig(
        fp16_model_dir=str(args.fp16_model_dir),
        quant_model_dir=str(args.quant_model_dir),
        captions_path=str(args.captions_path),
        max_samples=args.max_samples,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=str(device),
    )

    # Perplexity for both models on the same eval data.
    print("[INFO] Computing perplexity for FP16 model")
    ppl_fp16 = compute_perplexity_for_model(
        model=fp16_model,
        dataloader=dataloader,
        pad_token_id=tokenizer.pad_token_id,
        device=device,
    )
    print(f"[INFO] FP16 perplexity: {ppl_fp16:.4f}")

    print("[INFO] Computing perplexity for quantized model")
    ppl_quant = compute_perplexity_for_model(
        model=quant_model,
        dataloader=dataloader,
        pad_token_id=tokenizer.pad_token_id,
        device=device,
    )
    print(f"[INFO] Quantized perplexity: {ppl_quant:.4f}")

    # Logit-level comparison on a subset of batches.
    print("[INFO] Computing logit MSE and KL divergence")
    mse, kl = compute_logit_metrics(
        fp16_model=fp16_model,
        quant_model=quant_model,
        dataloader=dataloader,
        device=device,
        max_batches=args.max_logit_batches,
    )
    print(f"[INFO] Logit MSE (last token): {mse:.6e}")
    print(f"[INFO] Logit KL (last token, KL(fp16 || quant)): {kl:.6e}")

    # Write metrics to JSON and Markdown summary.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "config": asdict(eval_config),
        "metrics": {
            "perplexity_fp16": ppl_fp16,
            "perplexity_quant": ppl_quant,
            "perplexity_ratio_quant_over_fp16": (
                float(ppl_quant) / float(ppl_fp16) if ppl_fp16 > 0 else None
            ),
            "logit_mse_last_token": mse,
            "logit_kl_last_token_fp16_to_quant": kl,
        },
    }

    json_path = args.out_dir / "metrics.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    md_path = args.out_dir / "summary.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Qwen2.5-VL-3B FP16 vs 10%-split FP8 comparison\n\n")
        f.write("## Configuration\n\n")
        f.write(f"- FP16 model: `{args.fp16_model_dir}`\n")
        f.write(f"- Quantized model: `{args.quant_model_dir}`\n")
        f.write(f"- Eval data: `{args.captions_path}`\n")
        f.write(f"- Max samples: `{args.max_samples}`\n")
        f.write(f"- Max length: `{args.max_length}`\n")
        f.write(f"- Batch size: `{args.batch_size}`\n")
        f.write(f"- Device: `{device}`\n\n")

        f.write("## Metrics\n\n")
        f.write(f"- FP16 perplexity: `{ppl_fp16:.4f}`\n")
        f.write(f"- Quantized perplexity: `{ppl_quant:.4f}`\n")
        if ppl_fp16 > 0:
            ratio = float(ppl_quant) / float(ppl_fp16)
            f.write(f"- Perplexity ratio (quant / FP16): `{ratio:.4f}`\n")
        f.write(f"- Logit MSE (last token): `{mse:.6e}`\n")
        f.write(
            "- Logit KL (last token, KL(fp16 || quant)): "
            f"`{kl:.6e}`\n"
        )

    print(f"[INFO] Wrote metrics JSON to: {json_path}")
    print(f"[INFO] Wrote Markdown summary to: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
