#!/usr/bin/env python
"""
INT8 LM-only AutoQuant driver for Qwen3-VL-4B-Instruct.

This script:

- Loads the Qwen3-VL-4B-Instruct checkpoint.
- Extracts the language model component for LM-only AutoQuant, keeping the
  vision tower in higher precision.
- Builds a text-only calibration dataloader from COCO2017 captions.
- Runs NVIDIA ModelOpt AutoQuant with an INT8 configuration derived from
  ``INT8_LM_DEFAULT_CFG``.
- Emits per-layer sensitivity artifacts:
  - ``per-layer-sensitivity.md``
  - ``per-layer-sensitivity.json``

The resulting artifacts can be compared with FP8 all-layers runs to study
INT8 (W8A8) behavior for the text tower.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import torch

from auto_quantize_model.modelopt_autoquant import (
    AutoQuantSchemeConfig,
    write_layer_sensitivity_json,
    write_layer_sensitivity_md,
)
from auto_quantize_model.qwen.autoquant_sensitivity import (
    run_qwen3_vl_lm_autoquant_sensitivity,
    scheme_with_overrides,
)


AUTOQUANT_INT8_LM_DEFAULT = AutoQuantSchemeConfig(
    name="int8_autoquant_lm_default",
    auto_quantize_bits=8.0,
    auto_quantize_method="gradient",
    auto_quantize_score_size=128,
    coverage_mode="lm_default",
    coverage_fraction=1.0,
    quant_formats=["INT8_LM_DEFAULT_CFG"],
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the Qwen3-VL INT8 LM driver."""
    parser = argparse.ArgumentParser(
        description=(
            "Run ModelOpt AutoQuant INT8 LM-only sensitivity for Qwen3-VL-4B-Instruct "
            "and emit per-layer sensitivity artifacts."
        )
    )

    default_model_dir = (
        Path("models")
        / "qwen3_vl_4b_instruct"
        / "checkpoints"
        / "Qwen3-VL-4B-Instruct"
    )
    default_captions = (
        Path("datasets")
        / "vlm-quantize-calib"
        / "coco2017_captions_large.txt"
    )
    default_output_dir = Path("tmp") / "qwen3_vl_4b_autoquant_int8_lm"

    parser.add_argument(
        "--model-dir",
        type=Path,
        default=default_model_dir,
        help="Path to Qwen3-VL-4B-Instruct HF checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Directory to write the quantization manifest and artifacts.",
    )
    parser.add_argument(
        "--captions-path",
        type=Path,
        default=default_captions,
        help=(
            "Path to COCO2017 captions text file. Defaults to the shared "
            "large (512-sample) calibration subset."
        ),
    )
    parser.add_argument(
        "--max-calib-samples",
        type=int,
        default=512,
        help="Maximum number of text samples to use for calibration.",
    )
    parser.add_argument(
        "--calib-seq-len",
        type=int,
        default=512,
        help="Maximum sequence length for calibration tokens.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Calibration batch size for AutoQuant.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device to use (default: cuda).",
    )
    parser.add_argument(
        "--effective-bits",
        type=float,
        default=None,
        help="Override effective bits for AutoQuant (defaults to scheme value).",
    )
    parser.add_argument(
        "--auto-quantize-score-size",
        type=int,
        default=None,
        help="Override AutoQuant score size in samples (defaults to scheme value).",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        default=False,
        help=(
            "Do not run AutoQuant. Instead, read an existing quantization "
            "manifest from --output-dir and regenerate the per-layer "
            "sensitivity Markdown and JSON reports."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the Qwen3-VL INT8 LM-only AutoQuant driver."""
    args = parse_args(argv)

    scheme = scheme_with_overrides(
        AUTOQUANT_INT8_LM_DEFAULT,
        effective_bits=args.effective_bits,
        score_size=args.auto_quantize_score_size,
    )

    if args.report_only:
        if not args.output_dir.is_dir():
            print(
                f"[ERROR] Report-only mode requested but output dir does not exist: "
                f"{args.output_dir}",
                file=sys.stderr,
            )
            return 1
        manifest_path = args.output_dir / f"{scheme.name}_quant_manifest.json"
        if not manifest_path.is_file():
            print(
                "[ERROR] Report-only mode requested but manifest JSON not found at: "
                f"{manifest_path}",
                file=sys.stderr,
            )
            return 1
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(
                f"[ERROR] Failed to read manifest JSON at {manifest_path}: {exc}",
                file=sys.stderr,
            )
            return 1

        if "layer_sensitivity" not in manifest or "autoquant_state" not in manifest:
            print(
                "[ERROR] Manifest JSON is missing required keys "
                "`layer_sensitivity` or `autoquant_state`.",
                file=sys.stderr,
            )
            return 1

        sensitivity_md_path = args.output_dir / "per-layer-sensitivity.md"
        sensitivity_json_path = args.output_dir / "per-layer-sensitivity.json"
        model_id: Optional[str] = None
        model_meta = manifest.get("model") or {}
        if isinstance(model_meta, dict):
            model_id = model_meta.get("id")

        write_layer_sensitivity_md(
            layer_sensitivity=manifest["layer_sensitivity"],
            scheme=scheme,
            autoquant_state=manifest["autoquant_state"],
            out_path=sensitivity_md_path,
            model_id=model_id,
        )
        write_layer_sensitivity_json(
            manifest=manifest,
            out_path=sensitivity_json_path,
        )
        print(
            "[INFO] Report-only mode: regenerated per-layer sensitivity artifacts at "
            f"{args.output_dir}",
        )
        return 0

    if not args.model_dir.is_dir():
        print(
            f"[ERROR] Model directory not found: {args.model_dir}\n"
            "Hint: run models/qwen3_vl_4b_instruct/bootstrap.sh first.",
            file=sys.stderr,
        )
        return 1

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        print(
            "[WARN] CUDA is not available; running on CPU will be extremely slow.",
            file=sys.stderr,
        )

    print(f"[INFO] Running AutoQuant LM-only scheme: {scheme.name}")
    try:
        manifest, state_dict = run_qwen3_vl_lm_autoquant_sensitivity(
            model_dir=args.model_dir,
            captions_path=args.captions_path,
            scheme=scheme,
            max_calib_samples=args.max_calib_samples,
            calib_seq_len=args.calib_seq_len,
            batch_size=args.batch_size,
            device=args.device,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] AutoQuant run failed: {exc}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir / f"{scheme.name}_autoquant_state.pt"
    torch.save(state_dict, state_path)

    manifest_path = args.output_dir / f"{scheme.name}_quant_manifest.json"
    print(f"[INFO] Building quantization manifest at {manifest_path}")
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    sensitivity_md_path = args.output_dir / "per-layer-sensitivity.md"
    sensitivity_json_path = args.output_dir / "per-layer-sensitivity.json"
    model_meta = manifest.get("model") or {}
    model_id = model_meta.get("id") if isinstance(model_meta, dict) else None

    write_layer_sensitivity_md(
        layer_sensitivity=manifest["layer_sensitivity"],
        scheme=scheme,
        autoquant_state=manifest["autoquant_state"],
        out_path=sensitivity_md_path,
        model_id=model_id,
    )
    write_layer_sensitivity_json(
        manifest=manifest,
        out_path=sensitivity_json_path,
    )

    print("[INFO] AutoQuant INT8 LM-only run completed successfully.")
    print(f"[INFO] Quantization manifest written to: {manifest_path}")
    print(f"[INFO] AutoQuant state written to: {state_path}")
    print(f"[INFO] Per-layer sensitivity report: {sensitivity_md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
