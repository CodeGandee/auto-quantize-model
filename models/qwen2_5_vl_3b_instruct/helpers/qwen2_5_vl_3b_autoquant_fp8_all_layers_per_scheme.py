#!/usr/bin/env python
"""
Per-scheme AutoQuant FP8 all-layers driver for Qwen2.5-VL-3B.

This script re-runs NVIDIA ModelOpt AutoQuant for a single
all-layers FP8 coverage scheme derived from the
``fp8_autoquant_all_layers_fp8`` baseline.

The key idea is to:

- Treat the baseline all-layers AutoQuant run as the single source of
  sensitivity truth (``fp8_autoquant_all_layers_fp8_coco2017``).
- For a given coverage manifest
  (e.g. ``fp8_autoquant_all_layers_top10_coco2017_coverage_from_baseline.json``),
  interpret its ``dropped_layers`` entries as ``disabled_layers`` for
  ``mtq.auto_quantize``.
- Re-run AutoQuant with:
  - ``quantization_formats=[FP8_ALL_LAYERS_CFG]``
  - ``constraints={"effective_bits": baseline_bits}``
  - ``disabled_layers`` set to the coverage manifest's ``dropped_layers``.
- Export a scheme-specific Hugging Face checkpoint directory containing:
  - Config, processor, tokenizer.
  - Quantized weights via ``export_hf_checkpoint``.
  - A new AutoQuant state + layer sensitivity manifest keyed by the scheme.
  - The original baseline manifest and the per-scheme coverage manifest.

This script focuses on a single scheme at a time. A separate shell
wrapper can sweep all coverage manifests and invoke this driver once
per ``topXX`` scheme.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
)

import modelopt.torch.quantization as mtq
from auto_quantize_model.modelopt_configs import CUSTOM_QUANT_CONFIGS
from modelopt.torch.export import export_hf_checkpoint

# Ensure the helper module for Qwen2.5-VL AutoQuant is importable
# from the same directory when this file is executed as a script.
_THIS_DIR = Path(__file__).resolve().parent
_THIS_DIR_STR = str(_THIS_DIR)
if _THIS_DIR_STR not in sys.path:
    sys.path.insert(0, _THIS_DIR_STR)

from qwen2_5_vl_3b_autoquant_fp8_schemes import (  # noqa: E402
    AutoQuantSchemeConfig,
    build_quant_manifest,
    build_vlm_calib_dataloader,
    write_layer_sensitivity_md,
    write_layer_sensitivity_json,
)


def _load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file into a dictionary.

    Parameters
    ----------
    path :
        Path to the JSON file.

    Returns
    -------
    dict
        Parsed JSON contents.
    """
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to read JSON at {path}: {exc}") from exc


def _validate_coverage_partition(
    baseline_manifest: Mapping[str, Any],
    coverage_manifest: Mapping[str, Any],
    scheme_name: str,
) -> None:
    """Validate that the coverage manifest partitions the baseline layers.

    This function prints warnings to stderr when inconsistencies are
    detected but does not raise, to keep the driver usable even when
    manifests are slightly out of sync.

    Parameters
    ----------
    baseline_manifest :
        Baseline all-layers AutoQuant manifest.
    coverage_manifest :
        Per-scheme coverage manifest with ``selected_layers`` and
        ``dropped_layers`` lists.
    scheme_name :
        Human-readable scheme name used for logging.
    """
    ranking_entries = baseline_manifest.get("sensitivity_ranking")
    if not isinstance(ranking_entries, list):
        print(
            "[WARN] Baseline manifest is missing a valid 'sensitivity_ranking' list; "
            "coverage partition validation will be skipped.",
            file=sys.stderr,
        )
        return

    baseline_names = {
        str(entry.get("name"))
        for entry in ranking_entries
        if isinstance(entry, dict) and entry.get("name") is not None
    }

    selected_layers = coverage_manifest.get("selected_layers", [])
    dropped_layers = coverage_manifest.get("dropped_layers", [])
    if not isinstance(selected_layers, list) or not isinstance(dropped_layers, list):
        print(
            "[WARN] Coverage manifest is missing 'selected_layers' or 'dropped_layers' lists; "
            f"scheme={scheme_name!r}.",
            file=sys.stderr,
        )
        return

    selected_set = {str(item) for item in selected_layers}
    dropped_set = {str(item) for item in dropped_layers}

    if selected_set & dropped_set:
        print(
            "[WARN] Coverage manifest has overlap between selected and dropped layers; "
            f"scheme={scheme_name!r}.",
            file=sys.stderr,
        )

    combined = selected_set | dropped_set
    if combined != baseline_names:
        missing = sorted(baseline_names - combined)
        extra = sorted(combined - baseline_names)
        if missing:
            print(
                "[WARN] Coverage manifest does not cover all baseline layers; "
                f"missing={len(missing)}, scheme={scheme_name!r}.",
                file=sys.stderr,
            )
        if extra:
            print(
                "[WARN] Coverage manifest contains layers not present in baseline; "
                f"extra={len(extra)}, scheme={scheme_name!r}.",
                file=sys.stderr,
            )


def _create_vlm_forward_step(
    device: torch.device,
) -> Any:
    """Create a forward_step callable for VLM AutoQuant.

    The returned callable moves the batch to the specified device and
    forwards it through the model.

    Parameters
    ----------
    device :
        Target device for model inputs.

    Returns
    -------
    callable
        A function with signature ``forward_step(model, batch)``.
    """

    def _forward_step(model: torch.nn.Module, batch: Mapping[str, torch.Tensor]) -> Any:
        batch_on_device = {key: value.to(device) for key, value in batch.items()}
        return model(**batch_on_device)

    return _forward_step


def _create_vlm_loss_func() -> Any:
    """Create a loss function for gradient-based AutoQuant on Qwen2.5-VL.

    The loss is taken directly from the Hugging Face model output
    (``output.loss``), assuming the batch includes ``labels``.

    Returns
    -------
    callable
        Callable with signature ``loss_func(output, batch)`` returning a
        scalar tensor.
    """

    def _loss_func(output: Any, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        # The batch argument is currently unused but kept for signature
        # compatibility with ModelOpt expectations.
        del batch
        if hasattr(output, "loss") and output.loss is not None:
            return output.loss
        raise ValueError(
            "Model output does not contain a loss value. "
            "Ensure that the calibration batches include 'labels'."
        )

    return _loss_func


def _build_quantization_formats(scheme: AutoQuantSchemeConfig) -> List[Any]:
    """Build a list of quantization formats for AutoQuant.

    Parameters
    ----------
    scheme :
        Scheme configuration describing quantization formats by name.

    Returns
    -------
    list
        List of quantization configs accepted by ModelOpt.
    """
    formats: List[Any] = []
    for quant_name in scheme.quant_formats:
        if quant_name in CUSTOM_QUANT_CONFIGS:
            formats.append(CUSTOM_QUANT_CONFIGS[quant_name])
            continue
        if hasattr(mtq, quant_name):
            formats.append(getattr(mtq, quant_name))
            continue
        raise ValueError(f"Unknown quantization format in scheme: {quant_name}")
    if not formats:
        raise ValueError("At least one quantization format must be specified.")
    return formats


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the per-scheme driver."""
    parser = argparse.ArgumentParser(
        description=(
            "Re-run ModelOpt AutoQuant FP8 all-layers search for a single "
            "coverage scheme derived from the fp8_autoquant_all_layers_fp8 baseline."
        )
    )

    default_model_dir = (
        Path("models")
        / "qwen2_5_vl_3b_instruct"
        / "checkpoints"
        / "Qwen2.5-VL-3B-Instruct"
    )
    default_baseline_dir = (
        Path("models")
        / "qwen2_5_vl_3b_instruct"
        / "quantized"
        / "fp8_autoquant_all_layers_fp8_coco2017"
    )

    parser.add_argument(
        "--model-dir",
        type=Path,
        default=default_model_dir,
        help="Path to the base Qwen2.5-VL-3B-Instruct HF checkpoint.",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=default_baseline_dir,
        help=(
            "Path to the baseline all-layers AutoQuant FP8 checkpoint directory. "
            "Expected to contain a layer-sensitivity/ subdirectory."
        ),
    )
    parser.add_argument(
        "--baseline-manifest",
        type=Path,
        default=None,
        help=(
            "Optional path to the baseline all-layers quant manifest. If omitted, "
            "defaults to "
            "`<baseline-dir>/layer-sensitivity/fp8_autoquant_all_layers_fp8_quant_manifest.json`."
        ),
    )
    parser.add_argument(
        "--coverage-manifest",
        type=Path,
        required=True,
        help=(
            "Path to the per-scheme coverage manifest produced from the baseline "
            "sensitivity ranking (e.g., *_coverage_from_baseline.json)."
        ),
    )
    parser.add_argument(
        "--scheme-name",
        type=str,
        default=None,
        help=(
            "Name of the coverage scheme (e.g., fp8_autoquant_all_layers_top10_coco2017). "
            "If omitted, this is inferred from the coverage manifest."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help=(
            "Directory where the new scheme-specific HF checkpoint will be written. "
            "This directory must not contain an existing checkpoint unless --overwrite "
            "is specified."
        ),
    )
    parser.add_argument(
        "--max-calib-samples",
        type=int,
        default=512,
        help=(
            "Maximum number of VLM calibration samples to use. Defaults to the "
            "shared large (512-sample) calibration budget."
        ),
    )
    parser.add_argument(
        "--calib-seq-len",
        type=int,
        default=512,
        help="Maximum sequence length for VLM calibration samples.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Logical calibration batch size for AutoQuant score sizing.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device to use for AutoQuant (default: cuda).",
    )
    parser.add_argument(
        "--vlm-calib-db",
        type=Path,
        default=Path("datasets")
        / "vlm-quantize-calib"
        / "coco2017_vlm_calib_large.db",
        help=(
            "Path to the COCO2017 VLM calibration SQLite DB. Defaults to the "
            "shared large (512-sample) subset."
        ),
    )
    parser.add_argument(
        "--coco-root",
        type=Path,
        default=Path("datasets") / "coco2017" / "source-data",
        help=(
            "Root directory of COCO2017 (must contain train2017/, val2017/, "
            "annotations/). Used to resolve image_relpath entries in the VLM calib DB."
        ),
    )
    parser.add_argument(
        "--effective-bits",
        type=float,
        default=None,
        help=(
            "Optional override for the AutoQuant effective_bits constraint. "
            "Defaults to the baseline scheme's value."
        ),
    )
    parser.add_argument(
        "--auto-quantize-score-size",
        type=int,
        default=None,
        help=(
            "Optional override for the AutoQuant score size in samples. "
            "Defaults to the baseline scheme's value."
        ),
    )
    parser.add_argument(
        "--quant-format",
        type=str,
        default="fp8",
        choices=["fp8", "int8"],
        help=(
            "Quantization format family to use for the per-scheme run. "
            "'fp8' reuses the FP8 baseline config, 'int8' switches to "
            "INT8_ALL_LAYERS_CFG while keeping the same coverage manifest."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Allow overwriting an existing output directory if present.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the per-scheme AutoQuant all-layers driver."""
    args = parse_args(argv)

    if not args.model_dir.is_dir():
        print(
            f"[ERROR] Model directory not found: {args.model_dir}\n"
            "Hint: run models/qwen2_5_vl_3b_instruct/bootstrap.sh first.",
            file=sys.stderr,
        )
        return 1

    baseline_dir: Path = args.baseline_dir
    if not baseline_dir.is_dir():
        print(
            f"[ERROR] Baseline directory not found: {baseline_dir}",
            file=sys.stderr,
        )
        return 1

    if args.baseline_manifest is not None:
        baseline_manifest_path = args.baseline_manifest
    else:
        baseline_manifest_path = (
            baseline_dir
            / "layer-sensitivity"
            / "fp8_autoquant_all_layers_fp8_quant_manifest.json"
        )

    coverage_manifest_path: Path = args.coverage_manifest
    if not coverage_manifest_path.is_file():
        print(
            f"[ERROR] Coverage manifest not found: {coverage_manifest_path}",
            file=sys.stderr,
        )
        return 1

    try:
        baseline_manifest = _load_json(baseline_manifest_path)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Failed to load baseline manifest: {exc}", file=sys.stderr)
        return 1

    try:
        coverage_manifest = _load_json(coverage_manifest_path)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Failed to load coverage manifest: {exc}", file=sys.stderr)
        return 1

    # Determine scheme identity.
    scheme_name: Optional[str] = args.scheme_name or coverage_manifest.get("scheme_name")
    if not scheme_name:
        print(
            "[ERROR] Scheme name must be provided either via --scheme-name or "
            "coverage_manifest['scheme_name'].",
            file=sys.stderr,
        )
        return 1

    print(f"[INFO] Running per-scheme AutoQuant for scheme: {scheme_name}")

    # Validate coverage partition against the baseline ranking.
    _validate_coverage_partition(
        baseline_manifest=baseline_manifest,
        coverage_manifest=coverage_manifest,
        scheme_name=scheme_name,
    )

    # Build baseline scheme config and then adapt it for the current scheme.
    baseline_scheme_dict = baseline_manifest.get("scheme")
    if not isinstance(baseline_scheme_dict, dict):
        print(
            "[ERROR] Baseline manifest does not contain a 'scheme' dictionary.",
            file=sys.stderr,
        )
        return 1

    try:
        baseline_scheme = AutoQuantSchemeConfig(**baseline_scheme_dict)
    except TypeError as exc:
        print(
            f"[ERROR] Failed to reconstruct AutoQuantSchemeConfig from baseline: {exc}",
            file=sys.stderr,
        )
        return 1

    if baseline_scheme.auto_quantize_method != "gradient":
        print(
            "[ERROR] This driver currently supports only gradient-based AutoQuant "
            "for all-layers schemes.",
            file=sys.stderr,
        )
        return 1

    requested_bits: float = (
        float(args.effective_bits)
        if args.effective_bits is not None
        else float(baseline_scheme.auto_quantize_bits)
    )
    if args.quant_format == "int8" and args.effective_bits is None:
        effective_bits = 8.0
    else:
        effective_bits = requested_bits
    score_size: int = (
        int(args.auto_quantize_score_size)
        if args.auto_quantize_score_size is not None
        else int(baseline_scheme.auto_quantize_score_size)
    )

    coverage_fraction = float(coverage_manifest.get("coverage_fraction", 1.0))

    quant_formats: List[str] = ["FP8_ALL_LAYERS_CFG"]
    scheme_label = scheme_name
    if args.quant_format == "int8":
        quant_formats = ["INT8_ALL_LAYERS_CFG"]
        if scheme_label.startswith("fp8_autoquant_all_layers_"):
            suffix = scheme_label[len("fp8_autoquant_all_layers_") :]
            scheme_label = f"int8_autoquant_all_layers_{suffix}"
        else:
            scheme_label = f"{scheme_label}_int8"

    scheme_config = AutoQuantSchemeConfig(
        name=scheme_label,
        auto_quantize_bits=effective_bits,
        auto_quantize_method="gradient",
        auto_quantize_score_size=score_size,
        coverage_mode="all_layers_from_baseline",
        coverage_fraction=coverage_fraction,
        quant_formats=quant_formats,
    )

    # Prepare output directory.
    out_dir: Path = args.out_dir
    if out_dir.exists():
        if any(out_dir.iterdir()) and not args.overwrite:
            print(
                f"[ERROR] Output directory already exists and is not empty: {out_dir}. "
                "Use --overwrite to replace it.",
                file=sys.stderr,
            )
            return 1
        if args.overwrite:
            print(f"[INFO] Removing existing output directory: {out_dir}")
            shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        print(
            "[WARN] CUDA is not available; running AutoQuant on CPU will be extremely slow.",
            file=sys.stderr,
        )

    device = torch.device(args.device)
    torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print(f"[INFO] Loading base Qwen2.5-VL-3B-Instruct model from {args.model_dir}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(args.model_dir),
        torch_dtype=torch_dtype,
        device_map=None,
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir))
    tokenizer.padding_side = "left"
    processor = AutoProcessor.from_pretrained(str(args.model_dir))

    print(
        "[INFO] Building VLM calibration batches from "
        f"{args.vlm_calib_db} and COCO root {args.coco_root}"
    )
    calib_batches: Iterable[Mapping[str, torch.Tensor]] = build_vlm_calib_dataloader(
        calib_db=args.vlm_calib_db,
        coco_root=args.coco_root,
        tokenizer=tokenizer,
        processor=processor,
        batch_size=max(args.batch_size, 1),
        max_samples=args.max_calib_samples,
        max_length=args.calib_seq_len,
    )

    calib_list: List[Mapping[str, torch.Tensor]] = list(calib_batches)
    if not calib_list:
        print(
            "[ERROR] Calibration dataset is empty; cannot run AutoQuant.",
            file=sys.stderr,
        )
        return 1

    forward_step = _create_vlm_forward_step(device=device)
    loss_func = _create_vlm_loss_func()

    num_calib_steps = len(calib_list)
    # Convert score size (samples) into AutoQuant steps while keeping the
    # relationship with the logical batch size for consistency with other
    # drivers.
    num_score_steps = max(score_size // max(args.batch_size, 1), 1)
    num_score_steps = min(num_score_steps, num_calib_steps)

    quantization_formats = _build_quantization_formats(scheme_config)

    dropped_layers_raw = coverage_manifest.get("dropped_layers", [])
    if not isinstance(dropped_layers_raw, list):
        print(
            "[ERROR] Coverage manifest 'dropped_layers' is not a list.",
            file=sys.stderr,
        )
        return 1
    disabled_layers: List[str] = []
    for layer_name in dropped_layers_raw:
        base_name = str(layer_name)
        if base_name.endswith(".quant_recipe"):
            base_name = base_name[: -len(".quant_recipe")]
        # Use a wildcard so that both bare module names and their
        # fully qualified variants (e.g. with a leading 'model.')
        # are matched by the disabled_layers rule.
        pattern = f"*{base_name}*"
        disabled_layers.append(pattern)

    print(
        "[INFO] Invoking ModelOpt auto_quantize for all-layers scheme "
        f"{scheme_name} with effective_bits={effective_bits:.3f}, "
        f"coverage_fraction={coverage_fraction:.3f}, "
        f"num_calib_steps={num_calib_steps}, num_score_steps={num_score_steps}, "
        f"disabled_layers={len(disabled_layers)}"
    )

    quantized_model, state_dict = mtq.auto_quantize(
        model,
        constraints={"effective_bits": effective_bits},
        quantization_formats=quantization_formats,
        data_loader=calib_list,
        forward_step=forward_step,
        loss_func=loss_func,
        disabled_layers=disabled_layers,
        num_calib_steps=num_calib_steps,
        num_score_steps=num_score_steps,
        verbose=True,
    )

    # Build a new manifest describing the per-layer quantization state for
    # this scheme, mirroring the baseline manifest structure.
    print("[INFO] Building per-scheme quantization manifest.")
    manifest = build_quant_manifest(
        model=quantized_model,
        scheme=scheme_config,
        state_dict=state_dict,
        model_id=str(args.model_dir),
    )

    layer_sens_dir = out_dir / "layer-sensitivity"
    layer_sens_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = layer_sens_dir / f"{scheme_config.name}_quant_manifest.json"
    state_path = layer_sens_dir / f"{scheme_config.name}_autoquant_state.pt"
    sensitivity_md_path = layer_sens_dir / "per-layer-sensitivity.md"
    sensitivity_json_path = layer_sens_dir / "per-layer-sensitivity.json"

    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    torch.save(state_dict, state_path)

    write_layer_sensitivity_md(
        layer_sensitivity=manifest["layer_sensitivity"],
        scheme=scheme_config,
        autoquant_state=manifest["autoquant_state"],
        out_path=sensitivity_md_path,
        model_id=str(args.model_dir),
    )

    write_layer_sensitivity_json(
        manifest=manifest,
        out_path=sensitivity_json_path,
    )

    # Export a self-contained HF checkpoint for this scheme.
    print(f"[INFO] Exporting quantized HF checkpoint to {out_dir}")

    AutoConfig.from_pretrained(
        str(args.model_dir),
        trust_remote_code=True,
    ).save_pretrained(out_dir)

    try:
        AutoProcessor.from_pretrained(
            str(args.model_dir),
            trust_remote_code=True,
        ).save_pretrained(out_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Could not save processor config: {exc}", file=sys.stderr)

    export_hf_checkpoint(
        quantized_model,
        export_dir=out_dir,
    )

    tokenizer.save_pretrained(out_dir)

    # Copy baseline and coverage artifacts into the scheme directory so that
    # downstream tools can trace back to the original sensitivity analysis.
    for source_path in (
        baseline_manifest_path,
        coverage_manifest_path,
    ):
        try:
            if source_path.is_file():
                shutil.copy2(source_path, layer_sens_dir / source_path.name)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[WARN] Failed to copy {source_path} into {layer_sens_dir}: {exc}",
                file=sys.stderr,
            )

    print("[INFO] Per-scheme AutoQuant completed successfully.")
    print(f"[INFO] Quantized HF checkpoint directory: {out_dir}")
    print(f"[INFO] Quantization manifest: {manifest_path}")
    print(f"[INFO] AutoQuant state: {state_path}")
    print(f"[INFO] Per-layer sensitivity report: {sensitivity_md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
