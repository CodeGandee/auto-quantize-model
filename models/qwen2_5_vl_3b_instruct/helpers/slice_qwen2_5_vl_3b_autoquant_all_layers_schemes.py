#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import sys
from math import ceil
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Derive top-X%% all-layers AutoQuant FP8 schemes for "
            "Qwen2.5-VL-3B by slicing a full-sensitivity baseline manifest, "
            "and export scheme-specific directories."
        )
    )
    default_baseline_dir = (
        Path("models")
        / "qwen2_5_vl_3b_instruct"
        / "quantized"
        / "fp8_autoquant_all_layers_fp8_coco2017"
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=default_baseline_dir,
        help=(
            "Path to the baseline all-layers AutoQuant checkpoint directory "
            "(default: %(default)s). Expected to contain a layer-sensitivity/ "
            "subdirectory with the full-sensitivity manifest."
        ),
    )
    parser.add_argument(
        "--baseline-manifest",
        type=Path,
        default=None,
        help=(
            "Path to the full-sensitivity manifest JSON. If omitted, this "
            "defaults to "
            "`<baseline-dir>/layer-sensitivity/fp8_autoquant_all_layers_fp8_quant_manifest.json`."
        ),
    )
    parser.add_argument(
        "--top-percents",
        type=str,
        default="10,20,30,40,50,60,70,80,90,100",
        help=(
            "Comma-separated list of coverage percentages (e.g. '10,20,50,100') "
            "for which to derive top-X%% all-layers schemes."
        ),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("models") / "qwen2_5_vl_3b_instruct" / "quantized",
        help=(
            "Root directory under which scheme-specific directories will be "
            "created (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--scheme-name-prefix",
        type=str,
        default="fp8_autoquant_all_layers_top",
        help=(
            "Prefix for derived scheme directory names. The final directory "
            "will be `<prefix><percent>_coco2017` under --out-root."
        ),
    )
    parser.add_argument(
        "--copy-weights",
        action="store_true",
        default=True,
        help=(
            "If set (default), copy the entire baseline checkpoint directory "
            "for each derived scheme so that it is self-contained. The layer "
            "sensitivity baseline and per-scheme coverage manifest are always "
            "copied."
        ),
    )
    return parser.parse_args(argv)


def load_manifest(manifest_path: Path) -> Mapping[str, object]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest JSON not found: {manifest_path}")
    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to read manifest JSON at {manifest_path}: {exc}") from exc


def derive_coverage_from_ranking(
    ranking: List[Mapping[str, object]],
    top_percent: int,
) -> Dict[str, List[str]]:
    if not ranking:
        raise ValueError("sensitivity_ranking is empty; cannot derive coverage.")
    if top_percent <= 0 or top_percent > 100:
        raise ValueError(f"top_percent must be in (0, 100], got {top_percent}")

    num_layers = len(ranking)
    coverage_fraction = top_percent / 100.0
    k = max(1, ceil(coverage_fraction * num_layers))

    sorted_names = [entry["name"] for entry in ranking]
    selected = sorted_names[:k]
    dropped = sorted_names[k:]
    return {
        "selected_layers": list(selected),
        "dropped_layers": list(dropped),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    baseline_dir = args.baseline_dir
    if not baseline_dir.is_dir():
        print(
            f"[ERROR] Baseline directory not found: {baseline_dir}",
            file=sys.stderr,
        )
        return 1

    # Resolve manifest path.
    if args.baseline_manifest is not None:
        manifest_path = args.baseline_manifest
    else:
        manifest_path = (
            baseline_dir
            / "layer-sensitivity"
            / "fp8_autoquant_all_layers_fp8_quant_manifest.json"
        )

    print(f"[INFO] Loading baseline manifest from {manifest_path}")
    manifest = load_manifest(manifest_path)
    ranking = manifest.get("sensitivity_ranking")
    if not isinstance(ranking, list):
        print(
            "[ERROR] Manifest does not contain a valid 'sensitivity_ranking' "
            "list; ensure the baseline was produced by the updated "
            "AutoQuant driver.",
            file=sys.stderr,
        )
        return 1

    # Normalize top-percents list.
    try:
        top_percents = [
            int(p.strip()) for p in args.top_percents.split(",") if p.strip()
        ]
    except Exception as exc:  # noqa: BLE001
        print(
            f"[ERROR] Failed to parse --top-percents '{args.top_percents}': {exc}",
            file=sys.stderr,
        )
        return 1

    top_percents = sorted(set(p for p in top_percents if 0 < p <= 100))
    if not top_percents:
        print(
            "[ERROR] No valid coverage percentages provided via --top-percents.",
            file=sys.stderr,
        )
        return 1

    layer_sens_src_dir = baseline_dir / "layer-sensitivity"
    if not layer_sens_src_dir.is_dir():
        print(
            f"[WARN] Baseline directory is missing layer-sensitivity/ at "
            f"{layer_sens_src_dir}; per-scheme coverage manifests will still "
            "be created, but the baseline artifacts will not be copied.",
            file=sys.stderr,
        )

    args.out_root.mkdir(parents=True, exist_ok=True)

    for top_percent in top_percents:
        scheme_dir_name = f"{args.scheme_name_prefix}{top_percent}_coco2017"
        out_dir = args.out_root / scheme_dir_name

        if out_dir.exists():
            print(
                f"[WARN] Output directory already exists, will reuse: {out_dir}",
                file=sys.stderr,
            )
        elif args.copy_weights:
            print(
                f"[INFO] Copying baseline checkpoint from {baseline_dir} to {out_dir}"
            )
            shutil.copytree(baseline_dir, out_dir)
        else:
            print(f"[INFO] Creating empty scheme directory at {out_dir}")
            out_dir.mkdir(parents=True, exist_ok=True)

        coverage = derive_coverage_from_ranking(ranking, top_percent=top_percent)
        num_layers = len(ranking)

        coverage_manifest = {
            "baseline_dir": str(baseline_dir),
            "baseline_manifest": str(manifest_path),
            "scheme_name": scheme_dir_name,
            "top_percent": top_percent,
            "coverage_fraction": top_percent / 100.0,
            "num_layers_total": num_layers,
            "num_layers_selected": len(coverage["selected_layers"]),
            "num_layers_dropped": len(coverage["dropped_layers"]),
            "selected_layers": coverage["selected_layers"],
            "dropped_layers": coverage["dropped_layers"],
        }

        layer_sens_out_dir = out_dir / "layer-sensitivity"
        layer_sens_out_dir.mkdir(parents=True, exist_ok=True)

        coverage_path = (
            layer_sens_out_dir / f"{scheme_dir_name}_coverage_from_baseline.json"
        )
        with coverage_path.open("w", encoding="utf-8") as f:
            json.dump(coverage_manifest, f, indent=2)

        # Also copy the baseline manifest / state / Markdown into each scheme
        # directory for convenience, if they exist.
        if layer_sens_src_dir.is_dir():
            for name in (
                manifest_path.name,
                "fp8_autoquant_all_layers_fp8_autoquant_state.pt",
                "layer-sensitivity-report.md",
            ):
                src_path = layer_sens_src_dir / name
                if src_path.is_file():
                    dst_path = layer_sens_out_dir / name
                    try:
                        shutil.copy2(src_path, dst_path)
                    except Exception as exc:  # noqa: BLE001
                        print(
                            f"[WARN] Failed to copy {src_path} to {dst_path}: {exc}",
                            file=sys.stderr,
                        )

        print(
            f"[INFO] Derived top-{top_percent}% all-layers coverage manifest at "
            f"{coverage_path} (selected {coverage_manifest['num_layers_selected']} "
            f"of {num_layers} layers)."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
