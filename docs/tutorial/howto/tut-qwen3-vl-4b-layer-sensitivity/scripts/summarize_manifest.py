from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def _sanitize_path(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    if value.startswith("/"):
        return "<ABSOLUTE_PATH>"
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_md(path: Path, payload: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Layer Sensitivity Summary")
    lines.append("")
    lines.append("| Key | Value |")
    lines.append("|---|---|")
    for key in sorted(payload.keys()):
        value = payload[key]
        if isinstance(value, (dict, list)):
            rendered = f"`{json.dumps(value, sort_keys=True)}`"
        else:
            rendered = f"`{value}`"
        lines.append(f"| {key} | {rendered} |")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("Usage: summarize_manifest.py <manifest.json> <output_dir>", file=sys.stderr)
        return 2

    manifest_path = Path(argv[1])
    out_dir = Path(argv[2])
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError("Manifest JSON must be an object.")

    scheme = manifest.get("scheme") if isinstance(manifest.get("scheme"), dict) else {}
    model = manifest.get("model") if isinstance(manifest.get("model"), dict) else {}
    dataset = manifest.get("dataset") if isinstance(manifest.get("dataset"), dict) else {}
    autoquant_state = manifest.get("autoquant_state") if isinstance(manifest.get("autoquant_state"), dict) else {}

    layer_sensitivity = manifest.get("layer_sensitivity")
    layer_sensitivity_count = len(layer_sensitivity) if isinstance(layer_sensitivity, dict) else 0

    stable_summary: dict[str, Any] = {
        "scheme_name": scheme.get("name"),
        "quant_formats": scheme.get("quant_formats"),
        "coverage_mode": scheme.get("coverage_mode"),
        "coverage_fraction": scheme.get("coverage_fraction"),
        "auto_quantize_method": scheme.get("auto_quantize_method"),
        "auto_quantize_score_size": scheme.get("auto_quantize_score_size"),
        "model_id": _sanitize_path(model.get("id")),
        "dataset_captions_path": _sanitize_path(dataset.get("captions_path")),
        "dataset_vlm_calib_db": _sanitize_path(dataset.get("vlm_calib_db")),
        "has_layer_sensitivity": isinstance(manifest.get("layer_sensitivity"), dict),
        "has_autoquant_state": isinstance(manifest.get("autoquant_state"), dict),
        "manifest_keys": sorted([str(key) for key in manifest.keys()]),
    }

    full_summary: dict[str, Any] = {
        **stable_summary,
        "num_quantized_layers": manifest.get("num_quantized_layers"),
        "layer_sensitivity_count": layer_sensitivity_count,
        "dataset_name": dataset.get("name"),
        "dataset_size": dataset.get("size"),
        "autoquant_constraints": autoquant_state.get("constraints"),
        "autoquant_score": autoquant_state.get("score"),
        "autoquant_is_satisfied": autoquant_state.get("is_satisfied"),
        "effective_bits": scheme.get("auto_quantize_bits"),
    }

    _write_json(out_dir / "summary.json", stable_summary)
    _write_md(out_dir / "summary.md", stable_summary)
    _write_json(out_dir / "summary_full.json", full_summary)
    _write_md(out_dir / "summary_full.md", full_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
