#!/usr/bin/env python
"""Generate YOLOv10m low-bit candidate schemes from a Torch sensitivity report.

This script consumes the `layer-sensitivity-report.json` produced by:
  tests/manual/yolo10_layer_sensitivity_sweep/scripts/run_layer_sensitivity_sweep.py

and emits a small set of candidate schemes intended to be materialized via
ModelOpt ONNX PTQ (FP8 with per-node exclusions).

It maps Torch layer names (e.g. `model.2.m.0.cv1.conv`) to ONNX node names
in the exported YOLOv10m ONNX graph (e.g. `/model.2/m.0/cv1/conv/Conv`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import onnx

from auto_quantize_model.cv_models.yolo_preprocess import find_repo_root


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate YOLOv10m candidate mixed/low-bit schemes from a layer sensitivity report.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        required=True,
        help="Path to a layer-sensitivity-report.json file.",
    )
    parser.add_argument(
        "--onnx-path",
        type=Path,
        default=Path("models/cv-models/yolov10m/checkpoints/yolov10m.onnx"),
        help="YOLOv10m ONNX path used to validate node-name mapping.",
    )
    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=[0, 5, 10, 20],
        help="Top-K sensitive layers to keep at high precision (default: 0 5 10 20).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for per-K scheme JSON/txt files.",
    )
    parser.add_argument(
        "--top-n-md",
        type=int,
        default=25,
        help="Number of top sensitive layers to include in the Markdown summary (default: 25).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _merge_numeric_segments(segments: List[str]) -> List[str]:
    merged: List[str] = []
    idx = 0
    while idx < len(segments):
        if idx + 1 < len(segments) and segments[idx + 1].isdigit():
            merged.append(f"{segments[idx]}.{segments[idx + 1]}")
            idx += 2
            continue
        merged.append(segments[idx])
        idx += 1
    return merged


def torch_conv_layer_to_onnx_node(layer_name: str) -> str:
    """Return a likely ONNX node name for a Torch conv layer.

    Note: This is best-effort and does not guarantee existence in a specific ONNX
    export; prefer `resolve_onnx_conv_node()` when you have the ONNX node list.
    """

    segments = layer_name.split(".")
    merged = _merge_numeric_segments(segments)
    return "/" + "/".join(merged) + "/Conv"


def _layer_name_tokens(layer_name: str) -> List[str]:
    segments = layer_name.split(".")
    merged = _merge_numeric_segments(segments)
    if merged and merged[-1] == "conv":
        merged = merged[:-1]
    return merged


def _tokens_in_order(parts: List[str], tokens: List[str]) -> bool:
    start = 0
    for token in tokens:
        try:
            start = parts.index(token, start) + 1
        except ValueError:
            return False
    return True


def resolve_onnx_conv_node(layer_name: str, *, onnx_node_names: List[str]) -> str | None:
    """Resolve a Torch conv module name to an ONNX Conv node name.

    Ultralytics exports often insert extra grouping segments (e.g. `cv1/cv1.2`),
    so we match by checking whether the Torch tokens appear in-order in the ONNX
    node path and selecting the most specific match.
    """

    tokens = _layer_name_tokens(layer_name)
    if not tokens:
        return None

    candidates: List[str] = []
    for node_name in onnx_node_names:
        if not node_name or not node_name.endswith("/Conv"):
            continue
        parts = node_name.strip("/").split("/")
        if _tokens_in_order(parts, tokens):
            candidates.append(node_name)

    if not candidates:
        return None

    def _rank(name: str) -> tuple[int, int, int, str]:
        parts = name.strip("/").split("/")
        has_conv_dir = 1 if parts[-2:-1] == ["conv"] else 0
        return (-has_conv_dir, len(parts), len(name), name)

    return sorted(candidates, key=_rank)[0]


def load_sorted_conv_layers(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = report.get("layer_sensitivity")
    if not isinstance(rows, list):
        raise TypeError("Expected report['layer_sensitivity'] to be a list.")

    conv_rows: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        layer = row.get("layer")
        if not isinstance(layer, str):
            continue
        if not layer.endswith(".conv"):
            continue
        conv_rows.append(row)
    return conv_rows


def validate_onnx_nodes(onnx_path: Path, nodes: List[str]) -> List[str]:
    if not onnx_path.is_file():
        return nodes
    model = onnx.load(str(onnx_path))
    node_names = {node.name for node in model.graph.node if node.name}
    return [name for name in nodes if name and name not in node_names]


def write_markdown_summary(
    out_path: Path,
    *,
    top_rows: List[Dict[str, Any]],
    onnx_node_names: List[str],
) -> None:
    lines: List[str] = [
        "# YOLOv10m sensitivity → candidate scheme mapping",
        "",
        "This file is generated by `scripts/cv-models/make_yolov10m_candidate_schemes.py`.",
        "",
        "| Rank | Torch layer | Sensitivity | ONNX node |",
        "| ---: | --- | ---: | --- |",
    ]
    for idx, row in enumerate(top_rows, start=1):
        layer = str(row.get("layer", ""))
        sens = row.get("sensitivity")
        sens_str = f"{float(sens):.6f}" if isinstance(sens, (float, int)) else ""
        mapped = resolve_onnx_conv_node(layer, onnx_node_names=onnx_node_names)
        onnx_node = mapped or torch_conv_layer_to_onnx_node(layer)
        lines.append(f"| {idx} | `{layer}` | {sens_str} | `{onnx_node}` |")
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    repo_root = find_repo_root(Path(__file__))
    report_path = args.report_json
    if not report_path.is_absolute():
        report_path = (repo_root / report_path).resolve()
    onnx_path = args.onnx_path
    if not onnx_path.is_absolute():
        onnx_path = (repo_root / onnx_path).resolve()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    conv_rows = load_sorted_conv_layers(report)
    if not conv_rows:
        raise RuntimeError(f"No conv layers found in report: {report_path}")

    onnx_node_names: List[str] = []
    if onnx_path.is_file():
        model = onnx.load(str(onnx_path))
        onnx_node_names = [node.name for node in model.graph.node if node.name]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    write_markdown_summary(
        args.out_dir / "candidates.md",
        top_rows=conv_rows[: int(args.top_n_md)],
        onnx_node_names=onnx_node_names,
    )

    for k in sorted({int(val) for val in args.ks}):
        selected = conv_rows[: max(k, 0)]
        layers = [str(row.get("layer")) for row in selected if isinstance(row.get("layer"), str)]
        resolved_nodes: List[str] = []
        missing_layers: List[str] = []
        for name in layers:
            mapped = resolve_onnx_conv_node(name, onnx_node_names=onnx_node_names)
            if mapped is None:
                missing_layers.append(name)
                continue
            resolved_nodes.append(mapped)

        missing_nodes = validate_onnx_nodes(onnx_path, resolved_nodes)

        scheme: Dict[str, Any] = {
            "source_report": str(report_path),
            "onnx_path": str(onnx_path),
            "k": k,
            "excluded_layers": layers,
            "excluded_onnx_nodes": resolved_nodes,
            "missing_onnx_nodes": missing_nodes,
            "missing_layers": missing_layers,
            "materialization": {
                "path": "modelopt_onnx_ptq",
                "quantize_mode": "fp8",
                "calibration_method": "entropy",
                "high_precision_dtype": "fp16",
                "notes": (
                    "Keep top-K sensitive Conv nodes in high precision via nodes_to_exclude; "
                    "quantize remaining nodes with ModelOpt FP8 PTQ."
                ),
            },
        }

        json_path = args.out_dir / f"scheme_k{k}.json"
        json_path.write_text(json.dumps(scheme, indent=2), encoding="utf-8")

        txt_path = args.out_dir / f"nodes_to_exclude_k{k}.txt"
        txt_path.write_text("\n".join(resolved_nodes) + ("\n" if resolved_nodes else ""), encoding="utf-8")

        print(f"Wrote {json_path}")
        print(f"Wrote {txt_path}")
        if missing_nodes or missing_layers:
            missing_total = len(missing_nodes) + len(missing_layers)
            print(f"[WARN] {missing_total} missing mappings for k={k} (see scheme_k{k}.json).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
