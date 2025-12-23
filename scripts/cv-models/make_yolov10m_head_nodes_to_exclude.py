#!/usr/bin/env python
"""Generate a conservative YOLOv10m INT8 exclusion list for the detection head.

This helper emits ONNX node names to exclude from ModelOpt INT8 PTQ so the
final detection head remains in higher precision. It's a pragmatic default
when first validating INT8 Q/DQ accuracy on detectors.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

import onnx

from auto_quantize_model.cv_models.yolo_preprocess import find_repo_root


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create YOLOv10m head exclusion list (ONNX node names).")
    parser.add_argument(
        "--onnx-path",
        type=Path,
        default=Path("models/cv-models/yolov10m/checkpoints/yolov10m.onnx"),
        help="YOLOv10m ONNX checkpoint path.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output text file path (one node name per line).",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="/model.23/",
        help="Node-name prefix selecting the head region (default: /model.23/).",
    )
    parser.add_argument(
        "--op-types",
        type=str,
        nargs="+",
        default=["Conv"],
        help="Op types to exclude under the prefix (default: Conv).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    repo_root = find_repo_root(Path(__file__))
    onnx_path = args.onnx_path
    if not onnx_path.is_absolute():
        onnx_path = (repo_root / onnx_path).resolve()
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    model = onnx.load(str(onnx_path))
    wanted_ops = {str(op) for op in args.op_types}
    nodes: List[str] = []
    for node in model.graph.node:
        if not node.name:
            continue
        if not node.name.startswith(str(args.prefix)):
            continue
        if str(node.op_type) not in wanted_ops:
            continue
        nodes.append(str(node.name))

    nodes = sorted(set(nodes))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(nodes) + ("\n" if nodes else ""), encoding="utf-8")
    print(f"Wrote {len(nodes)} nodes to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

