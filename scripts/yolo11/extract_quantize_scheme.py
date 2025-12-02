#!/usr/bin/env python
"""
Extract per-node quantization scheme from a Q/DQ ONNX model.

Given a Q/DQ-style ONNX model (e.g., the output of NVIDIA ModelOpt
INT8 PTQ), this script analyzes which nodes are effectively quantized
to INT8 (i.e., operate between DequantizeLinear inputs and
QuantizeLinear outputs) and which nodes are left in higher precision
(FP16/FP32).

Usage
-----
Basic usage with a YOLO11n INT8 Q/DQ ONNX model:

    pixi run python scripts/yolo11/extract_quantize_scheme.py \\
        models/yolo11/onnx/yolo11n-int8-qdq-proto.onnx \\
        --output-dir datasets/quantize-calib/yolo11n-int8-scheme

The script produces:

- ``precision-scheme.json``: a machine-readable mapping of nodes to
  their inferred precision category, and
- ``precision-scheme.md``: a human-readable Markdown summary with a
  table of node types and precision categories.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import onnx
from mdutils import MdUtils


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv :
        Optional list of argument strings; if None, `sys.argv` is used.

    Returns
    -------
    argparse.Namespace
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Extract per-node quantization scheme from a Q/DQ ONNX model.",
    )
    parser.add_argument(
        "onnx_model",
        type=str,
        help="Path to a Q/DQ ONNX model (e.g., ModelOpt INT8 PTQ output).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        required=True,
        help=(
            "Output directory. The script will create 'precision-scheme.json' "
            "and 'precision-scheme.md' inside this directory."
        ),
    )
    return parser.parse_args(argv)


def build_tensor_producers(graph: onnx.GraphProto) -> Dict[str, onnx.NodeProto]:
    """Build a mapping from tensor name to producing node.

    Parameters
    ----------
    graph :
        ONNX graph protocol buffer.

    Returns
    -------
    dict
        Map from tensor name to the node that produces it.
    """
    producers: Dict[str, onnx.NodeProto] = {}
    for node in graph.node:
        for output in node.output:
            if output:
                producers[output] = node
    return producers


def build_tensor_consumers(graph: onnx.GraphProto) -> Dict[str, List[onnx.NodeProto]]:
    """Build a mapping from tensor name to list of consuming nodes."""
    consumers: Dict[str, List[onnx.NodeProto]] = defaultdict(list)
    for node in graph.node:
        for input_name in node.input:
            if input_name:
                consumers[input_name].append(node)
    return consumers


def classify_node_precision(
    node: onnx.NodeProto,
    tensor_producers: Dict[str, onnx.NodeProto],
    tensor_consumers: Dict[str, List[onnx.NodeProto]],
) -> str:
    """Classify a node as INT8-like or FP16/FP32-like.

    The classification is heuristic and based on Q/DQ placement:

    - If **any** of its non-empty inputs are produced by ``DequantizeLinear``
      nodes **or** any of its non-empty outputs feed into
      ``QuantizeLinear`` nodes, we treat it as ``\"int8\"``.
    - Otherwise, we treat it as ``\"fp32_fp16\"``.

    Parameters
    ----------
    node :
        Node to classify.
    tensor_producers :
        Mapping from tensor name to producing node.
    tensor_consumers :
        Mapping from tensor name to consuming nodes.

    Returns
    -------
    str
        Either ``\"int8\"`` or ``\"fp32_fp16\"``. Q/DQ nodes themselves
        are filtered out at a higher level and not classified here.
    """
    if node.op_type in {"QuantizeLinear", "DequantizeLinear"}:
        # Q/DQ nodes are considered implementation details and not
        # reported as separate layers in the scheme.
        return "qdq"

    # Check inputs
    input_nodes: List[onnx.NodeProto] = []
    for input_name in node.input:
        if not input_name:
            continue
        producer = tensor_producers.get(input_name)
        if producer is not None:
            input_nodes.append(producer)

    any_input_from_dq = bool(input_nodes) and any(
        n.op_type == "DequantizeLinear" for n in input_nodes
    )

    # Check outputs
    output_consumers: List[onnx.NodeProto] = []
    for output_name in node.output:
        if not output_name:
            continue
        for consumer in tensor_consumers.get(output_name, []):
            output_consumers.append(consumer)

    any_output_to_q = bool(output_consumers) and any(
        n.op_type == "QuantizeLinear" for n in output_consumers
    )

    if any_input_from_dq or any_output_to_q:
        return "int8"
    return "fp32_fp16"


def extract_quant_scheme(model_path: Path) -> Dict[str, Any]:
    """Extract a quantization scheme summary from a Q/DQ ONNX model.

    Parameters
    ----------
    model_path :
        Path to the ONNX model.

    Returns
    -------
    dict
        Dictionary containing per-node classifications and summary stats.
    """
    model = onnx.load(str(model_path))
    onnx.checker.check_model(model)
    graph = model.graph

    tensor_producers = build_tensor_producers(graph)
    tensor_consumers = build_tensor_consumers(graph)

    nodes_info: List[Dict[str, Any]] = []
    counts: Dict[str, int] = defaultdict(int)

    for node in graph.node:
        precision = classify_node_precision(node, tensor_producers, tensor_consumers)
        if precision == "qdq":
            # Skip Q/DQ nodes; they are implementation details rather
            # than original layers.
            continue
        counts[precision] += 1

        info: Dict[str, Any] = {
            "name": node.name or "",
            "op_type": node.op_type,
            "precision": precision,
            "inputs": [i for i in node.input if i],
            "outputs": [o for o in node.output if o],
        }
        nodes_info.append(info)

    summary: Dict[str, Any] = {
        "model_path": str(model_path),
        "num_nodes": len(graph.node),
        "counts": {k: int(v) for k, v in counts.items()},
        "nodes": nodes_info,
    }
    return summary


def write_summary_md(
    summary: Dict[str, Any],
    out_path: Path,
) -> None:
    """Write a human-readable Markdown summary of the quantization scheme.

    Parameters
    ----------
    summary :
        Summary dictionary returned by :func:`extract_quant_scheme`.
    out_path :
        Target path for the Markdown file.
    """
    md_file = MdUtils(
        file_name=str(out_path.with_suffix("")),
        title="Quantization Scheme Summary",
    )

    md_file.new_paragraph(f"**Model path:** `{summary['model_path']}`")
    md_file.new_paragraph(f"**Total nodes:** {summary['num_nodes']}")

    # Summary of counts per precision category
    md_file.new_header(level=2, title="Precision category counts", add_table_of_contents="n")
    table_header = ["Precision category", "Node count"]
    table_rows: List[str] = []

    for key, value in sorted(summary["counts"].items()):
        table_rows.extend([key, str(value)])

    md_file.new_table(
        columns=2,
        rows=len(summary["counts"]) + 1,
        text=table_header + table_rows,
        text_align="left",
    )

    # Optional: small sample of nodes
    md_file.new_header(level=2, title="Sample nodes", add_table_of_contents="n")
    max_samples = 20
    samples = summary["nodes"][:max_samples]

    node_table_header = ["Name", "Op type", "Precision"]
    node_rows: List[str] = []
    for n in samples:
        node_rows.extend([n["name"], n["op_type"], n["precision"]])

    md_file.new_table(
        columns=3,
        rows=len(samples) + 1,
        text=node_table_header + node_rows,
        text_align="left",
    )

    md_file.create_md_file()


def main(argv: List[str] | None = None) -> int:
    """Entry point for extracting a quantization scheme from ONNX."""
    args = parse_args(argv)

    model_path = Path(args.onnx_model)
    out_dir = Path(args.output_dir)

    if not model_path.is_file():
        print(f"Error: ONNX model not found at {model_path}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "precision-scheme.json"
    md_path = out_dir / "precision-scheme.md"

    summary = extract_quant_scheme(model_path)
    json_path.write_text(json.dumps(summary, indent=2))

    write_summary_md(summary, md_path)

    print(f"Wrote quantization scheme JSON to {json_path}")
    print(f"Wrote Markdown summary to {md_path}")
    print("Summary counts by precision category:")
    for key, value in summary["counts"].items():
        print(f"  {key}: {value}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
