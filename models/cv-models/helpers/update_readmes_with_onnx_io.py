#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from math import prod
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import onnx
from onnx import TensorProto

MODEL_OVERVIEWS: Mapping[str, str] = {
    "inception-v3": (
        "Inception-v3 is an Inception-family convolutional network for ImageNet-style image "
        "classification, designed to be compute-efficient via factorized convolutions."
    ),
    "mobilenet_v1_1.0_224": (
        "MobileNetV1 is a lightweight convolutional network for image classification built around "
        "depthwise separable convolutions (this checkpoint is the 1.0 width / 224px configuration)."
    ),
    "resnet18": (
        "ResNet-18 is a residual convolutional network for ImageNet-style image classification "
        "using skip connections to enable deeper, easier-to-train models."
    ),
    "yolo_world_v2_l": (
        "YOLO-World is a real-time open-vocabulary object detector. This exported ONNX checkpoint "
        "takes images as input and emits detection boxes and per-class scores."
    ),
    "yolo_world_v2_m": (
        "YOLO-World is a real-time open-vocabulary object detector. This exported ONNX checkpoint "
        "takes images as input and emits detection boxes and per-class scores."
    ),
    "yolo_world_v2_s": (
        "YOLO-World is a real-time open-vocabulary object detector. This exported ONNX checkpoint "
        "takes images as input and emits detection boxes and per-class scores."
    ),
    "yolov10m": (
        "YOLOv10 is a real-time end-to-end object detector. This ONNX checkpoint is the medium "
        "variant exported for image-only inference."
    ),
    "yolov2-coco-9": (
        "YOLOv2 (YOLO9000) is a single-stage object detector. This ONNX checkpoint emits a "
        "YOLOv2-style prediction map (grid output) suitable for downstream decoding."
    ),
    "yolov3": (
        "YOLOv3 is a single-stage object detector with multi-scale prediction heads (e.g., 13x13, "
        "26x26, 52x52 feature maps)."
    ),
    "yolov5n": (
        "Ultralytics YOLOv5 is a real-time object detector; `yolov5n` is the nano variant optimized "
        "for speed and small size."
    ),
    "yolov5s": (
        "Ultralytics YOLOv5 is a real-time object detector; `yolov5s` is the small variant "
        "balancing speed and accuracy."
    ),
    "yolov8s": (
        "Ultralytics YOLOv8 is a modern real-time object detector; `yolov8s` is the small variant."
    ),
    "yolow_word_s": (
        "YOLO-World variant (as provided) exported to ONNX for object detection; takes images and "
        "emits detection boxes and per-class scores."
    ),
}

MODEL_REFERENCES: Mapping[str, Sequence[str]] = {
    "inception-v3": ("https://arxiv.org/abs/1512.00567",),
    "mobilenet_v1_1.0_224": ("https://arxiv.org/abs/1704.04861",),
    "resnet18": ("https://arxiv.org/abs/1512.03385",),
    "yolo_world_v2_l": ("https://arxiv.org/abs/2401.17270",),
    "yolo_world_v2_m": ("https://arxiv.org/abs/2401.17270",),
    "yolo_world_v2_s": ("https://arxiv.org/abs/2401.17270",),
    "yolov10m": ("https://arxiv.org/abs/2405.14458",),
    "yolov2-coco-9": ("https://arxiv.org/abs/1612.08242",),
    "yolov3": ("https://arxiv.org/abs/1804.02767",),
    "yolov5n": ("https://github.com/ultralytics/yolov5",),
    "yolov5s": ("https://github.com/ultralytics/yolov5",),
    "yolov8s": ("https://github.com/ultralytics/ultralytics",),
    "yolow_word_s": ("https://arxiv.org/abs/2401.17270",),
}


@dataclass(frozen=True)
class TensorIO:
    name: str
    dtype: str
    shape: list[str]


@dataclass(frozen=True)
class ComputeStats:
    macs_total: int
    macs_conv: int
    macs_matmul: int
    macs_gemm: int
    conv_nodes_counted: int
    matmul_nodes_counted: int
    gemm_nodes_counted: int
    nodes_skipped: int


@dataclass(frozen=True)
class CheckpointReport:
    checkpoint_path: Path
    source_path: Path
    file_size_bytes: int
    parameter_count: int
    parameter_bytes: int
    inputs: list[TensorIO]
    outputs: list[TensorIO]
    shape_inferred: bool
    compute: ComputeStats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect ONNX model I/O + compute simple stats (params/size/MACs) and update "
            "models/cv-models/*/README.md files."
        ),
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        default=Path("models/cv-models"),
        help="Root directory containing model variant directories.",
    )
    parser.add_argument(
        "--tmp-root",
        type=Path,
        default=Path("tmp/cv-models-onnx-metadata"),
        help="Root directory for JSON inspection outputs (not committed; tmp/).",
    )
    parser.add_argument(
        "--infer-shapes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Attempt ONNX shape inference (needed for MACs) before extracting shapes.",
    )
    parser.add_argument(
        "--assumed-batch",
        type=int,
        default=1,
        help="Batch size to assume for symbolic batch dims when computing MACs.",
    )
    return parser.parse_args()


def iter_model_variant_dirs(models_root: Path) -> Iterable[Path]:
    for child in sorted(models_root.iterdir()):
        if not child.is_dir():
            continue
        if child.name == "helpers":
            continue
        yield child


def format_dim(dim: Any) -> str:
    if isinstance(dim, int) and dim > 0:
        return str(dim)
    if isinstance(dim, str) and dim:
        return dim
    return "?"


def tensor_shape_to_list(tensor_type: Any) -> list[str]:
    if not tensor_type.HasField("shape"):
        return []
    dims = []
    for dim in tensor_type.shape.dim:
        if dim.dim_value:
            dims.append(format_dim(dim.dim_value))
        elif dim.dim_param:
            dims.append(format_dim(dim.dim_param))
        else:
            dims.append("?")
    return dims


def describe_value_info(value_info: Any) -> TensorIO:
    if not value_info.type.HasField("tensor_type"):
        raise ValueError(f"Unsupported ONNX value type for {value_info.name!r}")

    tensor_type = value_info.type.tensor_type
    dtype = TensorProto.DataType.Name(tensor_type.elem_type)
    shape = tensor_shape_to_list(tensor_type)
    return TensorIO(name=value_info.name, dtype=dtype, shape=shape)

def _dtype_itemsize(data_type: int) -> int:
    sizes = {
        TensorProto.FLOAT: 4,
        TensorProto.UINT8: 1,
        TensorProto.INT8: 1,
        TensorProto.UINT16: 2,
        TensorProto.INT16: 2,
        TensorProto.INT32: 4,
        TensorProto.INT64: 8,
        TensorProto.BOOL: 1,
        TensorProto.FLOAT16: 2,
        TensorProto.DOUBLE: 8,
        TensorProto.UINT32: 4,
        TensorProto.UINT64: 8,
        TensorProto.BFLOAT16: 2,
    }
    if data_type not in sizes:
        raise ValueError(f"Unsupported TensorProto dtype id: {data_type}")
    return sizes[data_type]


def count_parameters_and_bytes(model: onnx.ModelProto) -> tuple[int, int]:
    parameter_count = 0
    parameter_bytes = 0
    for init in model.graph.initializer:
        numel = prod(init.dims) if init.dims else 0
        parameter_count += int(numel)
        parameter_bytes += int(numel) * _dtype_itemsize(init.data_type)
    return parameter_count, parameter_bytes


def value_shapes(model: onnx.ModelProto) -> dict[str, list[int | str]]:
    shapes: dict[str, list[int | str]] = {}

    def add_value_info(value_info: onnx.ValueInfoProto) -> None:
        if not value_info.type.HasField("tensor_type"):
            return
        tensor_type = value_info.type.tensor_type
        if not tensor_type.HasField("shape"):
            return

        dims: list[int | str] = []
        for dim in tensor_type.shape.dim:
            if dim.dim_value:
                dims.append(int(dim.dim_value))
            elif dim.dim_param:
                dims.append(str(dim.dim_param))
            else:
                dims.append("?")
        shapes[value_info.name] = dims

    for value_info in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        add_value_info(value_info)
    return shapes


def shape_to_ints(shape: Sequence[int | str], assumed_batch: int) -> list[int] | None:
    resolved: list[int] = []
    for idx, dim in enumerate(shape):
        if isinstance(dim, int) and dim > 0:
            resolved.append(dim)
            continue
        if idx == 0:
            resolved.append(assumed_batch)
            continue
        return None
    return resolved


def conv_macs(weight_shape: Sequence[int], output_shape: Sequence[int]) -> int:
    # ONNX Conv weights are [C_out, C_in/group, k1, k2, ...].
    if len(weight_shape) < 3 or len(output_shape) < 3:
        return 0

    cout = weight_shape[0]
    cin_per_group = weight_shape[1]
    kernel = prod(weight_shape[2:])

    batch = output_shape[0]
    out_spatial = prod(output_shape[2:])
    return int(batch * cout * out_spatial * cin_per_group * kernel)


def matmul_macs(
    input_a_shape: Sequence[int],
    input_b_shape: Sequence[int],
    output_shape: Sequence[int],
) -> int:
    # Handle the common MatMul case where output is [..., M, N].
    if len(input_a_shape) < 2 or len(input_b_shape) < 2 or len(output_shape) < 2:
        return 0

    m = output_shape[-2]
    n = output_shape[-1]
    k = input_a_shape[-1]
    if input_b_shape[-2] != k:
        return 0

    batch = prod(output_shape[:-2]) if len(output_shape) > 2 else 1
    return int(batch * m * n * k)


def gemm_macs(
    input_a_shape: Sequence[int],
    input_b_shape: Sequence[int],
    output_shape: Sequence[int],
    trans_a: int,
    trans_b: int,
) -> int:
    # GEMM is 2D in most exports: Y = alpha * A * B + beta * C.
    if len(input_a_shape) != 2 or len(input_b_shape) != 2 or len(output_shape) != 2:
        return 0

    a_m, a_k = input_a_shape if trans_a == 0 else (input_a_shape[1], input_a_shape[0])
    b_k, b_n = input_b_shape if trans_b == 0 else (input_b_shape[1], input_b_shape[0])
    if a_k != b_k:
        return 0
    if output_shape != [a_m, b_n]:
        return 0
    return int(a_m * b_n * a_k)


def compute_macs_stats(
    model: onnx.ModelProto,
    shapes: Mapping[str, list[int | str]],
    initializer_shapes: Mapping[str, list[int]],
    assumed_batch: int,
) -> ComputeStats:
    macs_conv = 0
    macs_matmul = 0
    macs_gemm = 0
    conv_nodes_counted = 0
    matmul_nodes_counted = 0
    gemm_nodes_counted = 0
    nodes_skipped = 0

    for node in model.graph.node:
        if node.op_type == "Conv":
            if len(node.input) < 2 or not node.output:
                nodes_skipped += 1
                continue

            weight_name = node.input[1]
            weight_shape = initializer_shapes.get(weight_name)
            output_name = node.output[0]
            output_shape_raw = shapes.get(output_name)
            if weight_shape is None or output_shape_raw is None:
                nodes_skipped += 1
                continue

            output_shape = shape_to_ints(output_shape_raw, assumed_batch=assumed_batch)
            if output_shape is None:
                nodes_skipped += 1
                continue

            macs_conv += conv_macs(weight_shape, output_shape)
            conv_nodes_counted += 1
            continue

        if node.op_type == "MatMul":
            if len(node.input) < 2 or not node.output:
                nodes_skipped += 1
                continue

            a_shape_raw = shapes.get(node.input[0])
            b_shape_raw = shapes.get(node.input[1])
            y_shape_raw = shapes.get(node.output[0])
            if a_shape_raw is None or b_shape_raw is None or y_shape_raw is None:
                nodes_skipped += 1
                continue

            a_shape = shape_to_ints(a_shape_raw, assumed_batch=assumed_batch)
            b_shape = shape_to_ints(b_shape_raw, assumed_batch=assumed_batch)
            y_shape = shape_to_ints(y_shape_raw, assumed_batch=assumed_batch)
            if a_shape is None or b_shape is None or y_shape is None:
                nodes_skipped += 1
                continue

            macs_matmul += matmul_macs(a_shape, b_shape, y_shape)
            matmul_nodes_counted += 1
            continue

        if node.op_type == "Gemm":
            if len(node.input) < 2 or not node.output:
                nodes_skipped += 1
                continue

            a_shape_raw = shapes.get(node.input[0])
            b_shape_raw = shapes.get(node.input[1])
            y_shape_raw = shapes.get(node.output[0])
            if a_shape_raw is None or b_shape_raw is None or y_shape_raw is None:
                nodes_skipped += 1
                continue

            a_shape = shape_to_ints(a_shape_raw, assumed_batch=assumed_batch)
            b_shape = shape_to_ints(b_shape_raw, assumed_batch=assumed_batch)
            y_shape = shape_to_ints(y_shape_raw, assumed_batch=assumed_batch)
            if a_shape is None or b_shape is None or y_shape is None:
                nodes_skipped += 1
                continue

            trans_a = 0
            trans_b = 0
            for attr in node.attribute:
                if attr.name == "transA":
                    trans_a = int(attr.i)
                elif attr.name == "transB":
                    trans_b = int(attr.i)

            macs_gemm += gemm_macs(a_shape, b_shape, y_shape, trans_a=trans_a, trans_b=trans_b)
            gemm_nodes_counted += 1
            continue

    macs_total = macs_conv + macs_matmul + macs_gemm
    return ComputeStats(
        macs_total=macs_total,
        macs_conv=macs_conv,
        macs_matmul=macs_matmul,
        macs_gemm=macs_gemm,
        conv_nodes_counted=conv_nodes_counted,
        matmul_nodes_counted=matmul_nodes_counted,
        gemm_nodes_counted=gemm_nodes_counted,
        nodes_skipped=nodes_skipped,
    )


def load_checkpoint_report(model_path: Path, infer_shapes: bool, assumed_batch: int) -> CheckpointReport:
    model = onnx.load_model(str(model_path), load_external_data=False)

    analyzed_model = model
    shape_inferred = False
    if infer_shapes:
        try:
            analyzed_model = onnx.shape_inference.infer_shapes(model)
            shape_inferred = True
        except Exception:
            analyzed_model = model
            shape_inferred = False

    parameter_count, parameter_bytes = count_parameters_and_bytes(model)
    file_size_bytes = model_path.stat().st_size
    source_path = model_path.resolve()

    initializer_names = {init.name for init in analyzed_model.graph.initializer}
    inputs = [
        describe_value_info(value_info)
        for value_info in analyzed_model.graph.input
        if value_info.name not in initializer_names
    ]
    outputs = [describe_value_info(value_info) for value_info in analyzed_model.graph.output]

    shapes = value_shapes(analyzed_model)
    initializer_shapes = {init.name: list(init.dims) for init in analyzed_model.graph.initializer}
    compute = compute_macs_stats(
        analyzed_model,
        shapes=shapes,
        initializer_shapes=initializer_shapes,
        assumed_batch=assumed_batch,
    )

    return CheckpointReport(
        checkpoint_path=model_path,
        source_path=source_path,
        file_size_bytes=file_size_bytes,
        parameter_count=parameter_count,
        parameter_bytes=parameter_bytes,
        inputs=inputs,
        outputs=outputs,
        shape_inferred=shape_inferred,
        compute=compute,
    )


def format_shape(shape: Sequence[str]) -> str:
    return "[" + ", ".join(shape) + "]"


def normalize_batch_shape(shape: Sequence[str], assumed_batch: int) -> list[str]:
    if not shape:
        return []
    if shape[0].isdigit():
        return list(shape)
    return [str(assumed_batch), *shape[1:]]


def markdown_for_checkpoint_io(checkpoint_report: CheckpointReport, include_heading: bool) -> str:
    rel_checkpoint = f"checkpoints/{checkpoint_report.checkpoint_path.name}"
    lines: list[str] = []
    if include_heading:
        lines.append(f"### `{rel_checkpoint}`")
        lines.append("")

    inputs_note = " (shape-inferred)" if checkpoint_report.shape_inferred else ""

    lines.append(f"**Inputs**{inputs_note}")
    if checkpoint_report.inputs:
        for item in checkpoint_report.inputs:
            lines.append(f"- `{item.name}`: `{item.dtype}` `{format_shape(item.shape)}`")
    else:
        lines.append("- None")
    lines.append("")

    lines.append("**Outputs**")
    if checkpoint_report.outputs:
        for item in checkpoint_report.outputs:
            lines.append(f"- `{item.name}`: `{item.dtype}` `{format_shape(item.shape)}`")
    else:
        lines.append("- None")
    lines.append("")

    return "\n".join(lines)


def write_checkpoint_json(tmp_root: Path, variant_name: str, checkpoint_report: CheckpointReport) -> Path:
    output_dir = tmp_root / variant_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{checkpoint_report.checkpoint_path.stem}.json"

    payload = {
        "checkpoint_path": str(checkpoint_report.checkpoint_path),
        "source_path": str(checkpoint_report.source_path),
        "file_size_bytes": checkpoint_report.file_size_bytes,
        "parameter_count": checkpoint_report.parameter_count,
        "parameter_bytes": checkpoint_report.parameter_bytes,
        "inputs": [item.__dict__ for item in checkpoint_report.inputs],
        "outputs": [item.__dict__ for item in checkpoint_report.outputs],
        "shape_inferred": checkpoint_report.shape_inferred,
        "compute": checkpoint_report.compute.__dict__,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return output_path


def _human_mib(num_bytes: int) -> str:
    return f"{num_bytes / (1024 * 1024):.2f} MiB"


def _human_gmacs(macs: int) -> str:
    return f"{macs / 1e9:.3f} GMACs"


def _human_gflops_from_macs(macs: int) -> str:
    return f"{(2 * macs) / 1e9:.3f} GFLOPs"


def overview_for_variant(variant_name: str) -> str:
    return MODEL_OVERVIEWS.get(
        variant_name,
        (
            "Computer vision ONNX checkpoint. This directory provides a stable path and metadata "
            "for running experiments (export, inference, quantization) in this repository."
        ),
    )


def references_for_variant(variant_name: str) -> Sequence[str]:
    return MODEL_REFERENCES.get(variant_name, ())


def render_readme(
    variant_name: str,
    checkpoint_reports: Sequence[CheckpointReport],
    infer_shapes: bool,
    assumed_batch: int,
) -> str:
    today = date.today().isoformat()
    lines: list[str] = []

    lines.append(f"# CV Model: {variant_name}")
    lines.append("")
    lines.append("## HEADER")
    lines.append(f"- **Purpose**: Provide a stable local path for the {variant_name} ONNX checkpoint")
    lines.append("- **Status**: Active")
    lines.append(f"- **Date**: {today}")
    lines.append("- **Dependencies**: Local model storage at /workspace/model-to-quantize/General-CV-Models")
    lines.append("- **Target**: AI assistants and developers")
    lines.append("")

    lines.append("## Overview")
    lines.append("")
    lines.append(overview_for_variant(variant_name))
    refs = references_for_variant(variant_name)
    if refs:
        lines.append("")
        lines.append("References:")
        for ref in refs:
            lines.append(f"- {ref}")
    lines.append("")

    lines.append("## Content")
    lines.append("")
    lines.append("This directory contains symlinks to externally stored ONNX checkpoints:")
    lines.append("")
    for report in checkpoint_reports:
        link_name = report.checkpoint_path.name
        lines.append(f"- `checkpoints/{link_name}` -> `{report.source_path}`")
    lines.append("")
    lines.append(
        "These checkpoint files are not committed to the repository. Update the symlink or "
        "replace the source file if the model location changes."
    )
    lines.append("")

    lines.append("## Model Stats")
    lines.append("")

    include_checkpoint_heading = len(checkpoint_reports) > 1
    for report in checkpoint_reports:
        input_shape = (
            format_shape(normalize_batch_shape(report.inputs[0].shape, assumed_batch=assumed_batch))
            if report.inputs
            else "[]"
        )
        if include_checkpoint_heading:
            lines.append(f"### `checkpoints/{report.checkpoint_path.name}`")
            lines.append("")
        lines.append(f"- **ONNX file size**: {_human_mib(report.file_size_bytes)}")
        lines.append(f"- **Parameter count**: {report.parameter_count:,}")
        lines.append(f"- **Parameter bytes**: {_human_mib(report.parameter_bytes)}")
        if infer_shapes and not report.shape_inferred:
            lines.append("- **Compute (MACs/FLOPs)**: unavailable (shape inference failed)")
        else:
            lines.append(
                f"- **Compute (MACs)**: {_human_gmacs(report.compute.macs_total)} @ input {input_shape} (Conv/MatMul/Gemm)"
            )
            lines.append(
                f"- **Compute (FLOPs)**: {_human_gflops_from_macs(report.compute.macs_total)} @ 2 FLOPs per MAC"
            )
            lines.append(f"- **Assumed batch**: {assumed_batch} (for symbolic batch dims)")
        lines.append("")

    lines.append(
        "- **Notes**: Parameter count/bytes are computed from ONNX initializer tensors; compute is "
        "MACs from Conv/MatMul/Gemm nodes only (elementwise ops excluded)."
    )
    lines.append("")

    lines.append("## ONNX I/O")
    lines.append("")
    for report in checkpoint_reports:
        lines.append(
            markdown_for_checkpoint_io(
                report,
                include_heading=include_checkpoint_heading,
            ).rstrip()
        )

    return "\n".join(lines).rstrip() + "\n"


def update_variant_readme(
    variant_dir: Path,
    checkpoint_reports: Sequence[CheckpointReport],
    infer_shapes: bool,
    assumed_batch: int,
) -> None:
    readme_path = variant_dir / "README.md"
    if not readme_path.exists():
        raise FileNotFoundError(f"Missing README: {readme_path}")
    readme_text = render_readme(
        variant_name=variant_dir.name,
        checkpoint_reports=checkpoint_reports,
        infer_shapes=infer_shapes,
        assumed_batch=assumed_batch,
    )
    readme_path.write_text(readme_text, encoding="utf-8")


def main() -> int:
    args = parse_args()

    models_root: Path = args.models_root
    tmp_root: Path = args.tmp_root
    infer_shapes: bool = args.infer_shapes
    assumed_batch: int = args.assumed_batch

    if not models_root.exists():
        raise FileNotFoundError(f"models-root not found: {models_root}")

    for variant_dir in iter_model_variant_dirs(models_root):
        checkpoints_dir = variant_dir / "checkpoints"
        checkpoint_paths = sorted(checkpoints_dir.glob("*.onnx"))
        if not checkpoint_paths:
            continue

        checkpoint_reports: list[CheckpointReport] = []
        for checkpoint_path in checkpoint_paths:
            checkpoint_report = load_checkpoint_report(
                checkpoint_path,
                infer_shapes=infer_shapes,
                assumed_batch=assumed_batch,
            )
            checkpoint_reports.append(checkpoint_report)
            write_checkpoint_json(tmp_root, variant_dir.name, checkpoint_report)

        update_variant_readme(
            variant_dir,
            checkpoint_reports,
            infer_shapes=infer_shapes,
            assumed_batch=assumed_batch,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
