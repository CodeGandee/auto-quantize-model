#!/usr/bin/env python
"""
Compare baseline and quantized YOLO11 ONNX models on calibration data.

This script runs two ONNX models:

- a baseline FP32/FP16-style model (no Q/DQ), and
- a quantized Q/DQ model (e.g., ModelOpt INT8 PTQ output),

on the same calibration inputs and reports numeric differences between
their outputs (L2 / max-abs error), as well as layer-level activation
differences (MSE, SQNR, cosine similarity, etc.).

Calibration data can be provided either as:

- a `.npy` / `.npz` tensor file with shape `[N, C, H, W]` or `[C, H, W]`,
  or
- a text file with one image path per line; images are read via
  `imageio` and letterboxed + normalized to match the ONNX input shape.

Usage
-----
Example comparing YOLO11n baseline vs. INT8 Q/DQ model:

    pixi run python scripts/yolo11/compare_quantize_precision.py \\
        --input-fp models/yolo11/onnx/yolo11n.onnx \\
        --input-qt models/yolo11/onnx/yolo11n-int8-qdq-proto.onnx \\
        --input-calib datasets/quantize-calib/calib_yolo11_640.npy \\
        --output-dir datasets/quantize-calib/yolo11n-int8-compare

This writes a `metrics.json` file with:

- aggregate output error statistics,
- per-sample output errors, and
- per-layer aggregate error metrics computed over the calibration set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
from collections import defaultdict

import imageio.v3 as iio
import numpy as np
import onnx
import onnxruntime as ort
from mdutils import MdUtils

# CuPy for GPU-accelerated comparison (required in this workflow)
import cupy as cp

HAS_CUPY = True


METRIC_DEFINITIONS: Dict[str, Dict[str, str]] = {
    # Per-sample metrics.
    "mse": {
        "short_name": "MSE",
        "full_name": "Mean squared error",
        "description": (
            "Mean of squared element-wise differences between baseline and "
            "quantized tensors for a single sample."
        ),
    },
    "rmse": {
        "short_name": "RMSE",
        "full_name": "Root mean squared error",
        "description": (
            "Square root of the mean squared error for a single sample; "
            "has the same units as the tensor values."
        ),
    },
    "mean_abs": {
        "short_name": "Mean |Δ|",
        "full_name": "Mean absolute error",
        "description": (
            "Mean of absolute element-wise differences between baseline and "
            "quantized tensors for a single sample."
        ),
    },
    "max_abs": {
        "short_name": "Max |Δ|",
        "full_name": "Maximum absolute error",
        "description": (
            "Largest absolute element-wise difference between baseline and "
            "quantized tensors for a single sample."
        ),
    },
    "sqnr_db": {
        "short_name": "SQNR (dB)",
        "full_name": "Signal-to-quantization-noise ratio (dB)",
        "description": (
            "Signal-to-quantization-noise ratio in decibels for a single sample; "
            "higher values mean less quantization noise."
        ),
    },
    "cosine_sim": {
        "short_name": "Cosine sim.",
        "full_name": "Cosine similarity",
        "description": (
            "Cosine similarity between flattened baseline and quantized tensors "
            "for a single sample; 1.0 means identical direction."
        ),
    },
    # Aggregated metrics over samples (global output or per-layer).
    "mean_mse": {
        "short_name": "Mean MSE",
        "full_name": "Mean MSE over samples",
        "description": (
            "Average of per-sample mean squared error values across the "
            "calibration set."
        ),
    },
    "max_mse": {
        "short_name": "Max MSE",
        "full_name": "Maximum MSE over samples",
        "description": (
            "Largest per-sample mean squared error observed across the "
            "calibration set."
        ),
    },
    "mean_rmse": {
        "short_name": "Mean RMSE",
        "full_name": "Mean RMSE over samples",
        "description": (
            "Average of per-sample root mean squared error values across the "
            "calibration set."
        ),
    },
    "max_rmse": {
        "short_name": "Max RMSE",
        "full_name": "Maximum RMSE over samples",
        "description": (
            "Largest per-sample root mean squared error observed across the "
            "calibration set."
        ),
    },
    "mean_mean_abs": {
        "short_name": "Mean mean |Δ|",
        "full_name": "Mean of mean absolute error over samples",
        "description": (
            "Average of per-sample mean absolute error values across the "
            "calibration set."
        ),
    },
    "max_mean_abs": {
        "short_name": "Max mean |Δ|",
        "full_name": "Maximum mean absolute error over samples",
        "description": (
            "Largest per-sample mean absolute error observed across the "
            "calibration set."
        ),
    },
    "mean_max_abs": {
        "short_name": "Mean max |Δ|",
        "full_name": "Mean of maximum absolute error over samples",
        "description": (
            "Average of per-sample maximum absolute error values across the "
            "calibration set."
        ),
    },
    "max_max_abs": {
        "short_name": "Max max |Δ|",
        "full_name": "Maximum absolute error over all samples",
        "description": (
            "Largest absolute difference between baseline and quantized tensors "
            "observed anywhere in the calibration set."
        ),
    },
    "mean_sqnr_db": {
        "short_name": "Mean SQNR (dB)",
        "full_name": "Mean SQNR over samples (dB)",
        "description": (
            "Average signal-to-quantization-noise ratio in decibels across "
            "the calibration set; higher is better."
        ),
    },
    "min_sqnr_db": {
        "short_name": "Min SQNR (dB)",
        "full_name": "Minimum SQNR over samples (dB)",
        "description": (
            "Worst (lowest) signal-to-quantization-noise ratio in decibels "
            "observed across the calibration set."
        ),
    },
    "mean_cosine_sim": {
        "short_name": "Mean cosine sim.",
        "full_name": "Mean cosine similarity over samples",
        "description": (
            "Average cosine similarity between flattened baseline and quantized "
            "tensors across the calibration set; values near 1.0 are best."
        ),
    },
    "min_cosine_sim": {
        "short_name": "Min cosine sim.",
        "full_name": "Minimum cosine similarity over samples",
        "description": (
            "Worst-case cosine similarity between flattened baseline and "
            "quantized tensors across the calibration set."
        ),
    },
}


def build_metric_definitions() -> Dict[str, Dict[str, str]]:
    """Return metric definition metadata keyed by metric name."""
    out: Dict[str, Dict[str, str]] = {}
    for key, meta in METRIC_DEFINITIONS.items():
        entry = {"key_name": key}
        entry.update(meta)
        out[key] = entry
    return out


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
        description="Compare baseline vs quantized YOLO11 ONNX outputs on calibration data.",
    )
    parser.add_argument(
        "--input-fp",
        type=str,
        required=True,
        help="Path to baseline (FP32/FP16) ONNX model.",
    )
    parser.add_argument(
        "--input-qt",
        type=str,
        required=True,
        help="Path to quantized (Q/DQ) ONNX model.",
    )
    parser.add_argument(
        "--input-calib",
        type=str,
        required=True,
        help=(
            "Path to calibration data: either a .npy/.npz tensor with shape "
            "[N, C, H, W] or [C, H, W], or a .txt with one image path per line."
        ),
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        required=True,
        help="Directory to write comparison results (metrics.json, etc.).",
    )
    parser.add_argument(
        "--providers",
        type=str,
        nargs="+",
        default=["CUDAExecutionProvider"],
        help="ONNX Runtime execution providers (default: CUDAExecutionProvider).",
    )
    return parser.parse_args(argv)


def get_input_shape(session: ort.InferenceSession) -> Tuple[int, int, int]:
    """Infer input tensor shape (C, H, W) from an ONNX Runtime session.

    Parameters
    ----------
    session :
        ONNX Runtime inference session.

    Returns
    -------
    tuple of int
        (channels, height, width) inferred from the first input.
    """
    input_meta = session.get_inputs()[0]
    shape = input_meta.shape
    if shape is None or len(shape) < 3:
        raise ValueError(f"Unexpected input shape for model: {shape}")

    # Assume NCHW-like (batch, channels, height, width).
    c = shape[-3]
    h = shape[-2]
    w = shape[-1]
    if not isinstance(c, int) or not isinstance(h, int) or not isinstance(w, int):
        raise ValueError(f"Non-static input shape: {shape}")
    return int(c), int(h), int(w)


def letterbox(
    image: np.ndarray,
    new_shape: Tuple[int, int] | int,
    color: Tuple[int, int, int] = (114, 114, 114),
) -> np.ndarray:
    """Resize and pad image to a square/rectangular shape, preserving aspect ratio.

    Parameters
    ----------
    image :
        Input image as an HWC uint8 array (RGB).
    new_shape :
        Target shape as an integer (square) or (height, width) tuple.
    color :
        Padding color (RGB).

    Returns
    -------
    np.ndarray
        Letterboxed image of shape `(new_height, new_width, 3)`.
    """
    import cv2  # Imported lazily to avoid unconditional dependency at import time.

    height, width = image.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    new_height, new_width = new_shape
    scale = min(new_width / width, new_height / height)

    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))

    resized = cv2.resize(
        image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
    )

    pad_width = new_width - resized_width
    pad_height = new_height - resized_height

    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top

    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=color,
    )
    return padded


def load_calibration_array(
    calib_path: Path,
    expected_chw: Tuple[int, int, int],
) -> np.ndarray:
    """Load calibration data from a .npy/.npz file.

    Parameters
    ----------
    calib_path :
        Path to a `.npy` or `.npz` file.
    expected_chw :
        Expected (C, H, W) shape tuple from the ONNX input.

    Returns
    -------
    np.ndarray
        Calibration tensor with shape `(N, C, H, W)` and dtype `float32`.
    """
    arr = np.load(calib_path)
    if isinstance(arr, np.lib.npyio.NpzFile):
        # Take the first array for simplicity.
        key = list(arr.files)[0]
        arr = arr[key]

    if arr.ndim == 3:
        arr = arr[None, ...]
    if arr.ndim != 4:
        raise ValueError(f"Expected calibration tensor with 3 or 4 dims, got shape {arr.shape}")

    # Try to interpret layout.
    c, h, w = expected_chw
    if arr.shape[1] == c:
        # NCHW
        pass
    elif arr.shape[-1] == c:
        # NHWC -> NCHW
        arr = np.transpose(arr, (0, 3, 1, 2))
    else:
        raise ValueError(
            f"Cannot map calibration tensor shape {arr.shape} to expected channels {c}"
        )

    # Resize if spatial dims do not match expected H, W.
    if arr.shape[2] != h or arr.shape[3] != w:
        import cv2

        resized = []
        for sample in arr:
            # CHW -> HWC
            img = np.transpose(sample, (1, 2, 0))
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
            img = np.transpose(img, (2, 0, 1))
            resized.append(img)
        arr = np.stack(resized, axis=0)

    return arr.astype("float32")


def load_calibration_from_images(
    list_path: Path,
    expected_chw: Tuple[int, int, int],
) -> np.ndarray:
    """Load calibration data from a text file of image paths.

    Parameters
    ----------
    list_path :
        Path to a `.txt` file containing one image path per line.
    expected_chw :
        Expected (C, H, W) shape tuple from the ONNX input.

    Returns
    -------
    np.ndarray
        Calibration tensor with shape `(N, C, H, W)` and dtype `float32`.
    """
    if not list_path.is_file():
        raise FileNotFoundError(f"Calibration list not found at {list_path}")

    paths: List[Path] = []
    with list_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            paths.append(Path(stripped).expanduser())
    if not paths:
        raise ValueError(f"No image paths found in {list_path}")

    c, h, w = expected_chw
    if c != 3:
        raise ValueError(f"Expected 3-channel RGB inputs, got channels={c}")

    tensors: List[np.ndarray] = []
    for img_path in paths:
        image = iio.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Failed to read image at {img_path}")
        if image.ndim == 2:
            # Grayscale -> RGB
            image = np.stack([image] * 3, axis=-1)
        if image.shape[-1] > 3:
            image = image[..., :3]

        # Assume imageio gives RGB; letterbox to expected H, W.
        letterboxed = letterbox(image, new_shape=(h, w))

        # HWC -> CHW, normalize to [0, 1].
        tensor = letterboxed.transpose(2, 0, 1).astype("float32") / 255.0
        tensors.append(tensor)

    return np.stack(tensors, axis=0)


def load_calibration(
    calib_path: Path,
    expected_chw: Tuple[int, int, int],
) -> np.ndarray:
    """Load calibration data (tensor or images) and return NCHW float32 array."""
    suffix = calib_path.suffix.lower()
    if suffix in {".npy", ".npz"}:
        return load_calibration_array(calib_path, expected_chw)
    if suffix == ".txt":
        return load_calibration_from_images(calib_path, expected_chw)
    raise ValueError(
        f"Unsupported calibration data format for {calib_path} "
        "(expected .npy, .npz, or .txt)"
    )


def compute_output_errors(
    y_fp: np.ndarray,
    y_qt: np.ndarray,
    use_gpu: bool = False,
) -> Dict[str, float]:
    """Compute scalar error metrics between baseline and quantized tensors.

    Parameters
    ----------
    y_fp :
        Baseline tensor.
    y_qt :
        Quantized tensor.
    use_gpu :
        If True and CuPy is available, use GPU for computation.

    Returns
    -------
    dict
        Dictionary with `mse`, `rmse`, `mean_abs`, `max_abs`, `sqnr_db`,
        and `cosine_sim`.
    """
    if y_fp.shape != y_qt.shape:
        raise ValueError(f"Output shapes differ: {y_fp.shape} vs {y_qt.shape}")

    # Select array library: CuPy for GPU, NumPy for CPU
    if use_gpu and HAS_CUPY:
        xp = cp
        # Transfer to GPU
        x = cp.asarray(y_fp, dtype=cp.float64)
        y = cp.asarray(y_qt, dtype=cp.float64)
    else:
        xp = np
        x = y_fp.astype("float64")
        y = y_qt.astype("float64")

    diff = y - x

    mse = float(xp.mean(diff * diff))
    rmse = float(xp.sqrt(mse))
    mean_abs = float(xp.mean(xp.abs(diff)))
    max_abs = float(xp.max(xp.abs(diff)))

    # SQNR: E[x^2] / E[(x - y)^2]
    signal_power = float(xp.mean(x * x))
    noise_power = mse
    if noise_power <= 0.0 or signal_power <= 0.0:
        sqnr_db = float("inf")
    else:
        sqnr = signal_power / noise_power
        sqnr_db = float(10.0 * xp.log10(sqnr))

    # Cosine similarity between flattened tensors.
    x_flat = x.ravel()
    y_flat = y.ravel()
    x_norm = float(xp.linalg.norm(x_flat))
    y_norm = float(xp.linalg.norm(y_flat))
    if x_norm == 0.0 or y_norm == 0.0:
        cosine_sim = 1.0
    else:
        cosine_sim = float(xp.dot(x_flat, y_flat) / (x_norm * y_norm))

    return {
        "mse": mse,
        "rmse": rmse,
        "mean_abs": mean_abs,
        "max_abs": max_abs,
        "sqnr_db": sqnr_db,
        "cosine_sim": cosine_sim,
    }


def _build_tensor_producers(graph: onnx.GraphProto) -> Dict[str, onnx.NodeProto]:
    """Build a mapping from tensor name to producing node."""
    producers: Dict[str, onnx.NodeProto] = {}
    for node in graph.node:
        for output in node.output:
            if output:
                producers[output] = node
    return producers


def _build_tensor_consumers(graph: onnx.GraphProto) -> Dict[str, List[onnx.NodeProto]]:
    """Build a mapping from tensor name to list of consuming nodes."""
    consumers: Dict[str, List[onnx.NodeProto]] = defaultdict(list)
    for node in graph.node:
        for input_name in node.input:
            if input_name:
                consumers[input_name].append(node)
    return consumers


def _classify_node_precision(
    node: onnx.NodeProto,
    tensor_producers: Dict[str, onnx.NodeProto],
    tensor_consumers: Dict[str, List[onnx.NodeProto]],
) -> str:
    """Classify a node as INT8-like or FP16/FP32-like based on Q/DQ placement."""
    if node.op_type in {"QuantizeLinear", "DequantizeLinear"}:
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


def write_quant_compare_summary_md(metrics: Dict[str, Any], out_path: Path) -> None:
    """Write a Markdown summary of per-layer quantization errors.

    Parameters
    ----------
    metrics :
        Metrics dictionary produced by this script (parsed from metrics.json).
    out_path :
        Path to the Markdown file to write.
    """
    metric_definitions: Dict[str, Dict[str, str]] = metrics.get("metric_definitions", {})

    md_file = MdUtils(
        file_name=str(out_path.with_suffix("")),
        title="Quantization Comparison Summary",
    )

    md_file.new_paragraph(f"**Baseline model:** `{metrics['baseline_model']}`")
    md_file.new_paragraph(f"**Quantized model:** `{metrics['quantized_model']}`")
    md_file.new_paragraph(f"**Calibration data:** `{metrics['calibration_path']}`")
    md_file.new_paragraph(f"**Num samples:** {metrics['num_samples']}")
    md_file.new_paragraph(f"**Input (C,H,W):** {metrics['input_chw']}")

    md_file.new_header(level=2, title="Global output error (model outputs)", add_table_of_contents="n")
    global_err = metrics["errors"]

    # Display a curated, human-readable subset of global metrics and
    # use short metric names where available.
    global_metric_keys = [
        "mean_mse",
        "max_mse",
        "mean_rmse",
        "max_rmse",
        "mean_mean_abs",
        "max_mean_abs",
        "mean_max_abs",
        "max_max_abs",
        "mean_sqnr_db",
        "min_sqnr_db",
        "mean_cosine_sim",
        "min_cosine_sim",
    ]
    global_rows: List[str] = []
    for key in global_metric_keys:
        if key not in global_err:
            continue
        label = metric_definitions.get(key, {}).get("short_name", key)
        value = global_err[key]
        global_rows.extend([label, f"{value:.6g}"])

    md_file.new_table(
        columns=2,
        rows=1 + len(global_rows) // 2,
        text=["Metric", "Value"] + global_rows,
        text_align="left",
    )

    md_file.new_header(level=2, title="Per-layer quantization errors", add_table_of_contents="n")

    per_layer = metrics.get("per_layer", [])
    # Sort layers by worst (lowest) mean SQNR dB (inf treated as best).
    def _sqnr_key(layer: Dict[str, Any]) -> float:
        v = layer["metrics"].get("mean_sqnr_db", float("inf"))
        return float(v) if np.isfinite(v) else float("inf")

    per_layer_sorted = sorted(per_layer, key=_sqnr_key)

    per_layer_metric_keys = [
        "mean_sqnr_db",
        "min_sqnr_db",
        "mean_cosine_sim",
        "min_cosine_sim",
        "mean_mse",
        "mean_mean_abs",
        "mean_max_abs",
    ]

    header = [
        "Layer name",
        "Op type",
        "Precision",
    ] + [metric_definitions.get(k, {}).get("short_name", k) for k in per_layer_metric_keys]
    rows: List[str] = []
    for layer in per_layer_sorted:
        name = layer["name"]
        op_type = layer["op_type"]
        precision = layer.get("precision", "")
        m = layer["metrics"]
        metric_values = [
            f"{m.get('mean_sqnr_db', float('inf')):.3f}",
            f"{m.get('min_sqnr_db', float('inf')):.3f}",
            f"{m.get('mean_cosine_sim', 1.0):.6f}",
            f"{m.get('min_cosine_sim', 1.0):.6f}",
            f"{m.get('mean_mse', 0.0):.6e}",
            f"{m.get('mean_mean_abs', 0.0):.6e}",
            f"{m.get('mean_max_abs', 0.0):.6e}",
        ]
        rows.extend(
            [
                name,
                op_type,
                precision,
                *metric_values,
            ]
        )

    md_file.new_table(
        columns=len(header),
        rows=len(per_layer_sorted) + 1,
        text=header + rows,
        text_align="left",
    )

    # Add a section describing the metrics used above.
    used_metric_keys = set(global_metric_keys + per_layer_metric_keys)
    used_metric_keys = [k for k in used_metric_keys if k in metric_definitions]
    if used_metric_keys:
        md_file.new_header(
            level=2,
            title="Metric definitions",
            add_table_of_contents="n",
        )
        for key in sorted(
            used_metric_keys,
            key=lambda k: metric_definitions[k].get("short_name", k).lower(),
        ):
            info = metric_definitions[key]
            short_name = info.get("short_name", key)
            full_name = info.get("full_name", "")
            description = info.get("description", "")
            details = " ".join([part for part in (full_name, description) if part])
            md_file.new_paragraph(f"- **{short_name}** (`{key}`): {details}")

    md_file.create_md_file()


def main(argv: List[str] | None = None) -> int:
    """Entry point for comparing baseline and quantized ONNX model outputs."""
    args = parse_args(argv)

    fp_path = Path(args.input_fp)
    qt_path = Path(args.input_qt)
    calib_path = Path(args.input_calib)
    out_dir = Path(args.output_dir)

    if not fp_path.is_file():
        print(f"Error: baseline ONNX model not found at {fp_path}")
        return 1
    if not qt_path.is_file():
        print(f"Error: quantized ONNX model not found at {qt_path}")
        return 1
    if not calib_path.is_file():
        print(f"Error: calibration data not found at {calib_path}")
        return 1

    # Always use GPU (CuPy) for comparisons in this workflow.
    use_gpu_compare = True
    print("Using CuPy for GPU-accelerated comparison metrics.")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare node-level matching between baseline and quantized models.
    fp_model = onnx.load(str(fp_path))
    qt_model = onnx.load(str(qt_path))

    qt_producers = _build_tensor_producers(qt_model.graph)
    qt_consumers = _build_tensor_consumers(qt_model.graph)

    fp_nodes = {
        n.name: n
        for n in fp_model.graph.node
        if n.name and n.op_type not in ("QuantizeLinear", "DequantizeLinear")
    }
    qt_nodes = {
        n.name: n
        for n in qt_model.graph.node
        if n.name and n.op_type not in ("QuantizeLinear", "DequantizeLinear")
    }

    matched_layers: List[Dict[str, str]] = []
    for name, fp_node in fp_nodes.items():
        qt_node = qt_nodes.get(name)
        if qt_node is None:
            continue
        if fp_node.op_type != qt_node.op_type:
            continue
        precision = _classify_node_precision(qt_node, qt_producers, qt_consumers)
        if precision == "qdq":
            continue
        # Use first non-empty output tensor as the activation to compare.
        fp_out = next((o for o in fp_node.output if o), None)
        qt_out = next((o for o in qt_node.output if o), None)
        if not fp_out or not qt_out:
            continue
        matched_layers.append(
            {
                "name": name,
                "op_type": fp_node.op_type,
                "fp_tensor": fp_out,
                "qt_tensor": qt_out,
                "precision": precision,
            }
        )

    print(f"Matched {len(matched_layers)} layers between baseline and quantized models.")

    fp_output_name = fp_model.graph.output[0].name
    qt_output_name = qt_model.graph.output[0].name

    # Extend graph outputs to include matched intermediate tensors so ONNX Runtime
    # can return them directly.
    existing_fp_outputs = {o.name for o in fp_model.graph.output}
    existing_qt_outputs = {o.name for o in qt_model.graph.output}
    for meta in matched_layers:
        if meta["fp_tensor"] not in existing_fp_outputs:
            fp_model.graph.output.append(
                onnx.ValueInfoProto(name=meta["fp_tensor"])
            )
        if meta["qt_tensor"] not in existing_qt_outputs:
            qt_model.graph.output.append(
                onnx.ValueInfoProto(name=meta["qt_tensor"])
            )

    # Prepare ORT output lists for both models: per-layer activations + final output.
    fp_output_names = [m["fp_tensor"] for m in matched_layers] + [fp_output_name]
    qt_output_names = [m["qt_tensor"] for m in matched_layers] + [qt_output_name]

    # Create sessions from the modified models.
    fp_sess = ort.InferenceSession(
        fp_model.SerializeToString(),
        providers=args.providers,
    )
    qt_sess = ort.InferenceSession(
        qt_model.SerializeToString(),
        providers=args.providers,
    )

    fp_input = fp_sess.get_inputs()[0]
    qt_input = qt_sess.get_inputs()[0]
    if fp_input.name != qt_input.name:
        print(
            f"Warning: input tensor names differ: {fp_input.name} vs {qt_input.name}. "
            f"Using {fp_input.name} for both.",
        )
    input_name = fp_input.name

    expected_chw = get_input_shape(fp_sess)
    calib = load_calibration(calib_path, expected_chw)

    print(
        f"Loaded calibration tensor with shape {calib.shape} "
        f"for input (C,H,W)={expected_chw}",
    )

    # Basic consistency check: quantized model input shape should match.
    qt_chw = get_input_shape(qt_sess)
    if qt_chw != expected_chw:
        print(
            f"Warning: baseline input shape {expected_chw} "
            f"differs from quantized input shape {qt_chw}",
        )

    per_sample_errors: List[Dict[str, Any]] = []
    all_mse: List[float] = []
    all_rmse: List[float] = []
    all_mean_abs: List[float] = []
    all_max_abs: List[float] = []
    all_sqnr_db: List[float] = []
    all_cosine_sim: List[float] = []

    # Per-layer aggregate error accumulators
    layer_errors: Dict[str, Dict[str, List[float]]] = {
        m["name"]: {
            "mse": [],
            "rmse": [],
            "mean_abs": [],
            "max_abs": [],
            "sqnr_db": [],
            "cosine_sim": [],
        }
        for m in matched_layers
    }

    num_samples = calib.shape[0]
    for idx in range(num_samples):
        x = calib[idx : idx + 1]
        fp_outs = fp_sess.run(fp_output_names, {input_name: x})
        qt_outs = qt_sess.run(qt_output_names, {input_name: x})

        # Final model outputs are the last entries
        err = compute_output_errors(fp_outs[-1], qt_outs[-1], use_gpu=use_gpu_compare)
        all_mse.append(err["mse"])
        all_rmse.append(err["rmse"])
        all_mean_abs.append(err["mean_abs"])
        all_max_abs.append(err["max_abs"])
        all_sqnr_db.append(err["sqnr_db"])
        all_cosine_sim.append(err["cosine_sim"])

        per_sample_errors.append(
            {
                "index": idx,
                "mse": err["mse"],
                "rmse": err["rmse"],
                "mean_abs": err["mean_abs"],
                "max_abs": err["max_abs"],
                "sqnr_db": err["sqnr_db"],
                "cosine_sim": err["cosine_sim"],
            }
        )

        # Per-layer errors for this sample
        for layer_idx, meta in enumerate(matched_layers):
            layer_name = meta["name"]
            layer_err = compute_output_errors(fp_outs[layer_idx], qt_outs[layer_idx], use_gpu=use_gpu_compare)
            layer_errors[layer_name]["mse"].append(layer_err["mse"])
            layer_errors[layer_name]["rmse"].append(layer_err["rmse"])
            layer_errors[layer_name]["mean_abs"].append(layer_err["mean_abs"])
            layer_errors[layer_name]["max_abs"].append(layer_err["max_abs"])
            layer_errors[layer_name]["sqnr_db"].append(layer_err["sqnr_db"])
            layer_errors[layer_name]["cosine_sim"].append(layer_err["cosine_sim"])

    # Aggregate per-layer metrics
    per_layer_summary: List[Dict[str, Any]] = []
    for meta in matched_layers:
        name = meta["name"]
        stats = layer_errors[name]
        per_layer_summary.append(
            {
                "name": name,
                "op_type": meta["op_type"],
                "precision": meta.get("precision", ""),
                "fp_tensor": meta["fp_tensor"],
                "qt_tensor": meta["qt_tensor"],
                "metrics": {
                    "mean_mse": float(np.mean(stats["mse"])) if stats["mse"] else 0.0,
                    "max_mse": float(np.max(stats["mse"])) if stats["mse"] else 0.0,
                    "mean_rmse": float(np.mean(stats["rmse"])) if stats["rmse"] else 0.0,
                    "max_rmse": float(np.max(stats["rmse"])) if stats["rmse"] else 0.0,
                    "mean_mean_abs": float(np.mean(stats["mean_abs"]))
                    if stats["mean_abs"]
                    else 0.0,
                    "max_mean_abs": float(np.max(stats["mean_abs"]))
                    if stats["mean_abs"]
                    else 0.0,
                    "mean_max_abs": float(np.mean(stats["max_abs"]))
                    if stats["max_abs"]
                    else 0.0,
                    "max_max_abs": float(np.max(stats["max_abs"]))
                    if stats["max_abs"]
                    else 0.0,
                    "mean_sqnr_db": float(np.mean(stats["sqnr_db"]))
                    if stats["sqnr_db"]
                    else float("inf"),
                    "min_sqnr_db": float(np.min(stats["sqnr_db"]))
                    if stats["sqnr_db"]
                    else float("inf"),
                    "mean_cosine_sim": float(np.mean(stats["cosine_sim"]))
                    if stats["cosine_sim"]
                    else 1.0,
                    "min_cosine_sim": float(np.min(stats["cosine_sim"]))
                    if stats["cosine_sim"]
                    else 1.0,
                },
            }
        )

    metrics = {
        "baseline_model": str(fp_path),
        "quantized_model": str(qt_path),
        "calibration_path": str(calib_path),
        "num_samples": num_samples,
        "providers": args.providers,
        "gpu_compare": use_gpu_compare,
        "input_chw": expected_chw,
        "metric_definitions": build_metric_definitions(),
        "errors": {
            "mean_mse": float(np.mean(all_mse)),
            "max_mse": float(np.max(all_mse)),
            "mean_rmse": float(np.mean(all_rmse)),
            "max_rmse": float(np.max(all_rmse)),
            "mean_mean_abs": float(np.mean(all_mean_abs)),
            "max_mean_abs": float(np.max(all_mean_abs)),
            "mean_max_abs": float(np.mean(all_max_abs)),
            "max_max_abs": float(np.max(all_max_abs)),
            "mean_sqnr_db": float(np.mean(all_sqnr_db)),
            "min_sqnr_db": float(np.min(all_sqnr_db)),
            "mean_cosine_sim": float(np.mean(all_cosine_sim)),
            "min_cosine_sim": float(np.min(all_cosine_sim)),
        },
        "per_sample": per_sample_errors,
        "per_layer": per_layer_summary,
    }

    out_json = out_dir / "metrics.json"
    out_json.write_text(json.dumps(metrics, indent=2))
    print(f"Wrote comparison metrics to {out_json}")

    # Write human-readable Markdown summary.
    out_md = out_dir / "quant-compare-summary.md"
    write_quant_compare_summary_md(metrics, out_md)
    print(f"Wrote quantization comparison summary to {out_md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
