#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence, Tuple

import numpy as np
import onnxruntime as ort

ORT_TYPE_TO_DTYPE = {
    "float": np.float32,
    "float16": np.float16,
    "double": np.float64,
    "int64": np.int64,
    "int32": np.int32,
    "int16": np.int16,
    "int8": np.int8,
    "uint64": np.uint64,
    "uint32": np.uint32,
    "uint16": np.uint16,
    "uint8": np.uint8,
    "bool": np.bool_,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run random-tensor inference for a single ONNX model.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the ONNX model file.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("tmp/cv-models-random-infer"),
        help="Root directory for outputs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for input generation.",
    )
    parser.add_argument(
        "--default-image-size",
        type=int,
        default=None,
        help="Default H/W when input shape is dynamic.",
    )
    parser.add_argument(
        "--default-batch",
        type=int,
        default=1,
        help="Default batch size when input shape is dynamic.",
    )
    parser.add_argument(
        "--default-channels",
        type=int,
        default=3,
        help="Default channel count when input shape is dynamic.",
    )
    parser.add_argument(
        "--providers",
        type=str,
        nargs="+",
        default=None,
        help=(
            "ONNX Runtime execution providers in priority order. "
            "When omitted, auto-selects best available (TensorRT > CUDA > CPU)."
        ),
    )
    parser.add_argument(
        "--use-cpu",
        action="store_true",
        help="Force CPU execution provider (overrides --providers).",
    )
    parser.add_argument(
        "--disable-cpu-fallback",
        action="store_true",
        help=(
            "Fail instead of silently falling back to CPU if a higher-priority EP "
            "(e.g., CUDA) is present but cannot initialize."
        ),
    )
    return parser.parse_args()


def model_name_from_path(model_path: Path) -> str:
    try:
        return model_path.parents[1].name
    except IndexError:
        return model_path.stem


def default_image_size_for_model(model_name: str, override: int | None) -> int:
    if override is not None:
        return override
    if "yolo" in model_name.lower():
        return 640
    return 224


def parse_ort_type(ort_type: str) -> np.dtype:
    if ort_type.startswith("tensor(") and ort_type.endswith(")"):
        ort_type = ort_type[len("tensor(") : -1]
    if ort_type not in ORT_TYPE_TO_DTYPE:
        raise ValueError(f"Unsupported ONNX tensor type: {ort_type}")
    return ORT_TYPE_TO_DTYPE[ort_type]


def resolve_shape(
    raw_shape: Sequence[Any],
    default_batch: int,
    default_channels: int,
    default_image_size: int,
) -> Tuple[list[int], bool]:
    rank = len(raw_shape)
    dynamic = False

    if rank == 4:
        fallbacks = [default_batch, default_channels, default_image_size, default_image_size]
    elif rank == 3:
        fallbacks = [default_channels, default_image_size, default_image_size]
    elif rank == 2:
        fallbacks = [default_batch, default_image_size]
    elif rank == 1:
        fallbacks = [default_batch]
    else:
        fallbacks = [1 for _ in range(rank)]

    resolved: list[int] = []
    for idx, dim in enumerate(raw_shape):
        if isinstance(dim, int) and dim > 0:
            resolved.append(dim)
            continue
        dynamic = True
        resolved.append(fallbacks[idx])

    return resolved, dynamic


def random_tensor(shape: Sequence[int], dtype: np.dtype, rng: np.random.Generator) -> np.ndarray:
    if dtype == np.bool_:
        return rng.integers(0, 2, size=shape, dtype=np.int8).astype(np.bool_)
    if np.issubdtype(dtype, np.integer):
        return rng.integers(0, 10, size=shape, dtype=dtype)
    return rng.random(size=shape, dtype=np.float32).astype(dtype)


def build_providers(args: argparse.Namespace) -> list[str]:
    if args.use_cpu:
        return ["CPUExecutionProvider"]
    available = set(ort.get_available_providers())
    if args.providers is not None:
        requested = [str(p) for p in args.providers]
        filtered = [p for p in requested if p in available]
        if not filtered:
            raise RuntimeError(
                f"None of the requested providers are available. requested={requested}, available={sorted(available)}"
            )
        return filtered

    preferred = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    return [p for p in preferred if p in available] or ["CPUExecutionProvider"]


def run_inference(model_path: Path, args: argparse.Namespace) -> Path:
    model_name = model_name_from_path(model_path)
    output_dir = args.output_root / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    default_image_size = default_image_size_for_model(model_name, args.default_image_size)
    sess_options = ort.SessionOptions()
    if args.disable_cpu_fallback:
        sess_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(
        str(model_path),
        sess_options=sess_options,
        providers=build_providers(args),
    )
    rng = np.random.default_rng(args.seed)

    inputs: dict[str, np.ndarray] = {}
    input_summaries: list[dict[str, Any]] = []

    for input_meta in session.get_inputs():
        raw_shape = list(input_meta.shape or [])
        resolved_shape, is_dynamic = resolve_shape(
            raw_shape,
            args.default_batch,
            args.default_channels,
            default_image_size,
        )
        dtype = parse_ort_type(input_meta.type)
        inputs[input_meta.name] = random_tensor(resolved_shape, dtype, rng)
        input_summaries.append(
            {
                "name": input_meta.name,
                "dtype": str(dtype),
                "original_shape": raw_shape,
                "resolved_shape": resolved_shape,
                "used_fallbacks": is_dynamic,
            }
        )

    start = time.perf_counter()
    outputs = session.run(None, inputs)
    runtime_ms = (time.perf_counter() - start) * 1000.0

    output_summaries: list[dict[str, Any]] = []
    for output_meta, output_value in zip(session.get_outputs(), outputs):
        dtype = parse_ort_type(output_meta.type)
        output_summaries.append(
            {
                "name": output_meta.name,
                "dtype": str(dtype),
                "shape": list(output_value.shape),
            }
        )

    summary = {
        "model_name": model_name,
        "model_path": str(model_path),
        "providers": session.get_providers(),
        "seed": args.seed,
        "default_image_size": default_image_size,
        "inputs": input_summaries,
        "outputs": output_summaries,
        "runtime_ms": round(runtime_ms, 3),
    }

    summary_path = output_dir / "infer-summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return summary_path


def main() -> int:
    args = parse_args()
    model_path = args.model
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    summary_path = run_inference(model_path, args)
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
