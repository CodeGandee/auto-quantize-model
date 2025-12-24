"""YOLOv10m + Brevitas PTQ/QAT helpers.

This module provides a small set of utilities used by the Brevitas-based
YOLOv10m PTQ/QAT scripts under `scripts/cv-models/`.

Notes
-----
- Ultralytics YOLOv10 checkpoints rely on pickled custom classes; with
  PyTorch 2.6+ the default `torch.load(weights_only=True)` fails. We keep a
  small context manager that forces `weights_only=False` for trusted local
  checkpoints.
- Brevitas ONNX export requires a Torch-2.9 compatibility shim (see
  `auto_quantize_model.brevitas_onnx_export_compat`).
"""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional, Sequence, cast

import numpy as np
import onnx
import onnxoptimizer  # type: ignore[import-untyped]
import torch
from torch import nn

from auto_quantize_model.brevitas_onnx_export_compat import apply_brevitas_torch_onnx_compat
from auto_quantize_model.cv_models.yolo_preprocess import (
    batched,
    find_repo_root,
    preprocess_image_path,
    read_image_list,
)


@contextlib.contextmanager
def torch_load_weights_only_disabled() -> Iterator[None]:
    """Temporarily force `torch.load(weights_only=False)` for Ultralytics checkpoints."""

    original_torch_load = torch.load

    def _patched_torch_load(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = _patched_torch_load  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.load = original_torch_load  # type: ignore[assignment]


def ensure_local_yolo10_src_on_path(*, repo_root: Path) -> None:
    """Prefer the local YOLOv10 clone under `models/yolo10/src/` if present."""

    src_dir = (repo_root / "models" / "yolo10" / "src").resolve()
    if src_dir.is_dir() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    _patch_ultralytics_dataset_cache_for_ddp()


def _patch_ultralytics_dataset_cache_for_ddp() -> None:
    try:
        from ultralytics.data import dataset as dataset_mod  # type: ignore[import-not-found]
    except Exception:
        return

    if getattr(dataset_mod, "__autoq_ddp_cache_patch__", False):
        return

    original_save = cast(Callable[..., Any], dataset_mod.save_dataset_cache_file)

    def _save_dataset_cache_file(*args: Any, **kwargs: Any) -> Any:
        from ultralytics.utils import LOCAL_RANK  # type: ignore[import-not-found]

        if LOCAL_RANK not in (-1, 0):
            return
        return original_save(*args, **kwargs)

    dataset_mod.save_dataset_cache_file = cast(Any, _save_dataset_cache_file)
    setattr(dataset_mod, "__autoq_ddp_cache_patch__", True)


def load_ultralytics_yolov10(*, checkpoint_path: Path, repo_root: Path) -> Any:
    """Load an Ultralytics YOLOv10 wrapper from a `.pt` checkpoint."""

    ensure_local_yolo10_src_on_path(repo_root=repo_root)
    from ultralytics import YOLOv10  # type: ignore[import-not-found]

    with torch_load_weights_only_disabled():
        return YOLOv10(str(checkpoint_path))


def export_ultralytics_onnx(
    *,
    checkpoint_path: Path,
    out_path: Path,
    repo_root: Path,
    imgsz: int = 640,
    prefer_fp16: bool = True,
    opset: int = 13,
    cleanup_intermediate: bool = True,
) -> Dict[str, Any]:
    """Export a baseline YOLOv10 ONNX model via Ultralytics and copy it to `out_path`."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    model = load_ultralytics_yolov10(checkpoint_path=checkpoint_path, repo_root=repo_root)

    export_kwargs: dict[str, Any] = {"format": "onnx", "imgsz": int(imgsz), "opset": int(opset)}
    exported_path: Optional[Path] = None
    fp16_used = False
    export_error: Optional[str] = None

    if prefer_fp16:
        try:
            exported_path = Path(model.export(**export_kwargs, half=True)).resolve()
            fp16_used = True
        except Exception as exc:  # pragma: no cover - depends on runtime exporter stack
            export_error = f"{type(exc).__name__}: {exc}"
            exported_path = None

    if exported_path is None:
        exported_path = Path(model.export(**export_kwargs, half=False)).resolve()

    if not exported_path.is_file():
        raise FileNotFoundError(f"Ultralytics export returned {exported_path} but it does not exist.")

    out_path.write_bytes(exported_path.read_bytes())
    if cleanup_intermediate:
        try:
            if exported_path != out_path and exported_path.is_file():
                exported_path.unlink()
        except OSError:
            pass

    return {
        "out_path": str(out_path),
        "exported_path": str(exported_path),
        "imgsz": int(imgsz),
        "opset": int(opset),
        "fp16_used": bool(fp16_used),
        "fp16_error": export_error,
    }


def load_yolov10_detection_model(*, checkpoint_path: Path, repo_root: Path) -> nn.Module:
    """Load YOLOv10 as a Torch `nn.Module` suitable for Brevitas wrapping/export."""

    yolo = load_ultralytics_yolov10(checkpoint_path=checkpoint_path, repo_root=repo_root)
    detection_model = getattr(yolo, "model", None)
    if not isinstance(detection_model, nn.Module):
        raise TypeError(f"Unexpected Ultralytics YOLOv10 .model type: {type(detection_model)}")
    return detection_model


class Yolov10HeadOutput(nn.Module):
    """Return a single YOLOv10 head tensor suitable for ONNX export.

    Ultralytics YOLOv10 returns a dict with `one2many` and `one2one` outputs in
    eval mode. Each value is typically a tuple: (pred, aux), where `pred` is a
    Tensor shaped like `[B, 84, 8400]`.
    """

    def __init__(self, model: nn.Module, *, head: str = "one2many") -> None:
        super().__init__()
        self.m_model = model
        self.m_head = str(head)

    def forward(self, images: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        outputs = self.m_model(images)

        if isinstance(outputs, dict):
            if self.m_head not in outputs:
                raise KeyError(f"YOLOv10 output missing key {self.m_head!r}. keys={sorted(outputs.keys())}")
            outputs = outputs[self.m_head]

        if isinstance(outputs, (tuple, list)) and outputs and torch.is_tensor(outputs[0]):
            return outputs[0]
        if torch.is_tensor(outputs):
            return outputs

        raise TypeError(f"Unsupported YOLOv10 output type for ONNX export: {type(outputs)}")


def export_yolov10_head_onnx(
    model: nn.Module,
    *,
    out_path: Path,
    head: str = "one2many",
    imgsz: int = 640,
    opset: int = 13,
    prefer_fp16: bool = True,
    fp16_device: str = "cuda:0",
) -> Dict[str, Any]:
    """Export a YOLOv10 head output ONNX (raw preds, not postprocessed NMS)."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = Yolov10HeadOutput(model, head=head).eval()

    fp16_error: Optional[str] = None
    if prefer_fp16:
        try:
            device_obj = torch.device(fp16_device)
            wrapper = wrapper.to(device_obj).half()
            example = torch.randn(
                1,
                3,
                int(imgsz),
                int(imgsz),
                device=device_obj,
                dtype=torch.float16,
            )
            torch.onnx.export(
                wrapper,
                args=(example,),
                f=str(out_path),
                opset_version=int(opset),
                input_names=["images"],
                output_names=["output0"],
                dynamo=False,
                do_constant_folding=False,
            )
            return {
                "out_path": str(out_path),
                "head": str(head),
                "imgsz": int(imgsz),
                "opset": int(opset),
                "fp16_used": True,
                "fp16_error": None,
            }
        except Exception as exc:  # pragma: no cover - depends on export stack/device
            fp16_error = f"{type(exc).__name__}: {exc}"

    wrapper = wrapper.cpu().float()
    example = torch.randn(1, 3, int(imgsz), int(imgsz), dtype=torch.float32)
    torch.onnx.export(
        wrapper,
        args=(example,),
        f=str(out_path),
        opset_version=int(opset),
        input_names=["images"],
        output_names=["output0"],
        dynamo=False,
    )
    return {
        "out_path": str(out_path),
        "head": str(head),
        "imgsz": int(imgsz),
        "opset": int(opset),
        "fp16_used": False,
        "fp16_error": fp16_error,
    }


def quantize_model_brevitas_ptq(
    model: nn.Module,
    *,
    weight_bit_width: int = 4,
    act_bit_width: Optional[int] = None,
    name_blacklist: Optional[Sequence[str]] = None,
) -> nn.Module:
    """Quantize a model with Brevitas layerwise graph transforms.

    Parameters
    ----------
    model
        Model to quantize.
    weight_bit_width
        Weight bit-width (default: 4).
    act_bit_width
        Activation bit-width. When None, leaves activations floating (W4A16-like).
    name_blacklist
        Optional module-name blacklist for Brevitas `layerwise_quantize`.
    """

    from brevitas.graph.quantize import layerwise_quantize  # type: ignore[import-untyped]
    from brevitas.quant.scaled_int import (  # type: ignore[import-untyped]
        Int4WeightPerTensorFloatDecoupled,
        Int8WeightPerChannelFloat,
        Int8ActPerTensorFloat,
    )
    import brevitas.nn as qnn  # type: ignore[import-untyped]

    weight_bit_width_int = int(weight_bit_width)
    if weight_bit_width_int == 4:
        weight_quant = Int4WeightPerTensorFloatDecoupled
    elif weight_bit_width_int == 8:
        weight_quant = Int8WeightPerChannelFloat
    else:
        raise ValueError(f"Only weight_bit_width in {{4, 8}} is supported for this task, got {weight_bit_width}.")

    compute_layer_map: dict[type[nn.Module], tuple[type[nn.Module], dict[str, Any]]] = {
        nn.Conv2d: (
            qnn.QuantConv2d,
            {
                "weight_quant": weight_quant,
                "weight_bit_width": weight_bit_width_int,
                "input_quant": Int8ActPerTensorFloat if act_bit_width is not None else None,
                "return_quant_tensor": False,
            },
        ),
    }

    if act_bit_width is not None and int(act_bit_width) != 8:
        raise ValueError(f"Only act_bit_width=8 is supported for this task, got {act_bit_width}.")

    quantized = layerwise_quantize(model, compute_layer_map=compute_layer_map, name_blacklist=name_blacklist)
    return quantized


def calibrate_activation_quantizers(
    model: nn.Module,
    *,
    image_list_path: Path,
    repo_root: Path,
    imgsz: int = 640,
    batch_size: int = 4,
    device: str = "cuda:0",
    max_images: Optional[int] = None,
) -> Dict[str, Any]:
    """Run a Brevitas PTQ calibration pass for activation quantizers."""

    from brevitas.graph.calibrate import calibration_mode  # type: ignore[import-untyped]

    image_paths_all = read_image_list(image_list_path, repo_root=repo_root)
    image_paths = image_paths_all
    if max_images is not None:
        image_paths = image_paths[: int(max_images)]

    device_obj = torch.device(device)
    model = model.to(device_obj)
    model.eval()

    seen = 0
    with torch.no_grad(), calibration_mode(model):
        for batch_paths in batched(image_paths, batch_size=int(batch_size)):
            batch_tensors: list[np.ndarray] = []
            for image_path in batch_paths:
                tensor, _ = preprocess_image_path(image_path, img_size=int(imgsz), add_batch_dim=False)
                batch_tensors.append(tensor)

            x_np = np.stack(batch_tensors, axis=0).astype(np.float32, copy=False)
            x = torch.from_numpy(x_np).to(device_obj, non_blocking=True)
            _ = model(x)
            seen += len(batch_paths)

    payload: Dict[str, Any] = {
        "image_list_path": str(image_list_path),
        "list_images": int(len(image_paths_all)),
        "calib_images": int(seen),
        "imgsz": int(imgsz),
        "batch_size": int(batch_size),
        "device": str(device_obj),
    }
    if max_images is not None:
        payload["max_images"] = int(max_images)
    return payload


def export_brevitas_qcdq_onnx(
    model: nn.Module,
    *,
    out_path: Path,
    imgsz: int = 640,
    opset: int = 13,
    fp16_input: bool = False,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Export a quantized model to QCDQ ONNX using Brevitas."""

    from brevitas.export import export_onnx_qcdq  # type: ignore[import-untyped]

    out_path.parent.mkdir(parents=True, exist_ok=True)

    apply_brevitas_torch_onnx_compat()

    device_obj = torch.device(device)
    model = model.eval().to(device_obj)
    if fp16_input:
        model = model.half()
    else:
        model = model.float()
    example = torch.randn(
        1,
        3,
        int(imgsz),
        int(imgsz),
        device=device_obj,
        dtype=torch.float16 if fp16_input else torch.float32,
    )
    export_onnx_qcdq(
        model,
        args=example,
        export_path=str(out_path),
        opset_version=int(opset),
        dynamo=False,
    )

    if not out_path.is_file():
        raise FileNotFoundError(f"Brevitas export did not create {out_path}")

    return {
        "out_path": str(out_path),
        "imgsz": int(imgsz),
        "opset": int(opset),
        "fp16_input": bool(fp16_input),
        "device": str(device_obj),
    }


def count_qdq_nodes(onnx_path: Path) -> Dict[str, int]:
    """Count quantization-related node types in an ONNX graph."""

    model = onnx.load(str(onnx_path))
    counts: dict[str, int] = {"QuantizeLinear": 0, "DequantizeLinear": 0, "Clip": 0}
    for node in model.graph.node:
        if node.op_type in counts:
            counts[node.op_type] += 1
    return counts


def optimize_onnx_keep_qdq(
    *,
    onnx_path: Path,
    out_path: Path,
    passes: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Run conservative `onnxoptimizer` passes and write to `out_path`.

    The default pass list is intentionally conservative to avoid erasing
    QuantizeLinear/DequantizeLinear nodes that make the model inspectable.
    """

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if passes is None:
        passes = (
            "eliminate_deadend",
            "eliminate_identity",
            "eliminate_nop_dropout",
            "eliminate_nop_pad",
            "eliminate_nop_transpose",
            "eliminate_unused_initializer",
            "fuse_consecutive_transposes",
            "fuse_consecutive_squeezes",
            "fuse_consecutive_unsqueezes",
            "set_unique_name_for_nodes",
        )

    before = count_qdq_nodes(onnx_path)
    model = onnx.load(str(onnx_path))
    optimized = onnxoptimizer.optimize(model, list(passes))
    onnx.save(optimized, str(out_path))
    after = count_qdq_nodes(out_path)

    return {"in_path": str(onnx_path), "out_path": str(out_path), "qdq_before": before, "qdq_after": after}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def infer_onnx_io_contract(onnx_path: Path) -> Dict[str, Any]:
    """Return a lightweight I/O summary for a model (names + shapes + dtypes)."""

    model = onnx.load(str(onnx_path))
    inputs: list[dict[str, Any]] = []
    for value_info in model.graph.input:
        t = value_info.type.tensor_type
        dims = [d.dim_value if d.dim_value > 0 else d.dim_param for d in t.shape.dim]
        inputs.append(
            {"name": value_info.name, "dtype": int(t.elem_type), "shape": dims},
        )

    outputs: list[dict[str, Any]] = []
    for value_info in model.graph.output:
        t = value_info.type.tensor_type
        dims = [d.dim_value if d.dim_value > 0 else d.dim_param for d in t.shape.dim]
        outputs.append(
            {"name": value_info.name, "dtype": int(t.elem_type), "shape": dims},
        )

    return {"inputs": inputs, "outputs": outputs}


def default_repo_root() -> Path:
    return find_repo_root(Path.cwd())
