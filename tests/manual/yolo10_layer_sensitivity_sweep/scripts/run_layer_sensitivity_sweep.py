#!/usr/bin/env python
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import json
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch
import yaml  # type: ignore[import-untyped]

import modelopt.torch.quantization as mtq  # type: ignore[import-untyped]

from auto_quantize_model.modelopt_autoquant import (
    AutoQuantSchemeConfig,
    build_quant_manifest,
    compute_num_score_steps,
    write_layer_sensitivity_json,
    write_layer_sensitivity_md,
)
from auto_quantize_model.modelopt_configs import resolve_quant_config
from auto_quantize_model.modelopt_quant_overrides import apply_quant_cfg_overrides


_SUPPORTED_DTYPES: Tuple[str, ...] = ("int4", "int8", "fp4", "fp8")
_SUPPORTED_GRANULARITIES: Tuple[str, ...] = ("per_channel", "per_layer")


def _find_repo_root(start: Path) -> Path:
    current = start.resolve()
    for _ in range(12):
        if (current / "pyproject.toml").is_file():
            return current
        if current.parent == current:
            break
        current = current.parent
    raise RuntimeError(f"Failed to locate repo root from {start}")


def _setup_ultralytics(repo_root: Path) -> None:
    src_dir = repo_root / "models" / "yolo10" / "src"
    if src_dir.is_dir():
        sys.path.insert(0, str(src_dir))


@contextlib.contextmanager
def _patched_torch_load_weights_only_false() -> Iterator[None]:
    original_torch_load = torch.load

    def _patched_torch_load(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = _patched_torch_load  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.load = original_torch_load  # type: ignore[assignment]


def _resolve_checkpoint(repo_root: Path, model_name: str) -> Path:
    checkpoint = repo_root / "models" / "yolo10" / "checkpoints" / f"{model_name}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"YOLOv10 checkpoint not found: {checkpoint}")
    return checkpoint


def _resolve_yolo_model(repo_root: Path, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    _setup_ultralytics(repo_root)

    with _patched_torch_load_weights_only_false():
        from ultralytics import YOLOv10  # type: ignore[import-not-found]

        yolo = YOLOv10(str(checkpoint))
    model = getattr(yolo, "model", None)
    if not isinstance(model, torch.nn.Module):
        raise RuntimeError("Ultralytics YOLOv10 wrapper did not expose a torch module at `.model`.")

    model.to(device)
    model.eval()
    return model


def _read_image_paths(repo_root: Path, list_path: Path, max_samples: int) -> List[Path]:
    if not list_path.is_file():
        raise FileNotFoundError(f"Image list not found: {list_path}")

    resolved: List[Path] = []
    with list_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            path = Path(raw)
            if not path.is_absolute():
                path = repo_root / path
            path = path.resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Image not found: {path}")
            resolved.append(path)
            if len(resolved) >= max_samples:
                break

    if not resolved:
        raise RuntimeError(f"No image paths loaded from {list_path}")
    return resolved


def _letterbox(image: Any, img_size: int) -> Any:
    import cv2

    height, width = image.shape[:2]
    scale = min(img_size / width, img_size / height)
    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

    pad_width = img_size - resized_width
    pad_height = img_size - resized_height
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
        value=(114, 114, 114),
    )
    return padded


def _build_calib_batches(
    image_paths: Sequence[Path],
    *,
    img_size: int,
    batch_size: int,
) -> List[Mapping[str, torch.Tensor]]:
    import cv2
    import numpy as np

    tensors: List[torch.Tensor] = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = _letterbox(image, img_size=img_size)
        chw = image.transpose(2, 0, 1).astype("float32") / 255.0
        tensors.append(torch.from_numpy(np.ascontiguousarray(chw)))

    batches: List[Mapping[str, torch.Tensor]] = []
    safe_batch = max(int(batch_size), 1)
    for start in range(0, len(tensors), safe_batch):
        chunk = torch.stack(tensors[start : start + safe_batch], dim=0)
        batches.append({"images": chunk})
    return batches


def _iter_tensor_leaves(value: Any) -> Iterator[torch.Tensor]:
    if torch.is_tensor(value):
        yield value
        return
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_tensor_leaves(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensor_leaves(item)
        return


def _forward_step(device: torch.device):
    def _step(model: torch.nn.Module, batch: Mapping[str, torch.Tensor]) -> Any:
        images = batch["images"].to(device)
        return model(images)

    return _step


def _loss_func(device: torch.device):
    def _loss(output: Any, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        _ = batch
        leaves = [leaf.to(device) for leaf in _iter_tensor_leaves(output)]
        if not leaves:
            raise TypeError(f"Unsupported model output type: {type(output)!r}")
        loss = torch.stack([leaf.float().pow(2).mean() for leaf in leaves]).mean()
        return loss

    return _loss


def _summarize_quant_cfg(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    quant_cfg = cfg.get("quant_cfg") if isinstance(cfg, Mapping) else None
    if not isinstance(quant_cfg, Mapping):
        return {"quant_cfg": type(quant_cfg).__name__}

    def _summ(value: Any) -> Dict[str, Any]:
        if not isinstance(value, Mapping):
            return {"value_type": type(value).__name__}
        out: Dict[str, Any] = {}
        for key in ("type", "num_bits", "axis", "enable", "block_sizes"):
            if key in value:
                out[key] = value.get(key)
        return out

    return {
        "*weight_quantizer": _summ(quant_cfg.get("*weight_quantizer")),
        "*input_quantizer": _summ(quant_cfg.get("*input_quantizer")),
        "default": _summ(quant_cfg.get("default")),
    }


def _quantizer_uses_block_sizes(cfg: Mapping[str, Any], quant_key: str) -> bool:
    quant_cfg = cfg.get("quant_cfg")
    if not isinstance(quant_cfg, Mapping):
        return False
    value = quant_cfg.get(quant_key)
    if isinstance(value, Mapping):
        return bool(value.get("block_sizes"))
    if isinstance(value, list):
        return any(isinstance(item, Mapping) and bool(item.get("block_sizes")) for item in value)
    return False


def _default_effective_bits(weight_dtype: str) -> float:
    if weight_dtype == "int4":
        return 4.0
    if weight_dtype == "int8":
        return 8.0
    if weight_dtype == "fp4":
        return 4.0
    if weight_dtype == "fp8":
        return 11.0
    raise ValueError(f"Unsupported weight dtype: {weight_dtype!r}")


def _copy_quant_entry(src_cfg: Mapping[str, Any], key: str) -> Dict[str, Any]:
    quant_cfg = src_cfg.get("quant_cfg")
    if not isinstance(quant_cfg, Mapping):
        raise TypeError(f"Source cfg missing mapping key quant_cfg for {key}")
    entry = quant_cfg.get(key)
    if not isinstance(entry, Mapping):
        raise TypeError(f"Expected quant_cfg[{key}] mapping but got {type(entry)!r}")
    return copy.deepcopy(dict(entry))


def _select_weight_base_format(weight_dtype: str, fp4_preset: str, int4_preset: str) -> str:
    if weight_dtype == "int8":
        return "INT8_DEFAULT_CFG"
    if weight_dtype == "fp8":
        return "FP8_DEFAULT_CFG"
    if weight_dtype == "fp4":
        if fp4_preset == "mxfp4":
            return "MXFP4_DEFAULT_CFG"
        if fp4_preset == "nvfp4":
            return "NVFP4_DEFAULT_CFG"
        raise ValueError(f"Unsupported fp4 preset: {fp4_preset!r}")
    if weight_dtype == "int4":
        if int4_preset == "awq":
            return "INT4_AWQ_CFG"
        if int4_preset == "blockwise":
            return "INT4_BLOCKWISE_WEIGHT_ONLY_CFG"
        raise ValueError(f"Unsupported int4 preset: {int4_preset!r}")
    raise ValueError(f"Unsupported weight dtype: {weight_dtype!r}")


def build_base_quant_config(
    weight_dtype: str,
    act_dtype: str,
    *,
    fp4_preset: str,
    int4_preset: str,
) -> Tuple[str, Dict[str, Any]]:
    base_name = _select_weight_base_format(weight_dtype, fp4_preset=fp4_preset, int4_preset=int4_preset)
    cfg = copy.deepcopy(resolve_quant_config(base_name))

    quant_cfg_obj = cfg.get("quant_cfg")
    if not isinstance(quant_cfg_obj, MutableMapping):
        raise TypeError(f"Resolved config {base_name} missing mapping key quant_cfg.")
    quant_cfg: MutableMapping[str, Any] = quant_cfg_obj

    weight_entry = quant_cfg.get("*weight_quantizer")
    if isinstance(weight_entry, Mapping):
        weight_patch = dict(weight_entry)
        weight_patch["enable"] = True
        quant_cfg["*weight_quantizer"] = weight_patch

    if act_dtype == weight_dtype:
        return base_name, cfg

    if act_dtype == "int8":
        src = resolve_quant_config("INT8_DEFAULT_CFG")
        patched = _copy_quant_entry(src, "*input_quantizer")
        patched["enable"] = True
        quant_cfg["*input_quantizer"] = patched
        return base_name, cfg

    if act_dtype == "fp8":
        src = resolve_quant_config("FP8_DEFAULT_CFG")
        patched = _copy_quant_entry(src, "*input_quantizer")
        patched["enable"] = True
        quant_cfg["*input_quantizer"] = patched
        return base_name, cfg

    if act_dtype == "int4":
        quant_cfg["*input_quantizer"] = {"num_bits": 4, "axis": None, "enable": True}
        return base_name, cfg

    if act_dtype == "fp4":
        fp4_name = "MXFP4_DEFAULT_CFG" if fp4_preset == "mxfp4" else "NVFP4_DEFAULT_CFG"
        src = resolve_quant_config(fp4_name)
        patched = _copy_quant_entry(src, "*input_quantizer")
        patched["enable"] = True
        quant_cfg["*input_quantizer"] = patched
        return base_name, cfg

    raise ValueError(f"Unsupported activation dtype: {act_dtype!r}")


def build_granularity_overrides(
    base_cfg: Mapping[str, Any],
    granularity: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if granularity not in _SUPPORTED_GRANULARITIES:
        raise ValueError(f"Unsupported granularity: {granularity!r}")

    weight_has_blocks = _quantizer_uses_block_sizes(base_cfg, "*weight_quantizer")
    input_has_blocks = _quantizer_uses_block_sizes(base_cfg, "*input_quantizer")

    if granularity == "per_channel" and (weight_has_blocks or input_has_blocks):
        parts: List[str] = []
        if weight_has_blocks:
            parts.append("*weight_quantizer uses block_sizes")
        if input_has_blocks:
            parts.append("*input_quantizer uses block_sizes")
        reason = "per_channel requires axis-based quantization but " + " and ".join(parts)
        return None, reason

    overrides: Dict[str, Any] = {}

    if granularity == "per_channel":
        overrides["*weight_quantizer"] = {"axis": 0}
        overrides["*input_quantizer"] = {"axis": 1}
        return overrides, None

    if granularity == "per_layer":
        if not weight_has_blocks:
            overrides["*weight_quantizer"] = {"axis": None}
        if not input_has_blocks:
            overrides["*input_quantizer"] = {"axis": None}
        return overrides, None

    raise RuntimeError("Unreachable granularity branch.")


def _top_sensitive_layers(
    layer_sensitivity: Mapping[str, Mapping[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    rows: List[Tuple[str, float]] = []
    for layer_name, entry in layer_sensitivity.items():
        formats = entry.get("formats") or []
        scores = entry.get("scores") or []

        filtered_scores: List[float] = []
        for fmt, score_value in zip(formats, scores):
            if str(fmt).startswith("NONE("):
                continue
            try:
                filtered_scores.append(float(score_value))
            except Exception:
                continue
        if not filtered_scores:
            continue
        rows.append((layer_name.replace(".quant_recipe", ""), max(filtered_scores)))

    rows.sort(key=lambda item: item[1], reverse=True)
    return [{"layer": name, "sensitivity": score} for name, score in rows[: max(int(top_k), 0)]]


@dataclass(frozen=True)
class RunKey:
    model: str
    weight_dtype: str
    act_dtype: str
    granularity: str


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLOv10 AutoQuant layer sensitivity sweeps.")
    repo_root = _find_repo_root(Path(__file__))
    parser.add_argument("--repo-root", type=Path, default=repo_root, help="Repository root (auto-detected).")
    parser.add_argument(
        "--image-list",
        type=Path,
        default=repo_root / "datasets" / "quantize-calib" / "quant100.txt",
        help="Text file listing image paths (one per line).",
    )
    parser.add_argument("--output-root", type=Path, required=True, help="Root directory for per-run outputs.")
    parser.add_argument("--log-root", type=Path, required=True, help="Root directory for per-run logs.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device (default: cuda).")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size (default: 640).")
    parser.add_argument("--max-calib-samples", type=int, default=100, help="Calibration images to load (default: 100).")
    parser.add_argument("--batch-size", type=int, default=1, help="Calibration batch size (default: 1).")
    parser.add_argument(
        "--auto-quantize-score-size",
        type=int,
        default=16,
        help="AutoQuant score-size (in samples).",
    )
    parser.add_argument(
        "--effective-bits",
        type=float,
        default=None,
        help="Override AutoQuant effective-bits constraint (default: derived from weight dtype).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["yolov10n", "yolov10s", "yolov10m"],
        help="Model names to sweep (default: yolov10n yolov10s yolov10m).",
    )
    parser.add_argument(
        "--weight-dtypes",
        nargs="+",
        default=list(_SUPPORTED_DTYPES),
        help="Weight dtypes to include (default: int4 int8 fp4 fp8).",
    )
    parser.add_argument(
        "--act-dtypes",
        nargs="+",
        default=list(_SUPPORTED_DTYPES),
        help="Activation dtypes to include (default: int4 int8 fp4 fp8).",
    )
    parser.add_argument(
        "--granularities",
        nargs="+",
        default=list(_SUPPORTED_GRANULARITIES),
        help="Granularities to include (default: per_channel per_layer).",
    )
    parser.add_argument("--max-runs", type=int, default=None, help="Stop after N attempted runs.")
    parser.add_argument(
        "--fp4-preset",
        type=str,
        choices=("nvfp4", "mxfp4"),
        default="nvfp4",
        help="FP4 preset family to use when fp4 is requested (default: nvfp4).",
    )
    parser.add_argument(
        "--int4-preset",
        type=str,
        choices=("blockwise", "awq"),
        default="blockwise",
        help="INT4 preset family to use when int4 weights are requested (default: blockwise).",
    )
    parser.add_argument("--top-k", type=int, default=25, help="Top-k layers to record in the sweep index.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if int(args.max_calib_samples) < 10:
        raise ValueError("--max-calib-samples must be >= 10 for a meaningful sensitivity run.")

    device = torch.device(str(args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False.")

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.log_root.mkdir(parents=True, exist_ok=True)

    models = [str(name) for name in args.models]
    weight_dtypes = [str(name) for name in args.weight_dtypes]
    act_dtypes = [str(name) for name in args.act_dtypes]
    granularities = [str(name) for name in args.granularities]

    for dtype in weight_dtypes + act_dtypes:
        if dtype not in _SUPPORTED_DTYPES:
            raise ValueError(f"Unsupported dtype {dtype!r}; expected one of {_SUPPORTED_DTYPES}.")
    for granularity in granularities:
        if granularity not in _SUPPORTED_GRANULARITIES:
            raise ValueError(f"Unsupported granularity {granularity!r}; expected one of {_SUPPORTED_GRANULARITIES}.")

    image_paths = _read_image_paths(args.repo_root, args.image_list, max_samples=int(args.max_calib_samples))
    calib_batches = _build_calib_batches(
        image_paths,
        img_size=int(args.imgsz),
        batch_size=int(args.batch_size),
    )

    failures: List[Dict[str, Any]] = []
    successes: List[Dict[str, Any]] = []

    attempted = 0
    for model_name in models:
        checkpoint = _resolve_checkpoint(args.repo_root, model_name)

        for weight_dtype in weight_dtypes:
            for act_dtype in act_dtypes:
                for granularity in granularities:
                    run_key = RunKey(
                        model=model_name,
                        weight_dtype=weight_dtype,
                        act_dtype=act_dtype,
                        granularity=granularity,
                    )

                    attempted += 1
                    if args.max_runs is not None and attempted > int(args.max_runs):
                        break

                    run_name = f"{model_name}_w{weight_dtype}_a{act_dtype}_{granularity}"
                    out_dir = args.output_root / model_name / f"{weight_dtype}-{act_dtype}" / granularity
                    out_dir.mkdir(parents=True, exist_ok=True)
                    log_path = args.log_root / model_name / f"{weight_dtype}-{act_dtype}" / f"{granularity}.log"
                    log_path.parent.mkdir(parents=True, exist_ok=True)

                    started = time.time()

                    try:
                        base_format_name, base_cfg = build_base_quant_config(
                            weight_dtype,
                            act_dtype,
                            fp4_preset=str(args.fp4_preset),
                            int4_preset=str(args.int4_preset),
                        )

                        quant_cfg_overrides, unsupported_reason = build_granularity_overrides(base_cfg, granularity)
                        if quant_cfg_overrides is None:
                            log_path.write_text(
                                "\n".join(
                                    [
                                        f"[INFO] run={run_name}",
                                        f"[INFO] checkpoint={checkpoint}",
                                        f"[INFO] weight_dtype={weight_dtype} act_dtype={act_dtype} granularity={granularity}",
                                        f"[INFO] base_format={base_format_name}",
                                        "[INFO] status=skipped",
                                        f"[INFO] reason={unsupported_reason}",
                                    ]
                                )
                                + "\n",
                                encoding="utf-8",
                            )
                            failures.append(
                                {
                                    "run": run_key.__dict__,
                                    "status": "skipped",
                                    "reason": unsupported_reason,
                                    "base_format_name": base_format_name,
                                    "out_dir": str(out_dir),
                                    "log_path": str(log_path),
                                }
                            )
                            continue

                        effective_cfg = apply_quant_cfg_overrides(base_cfg, quant_cfg_overrides)
                        config_summary = _summarize_quant_cfg(effective_cfg)

                        effective_bits = (
                            float(args.effective_bits)
                            if args.effective_bits is not None
                            else _default_effective_bits(weight_dtype)
                        )

                        scheme = AutoQuantSchemeConfig(
                            name=run_name,
                            auto_quantize_bits=effective_bits,
                            auto_quantize_method="gradient",
                            auto_quantize_score_size=int(args.auto_quantize_score_size),
                            coverage_mode="full",
                            coverage_fraction=1.0,
                            quant_formats=[base_format_name],
                        )

                        with log_path.open("w", encoding="utf-8") as log, contextlib.redirect_stdout(
                            log
                        ), contextlib.redirect_stderr(log):
                            print(f"[INFO] run={run_name}")
                            print(f"[INFO] checkpoint={checkpoint}")
                            print(f"[INFO] weight_dtype={weight_dtype} act_dtype={act_dtype} granularity={granularity}")
                            print(f"[INFO] base_format={base_format_name}")
                            print(f"[INFO] quant_cfg_overrides={json.dumps(quant_cfg_overrides)}")
                            print(f"[INFO] effective_quant_cfg_summary={json.dumps(config_summary)}")
                            print(f"[INFO] effective_bits={effective_bits}")
                            sys.stdout.flush()

                            model = _resolve_yolo_model(args.repo_root, checkpoint, device=device)

                            num_score_steps = compute_num_score_steps(
                                score_size=scheme.auto_quantize_score_size,
                                batch_size=max(int(args.batch_size), 1),
                                num_batches=len(calib_batches),
                            )

                            quantized_model, state_dict = mtq.auto_quantize(
                                model,
                                constraints={"effective_bits": scheme.auto_quantize_bits},
                                quantization_formats=[effective_cfg],
                                data_loader=calib_batches,
                                forward_step=_forward_step(device),
                                loss_func=_loss_func(device),
                                num_calib_steps=len(calib_batches),
                                num_score_steps=num_score_steps,
                                verbose=True,
                            )

                            manifest = build_quant_manifest(
                                model=quantized_model,
                                scheme=scheme,
                                state_dict=state_dict,
                                model_id=str(checkpoint),
                            )
                            manifest["dataset"] = {
                                "image_list": str(args.image_list),
                                "imgsz": int(args.imgsz),
                                "num_calib_samples": len(image_paths),
                                "max_calib_samples": int(args.max_calib_samples),
                                "num_calib_batches": len(calib_batches),
                                "batch_size": int(args.batch_size),
                            }
                            manifest["quantization"] = {
                                "base_format_name": base_format_name,
                                "quant_pair": {"weight": weight_dtype, "activation": act_dtype},
                                "quant_granularity": {
                                    "name": granularity,
                                    "quant_cfg_overrides": quant_cfg_overrides,
                                },
                                "effective_quant_cfg_summary": config_summary,
                                "presets": {
                                    "fp4_preset": str(args.fp4_preset),
                                    "int4_preset": str(args.int4_preset),
                                },
                            }

                            state_path = out_dir / f"{run_name}_autoquant_state.pt"
                            manifest_path = out_dir / f"{run_name}_quant_manifest.json"
                            composed_config_path = out_dir / "composed-config.yaml"
                            md_path = out_dir / "layer-sensitivity-report.md"
                            json_path = out_dir / "layer-sensitivity-report.json"

                            composed_config_payload = {
                                "run": run_key.__dict__,
                                "checkpoint": str(checkpoint),
                                "dataset": manifest.get("dataset"),
                                "autoquant": {
                                    "effective_bits": scheme.auto_quantize_bits,
                                    "method": scheme.auto_quantize_method,
                                    "score_size": scheme.auto_quantize_score_size,
                                    "batch_size": int(args.batch_size),
                                    "imgsz": int(args.imgsz),
                                },
                                "quantization": manifest.get("quantization"),
                            }
                            composed_config_path.write_text(
                                yaml.safe_dump(composed_config_payload, sort_keys=False),
                                encoding="utf-8",
                            )
                            manifest["run_config"] = {"composed_yaml_path": composed_config_path.name}

                            torch.save(state_dict, state_path)
                            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

                            write_layer_sensitivity_md(
                                layer_sensitivity=manifest["layer_sensitivity"],
                                scheme=scheme,
                                autoquant_state=manifest["autoquant_state"],
                                out_path=md_path,
                                model_id=str(checkpoint),
                                dataset=manifest.get("dataset"),
                                quantization=manifest.get("quantization")
                                if isinstance(manifest.get("quantization"), dict)
                                else None,
                                run_config=manifest.get("run_config")
                                if isinstance(manifest.get("run_config"), dict)
                                else None,
                            )
                            write_layer_sensitivity_json(manifest=manifest, out_path=json_path)

                            elapsed = time.time() - started
                            print(f"[INFO] completed in {elapsed:.2f}s")
                            print(f"[INFO] out_dir={out_dir}")
                            print(f"[INFO] manifest={manifest_path}")
                            print(f"[INFO] report={md_path}")

                        top_layers = _top_sensitive_layers(manifest["layer_sensitivity"], top_k=int(args.top_k))
                        successes.append(
                            {
                                "run": run_key.__dict__,
                                "status": "ok",
                                "elapsed_s": time.time() - started,
                                "base_format_name": base_format_name,
                                "out_dir": str(out_dir),
                                "log_path": str(log_path),
                                "manifest_path": str(out_dir / f"{run_name}_quant_manifest.json"),
                                "report_path": str(out_dir / "layer-sensitivity-report.md"),
                                "top_layers": top_layers,
                            }
                        )
                        print(f"[INFO] Completed {run_name}: {out_dir}")

                    except Exception as exc:
                        failures.append(
                            {
                                "run": run_key.__dict__,
                                "status": "failed",
                                "error": str(exc),
                                "traceback": traceback.format_exc(),
                                "out_dir": str(out_dir),
                                "log_path": str(log_path),
                            }
                        )
                        print(f"[WARN] Failed {run_name}: {exc}")
                        try:
                            log_path.parent.mkdir(parents=True, exist_ok=True)
                            mode = "a" if log_path.exists() else "w"
                            with log_path.open(mode, encoding="utf-8") as log:
                                log.write(f"[ERROR] run={run_name}\n")
                                log.write(f"[ERROR] {exc}\n")
                                log.write(traceback.format_exc())
                                log.write("\n")
                        except Exception:
                            pass
                    finally:
                        if device.type == "cuda":
                            torch.cuda.empty_cache()

                if args.max_runs is not None and attempted > int(args.max_runs):
                    break
            if args.max_runs is not None and attempted > int(args.max_runs):
                break
        if args.max_runs is not None and attempted > int(args.max_runs):
            break

    index_json = args.output_root / "sweep_index.json"
    index_csv = args.output_root / "sweep_index.csv"
    failures_json = args.output_root / "failures.json"

    index_json.write_text(json.dumps({"successes": successes, "failures": failures}, indent=2), encoding="utf-8")
    failures_json.write_text(json.dumps(failures, indent=2), encoding="utf-8")

    with index_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "model",
                "weight_dtype",
                "act_dtype",
                "granularity",
                "status",
                "base_format",
                "elapsed_s",
                "out_dir",
                "log_path",
                "top_layers",
            ]
        )
        for entry in successes:
            run = entry.get("run", {})
            top_layers = entry.get("top_layers") or []
            top_str = ";".join(f"{item.get('layer')}:{item.get('sensitivity')}" for item in top_layers)
            writer.writerow(
                [
                    run.get("model"),
                    run.get("weight_dtype"),
                    run.get("act_dtype"),
                    run.get("granularity"),
                    entry.get("status"),
                    entry.get("base_format_name"),
                    f"{entry.get('elapsed_s', 0.0):.3f}",
                    entry.get("out_dir"),
                    entry.get("log_path"),
                    top_str,
                ]
            )
        for entry in failures:
            run = entry.get("run", {})
            writer.writerow(
                [
                    run.get("model"),
                    run.get("weight_dtype"),
                    run.get("act_dtype"),
                    run.get("granularity"),
                    entry.get("status"),
                    entry.get("base_format_name", ""),
                    "",
                    entry.get("out_dir"),
                    entry.get("log_path"),
                    entry.get("reason", entry.get("error", "")),
                ]
            )

    print(f"[INFO] Sweep completed. Outputs: {args.output_root}")
    print(f"[INFO] Index: {index_json}")
    print(f"[INFO] Failures: {failures_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
