#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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
    src_dir = repo_root / "models" / "yolo11" / "src"
    if src_dir.is_dir():
        sys.path.insert(0, str(src_dir))


def _read_image_paths(list_path: Path, max_samples: int) -> List[Path]:
    if not list_path.is_file():
        raise FileNotFoundError(f"Image list not found: {list_path}")

    paths: List[Path] = []
    with list_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if not value:
                continue
            paths.append(Path(value))
            if len(paths) >= max_samples:
                break
    if not paths:
        raise RuntimeError(f"No image paths loaded from {list_path}")
    return paths


def _letterbox(image, img_size: int) -> Any:
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


def _as_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (list, tuple)) and output:
        if torch.is_tensor(output[0]):
            return output[0]
    raise TypeError(f"Unsupported model output type: {type(output)!r}")


def _forward_step(device: torch.device):
    def _step(model: torch.nn.Module, batch: Mapping[str, torch.Tensor]) -> Any:
        images = batch["images"].to(device)
        return model(images)

    return _step


def _loss_func(device: torch.device):
    def _loss(output: Any, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        _ = batch
        tensor = _as_tensor(output).to(device)
        return tensor.float().pow(2).mean()

    return _loss


def _resolve_yolo_model(repo_root: Path, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    _setup_ultralytics(repo_root)
    from ultralytics import YOLO  # type: ignore[import]

    yolo = YOLO(str(checkpoint))
    model = getattr(yolo, "model", None)
    if not isinstance(model, torch.nn.Module):
        raise RuntimeError("Ultralytics YOLO wrapper did not expose a torch module at `.model`.")
    model.to(device)
    model.eval()
    return model


def _summarize_quant_cfg(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    quant_cfg = cfg.get("quant_cfg") if isinstance(cfg, Mapping) else None
    if not isinstance(quant_cfg, Mapping):
        return {"quant_cfg": type(quant_cfg).__name__}

    def _summ(value: Any) -> Dict[str, Any]:
        if not isinstance(value, Mapping):
            return {"value_type": type(value).__name__}
        out: Dict[str, Any] = {}
        if "axis" in value:
            out["axis"] = value.get("axis")
        if "block_sizes" in value:
            out["block_sizes"] = value.get("block_sizes")
        if "type" in value:
            out["type"] = value.get("type")
        return out

    return {
        "*weight_quantizer": _summ(quant_cfg.get("*weight_quantizer")),
        "*input_quantizer": _summ(quant_cfg.get("*input_quantizer")),
        "default": _summ(quant_cfg.get("default")),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO11n FP8/FP8 AutoQuant layer-sensitivity for multiple granularity variants.",
    )
    repo_root = _find_repo_root(Path(__file__))
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=repo_root,
        help="Repository root (auto-detected).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=repo_root / "models" / "yolo11" / "checkpoints" / "yolo11n.pt",
        help="YOLO11n checkpoint path.",
    )
    parser.add_argument(
        "--image-list",
        type=Path,
        default=repo_root / "datasets" / "quantize-calib" / "quant100.txt",
        help="Text file listing image paths (one per line).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Root directory for per-variant output folders.",
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        required=True,
        help="Directory to write per-variant logs.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Torch device (default: cuda).")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size (default: 640).")
    parser.add_argument("--max-calib-samples", type=int, default=16, help="Calibration images to load (default: 16).")
    parser.add_argument("--batch-size", type=int, default=1, help="Calibration batch size (default: 1).")
    parser.add_argument("--effective-bits", type=float, default=11.0, help="AutoQuant effective-bits constraint.")
    parser.add_argument(
        "--auto-quantize-score-size",
        type=int,
        default=16,
        help="AutoQuant score-size (in samples).",
    )
    parser.add_argument(
        "--base-format",
        type=str,
        default="FP8_DEFAULT_CFG",
        help="Base ModelOpt quantization format name.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    device = torch.device(str(args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False.")

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.log_root.mkdir(parents=True, exist_ok=True)

    image_paths = _read_image_paths(args.image_list, max_samples=int(args.max_calib_samples))
    calib_batches = _build_calib_batches(
        image_paths,
        img_size=int(args.imgsz),
        batch_size=int(args.batch_size),
    )

    base_cfg = resolve_quant_config(str(args.base_format))

    variants: List[Tuple[str, Dict[str, Any]]] = [
        ("baseline_default", {}),
        ("per_axis_channel", {"*input_quantizer": {"axis": 1}}),
        ("per_axis_height", {"*input_quantizer": {"axis": 2}}),
        ("per_axis_width", {"*input_quantizer": {"axis": 3}}),
        ("per_block_channel_8", {"*input_quantizer": {"block_sizes": {1: 8}}}),
        ("per_block_channel_16", {"*input_quantizer": {"block_sizes": {1: 16}}}),
        ("per_block_channel_32", {"*input_quantizer": {"block_sizes": {1: 32}}}),
    ]

    for variant_name, quant_cfg_overrides in variants:
        out_dir = args.output_root / variant_name
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = args.log_root / f"{variant_name}.log"

        scheme = AutoQuantSchemeConfig(
            name=f"yolo11n_fp8fp8_{variant_name}",
            auto_quantize_bits=float(args.effective_bits),
            auto_quantize_method="gradient",
            auto_quantize_score_size=int(args.auto_quantize_score_size),
            coverage_mode="full",
            coverage_fraction=1.0,
            quant_formats=[str(args.base_format)],
        )

        effective_cfg = apply_quant_cfg_overrides(base_cfg, quant_cfg_overrides)
        config_summary = _summarize_quant_cfg(effective_cfg)

        started = time.time()
        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"[INFO] variant={variant_name}\n")
            log.write(f"[INFO] checkpoint={args.checkpoint}\n")
            log.write(f"[INFO] base_format={args.base_format}\n")
            log.write(f"[INFO] quant_cfg_overrides={json.dumps(quant_cfg_overrides)}\n")
            log.write(f"[INFO] effective_quant_cfg_summary={json.dumps(config_summary)}\n")
            log.flush()

            model = _resolve_yolo_model(args.repo_root, args.checkpoint, device=device)

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
                model_id=str(args.checkpoint),
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
                "base_format_name": str(args.base_format),
                "quant_pair": {"weight": "fp8", "activation": "fp8"},
                "quant_granularity": {
                    "name": variant_name,
                    "quant_cfg_overrides": quant_cfg_overrides,
                },
                "effective_quant_cfg_summary": config_summary,
            }

            state_path = out_dir / f"{scheme.name}_autoquant_state.pt"
            manifest_path = out_dir / f"{scheme.name}_quant_manifest.json"
            composed_config_path = out_dir / "composed-config.yaml"
            md_path = out_dir / "layer-sensitivity-report.md"
            json_path = out_dir / "layer-sensitivity-report.json"

            composed_config_payload = {
                "variant": variant_name,
                "checkpoint": str(args.checkpoint),
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
                model_id=str(args.checkpoint),
                dataset=manifest.get("dataset"),
                quantization=manifest.get("quantization") if isinstance(manifest.get("quantization"), dict) else None,
                run_config=manifest.get("run_config") if isinstance(manifest.get("run_config"), dict) else None,
            )
            write_layer_sensitivity_json(manifest=manifest, out_path=json_path)

            elapsed = time.time() - started
            log.write(f"[INFO] completed in {elapsed:.2f}s\n")
            log.write(f"[INFO] out_dir={out_dir}\n")
            log.write(f"[INFO] manifest={manifest_path}\n")
            log.write(f"[INFO] report={md_path}\n")

        print(f"[INFO] Completed {variant_name}: {out_dir}")

    print(f"[INFO] Sweep completed. Outputs: {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
