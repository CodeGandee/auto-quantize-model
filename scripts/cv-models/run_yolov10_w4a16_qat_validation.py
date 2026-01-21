#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from omegaconf import OmegaConf
import torch
from torch import nn

from auto_quantize_model.cv_models.yolov10_brevitas import torch_load_weights_only_disabled
from auto_quantize_model.cv_models.yolov10_coco_subset_dataset import prepare_coco2017_yolo_subset_dataset
from auto_quantize_model.cv_models.yolov10_qc import insert_qc_modules, run_qc_training
from auto_quantize_model.cv_models.yolov10_results_csv import read_metric_series
from auto_quantize_model.cv_models.yolov10_stability import classify_collapse, classify_run_status, summarize_series
from auto_quantize_model.cv_models.yolov10_ultralytics_trainers import Yolov10BrevitasW4A16Trainer
from auto_quantize_model.cv_models.yolov10_w4a16_validation import (
    compose_validation_cfg,
    load_validation_config,
    write_run_summary_json,
    write_run_summary_markdown,
)
from auto_quantize_model.cv_models.yolo_preprocess import find_repo_root


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI args for a single QAT validation run.

    Contract: `specs/001-yolov10-qat-validation/contracts/cli.md`.
    """

    parser = argparse.ArgumentParser(description="Run one YOLOv10 W4A16 QAT validation experiment (EMA + QC).")
    parser.add_argument("--variant", choices=["yolo10n", "yolo10s", "yolo10m"], required=True)
    parser.add_argument("--method", choices=["baseline", "ema", "ema+qc"], required=True)
    parser.add_argument("--profile", choices=["smoke", "short", "full"], required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--coco-root", type=Path, default=Path("datasets/coco2017/source-data"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp", dest="amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _git_metadata(*, repo_root: Path) -> dict[str, Any]:
    def _run(cmd: list[str]) -> str:
        out = subprocess.run(cmd, cwd=repo_root, check=False, capture_output=True, text=True)
        return (out.stdout or "").strip()

    commit = _run(["git", "rev-parse", "HEAD"]) or "unknown"
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]) or "unknown"
    dirty = bool(_run(["git", "status", "--porcelain"]))
    return {"commit": commit, "branch": branch, "dirty": dirty}


def _resolve_device(device: str) -> str:
    device = str(device).strip()
    if device.lower() == "cpu":
        return "cpu"
    # Ultralytics accepts "0,1"; for torch.device we use the first entry.
    first = device.split(",")[0].strip()
    return first or "0"


def _resolve_checkpoint(*, repo_root: Path, checkpoint: str) -> Path:
    checkpoint_path = Path(str(checkpoint)).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = repo_root / checkpoint_path
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    return checkpoint_path


def run_one(
    *,
    variant: str,
    method: str,
    profile: str,
    run_root: Path,
    coco_root: Path,
    imgsz: int,
    epochs: Optional[int],
    batch: Optional[int],
    device: str,
    workers: int,
    seed: int,
    amp: bool,
) -> int:
    repo_root = find_repo_root(Path.cwd())
    config_dir = repo_root / "conf" / "cv-models" / "yolov10_w4a16_validation"

    overrides: Dict[str, Any] = {
        "variant": str(variant),
        "method": str(method),
        "profile": str(profile),
        "run_root": run_root,
        "coco_root": coco_root,
        "imgsz": int(imgsz),
        "device": str(device),
        "workers": int(workers),
        "seed": int(seed),
        "amp": bool(amp),
    }
    if epochs is not None:
        overrides["epochs"] = int(epochs)
    if batch is not None:
        overrides["batch"] = int(batch)

    cfg = compose_validation_cfg(config_dir=config_dir, overrides=overrides)
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(resolved_cfg, dict):
        raise TypeError("Resolved validation config must be a mapping.")

    vcfg = load_validation_config(config_dir=config_dir, overrides=overrides)

    dataset_cfg = cfg.get("dataset") or {}
    train_list_value = dataset_cfg.get("train_list") if hasattr(dataset_cfg, "get") else None
    train_list_path: Path | None = None
    if train_list_value:
        train_list_path = Path(str(train_list_value)).expanduser()
        if not train_list_path.is_absolute():
            train_list_path = repo_root / train_list_path
        train_list_path = train_list_path.resolve()

    train_max_images_value = dataset_cfg.get("train_max_images") if hasattr(dataset_cfg, "get") else None
    train_max_images: int | None = int(train_max_images_value) if train_max_images_value is not None else None

    val_max_images_value = dataset_cfg.get("val_max_images") if hasattr(dataset_cfg, "get") else None
    val_max_images: int | None = int(val_max_images_value) if val_max_images_value is not None else None

    selection_value = dataset_cfg.get("selection") if hasattr(dataset_cfg, "get") else "random"
    selection = str(selection_value or "random")

    coco_root_path = vcfg.coco_root
    if not coco_root_path.is_absolute():
        coco_root_path = (repo_root / coco_root_path).resolve()

    dataset_out_dir = vcfg.run_root / "dataset"
    dataset = prepare_coco2017_yolo_subset_dataset(
        repo_root=repo_root,
        coco_root=coco_root_path,
        out_dir=dataset_out_dir,
        train_list=train_list_path,
        train_max_images=train_max_images,
        val_max_images=val_max_images,
        seed=vcfg.seed,
        selection=selection,
    )

    checkpoint_value = cfg.model.checkpoint
    checkpoint_path = _resolve_checkpoint(repo_root=repo_root, checkpoint=str(checkpoint_value))

    artifacts_cfg = cfg.get("artifacts") or {}
    run_name_template = str(artifacts_cfg.get("run_name_template") or "{variant}-{method}-{profile}-seed{seed}")
    run_name = run_name_template.format(
        variant=vcfg.variant,
        method=vcfg.method.replace("+", "-"),
        profile=vcfg.profile,
        seed=vcfg.seed,
    )
    ultralytics_project_subdir = str(artifacts_cfg.get("ultralytics_project") or "ultralytics")
    project_dir = vcfg.run_root / ultralytics_project_subdir

    ema_cfg = cfg.method.ema if cfg.get("method") is not None and cfg.method.get("ema") is not None else {}
    ema_enabled = bool(ema_cfg.get("enabled", False)) if hasattr(ema_cfg, "get") else False
    ema_decay = float(ema_cfg.get("decay", 0.9999)) if hasattr(ema_cfg, "get") and ema_cfg.get("decay") is not None else None
    ema_tau = float(ema_cfg.get("tau", 2000.0)) if hasattr(ema_cfg, "get") and ema_cfg.get("tau") is not None else None

    trainer_overrides: Dict[str, Any] = {
        "mode": "train",
        "task": "detect",
        "model": str(checkpoint_path),
        "data": str(dataset.dataset_yaml),
        "imgsz": int(vcfg.imgsz),
        "epochs": int(vcfg.epochs),
        "batch": int(vcfg.batch),
        "device": str(vcfg.device),
        "workers": int(vcfg.workers),
        "seed": int(vcfg.seed),
        "deterministic": True,
        "amp": bool(vcfg.amp),
        "val": True,
        "pretrained": True,
        "resume": False,
        "save": True,
        "save_period": -1,
        "plots": False,
        "exist_ok": True,
        "project": str(project_dir),
        "name": str(run_name),
    }

    trainer = Yolov10BrevitasW4A16Trainer(
        overrides=trainer_overrides,
        method_variant=vcfg.method,
        ema_decay=ema_decay if ema_enabled else None,
        ema_tau=ema_tau if ema_enabled else None,
        _callbacks=None,
    )

    save_dir: Path
    results_csv: Path
    tb_dir: Path

    def _attach_tensorboard_logger(trainer_obj: Any) -> Path:
        from torch.utils.tensorboard import SummaryWriter

        save_dir_local = Path(getattr(trainer_obj, "save_dir"))
        tb_dir_local = save_dir_local / "tensorboard"
        tb_dir_local.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_dir_local))

        def on_fit_epoch_end(t: Any) -> None:
            step = int(getattr(t, "epoch", 0)) + 1
            metrics = getattr(t, "metrics", None)
            if isinstance(metrics, dict):
                for k, v in metrics.items():
                    try:
                        writer.add_scalar(str(k), float(v), step)
                    except Exception:
                        continue

        def on_train_end(_: Any) -> None:
            writer.flush()
            writer.close()

        trainer_obj.add_callback("on_fit_epoch_end", on_fit_epoch_end)
        trainer_obj.add_callback("on_train_end", on_train_end)
        return tb_dir_local

    tb_dir = _attach_tensorboard_logger(trainer)
    with torch_load_weights_only_disabled():
        trainer.train()

    save_dir = Path(trainer.save_dir)
    results_csv = save_dir / "results.csv"

    qc_result: dict[str, Any] | None = None
    qc_cfg = cfg.method.qc if cfg.get("method") is not None and cfg.method.get("qc") is not None else {}
    qc_enabled = bool(qc_cfg.get("enabled", False)) if hasattr(qc_cfg, "get") else False
    if qc_enabled:
        model_for_qc = getattr(getattr(trainer, "ema", None), "ema", None) or trainer.model
        if not isinstance(model_for_qc, nn.Module):
            raise TypeError(f"Unexpected trainer.model type: {type(model_for_qc)}")

        inserted = insert_qc_modules(model_for_qc)
        qc_lr = float(qc_cfg.get("lr", 1e-4)) if hasattr(qc_cfg, "get") else 1e-4
        qc_epochs = int(qc_cfg.get("epochs", 1)) if hasattr(qc_cfg, "get") else 1

        calib_batches = []
        train_loader = getattr(trainer, "train_loader", None)
        if train_loader is None:
            raise RuntimeError("Trainer has no train_loader; cannot run QC.")
        for idx, batch_item in enumerate(train_loader):
            calib_batches.append(batch_item)
            if idx >= 9:
                break

        device_first = _resolve_device(vcfg.device)
        device_obj = torch.device("cpu") if device_first == "cpu" else torch.device(f"cuda:{device_first}")
        qc_run = run_qc_training(
            model=model_for_qc,
            train_batches=calib_batches,
            device=device_obj,
            lr=qc_lr,
            epochs=qc_epochs,
        )
        qc_result = {"inserted_modules": int(inserted), "qc_steps": qc_run.qc_steps, "qc_epochs": qc_run.qc_epochs, "lr": qc_run.lr}

    primary_metric = "metrics/mAP50-95(B)"
    series = read_metric_series(results_csv=results_csv, metric_name=primary_metric)
    qat_best_value, qat_final_value = summarize_series(series)

    qc_final_value: float | None = None
    if qc_enabled and qc_result is not None:
        try:
            qc_metrics, _ = trainer.validate()
            if isinstance(qc_metrics, dict) and primary_metric in qc_metrics:
                qc_final_value = float(qc_metrics[primary_metric])
                qc_result["final_metric"] = qc_final_value
        except Exception as exc:  # noqa: BLE001
            qc_result["validate_error"] = str(exc)

    best_value = max(float(qat_best_value), float(qc_final_value)) if qc_final_value is not None else float(qat_best_value)
    final_value = float(qc_final_value) if qc_final_value is not None else float(qat_final_value)
    collapsed = bool(best_value > 0 and final_value < 0.5 * best_value)
    final_over_best_ratio = float(final_value / best_value) if best_value > 0 else 0.0
    status = classify_run_status(collapsed=collapsed, error=None)

    created_at = _now_utc_iso()
    run_id = f"{created_at.split('T')[0]}_{vcfg.variant}_{vcfg.method}_seed{vcfg.seed}"
    git_meta = _git_metadata(repo_root=repo_root)

    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "run_id": run_id,
        "created_at": created_at,
        "git": git_meta,
        "model": {"variant": vcfg.variant, "checkpoint": str(checkpoint_path), "imgsz": int(vcfg.imgsz)},
        "quantization": OmegaConf.to_container(cfg.quantization, resolve=True),
        "method": OmegaConf.to_container(cfg.method, resolve=True),
        "dataset": {
            "coco_root": str(coco_root_path),
            "dataset_yaml": str(dataset.dataset_yaml),
            "train_images": int(dataset.train_images),
            "val_images": int(dataset.val_images),
            "provenance_json": str(dataset.provenance_json),
        },
        "training": {
            "framework": "ultralytics",
            "epochs": int(vcfg.epochs),
            "batch": int(vcfg.batch),
            "seed": int(vcfg.seed),
            "device": str(vcfg.device),
            "amp": bool(vcfg.amp),
            "overrides": trainer_overrides,
        },
        "artifacts": {
            "run_root": str(vcfg.run_root),
            "results_csv": str(results_csv) if results_csv.is_file() else "",
            "tensorboard_log_dir": str(tb_dir) if tb_dir.is_dir() else "",
        },
        "metrics": {"primary_name": primary_metric, "best_value": float(best_value), "final_value": float(final_value)},
        "stability": {
            "collapse_threshold_ratio": 0.5,
            "final_over_best_ratio": float(final_over_best_ratio),
            "collapsed": bool(collapsed),
        },
        "status": status,
        "resolved_config": resolved_cfg,
    }
    if qc_result is not None:
        payload["qc_result"] = qc_result

    write_run_summary_json(out_path=vcfg.run_root / "run_summary.json", payload=payload)
    write_run_summary_markdown(out_path=vcfg.run_root / "summary.md", payload=payload)

    return 0


def main(argv: list[str] | None = None) -> int:
    """Run a single validation experiment and write run artifacts under --run-root.

    Contract: `specs/001-yolov10-qat-validation/contracts/cli.md`.
    """

    args = parse_args(argv)
    run_root = Path(args.run_root).expanduser()
    run_root.mkdir(parents=True, exist_ok=True)

    try:
        return run_one(
            variant=str(args.variant),
            method=str(args.method),
            profile=str(args.profile),
            run_root=run_root,
            coco_root=Path(args.coco_root).expanduser(),
            imgsz=int(args.imgsz),
            epochs=int(args.epochs) if args.epochs is not None else None,
            batch=int(args.batch) if args.batch is not None else None,
            device=str(args.device),
            workers=int(args.workers),
            seed=int(args.seed),
            amp=bool(args.amp),
        )
    except Exception as exc:  # noqa: BLE001
        error_text = traceback.format_exc()
        payload: dict[str, Any] = {
            "schema_version": "1.0",
            "run_id": f"failed_{args.variant}_{args.method}_seed{args.seed}",
            "created_at": _now_utc_iso(),
            "git": _git_metadata(repo_root=find_repo_root(Path.cwd())),
            "model": {"variant": str(args.variant), "checkpoint": "", "imgsz": int(args.imgsz)},
            "quantization": {"mode": "w4a16", "library": "brevitas", "weight_bit_width": 4, "activation_bit_width": None},
            "method": {"variant": str(args.method), "ema": {"enabled": str(args.method) != "baseline"}, "qc": {"enabled": str(args.method) == "ema+qc"}},
            "dataset": {"coco_root": str(Path(args.coco_root).expanduser()), "dataset_yaml": "", "train_images": 0, "val_images": 0},
            "training": {"framework": "ultralytics", "epochs": int(args.epochs or 0), "batch": int(args.batch or 1), "seed": int(args.seed), "device": str(args.device), "amp": bool(args.amp)},
            "artifacts": {"run_root": str(run_root), "results_csv": "", "tensorboard_log_dir": ""},
            "metrics": {"primary_name": "metrics/mAP50-95(B)", "best_value": 0.0, "final_value": 0.0},
            "stability": {"collapse_threshold_ratio": 0.5, "final_over_best_ratio": 0.0, "collapsed": False},
            "status": "incomplete",
            "error": error_text,
        }
        write_run_summary_json(out_path=run_root / "run_summary.json", payload=payload)
        write_run_summary_markdown(out_path=run_root / "summary.md", payload=payload)
        print(f"[ERROR] Run failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
