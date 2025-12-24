#!/usr/bin/env python
"""Train YOLOv10m from scratch (COCO2017): FP16 baseline vs Brevitas W4A16 QAT.

All artifacts are written under a caller-provided `--run-root` (expected to be
under `tmp/`), including:

- run-local YOLO-format COCO dataset (labels + image dir symlinks),
- Ultralytics training outputs (TensorBoard events, results.csv),
- checkpoints every N epochs (default: 5),
- loss curve CSV + PNG,
- ONNX exports (baseline head + Brevitas QCDQ for QAT).

Run in the RTX 5090 Pixi env:
  pixi run -e rtx5090 python scripts/cv-models/train_yolov10m_scratch_fp16_vs_w4a16_qat_brevitas.py ...
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from auto_quantize_model.cv_models.yolov10_brevitas import (
    Yolov10HeadOutput,
    export_brevitas_qcdq_onnx,
    export_yolov10_head_onnx,
    optimize_onnx_keep_qdq,
    quantize_model_brevitas_ptq,
    torch_load_weights_only_disabled,
)
from auto_quantize_model.cv_models.yolov10_coco_dataset import prepare_coco2017_yolo_dataset
from auto_quantize_model.cv_models.yolo_preprocess import find_repo_root


def _read_overrides_yaml(path: Path) -> Dict[str, Any]:
    import yaml  # type: ignore[import-untyped]

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a mapping in {path}, got {type(payload)}")
    return {str(k): v for k, v in payload.items()}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _plot_loss_curve(
    *,
    epochs: Sequence[int],
    loss_values: Sequence[float],
    out_png: Path,
    title: str,
) -> None:
    if not epochs:
        return
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(list(epochs), list(loss_values), linewidth=1.0)
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel("train_loss_total")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def _extract_total_train_loss(results_csv: Path) -> Tuple[List[int], List[float], List[str]]:
    if not results_csv.is_file():
        raise FileNotFoundError(f"Missing Ultralytics results.csv at {results_csv}")

    with results_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        epochs: List[int] = []
        totals: List[float] = []
        train_keys: List[str] = []
        for row in reader:
            normalized: Dict[str, str] = {str(k).strip(): str(v).strip() for k, v in row.items() if k is not None}
            epoch = int(float(normalized.get("epoch", "0")))
            if not train_keys:
                train_keys = sorted(k for k in normalized.keys() if k.startswith("train/"))
            total = 0.0
            for k in train_keys:
                raw = normalized.get(k, "")
                if not raw:
                    continue
                try:
                    total += float(raw)
                except ValueError:
                    continue
            epochs.append(epoch)
            totals.append(float(total))
        return epochs, totals, train_keys


def _write_loss_artifacts(*, results_csv: Path, out_dir: Path, title: str) -> Dict[str, str]:
    epochs, totals, train_keys = _extract_total_train_loss(results_csv)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "loss_curve.csv"
    out_png = out_dir / "loss_curve.png"

    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss_total"])
        for e, v in zip(epochs, totals):
            writer.writerow([int(e), f"{float(v):.8f}"])

    _plot_loss_curve(epochs=epochs, loss_values=totals, out_png=out_png, title=title)
    return {"loss_curve_csv": str(out_csv), "loss_curve_png": str(out_png), "train_keys": ",".join(train_keys)}


def _get_default_model_cfg(repo_root: Path) -> Path:
    return repo_root / "models" / "yolo10" / "src" / "ultralytics" / "cfg" / "models" / "v10" / "yolov10m.yaml"


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--run-root", type=Path, required=True)
    common.add_argument("--coco-root", type=Path, default=Path("datasets/coco2017/source-data"))
    common.add_argument("--model-cfg", type=Path, default=None)
    common.add_argument("--hyp", type=Path, default=Path("conf/cv-models/yolov10m/hyp.scratch.yaml"))
    common.add_argument("--imgsz", type=int, default=640)
    common.add_argument("--epochs", type=int, default=300)
    common.add_argument("--batch", type=int, default=32)
    common.add_argument("--device", type=str, default="0")
    common.add_argument("--workers", type=int, default=8)
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--fraction", type=float, default=1.0)
    common.add_argument("--amp", action="store_true", default=True)
    common.add_argument("--no-amp", dest="amp", action="store_false")
    common.add_argument("--val", action="store_true", default=True)
    common.add_argument("--no-val", dest="val", action="store_false")
    common.add_argument("--save-period", type=int, default=5)
    common.add_argument("--project-subdir", type=str, default=None)
    common.add_argument("--run-name", type=str, default=None)

    fp16 = sub.add_parser("fp16", parents=[common], help="Train FP16/AMP baseline from scratch.")
    fp16.add_argument("--export-opset", type=int, default=13)
    fp16.add_argument("--no-export", dest="export", action="store_false", default=True)

    qat = sub.add_parser("qat-w4a16", parents=[common], help="Train Brevitas W4A16 QAT from scratch.")
    qat.add_argument("--export-opset", type=int, default=13)
    qat.add_argument("--no-export", dest="export", action="store_false", default=True)
    qat.add_argument("--optimize-onnx", action="store_true", default=True)
    qat.add_argument("--no-optimize-onnx", dest="optimize_onnx", action="store_false")

    return parser.parse_args(argv)


def _resolve_model_cfg(args: argparse.Namespace, repo_root: Path) -> Path:
    if args.model_cfg is not None:
        return Path(args.model_cfg)
    return _get_default_model_cfg(repo_root)


def _run_ultralytics_train(
    *,
    trainer_cls: Any,
    overrides: Dict[str, Any],
) -> Any:
    trainer = trainer_cls(overrides=overrides, _callbacks=None)
    _attach_tensorboard_logger(trainer)
    with torch_load_weights_only_disabled():
        trainer.train()
    return trainer


def _attach_tensorboard_logger(trainer: Any) -> Path:
    """Attach a lightweight TensorBoard logger independent of Ultralytics SETTINGS."""

    save_dir = Path(getattr(trainer, "save_dir"))
    tb_dir = save_dir / "tensorboard"
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))

    def on_train_epoch_end(trainer_obj: Any) -> None:
        step = int(getattr(trainer_obj, "epoch", 0)) + 1
        loss_items = trainer_obj.label_loss_items(getattr(trainer_obj, "tloss", None), prefix="train")
        if isinstance(loss_items, dict):
            for k, v in loss_items.items():
                try:
                    writer.add_scalar(str(k), float(v), step)
                except Exception:
                    continue
        lr_items = getattr(trainer_obj, "lr", None)
        if isinstance(lr_items, dict):
            for k, v in lr_items.items():
                try:
                    writer.add_scalar(str(k), float(v), step)
                except Exception:
                    continue

    def on_fit_epoch_end(trainer_obj: Any) -> None:
        step = int(getattr(trainer_obj, "epoch", 0)) + 1
        metrics = getattr(trainer_obj, "metrics", None)
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                try:
                    writer.add_scalar(str(k), float(v), step)
                except Exception:
                    continue

    def on_train_end(_: Any) -> None:
        writer.flush()
        writer.close()

    trainer.add_callback("on_train_epoch_end", on_train_epoch_end)
    trainer.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    trainer.add_callback("on_train_end", on_train_end)
    return tb_dir


def _base_overrides(
    *,
    repo_root: Path,
    dataset_yaml: Path,
    model_cfg: Path,
    hyp_overrides: Dict[str, Any],
    out_dir: Path,
    run_name: str,
    imgsz: int,
    epochs: int,
    batch: int,
    device: str,
    workers: int,
    seed: int,
    fraction: float,
    amp: bool,
    val: bool,
    save_period: int,
) -> Dict[str, Any]:
    overrides: Dict[str, Any] = dict(hyp_overrides)
    overrides.update(
        {
            "mode": "train",
            "task": "detect",
            "model": str(model_cfg),
            "data": str(dataset_yaml),
            "imgsz": int(imgsz),
            "epochs": int(epochs),
            "batch": int(batch),
            "device": str(device),
            "workers": int(workers),
            "seed": int(seed),
            "deterministic": True,
            "fraction": float(fraction),
            "amp": bool(amp),
            "val": bool(val),
            "pretrained": False,
            "resume": False,
            "save": True,
            "save_period": int(save_period),
            "plots": False,
            "exist_ok": True,
            "project": str(out_dir),
            "name": str(run_name),
        }
    )
    return overrides


def run_fp16(args: argparse.Namespace, *, repo_root: Path) -> Dict[str, Any]:
    model_cfg = _resolve_model_cfg(args, repo_root)
    hyp_overrides = _read_overrides_yaml(args.hyp)

    dataset_out_dir = args.run_root / "dataset"
    dataset = prepare_coco2017_yolo_dataset(repo_root=repo_root, coco_root=args.coco_root, out_dir=dataset_out_dir)

    run_name = args.run_name or "yolov10m-scratch-fp16"
    project_dir = args.run_root / (args.project_subdir or "fp16") / "ultralytics"

    from auto_quantize_model.cv_models.yolov10_brevitas import ensure_local_yolo10_src_on_path

    ensure_local_yolo10_src_on_path(repo_root=repo_root)
    from ultralytics.models.yolov10.train import YOLOv10DetectionTrainer  # type: ignore[import-not-found]

    overrides = _base_overrides(
        repo_root=repo_root,
        dataset_yaml=dataset.dataset_yaml,
        model_cfg=model_cfg,
        hyp_overrides=hyp_overrides,
        out_dir=project_dir,
        run_name=run_name,
        imgsz=int(args.imgsz),
        epochs=int(args.epochs),
        batch=int(args.batch),
        device=str(args.device),
        workers=int(args.workers),
        seed=int(args.seed),
        fraction=float(args.fraction),
        amp=bool(args.amp),
        val=bool(args.val),
        save_period=int(args.save_period),
    )

    trainer = _run_ultralytics_train(trainer_cls=YOLOv10DetectionTrainer, overrides=overrides)

    save_dir = Path(trainer.save_dir)
    tb_dir = save_dir / "tensorboard"
    artifacts = _write_loss_artifacts(
        results_csv=save_dir / "results.csv",
        out_dir=save_dir / "loss",
        title="YOLOv10m scratch FP16: train loss (sum of train/*)",
    )

    export_info: Dict[str, Any] | None = None
    onnx_path: Path | None = None
    if bool(args.export):
        onnx_dir = args.run_root / "onnx"
        onnx_path = onnx_dir / "yolov10m-scratch-fp16.onnx"
        model_to_export = getattr(getattr(trainer, "ema", None), "ema", None) or trainer.model
        if not isinstance(model_to_export, nn.Module):
            raise TypeError(f"Unexpected trainer.model type: {type(model_to_export)}")
        export_info = export_yolov10_head_onnx(
            model_to_export,
            out_path=onnx_path,
            head="one2many",
            imgsz=int(args.imgsz),
            opset=int(args.export_opset),
            prefer_fp16=True,
            fp16_device="cuda:0",
        )

    summary = {
        "mode": "fp16",
        "save_dir": str(save_dir),
        "dataset_yaml": str(dataset.dataset_yaml),
        "dataset_provenance": str(dataset.provenance_json),
        "trainer_overrides": overrides,
        "tensorboard_log_dir": str(tb_dir),
        "loss_artifacts": artifacts,
        "onnx_export": export_info,
        "onnx_path": str(onnx_path) if onnx_path else None,
    }
    _write_json(args.run_root / "fp16" / "run_summary.json", summary)
    return summary


def run_qat_w4a16(args: argparse.Namespace, *, repo_root: Path) -> Dict[str, Any]:
    model_cfg = _resolve_model_cfg(args, repo_root)
    hyp_overrides = _read_overrides_yaml(args.hyp)

    dataset_out_dir = args.run_root / "dataset"
    dataset = prepare_coco2017_yolo_dataset(repo_root=repo_root, coco_root=args.coco_root, out_dir=dataset_out_dir)

    run_name = args.run_name or "yolov10m-scratch-qat-w4a16"
    project_dir = args.run_root / (args.project_subdir or "qat-w4a16") / "ultralytics"

    from auto_quantize_model.cv_models.yolov10_brevitas import ensure_local_yolo10_src_on_path

    ensure_local_yolo10_src_on_path(repo_root=repo_root)
    from ultralytics.models.yolov10.train import YOLOv10DetectionTrainer  # type: ignore[import-not-found]

    class Yolov10BrevitasW4A16Trainer(YOLOv10DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):  # type: ignore[override]
            model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            model = quantize_model_brevitas_ptq(model, weight_bit_width=4, act_bit_width=None)
            return model

        def save_model(self) -> None:  # type: ignore[override]
            """Save pickling-free checkpoints for Brevitas models.

            Ultralytics default checkpoints pickle the full model object, which
            can be brittle for graph-transformed Brevitas modules. We instead
            store `state_dict()` plus optimizer/EMA state for resumption.
            """

            from datetime import datetime

            import torch
            from ultralytics.utils.torch_utils import de_parallel  # type: ignore[import-not-found, attr-defined]

            model = de_parallel(self.model) if isinstance(self.model, nn.Module) else self.model
            ema_model = getattr(getattr(self, "ema", None), "ema", None)
            if isinstance(ema_model, nn.Module):
                ema_state = de_parallel(ema_model).state_dict()
            else:
                ema_state = None
            optimizer = getattr(self, "optimizer", None)

            ckpt: Dict[str, Any] = {
                "epoch": int(self.epoch + 1),
                "model_state_dict": model.state_dict() if isinstance(model, nn.Module) else None,
                "ema_state_dict": ema_state,
                "optimizer": optimizer.state_dict() if optimizer is not None else None,
                "train_args": vars(self.args),
                "date": datetime.now().isoformat(),
            }

            self.last.parent.mkdir(parents=True, exist_ok=True)
            torch.save(ckpt, self.last)
            if getattr(self, "best_fitness", None) == getattr(self, "fitness", None):
                torch.save(ckpt, self.best)

            save_period = int(getattr(self.args, "save_period", -1) or -1)
            if save_period > 0 and (int(self.epoch + 1) % save_period == 0):
                torch.save(ckpt, self.wdir / f"epoch{int(self.epoch + 1):03d}.pt")

        def final_eval(self) -> None:  # type: ignore[override]
            # Skip Ultralytics final_eval (expects pickle checkpoints).
            return

    overrides = _base_overrides(
        repo_root=repo_root,
        dataset_yaml=dataset.dataset_yaml,
        model_cfg=model_cfg,
        hyp_overrides=hyp_overrides,
        out_dir=project_dir,
        run_name=run_name,
        imgsz=int(args.imgsz),
        epochs=int(args.epochs),
        batch=int(args.batch),
        device=str(args.device),
        workers=int(args.workers),
        seed=int(args.seed),
        fraction=float(args.fraction),
        amp=bool(args.amp),
        val=bool(args.val),
        save_period=int(args.save_period),
    )

    trainer = _run_ultralytics_train(trainer_cls=Yolov10BrevitasW4A16Trainer, overrides=overrides)

    save_dir = Path(trainer.save_dir)
    tb_dir = save_dir / "tensorboard"
    artifacts = _write_loss_artifacts(
        results_csv=save_dir / "results.csv",
        out_dir=save_dir / "loss",
        title="YOLOv10m scratch QAT W4A16: train loss (sum of train/*)",
    )

    export_info: Dict[str, Any] | None = None
    onnx_path: Path | None = None
    if bool(args.export):
        onnx_dir = args.run_root / "onnx"
        onnx_path = onnx_dir / "yolov10m-scratch-w4a16-qcdq-qat.onnx"

        model_to_export = getattr(getattr(trainer, "ema", None), "ema", None) or trainer.model
        if not isinstance(model_to_export, nn.Module):
            raise TypeError(f"Unexpected trainer.model type: {type(model_to_export)}")

        export_model = Yolov10HeadOutput(model_to_export, head="one2many")
        export_info = export_brevitas_qcdq_onnx(
            export_model,
            out_path=onnx_path,
            imgsz=int(args.imgsz),
            opset=int(args.export_opset),
            fp16_input=True,
            device="cuda:0",
        )

        if bool(args.optimize_onnx):
            opt_path = onnx_path.with_suffix("").with_suffix(".opt.onnx")
            opt_info = optimize_onnx_keep_qdq(onnx_path=onnx_path, out_path=opt_path)
            export_info["optimize"] = opt_info
            onnx_path = Path(opt_info["out_path"])

    summary = {
        "mode": "qat-w4a16",
        "save_dir": str(save_dir),
        "dataset_yaml": str(dataset.dataset_yaml),
        "dataset_provenance": str(dataset.provenance_json),
        "trainer_overrides": overrides,
        "tensorboard_log_dir": str(tb_dir),
        "loss_artifacts": artifacts,
        "onnx_export": export_info,
        "onnx_path": str(onnx_path) if onnx_path else None,
    }
    _write_json(args.run_root / "qat-w4a16" / "run_summary.json", summary)
    return summary


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = find_repo_root(Path.cwd())

    args.run_root.mkdir(parents=True, exist_ok=True)

    if args.command == "fp16":
        _ = run_fp16(args, repo_root=repo_root)
        return 0
    if args.command == "qat-w4a16":
        _ = run_qat_w4a16(args, repo_root=repo_root)
        return 0
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
