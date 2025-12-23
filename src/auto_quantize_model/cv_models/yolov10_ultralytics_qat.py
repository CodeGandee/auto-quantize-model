"""Ultralytics-trainer QAT helpers for YOLOv10 models.

This module provides a small wrapper to run short QAT fine-tunes using the
Ultralytics `DetectionTrainer`, while also producing:

- TensorBoard event logs (train loss + components),
- a lightweight loss curve (CSV + PNG).

This keeps Ultralytics as the source of truth for training behavior, while still
making experiments easy to track and visualize.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter


@dataclass(frozen=True)
class UltralyticsQATRunOutputs:
    """Files and directories produced by an Ultralytics QAT run."""

    save_dir: Path
    tensorboard_log_dir: Path
    results_csv: Path
    loss_curve_csv: Path
    loss_curve_png: Path


def _loss_item_names(num_items: int) -> List[str]:
    if int(num_items) == 3:
        return ["box", "cls", "dfl"]
    if int(num_items) == 6:
        return ["box_om", "cls_om", "dfl_om", "box_oo", "cls_oo", "dfl_oo"]
    return [f"loss_{i}" for i in range(int(num_items))]


def _plot_loss_curve(
    *,
    steps: Sequence[int],
    values: Sequence[float],
    out_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    if not steps:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(list(steps), list(values), linewidth=1.0)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def run_ultralytics_qat(
    *,
    model: nn.Module,
    checkpoint_path: Path,
    dataset_yaml: Path,
    out_dir: Path,
    run_name: str,
    imgsz: int,
    epochs: int,
    batch: int,
    device: str,
    seed: int,
    workers: int,
    amp: bool,
    val: bool,
    lr0: float,
    weight_decay: float,
    log_every_n_steps: int = 10,
) -> Tuple[nn.Module, UltralyticsQATRunOutputs, Dict[str, Any]]:
    """Run a short Ultralytics QAT fine-tune with custom logging.

    Parameters
    ----------
    model
        Pre-loaded (and optionally quantized) YOLOv10 detection model.
    checkpoint_path
        Original checkpoint path used for metadata (`trainer.args.model`).
    dataset_yaml
        Ultralytics dataset YAML (YOLO-format labels).
    out_dir
        Base output directory. The trainer will write into `project/name` within
        this directory.
    run_name
        Experiment name for output directories.
    imgsz
        Training image size.
    epochs
        Number of training epochs.
    batch
        Batch size.
    device
        Ultralytics device string (`"0"`, `"cpu"`, `"0,1"`, ...).
    seed
        Random seed.
    workers
        Dataloader workers.
    amp
        Enable AMP in Ultralytics trainer.
    val
        Enable validation inside trainer.
    lr0
        Base learning rate (Ultralytics `lr0`).
    weight_decay
        Weight decay (Ultralytics `weight_decay`).
    log_every_n_steps
        TensorBoard/loss-curve logging frequency in batches.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_yaml = dataset_yaml.resolve()

    # Ultralytics local imports (must come from the YOLOv10 src on sys.path).
    from ultralytics.models.yolo.detect import DetectionTrainer  # type: ignore[import-not-found]
    from ultralytics.utils.torch_utils import de_parallel  # type: ignore[import-not-found, attr-defined]

    trainer_overrides: Dict[str, Any] = {
        "mode": "train",
        "task": "detect",
        "model": str(checkpoint_path),
        "data": str(dataset_yaml),
        "imgsz": int(imgsz),
        "epochs": int(epochs),
        "batch": int(batch),
        "device": str(device),
        "seed": int(seed),
        "workers": int(workers),
        "amp": bool(amp),
        "val": bool(val),
        "lr0": float(lr0),
        "weight_decay": float(weight_decay),
        # Keep trainer light and deterministic.
        "plots": False,
        "save": False,
        "exist_ok": True,
        "project": str(out_dir),
        "name": str(run_name),
    }

    trainer = DetectionTrainer(overrides=trainer_overrides, _callbacks=None)
    trainer.model = model  # ensures `setup_model()` is skipped

    # YOLOv10 returns a dict with `one2many` / `one2one` head outputs, but Ultralytics'
    # default validator expects a tensor. We rely on the repository's ONNX evaluation
    # script instead, so we skip in-training validation to avoid crashes.
    def validate_noop() -> Tuple[Dict[str, float], float]:
        loss_value = getattr(trainer, "loss", None)
        fitness = 0.0
        if torch.is_tensor(loss_value):
            try:
                fitness = -float(loss_value.detach().float().cpu().item())
            except Exception:
                fitness = 0.0
        return {}, fitness

    trainer.validate = validate_noop  # type: ignore[method-assign]

    # Ultralytics always calls save_model() on final epoch; avoid pickle/dill issues with Brevitas injectors.
    trainer.save_model = lambda: None  # type: ignore[method-assign]
    trainer.final_eval = lambda: None  # type: ignore[method-assign]

    save_dir = Path(trainer.save_dir)
    tb_log_dir = save_dir / "tensorboard"
    loss_dir = save_dir / "loss"
    loss_curve_csv = loss_dir / "loss_curve.csv"
    loss_curve_png = loss_dir / "loss_curve.png"

    tb_log_dir.mkdir(parents=True, exist_ok=True)
    loss_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(tb_log_dir))

    step_counter = 0
    logged_steps: List[int] = []
    logged_loss_mean: List[float] = []

    def on_train_batch_end(trainer_obj: Any) -> None:
        nonlocal step_counter

        if getattr(trainer_obj, "loss", None) is None:
            return

        step_counter += 1
        if int(log_every_n_steps) > 1 and step_counter % int(log_every_n_steps) != 0:
            return

        loss_tensor = trainer_obj.loss
        if not torch.is_tensor(loss_tensor):
            return

        batch_size = max(1, int(getattr(trainer_obj, "batch_size", 1)))
        loss_value = float(loss_tensor.detach().float().cpu().item())
        loss_mean_value = loss_value / float(batch_size)

        logged_steps.append(int(step_counter))
        logged_loss_mean.append(float(loss_mean_value))

        writer.add_scalar("train/loss_mean", loss_mean_value, step_counter)

        loss_items = getattr(trainer_obj, "loss_items", None)
        if torch.is_tensor(loss_items):
            items = loss_items.detach().float().cpu().flatten()
            names = _loss_item_names(int(items.numel()))
            for idx, name in enumerate(names):
                if idx >= int(items.numel()):
                    break
                writer.add_scalar(f"train/{name}", float(items[idx].item()), step_counter)

    def on_train_end(trainer_obj: Any) -> None:
        writer.flush()
        writer.close()

        loss_curve_csv.parent.mkdir(parents=True, exist_ok=True)
        with loss_curve_csv.open("w", newline="", encoding="utf-8") as handle:
            csv_writer = csv.writer(handle)
            csv_writer.writerow(["batch_step", "train_loss_mean"])
            for step, value in zip(logged_steps, logged_loss_mean):
                csv_writer.writerow([step, f"{value:.8f}"])

        _plot_loss_curve(
            steps=logged_steps,
            values=logged_loss_mean,
            out_path=loss_curve_png,
            title="Train loss (mean per-image)",
            xlabel="batch_step",
            ylabel="loss",
        )

    trainer.add_callback("on_train_batch_end", on_train_batch_end)
    trainer.add_callback("on_train_end", on_train_end)

    trainer.train()

    trained_model = trainer.model
    try:
        trained_model = de_parallel(trained_model)
    except Exception:
        pass

    outputs = UltralyticsQATRunOutputs(
        save_dir=save_dir,
        tensorboard_log_dir=tb_log_dir,
        results_csv=save_dir / "results.csv",
        loss_curve_csv=loss_curve_csv,
        loss_curve_png=loss_curve_png,
    )
    summary: Dict[str, Any] = {
        "trainer_overrides": trainer_overrides,
        "save_dir": str(save_dir),
        "tensorboard_log_dir": str(tb_log_dir),
        "loss_curve_csv": str(loss_curve_csv),
        "loss_curve_png": str(loss_curve_png),
        "results_csv": str(outputs.results_csv),
        "logged_steps": int(len(logged_steps)),
        "log_every_n_steps": int(log_every_n_steps),
    }
    return trained_model, outputs, summary
