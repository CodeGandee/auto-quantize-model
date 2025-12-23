"""YOLOv10 QAT training helpers using PyTorch Lightning.

This module provides a minimal PyTorch Lightning wrapper around the Ultralytics
YOLOv10 detection model so we can:

- run short QAT fine-tunes on Brevitas-quantized YOLOv10 models,
- log training/validation losses to TensorBoard,
- emit a simple loss curve (CSV + PNG) for quick inspection.

The goal is training management and observability, not a full Ultralytics
trainer reimplementation.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, cast

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn


def freeze_ultralytics_dfl_layers(model: nn.Module) -> int:
    """Freeze DFL-related parameters (Ultralytics default behavior).

    Ultralytics always freezes layers containing ".dfl" by default. This mirrors
    that behavior for our Lightning-based QAT runs.
    """

    frozen = 0
    for name, param in model.named_parameters():
        if ".dfl" in name:
            param.requires_grad = False
            frozen += 1
    return frozen


class Yolov10QATLightningModule(pl.LightningModule):
    """LightningModule wrapper for Ultralytics YOLOv10 detection training."""

    def __init__(
        self,
        model: nn.Module,
        *,
        lr: float,
        weight_decay: float,
        warmup_steps: int,
    ) -> None:
        super().__init__()
        self.m_model = model
        self.m_lr = float(lr)
        self.m_weight_decay = float(weight_decay)
        self.m_warmup_steps = int(warmup_steps)

    def _preprocess_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Ultralytics-style batches to match training expectations.

        Ultralytics `DetectionTrainer.preprocess_batch()` converts images to float
        and scales to `[0, 1]`. Without this, training and subsequent ONNX
        inference can diverge badly because the model is trained on `[0, 255]`.
        """

        images = batch.get("img")
        if torch.is_tensor(images):
            batch["img"] = images.to(self.device, non_blocking=True).float() / 255.0
        return batch

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:  # type: ignore[override]
        batch = self._preprocess_batch(batch)
        loss, loss_items = self.m_model(batch)
        batch_size = int(batch["img"].shape[0]) if "img" in batch else 1

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log(
            "train/loss_mean",
            loss / float(batch_size),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        self._log_loss_items(loss_items, prefix="train", batch_size=batch_size)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:  # type: ignore[override]
        batch = self._preprocess_batch(batch)
        loss, loss_items = self.m_model(batch)
        batch_size = int(batch["img"].shape[0]) if "img" in batch else 1
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("val/loss_mean", loss / float(batch_size), on_step=False, on_epoch=True, batch_size=batch_size)
        self._log_loss_items(loss_items, prefix="val", batch_size=batch_size)
        return loss

    def configure_optimizers(self) -> Any:  # type: ignore[override]
        """Configure optimizer and optional warmup scheduler (Ultralytics-like)."""

        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)
        decay_params: list[torch.nn.Parameter] = []
        norm_params: list[torch.nn.Parameter] = []
        bias_params: list[torch.nn.Parameter] = []

        for module_name, module in self.m_model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname:
                    bias_params.append(param)
                elif isinstance(module, bn):
                    norm_params.append(param)
                else:
                    decay_params.append(param)

        optimizer = torch.optim.AdamW(
            bias_params,
            lr=float(self.m_lr),
            betas=(0.9, 0.999),
            weight_decay=0.0,
        )
        optimizer.add_param_group({"params": decay_params, "weight_decay": float(self.m_weight_decay)})
        optimizer.add_param_group({"params": norm_params, "weight_decay": 0.0})

        warmup_steps = int(self.m_warmup_steps)
        if warmup_steps <= 0:
            return optimizer

        def lr_factor(step: int) -> float:
            # Ultralytics warms up per-batch; Lightning calls this on scheduler.step().
            return min(1.0, float(step + 1) / float(warmup_steps))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def _log_loss_items(self, loss_items: Any, *, prefix: str, batch_size: int) -> None:
        if not isinstance(loss_items, torch.Tensor):
            return

        items = loss_items.detach()
        if items.numel() < 3:
            return

        # Ultralytics YOLOv10 prints these as:
        # box_om cls_om dfl_om box_oo cls_oo dfl_oo
        names = [
            "box_om",
            "cls_om",
            "dfl_om",
            "box_oo",
            "cls_oo",
            "dfl_oo",
        ]
        for idx, name in enumerate(names[: items.numel()]):
            self.log(
                f"{prefix}/{name}",
                items[idx],
                on_step=(prefix == "train"),
                on_epoch=True,
                batch_size=batch_size,
            )


@dataclass(frozen=True)
class QATLightningRunOutputs:
    """Files and directories produced by a Lightning QAT run."""

    log_dir: Path
    checkpoint_dir: Path
    loss_curve_csv: Path
    loss_curve_png: Path


class LossCurveCallback(pl.Callback):
    """Record per-step training loss and write CSV + PNG at the end."""

    def __init__(self, *, out_dir: Path) -> None:
        super().__init__()
        self.m_out_dir = out_dir
        self.m_batch_steps: list[int] = []
        self.m_train_loss_mean: list[float] = []
        self.m_seen_batches = 0

    def on_train_batch_end(  # type: ignore[override]
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if outputs is None:
            return

        if isinstance(outputs, dict) and "loss" in outputs and torch.is_tensor(outputs["loss"]):
            loss = outputs["loss"]
        elif torch.is_tensor(outputs):
            loss = outputs
        else:
            return

        batch_size = int(batch["img"].shape[0]) if isinstance(batch, dict) and "img" in batch else 1
        self.m_seen_batches += 1
        self.m_batch_steps.append(int(self.m_seen_batches))
        self.m_train_loss_mean.append(float((loss.detach().cpu() / float(batch_size)).item()))

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # type: ignore[override]
        self.m_out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.m_out_dir / "loss_curve.csv"
        png_path = self.m_out_dir / "loss_curve.png"

        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["batch_step", "train_loss_mean"])
            for step, value in zip(self.m_batch_steps, self.m_train_loss_mean):
                writer.writerow([step, f"{value:.8f}"])

        if self.m_batch_steps:
            plt.figure(figsize=(10, 4))
            plt.plot(self.m_batch_steps, self.m_train_loss_mean, linewidth=1.0)
            plt.title("Train loss (mean per-image)")
            plt.xlabel("batch_step")
            plt.ylabel("loss")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(png_path, dpi=150)
            plt.close()


def run_lightning_qat(
    *,
    model: nn.Module,
    train_dataloader: Any,
    val_dataloader: Optional[Any],
    out_dir: Path,
    run_name: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
    accelerator: str = "gpu",
    devices: int = 1,
    precision: str = "32-true",
    log_every_n_steps: int = 10,
    gradient_clip_val: float = 10.0,
) -> tuple[nn.Module, QATLightningRunOutputs, Dict[str, Any]]:
    """Run a short QAT fine-tune using PyTorch Lightning."""

    out_dir.mkdir(parents=True, exist_ok=True)
    pl.seed_everything(int(seed), workers=True)

    frozen = freeze_ultralytics_dfl_layers(model)

    batches_per_epoch = len(train_dataloader)
    batch_size = int(getattr(train_dataloader, "batch_size", 1) or 1)
    args = getattr(model, "args", None)
    warmup_epochs = float(getattr(args, "warmup_epochs", 0.0)) if args is not None else 0.0
    nbs = int(getattr(args, "nbs", 64)) if args is not None else 64

    warmup_steps = 0
    if warmup_epochs > 0:
        warmup_steps = int(max(round(warmup_epochs * batches_per_epoch), 100))

    nbs_ratio = float(nbs) / float(batch_size)

    def warmup_accumulate(ni: int) -> int:
        if warmup_steps <= 0:
            return max(1, int(round(nbs_ratio)))
        progress = min(1.0, float(ni) / float(warmup_steps))
        value = 1.0 + (nbs_ratio - 1.0) * progress
        return max(1, int(round(value)))

    last_opt_step = -1
    optimizer_steps = 0
    for ni in range(batches_per_epoch):
        accumulate = warmup_accumulate(ni)
        if ni - last_opt_step >= accumulate:
            optimizer_steps += 1
            last_opt_step = ni
    accumulate_grad_batches = max(1, int(round(batches_per_epoch / max(optimizer_steps, 1))))

    log_dir = out_dir / "lightning"
    checkpoint_dir = out_dir / "checkpoints"
    loss_dir = out_dir / "loss"

    logger = TensorBoardLogger(save_dir=str(log_dir), name=str(run_name))
    checkpoint_cb = ModelCheckpoint(dirpath=str(checkpoint_dir), save_last=True, every_n_epochs=1)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    loss_curve = LossCurveCallback(out_dir=loss_dir)

    pl_module = Yolov10QATLightningModule(
        model,
        lr=float(lr),
        weight_decay=float(weight_decay),
        warmup_steps=warmup_steps,
    )
    trainer = pl.Trainer(
        max_epochs=int(epochs),
        accelerator=str(accelerator),
        devices=int(devices),
        precision=cast(Any, str(precision)),
        logger=logger,
        callbacks=[checkpoint_cb, lr_monitor, loss_curve],
        log_every_n_steps=int(log_every_n_steps),
        gradient_clip_val=float(gradient_clip_val),
        accumulate_grad_batches=int(accumulate_grad_batches),
        enable_checkpointing=True,
        enable_progress_bar=True,
        num_sanity_val_steps=0,
        default_root_dir=str(out_dir),
    )

    trainer.fit(pl_module, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    outputs = QATLightningRunOutputs(
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        loss_curve_csv=loss_dir / "loss_curve.csv",
        loss_curve_png=loss_dir / "loss_curve.png",
    )
    summary = {
        "epochs": int(epochs),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "seed": int(seed),
        "precision": str(precision),
        "frozen_params": int(frozen),
        "batches_per_epoch": int(batches_per_epoch),
        "batch_size": int(batch_size),
        "nbs": int(nbs),
        "warmup_epochs": float(warmup_epochs),
        "warmup_steps": int(warmup_steps),
        "accumulate_grad_batches": int(accumulate_grad_batches),
        "tensorboard_log_dir": str(logger.log_dir),
        "last_checkpoint": str(checkpoint_cb.last_model_path) if checkpoint_cb.last_model_path else None,
        "loss_curve_csv": str(outputs.loss_curve_csv),
        "loss_curve_png": str(outputs.loss_curve_png),
    }
    return model, outputs, summary
