"""Ultralytics trainer subclasses for YOLOv10.

Why this module exists
----------------------
Ultralytics spawns multi-GPU training via a generated DDP python file that does:

  from <trainer_module> import <trainer_class>

If the trainer class is defined inside a script executed as `__main__` (or is a
nested class), DDP import will fail. Keeping trainer subclasses in an importable
module avoids that issue and makes DDP usable.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import torch
from torch import nn

from auto_quantize_model.cv_models.yolov10_brevitas import (
    ensure_local_yolo10_src_on_path,
    quantize_model_brevitas_ptq,
)
from auto_quantize_model.cv_models.yolo_preprocess import find_repo_root


def _resolve_repo_root() -> Path:
    env_root = os.environ.get("PIXI_PROJECT_ROOT") or os.environ.get("PIXI_PROJECT_MANIFEST")
    if env_root:
        env_path = Path(env_root).expanduser().resolve()
        if env_path.is_file():
            return env_path.parent
        return env_path
    return find_repo_root(Path.cwd())


_REPO_ROOT = _resolve_repo_root()
ensure_local_yolo10_src_on_path(repo_root=_REPO_ROOT)

from ultralytics.models.yolov10.train import YOLOv10DetectionTrainer  # type: ignore[import-not-found]  # noqa: E402
from ultralytics.utils.torch_utils import de_parallel  # type: ignore[import-not-found, attr-defined]  # noqa: E402


class Yolov10BrevitasW4A16Trainer(YOLOv10DetectionTrainer):
    """YOLOv10 trainer with Brevitas weight-only int4 fake-quant (W4A16)."""

    def get_model(self, cfg=None, weights=None, verbose=True):  # type: ignore[override]
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        model = quantize_model_brevitas_ptq(model, weight_bit_width=4, act_bit_width=None)
        return model

    def save_model(self) -> None:  # type: ignore[override]
        """Save pickling-free checkpoints for Brevitas models.

        Ultralytics default checkpoints pickle the full model object, which can
        be brittle for graph-transformed Brevitas modules. We instead store
        `state_dict()` plus optimizer/EMA state for resumption/export.
        """

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
        return
