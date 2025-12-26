"""Quantization Correction (QC) helpers for YOLOv10 W4A16 validation.

This implements a minimal version of the WACV'24 post-hoc QC stage:

- Insert per-channel affine correction parameters (gamma, beta) around BN.
- Freeze the base QAT model weights.
- Train only QC parameters for a small number of epochs on a calibration subset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable

import torch
from torch import nn


class QcBatchNorm2d(nn.Module):
    """Apply a learnable affine correction before BN: x' = gamma*x + beta, then BN(x')."""

    def __init__(self, bn: nn.BatchNorm2d) -> None:
        super().__init__()
        self.m_bn = bn
        self.m_gamma = nn.Parameter(torch.ones(int(bn.num_features), dtype=torch.float32))
        self.m_beta = nn.Parameter(torch.zeros(int(bn.num_features), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        gamma = self.m_gamma.view(1, -1, 1, 1).to(dtype=x.dtype, device=x.device)
        beta = self.m_beta.view(1, -1, 1, 1).to(dtype=x.dtype, device=x.device)
        return self.m_bn(x * gamma + beta)


@dataclass(frozen=True)
class QcRunResult:
    qc_steps: int
    qc_epochs: int
    lr: float


def insert_qc_modules(model: nn.Module) -> int:
    """Replace eligible BN modules with QcBatchNorm2d wrappers and return count inserted."""

    inserted = 0
    for name, child in model.named_children():
        if isinstance(child, QcBatchNorm2d):
            continue
        if isinstance(child, nn.BatchNorm2d):
            setattr(model, name, QcBatchNorm2d(child))
            inserted += 1
            continue
        inserted += insert_qc_modules(child)
    return inserted


def _iter_qc_parameters(model: nn.Module) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for module in model.modules():
        if isinstance(module, QcBatchNorm2d):
            params.extend([module.m_gamma, module.m_beta])
    return params


def _to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {k: _to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_device(v, device) for v in value)
    return value


def _loss_tensor_from_output(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (list, tuple)) and output:
        first = output[0]
        if torch.is_tensor(first):
            return first
    if isinstance(output, dict):
        for value in output.values():
            if torch.is_tensor(value):
                return value
    raise TypeError(f"Unsupported loss output type: {type(output)!r}")


def run_qc_training(
    *,
    model: nn.Module,
    train_batches: Iterable[Dict[str, Any]],
    device: torch.device,
    lr: float,
    epochs: int,
) -> QcRunResult:
    """Train only QC parameters for `epochs`, freezing base weights and fixing BN stats."""

    model.to(device)
    model.train()

    for param in model.parameters():
        param.requires_grad_(False)

    qc_params = _iter_qc_parameters(model)
    for param in qc_params:
        param.requires_grad_(True)

    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
        if isinstance(module, QcBatchNorm2d):
            module.m_bn.eval()

    if not qc_params:
        raise ValueError("No QC parameters found; did you call insert_qc_modules()?")

    optimizer = torch.optim.Adam(qc_params, lr=float(lr))

    qc_steps = 0
    epochs_int = max(0, int(epochs))
    for _ in range(epochs_int):
        for batch in train_batches:
            batch_device = _to_device(batch, device)

            optimizer.zero_grad(set_to_none=True)

            loss_fn = getattr(model, "loss", None)
            if callable(loss_fn) and isinstance(batch_device, dict) and "img" in batch_device:
                loss_out = loss_fn(batch_device)
                loss = _loss_tensor_from_output(loss_out)
            else:
                if isinstance(batch_device, dict):
                    tensor_candidates = [v for v in batch_device.values() if torch.is_tensor(v)]
                    if not tensor_candidates:
                        raise ValueError("QC batch dict contains no tensors.")
                    x = tensor_candidates[0]
                else:
                    raise TypeError("QC train_batches must yield dicts when model.loss() is unavailable.")

                out = model(x)
                loss = _loss_tensor_from_output(out).float().pow(2).mean()

            loss.backward()
            optimizer.step()
            qc_steps += 1

    return QcRunResult(qc_steps=int(qc_steps), qc_epochs=int(epochs_int), lr=float(lr))
