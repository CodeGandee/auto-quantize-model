"""Brevitas ONNX export compatibility shims for newer PyTorch versions.

This repository uses `brevitas==0.12.x` together with the `rtx5090` Pixi
environment which pins a recent PyTorch (`torch==2.9.x`). Brevitas 0.12.x
expects older Torch ONNX internals (e.g. `torch.onnx._globals` and
`torch.onnx.symbolic_helper._export_onnx_opset_version`) that moved in
PyTorch 2.6+.

The helper in this module provides a small, reusable patch so that
`brevitas.export.export_onnx_qcdq(..., dynamo=False)` works without ad-hoc
monkeypatching in notebooks/scripts.
"""

from __future__ import annotations

from typing import Callable, Optional


def _resolve_torch_export_opset_version() -> int:
    """Return the currently configured ONNX opset version in Torch.

    Returns
    -------
    int
        Opset version used by `torch.onnx.export`.
    """

    try:
        from torch.onnx import symbolic_helper  # type: ignore

        maybe_version = getattr(symbolic_helper, "_export_onnx_opset_version", None)
        if isinstance(maybe_version, int) and maybe_version > 0:
            return int(maybe_version)
    except Exception:
        pass

    try:
        from torch.onnx._internal.torchscript_exporter._globals import GLOBALS  # type: ignore

        return int(GLOBALS.export_onnx_opset_version)
    except Exception:
        pass

    try:
        from torch.onnx._globals import GLOBALS  # type: ignore

        return int(GLOBALS.export_onnx_opset_version)
    except Exception as exc:
        raise RuntimeError(
            "Unable to locate Torch ONNX export opset version. "
            "Tried torch.onnx.symbolic_helper._export_onnx_opset_version, "
            "torch.onnx._internal.torchscript_exporter._globals.GLOBALS, and "
            "torch.onnx._globals.GLOBALS."
        ) from exc


def _make_opset_getter() -> Callable[[], int]:
    def _onnx_export_opset() -> int:
        return _resolve_torch_export_opset_version()

    return _onnx_export_opset


def apply_brevitas_torch_onnx_compat(*, force: bool = False) -> bool:
    """Patch Brevitas to use Torch 2.6+ ONNX opset internals.

    Parameters
    ----------
    force
        When True, overwrite an existing patched function.

    Returns
    -------
    bool
        True if a patch was applied, False if it was already applied.
    """

    import brevitas.export.onnx as be_onnx  # type: ignore[import-untyped]
    import brevitas.export.onnx.standard.function as be_std_fn  # type: ignore[import-untyped]

    sentinel = "_autoq_torch_onnx_compat_applied"
    already_applied = bool(getattr(be_onnx, sentinel, False)) and bool(getattr(be_std_fn, sentinel, False))
    if already_applied and not force:
        return False

    opset_getter: Callable[[], int] = _make_opset_getter()

    be_onnx.onnx_export_opset = opset_getter  # type: ignore[assignment]
    setattr(be_onnx, sentinel, True)

    be_std_fn.onnx_export_opset = opset_getter  # type: ignore[assignment]
    setattr(be_std_fn, sentinel, True)

    return True


def get_brevitas_onnx_compat_status() -> dict[str, Optional[int]]:
    """Return a small diagnostic dict for the current compatibility state."""

    opset: Optional[int] = None
    try:
        opset = _resolve_torch_export_opset_version()
    except Exception:
        opset = None

    try:
        import brevitas.export.onnx as be_onnx  # type: ignore[import-untyped]

        applied = bool(getattr(be_onnx, "_autoq_torch_onnx_compat_applied", False))
    except Exception:
        applied = False

    return {"patched": applied, "torch_export_opset": opset}
