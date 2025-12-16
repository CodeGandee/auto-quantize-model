"""
Custom ModelOpt quantization configs used by the AutoQuant tooling.

This module defines experimental quantization formats that extend or
override the built-in ModelOpt presets. These configs are intended for
research and analysis, for example:

- Enabling FP8 quantization for all layers (no name-pattern exclusions).
- Creating "weight-only" variants (disable input quantizers) to study
  weight precision independently from activation precision.
- Defining hybrid weight/activation formats that are not shipped as
  built-in presets (experimental).
- Comparing ModelOpt default LLM configs against more aggressive or
  exploratory schemes.

The configs returned here are regular ModelOpt quantization config
dicts and can be passed directly to ``modelopt.torch.quantization``
APIs such as ``mtq.quantize`` or ``mtq.auto_quantize``.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

import modelopt.torch.quantization as mtq  # type: ignore[import-untyped]


def resolve_quant_config(format_name: str) -> Dict[str, Any]:
    """Resolve a quantization format name to a ModelOpt quantization config dict.

    Parameters
    ----------
    format_name:
        Either a key in :data:`CUSTOM_QUANT_CONFIGS` or the name of a built-in
        preset exposed as an attribute of ``modelopt.torch.quantization`` (e.g.,
        ``FP8_DEFAULT_CFG``, ``INT8_WEIGHT_ONLY_CFG``).

    Returns
    -------
    dict
        A ModelOpt quantization config dictionary suitable for passing to
        ``mtq.quantize`` or ``mtq.auto_quantize``.

    Raises
    ------
    ValueError
        If the format name cannot be resolved.
    TypeError
        If the resolved attribute is not a dict-like config.
    """
    if format_name in CUSTOM_QUANT_CONFIGS:
        return CUSTOM_QUANT_CONFIGS[format_name]

    if hasattr(mtq, format_name):
        cfg = getattr(mtq, format_name)
        if not isinstance(cfg, dict):
            raise TypeError(
                f"Resolved ModelOpt format {format_name!r} is not a dict config: {type(cfg)!r}"
            )
        return cfg

    raise ValueError(
        f"Unknown ModelOpt quantization format: {format_name!r}. "
        f"Expected a key in CUSTOM_QUANT_CONFIGS or a preset in modelopt.torch.quantization."
    )


def _build_fp8_all_layers_cfg() -> Dict[str, Any]:
    """Build an FP8 quantization config that enables all layers.

    This config starts from ``mtq.FP8_DEFAULT_CFG`` but removes the
    default LLM name-pattern exclusions (e.g., ``*lm_head*``,
    ``*router*``) and instead enables quantization by default for any
    quantizer that exists.

    Notes
    -----
    - This is intended for analysis and comparison; it is not expected
      to be an accuracy-optimized configuration.
    - The structure matches the usual ModelOpt quantization config:
      a dict with keys ``\"quant_cfg\"`` and ``\"algorithm\"``.
    """
    cfg: Dict[str, Any] = deepcopy(mtq.FP8_DEFAULT_CFG)
    quant_cfg = cfg.get("quant_cfg", {})

    # Preserve only the generic weight/input quantizer patterns; drop
    # the LLM-specific disable patterns so we can study all layers.
    keep_keys = {"*weight_quantizer", "*input_quantizer"}
    keys_to_remove = [key for key in list(quant_cfg.keys()) if key not in keep_keys]
    for key in keys_to_remove:
        quant_cfg.pop(key, None)

    # Enable all remaining quantizers by default with FP8 attributes
    # mirroring the generic input quantizer settings.
    input_cfg = quant_cfg.get("*input_quantizer", {})
    num_bits = input_cfg.get("num_bits", (4, 3))
    axis = input_cfg.get("axis", None)
    quant_cfg["default"] = {
        "num_bits": num_bits,
        "axis": axis,
        "enable": True,
    }

    return cfg


def _build_int8_lm_default_cfg() -> Dict[str, Any]:
    """Build an INT8 quantization config for LM-only flows.

    This config is a direct copy of ``mtq.INT8_DEFAULT_CFG`` so that it
    preserves ModelOpt's built-in name-pattern filters and default
    enablement for large language models.

    Returns
    -------
    dict
        A deep copy of the default INT8 configuration suitable for LM
        AutoQuant runs.
    """
    cfg: Dict[str, Any] = deepcopy(mtq.INT8_DEFAULT_CFG)
    return cfg


def _build_int8_all_layers_cfg() -> Dict[str, Any]:
    """Build an INT8 quantization config that enables all layers.

    This config mirrors :func:`_build_fp8_all_layers_cfg` but starts
    from ``mtq.INT8_DEFAULT_CFG`` so that both the vision and language
    towers can be quantized with INT8 where supported.

    Returns
    -------
    dict
        INT8 configuration with generic quantizers enabled for all
        layers (subject to ModelOpt operator support).
    """
    cfg: Dict[str, Any] = deepcopy(mtq.INT8_DEFAULT_CFG)
    quant_cfg = cfg.get("quant_cfg", {})

    keep_keys = {"*weight_quantizer", "*input_quantizer"}
    keys_to_remove = [key for key in list(quant_cfg.keys()) if key not in keep_keys]
    for key in keys_to_remove:
        quant_cfg.pop(key, None)

    input_cfg = quant_cfg.get("*input_quantizer", {})
    num_bits = input_cfg.get("num_bits", 8)
    axis = input_cfg.get("axis", None)
    quant_cfg["default"] = {
        "num_bits": num_bits,
        "axis": axis,
        "enable": True,
    }

    return cfg


def _build_int8_weight_only_cfg() -> Dict[str, Any]:
    """Build an INT8 weight-only quantization config.

    Some ModelOpt releases expose this as ``INT8_WEIGHT_ONLY_CFG``; we
    provide it here as a stable config name for experiment tooling.
    """
    cfg: Dict[str, Any] = deepcopy(mtq.INT8_DEFAULT_CFG)
    quant_cfg = cfg.get("quant_cfg", {})
    quant_cfg["*input_quantizer"] = {"enable": False}
    return cfg


def _build_fp8_weight_only_cfg() -> Dict[str, Any]:
    """Build an FP8 weight-only quantization config.

    This is derived from ``mtq.FP8_DEFAULT_CFG`` but disables the input
    quantizer so activations remain in higher precision (e.g., FP16/BF16)
    while weights are quantized to FP8.
    """
    cfg: Dict[str, Any] = deepcopy(mtq.FP8_DEFAULT_CFG)
    quant_cfg = cfg.get("quant_cfg", {})
    quant_cfg["*input_quantizer"] = {"enable": False}
    return cfg


def _build_nvfp4_weight_only_cfg() -> Dict[str, Any]:
    """Build an NVFP4 weight-only quantization config.

    NVFP4 quantization uses NVIDIA FP4 formats and typically pairs with
    FP8 activation quantization in ModelOpt's built-in W4A8 presets. This
    variant disables the input quantizer so we can study FP4 weights while
    keeping activations in higher precision (e.g., FP16/BF16).
    """
    cfg: Dict[str, Any] = deepcopy(mtq.NVFP4_DEFAULT_CFG)
    quant_cfg = cfg.get("quant_cfg", {})
    quant_cfg["*input_quantizer"] = {"enable": False}
    return cfg


def _build_mxfp4_weight_only_cfg() -> Dict[str, Any]:
    """Build an MXFP4 weight-only quantization config.

    MXFP4 uses dynamic per-block FP4 quantization. This variant disables
    input quantization to isolate weight precision effects.
    """
    cfg: Dict[str, Any] = deepcopy(mtq.MXFP4_DEFAULT_CFG)
    quant_cfg = cfg.get("quant_cfg", {})
    quant_cfg["*input_quantizer"] = {"enable": False}
    return cfg


def _build_int8_weight_fp8_act_cfg() -> Dict[str, Any]:
    """Build an experimental INT8-weight + FP8-activation config.

    ModelOpt ships INT8 and FP8 presets, but does not always expose a
    dedicated "INT8 weights + FP8 activations" format. For sensitivity
    analysis we sometimes want this hybrid to disentangle weight vs
    activation effects.

    Notes
    -----
    - This is an experimental, non-standard configuration.
    - If the underlying backend does not support the hybrid quantizer
      types for a given operator, quantization may be skipped or the run
      may fail at runtime.
    """
    cfg: Dict[str, Any] = deepcopy(mtq.INT8_DEFAULT_CFG)
    quant_cfg = cfg.get("quant_cfg", {})
    fp8_input_cfg = deepcopy(getattr(mtq, "FP8_DEFAULT_CFG")["quant_cfg"]["*input_quantizer"])
    quant_cfg["*input_quantizer"] = fp8_input_cfg
    return cfg


FP8_ALL_LAYERS_CFG: Dict[str, Any] = _build_fp8_all_layers_cfg()
INT8_LM_DEFAULT_CFG: Dict[str, Any] = _build_int8_lm_default_cfg()
INT8_ALL_LAYERS_CFG: Dict[str, Any] = _build_int8_all_layers_cfg()
INT8_WEIGHT_ONLY_CFG: Dict[str, Any] = _build_int8_weight_only_cfg()
FP8_WEIGHT_ONLY_CFG: Dict[str, Any] = _build_fp8_weight_only_cfg()
INT8_WEIGHT_FP8_ACT_CFG: Dict[str, Any] = _build_int8_weight_fp8_act_cfg()
NVFP4_WEIGHT_ONLY_CFG: Dict[str, Any] | None = (
    _build_nvfp4_weight_only_cfg() if hasattr(mtq, "NVFP4_DEFAULT_CFG") else None
)
MXFP4_WEIGHT_ONLY_CFG: Dict[str, Any] | None = (
    _build_mxfp4_weight_only_cfg() if hasattr(mtq, "MXFP4_DEFAULT_CFG") else None
)

# Registry of custom quantization configs that can be referenced by name
# from higher-level drivers (e.g., AutoQuant schemes).
CUSTOM_QUANT_CONFIGS: Dict[str, Dict[str, Any]] = {
    "FP8_ALL_LAYERS_CFG": FP8_ALL_LAYERS_CFG,
    "INT8_LM_DEFAULT_CFG": INT8_LM_DEFAULT_CFG,
    "INT8_ALL_LAYERS_CFG": INT8_ALL_LAYERS_CFG,
    "INT8_WEIGHT_ONLY_CFG": INT8_WEIGHT_ONLY_CFG,
    "FP8_WEIGHT_ONLY_CFG": FP8_WEIGHT_ONLY_CFG,
    "INT8_WEIGHT_FP8_ACT_CFG": INT8_WEIGHT_FP8_ACT_CFG,
}

if NVFP4_WEIGHT_ONLY_CFG is not None:
    CUSTOM_QUANT_CONFIGS["NVFP4_WEIGHT_ONLY_CFG"] = NVFP4_WEIGHT_ONLY_CFG
if MXFP4_WEIGHT_ONLY_CFG is not None:
    CUSTOM_QUANT_CONFIGS["MXFP4_WEIGHT_ONLY_CFG"] = MXFP4_WEIGHT_ONLY_CFG
