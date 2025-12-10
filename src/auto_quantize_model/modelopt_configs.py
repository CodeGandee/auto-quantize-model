"""
Custom ModelOpt quantization configs used by the AutoQuant tooling.

This module defines experimental quantization formats that extend or
override the built-in ModelOpt presets. These configs are intended for
research and analysis, for example:

- Enabling FP8 quantization for all layers (no name-pattern exclusions).
- Comparing ModelOpt default LLM configs against more aggressive or
  exploratory schemes.

The configs returned here are regular ModelOpt quantization config
dicts and can be passed directly to ``modelopt.torch.quantization``
APIs such as ``mtq.quantize`` or ``mtq.auto_quantize``.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

import modelopt.torch.quantization as mtq


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


FP8_ALL_LAYERS_CFG: Dict[str, Any] = _build_fp8_all_layers_cfg()

# Registry of custom quantization configs that can be referenced by name
# from higher-level drivers (e.g., AutoQuant schemes).
CUSTOM_QUANT_CONFIGS: Dict[str, Dict[str, Any]] = {
    "FP8_ALL_LAYERS_CFG": FP8_ALL_LAYERS_CFG,
}
