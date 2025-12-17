"""Helpers for applying Hydra quantization overlays onto ModelOpt configs.

This module implements a small, deterministic overlay mechanism for ModelOpt
quantization configs (dicts with a ``quant_cfg`` key) so Hydra can sweep over
quantizer granularity choices (axis vs block_sizes) without duplicating full
configs.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping, MutableMapping, Set

_SUPPORTED_QUANT_CFG_KEYS: Set[str] = {"*weight_quantizer", "*input_quantizer", "default"}
_BLOCK_SIZES_SPECIAL_KEYS: Set[str] = {"type", "scale_bits", "scale_block_sizes"}


def apply_quant_cfg_overrides(
    base_cfg: Mapping[str, Any],
    quant_cfg_overrides: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    """Return a deep-copied ModelOpt config with ``quant_cfg_overrides`` applied.

    Parameters
    ----------
    base_cfg:
        Base ModelOpt quantization config dict (as returned by
        :func:`auto_quantize_model.modelopt_configs.resolve_quant_config`).
    quant_cfg_overrides:
        Overlay mapping keyed by entries in ``cfg["quant_cfg"]`` (currently
        limited to ``"*weight_quantizer"``, ``"*input_quantizer"``, and
        optionally ``"default"``). Each value is a dict of quantizer-attribute
        overrides (e.g., ``{"axis": 0}`` or ``{"block_sizes": {-1: 128}}``).

    Notes
    -----
    - The returned config is a deep copy; ``base_cfg`` is never mutated.
    - ModelOpt disallows setting both ``axis`` and ``block_sizes`` on the same
      quantizer attributes. When the overlay provides ``axis``, any existing
      ``block_sizes`` entry is removed (and vice versa).
    - YAML/OmegaConf may represent ``block_sizes`` axis keys as strings (e.g.,
      ``"-1"``). These are normalized to integers while preserving special
      string keys like ``"type"`` and ``"scale_block_sizes"``.
    """

    cfg: Dict[str, Any] = deepcopy(dict(base_cfg))

    overrides: Dict[str, Any] = {}
    if quant_cfg_overrides is not None:
        overrides = _to_container(quant_cfg_overrides)

    if not overrides:
        return cfg

    quant_cfg = cfg.get("quant_cfg")
    if not isinstance(quant_cfg, MutableMapping):
        raise TypeError(
            "ModelOpt config missing mapping key `quant_cfg`; "
            f"expected dict-like config but got {type(quant_cfg)!r}."
        )

    for key, override_value in overrides.items():
        if key not in _SUPPORTED_QUANT_CFG_KEYS:
            raise ValueError(
                f"Unsupported quant_cfg override key {key!r}; "
                f"expected one of {sorted(_SUPPORTED_QUANT_CFG_KEYS)}."
            )
        if not isinstance(override_value, Mapping):
            raise TypeError(
                f"Override for {key!r} must be a mapping of quantizer attributes; "
                f"got {type(override_value)!r}."
            )

        base_value = quant_cfg.get(key)
        quant_cfg[key] = _apply_quantizer_attr_overrides(
            base_value,
            override_value=_to_container(override_value),
            context=f"quant_cfg[{key}]",
        )

    return cfg


def _apply_quantizer_attr_overrides(
    base_value: Any,
    *,
    override_value: Dict[str, Any],
    context: str,
) -> Any:
    """Apply a quantizer-attribute overlay to a single quant_cfg value."""

    if isinstance(base_value, list):
        if not base_value:
            raise ValueError(f"{context} is an empty list; expected dict-like quantizer attributes.")

        updated: list[Any] = []
        for index, entry in enumerate(base_value):
            updated.append(
                _apply_quantizer_attr_overrides(
                    entry,
                    override_value=override_value,
                    context=f"{context}[{index}]",
                )
            )
        return updated

    if base_value is None:
        base_attrs: Dict[str, Any] = {}
    elif isinstance(base_value, Mapping):
        base_attrs = deepcopy(dict(base_value))
    else:
        raise TypeError(
            f"{context} must be a mapping of quantizer attributes (or a list of mappings); "
            f"got {type(base_value)!r}."
        )

    if "axis" in override_value and "block_sizes" in override_value:
        raise ValueError(
            f"{context} override cannot set both `axis` and `block_sizes` "
            "(ModelOpt requires selecting exactly one granularity mechanism)."
        )

    merged = deepcopy(base_attrs)
    if "block_sizes" in override_value:
        block_sizes_override = override_value.get("block_sizes")
        if not isinstance(block_sizes_override, Mapping):
            raise TypeError(f"{context}.block_sizes override must be a mapping; got {type(block_sizes_override)!r}.")

        base_block_sizes = merged.get("block_sizes")
        base_block_sizes_norm: Dict[Any, Any] = {}
        if base_block_sizes is None:
            base_block_sizes_norm = {}
        elif isinstance(base_block_sizes, Mapping):
            base_block_sizes_norm = _normalize_block_sizes_mapping(base_block_sizes)
        else:
            raise TypeError(f"{context}.block_sizes must be a mapping when present; got {type(base_block_sizes)!r}.")

        override_block_sizes_norm = _normalize_block_sizes_mapping(block_sizes_override)
        base_block_sizes_norm.update(override_block_sizes_norm)
        merged["block_sizes"] = base_block_sizes_norm

    for attr_key, attr_value in override_value.items():
        if attr_key == "block_sizes":
            continue
        merged[attr_key] = attr_value

    if "axis" in override_value:
        merged.pop("block_sizes", None)
    if "block_sizes" in override_value:
        merged.pop("axis", None)

    if "axis" in merged and "block_sizes" in merged:
        raise ValueError(f"{context} cannot contain both `axis` and `block_sizes` after merge.")

    if "block_sizes" in merged:
        merged["block_sizes"] = _normalize_block_sizes_mapping(merged["block_sizes"])

    return merged


def _normalize_block_sizes_mapping(block_sizes: Mapping[Any, Any]) -> Dict[Any, Any]:
    """Normalize YAML-friendly block_sizes mappings to ModelOpt-friendly keys."""

    normalized: Dict[Any, Any] = {}
    for raw_key, raw_value in block_sizes.items():
        key = _normalize_block_sizes_key(raw_key)
        value: Any = raw_value
        if key == "scale_block_sizes":
            if not isinstance(raw_value, Mapping):
                raise TypeError(
                    "block_sizes.scale_block_sizes must be a mapping; "
                    f"got {type(raw_value)!r}."
                )
            value = _normalize_block_sizes_mapping(raw_value)
        normalized[key] = value
    return normalized


def _normalize_block_sizes_key(key: Any) -> Any:
    if isinstance(key, int):
        return key
    if isinstance(key, str):
        if key in _BLOCK_SIZES_SPECIAL_KEYS:
            return key
        try:
            return int(key)
        except Exception:
            return key
    return key


def _to_container(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _to_container(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_to_container(item) for item in value]
    return value

