from __future__ import annotations

import copy

import pytest

from auto_quantize_model.modelopt_quant_overrides import apply_quant_cfg_overrides


def test_apply_quant_cfg_overrides_deep_copies_base() -> None:
    base = {
        "quant_cfg": {
            "*weight_quantizer": {"block_sizes": {-1: 128}, "enable": True},
            "*input_quantizer": {"axis": None, "enable": True},
        },
        "algorithm": "max",
    }
    base_snapshot = copy.deepcopy(base)

    effective = apply_quant_cfg_overrides(base, {"*weight_quantizer": {"axis": 0}})

    assert base == base_snapshot
    assert effective is not base
    assert effective["quant_cfg"]["*weight_quantizer"]["axis"] == 0
    assert "block_sizes" not in effective["quant_cfg"]["*weight_quantizer"]


def test_axis_override_removes_block_sizes() -> None:
    base = {
        "quant_cfg": {
            "*weight_quantizer": {"block_sizes": {-1: 128, "type": "static"}, "enable": True},
            "*input_quantizer": {"axis": None, "enable": True},
        },
        "algorithm": "max",
    }

    effective = apply_quant_cfg_overrides(base, {"*weight_quantizer": {"axis": 1}})
    attrs = effective["quant_cfg"]["*weight_quantizer"]

    assert attrs["axis"] == 1
    assert "block_sizes" not in attrs
    assert attrs["enable"] is True


def test_block_sizes_override_removes_axis_and_merges_block_sizes() -> None:
    base = {
        "quant_cfg": {
            "*weight_quantizer": {
                "axis": 0,
                "block_sizes": {"-1": 128, "type": "static", "scale_block_sizes": {"-1": 32}},
                "enable": True,
            },
        },
        "algorithm": "max",
    }

    effective = apply_quant_cfg_overrides(base, {"*weight_quantizer": {"block_sizes": {"-1": 64}}})
    attrs = effective["quant_cfg"]["*weight_quantizer"]

    assert "axis" not in attrs
    assert attrs["enable"] is True

    block_sizes = attrs["block_sizes"]
    assert block_sizes[-1] == 64
    assert block_sizes["type"] == "static"
    assert block_sizes["scale_block_sizes"][-1] == 32


def test_block_sizes_key_normalization_preserves_special_keys() -> None:
    base = {"quant_cfg": {"*input_quantizer": {}}, "algorithm": "max"}

    effective = apply_quant_cfg_overrides(
        base,
        {
            "*input_quantizer": {
                "block_sizes": {
                    "-1": None,
                    "-2": 128,
                    "type": "dynamic",
                    "scale_bits": (4, 3),
                    "scale_block_sizes": {"-1": 16},
                }
            }
        },
    )

    block_sizes = effective["quant_cfg"]["*input_quantizer"]["block_sizes"]
    assert block_sizes[-1] is None
    assert block_sizes[-2] == 128
    assert block_sizes["type"] == "dynamic"
    assert block_sizes["scale_bits"] == (4, 3)
    assert block_sizes["scale_block_sizes"][-1] == 16


def test_unknown_quant_cfg_override_key_raises() -> None:
    base = {"quant_cfg": {"*weight_quantizer": {}}, "algorithm": "max"}

    with pytest.raises(ValueError, match="Unsupported quant_cfg override key"):
        apply_quant_cfg_overrides(base, {"*unknown_quantizer": {"axis": 0}})


def test_override_setting_axis_and_block_sizes_raises() -> None:
    base = {"quant_cfg": {"*weight_quantizer": {}}, "algorithm": "max"}

    with pytest.raises(ValueError, match="cannot set both `axis` and `block_sizes`"):
        apply_quant_cfg_overrides(
            base,
            {"*weight_quantizer": {"axis": 0, "block_sizes": {-1: 128}}},
        )

