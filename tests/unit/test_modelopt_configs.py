from __future__ import annotations

import modelopt.torch.quantization as mtq  # type: ignore[import-untyped]

from auto_quantize_model import modelopt_configs


def test_custom_quant_configs_registry_contains_expected_entries() -> None:
    required = {
        "FP8_ALL_LAYERS_CFG",
        "FP8_WEIGHT_ONLY_CFG",
        "INT8_LM_DEFAULT_CFG",
        "INT8_ALL_LAYERS_CFG",
        "INT8_WEIGHT_ONLY_CFG",
        "INT8_WEIGHT_FP8_ACT_CFG",
    }
    assert required.issubset(modelopt_configs.CUSTOM_QUANT_CONFIGS.keys())

    if hasattr(mtq, "NVFP4_DEFAULT_CFG"):
        assert "NVFP4_WEIGHT_ONLY_CFG" in modelopt_configs.CUSTOM_QUANT_CONFIGS
    else:
        assert "NVFP4_WEIGHT_ONLY_CFG" not in modelopt_configs.CUSTOM_QUANT_CONFIGS

    if hasattr(mtq, "MXFP4_DEFAULT_CFG"):
        assert "MXFP4_WEIGHT_ONLY_CFG" in modelopt_configs.CUSTOM_QUANT_CONFIGS
    else:
        assert "MXFP4_WEIGHT_ONLY_CFG" not in modelopt_configs.CUSTOM_QUANT_CONFIGS


def _assert_input_quantizer_disabled(cfg: dict) -> None:
    quant_cfg = cfg.get("quant_cfg") or {}
    input_cfg = quant_cfg.get("*input_quantizer") or {}
    assert input_cfg.get("enable") is False


def test_weight_only_configs_disable_input_quantizer() -> None:
    _assert_input_quantizer_disabled(modelopt_configs.FP8_WEIGHT_ONLY_CFG)
    _assert_input_quantizer_disabled(modelopt_configs.INT8_WEIGHT_ONLY_CFG)

    if modelopt_configs.NVFP4_WEIGHT_ONLY_CFG is not None:
        _assert_input_quantizer_disabled(modelopt_configs.NVFP4_WEIGHT_ONLY_CFG)
    if modelopt_configs.MXFP4_WEIGHT_ONLY_CFG is not None:
        _assert_input_quantizer_disabled(modelopt_configs.MXFP4_WEIGHT_ONLY_CFG)


def test_int8_weight_fp8_act_cfg_mixes_quantizer_types() -> None:
    cfg = modelopt_configs.INT8_WEIGHT_FP8_ACT_CFG
    quant_cfg = cfg.get("quant_cfg") or {}
    weight_cfg = quant_cfg.get("*weight_quantizer") or {}
    input_cfg = quant_cfg.get("*input_quantizer") or {}

    assert weight_cfg.get("num_bits") == 8
    assert input_cfg.get("num_bits") == (4, 3)


def test_resolve_quant_config_handles_custom_and_builtin_names() -> None:
    cfg_custom = modelopt_configs.resolve_quant_config("FP8_WEIGHT_ONLY_CFG")
    assert cfg_custom is modelopt_configs.CUSTOM_QUANT_CONFIGS["FP8_WEIGHT_ONLY_CFG"]

    cfg_builtin = modelopt_configs.resolve_quant_config("FP8_DEFAULT_CFG")
    assert isinstance(cfg_builtin, dict)
    assert cfg_builtin.get("quant_cfg") == mtq.FP8_DEFAULT_CFG.get("quant_cfg")
