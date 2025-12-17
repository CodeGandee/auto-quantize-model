from __future__ import annotations

from pathlib import Path

from auto_quantize_model.experiment_layout import (
    resolve_lm_run_dir,
    resolve_pair_dir,
    resolve_publish_output_dir,
    resolve_scheme_name,
)


def test_resolve_scheme_name_default_and_override() -> None:
    assert resolve_scheme_name("wfp8_afp8") == "wfp8_afp8_autoquant_lm"
    assert resolve_scheme_name("wfp8_afp8", scheme_name_override="custom") == "custom"


def test_resolve_pair_and_run_dir_defaults() -> None:
    assert resolve_pair_dir("fp8", "fp16") == "weight-fp8-act-fp16"
    assert (
        resolve_lm_run_dir(
            model_name="qwen3_vl_4b_instruct",
            quant_pair_name="wfp8_afp8",
            dataset_size="medium",
        )
        == "qwen3_vl_4b_instruct_autoquant_wfp8_afp8_lm_medium"
    )


def test_resolve_publish_output_dir_defaults() -> None:
    root = Path("/tmp/publish-root")
    out_dir = resolve_publish_output_dir(
        root,
        weight="fp8",
        activation="fp8",
        model_name="qwen3_vl_4b_instruct",
        quant_pair_name="wfp8_afp8",
        quant_granularity_name="default",
        dataset_size="medium",
    )
    assert out_dir == (
        root / "weight-fp8-act-fp8" / "default" / "qwen3_vl_4b_instruct_autoquant_wfp8_afp8_lm_medium"
    )


def test_resolve_publish_output_dir_overrides() -> None:
    root = Path("/tmp/publish-root")
    out_dir = resolve_publish_output_dir(
        root,
        weight="int8",
        activation="int8",
        model_name="qwen3_vl_4b_instruct",
        quant_pair_name="wint8_aint8",
        quant_granularity_name="recipe_match_channel_token",
        dataset_size="small",
        run_dir_override="qwen3_vl_4b_autoquant_int8_lm_small",
    )
    assert out_dir == (
        root / "weight-int8-act-int8" / "recipe_match_channel_token" / "qwen3_vl_4b_autoquant_int8_lm_small"
    )

    out_dir = resolve_publish_output_dir(
        root,
        weight="fp8",
        activation="fp8",
        model_name="unused",
        quant_pair_name="unused",
        quant_granularity_name="default",
        dataset_size="unused",
        pair_dir_override="custom-pair",
        run_dir_override="custom-run",
    )
    assert out_dir == root / "custom-pair" / "default" / "custom-run"
