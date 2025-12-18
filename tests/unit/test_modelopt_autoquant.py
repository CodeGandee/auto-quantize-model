from __future__ import annotations

import json
from pathlib import Path

from auto_quantize_model.modelopt_autoquant import (
    AutoQuantSchemeConfig,
    compute_num_score_steps,
    write_layer_sensitivity_json,
    write_layer_sensitivity_md,
)


def test_compute_num_score_steps_clamps_to_num_batches() -> None:
    assert compute_num_score_steps(score_size=128, batch_size=8, num_batches=4) == 4


def test_compute_num_score_steps_minimum_is_one() -> None:
    assert compute_num_score_steps(score_size=1, batch_size=8, num_batches=10) == 1
    assert compute_num_score_steps(score_size=0, batch_size=8, num_batches=10) == 1
    assert compute_num_score_steps(score_size=128, batch_size=8, num_batches=0) == 1


def test_layer_sensitivity_report_includes_none_only_layers(tmp_path: Path) -> None:
    scheme = AutoQuantSchemeConfig(
        name="test_scheme",
        auto_quantize_bits=8.0,
        auto_quantize_method="gradient",
        auto_quantize_score_size=128,
        quant_formats=["CUSTOM_CFG"],
    )
    layer_sensitivity = {
        "layer0.quant_recipe": {
            "formats": ["NONE(effective-bits: 16.0)"],
            "scores": [0.0],
            "costs": [123.0],
        }
    }
    autoquant_state = {
        "constraints": {"effective_bits": 8.0},
        "score": 1.23,
        "is_satisfied": True,
    }

    quantization = {
        "base_format_name": "CUSTOM_CFG",
        "quant_granularity": {"name": "per_channel", "quant_cfg_overrides": {"*input_quantizer": {"axis": 1}}},
    }
    run_config = {"composed_yaml_path": "composed-config.yaml"}
    (tmp_path / "composed-config.yaml").write_text("foo: bar\n", encoding="utf-8")

    out_md = tmp_path / "layer-sensitivity-report.md"
    write_layer_sensitivity_md(
        layer_sensitivity=layer_sensitivity,
        scheme=scheme,
        autoquant_state=autoquant_state,
        out_path=out_md,
        model_id="dummy-model",
        quantization=quantization,
        run_config=run_config,
    )
    rendered = out_md.read_text(encoding="utf-8")
    assert "## Quantization" in rendered
    assert "Granularity" in rendered
    assert "composed-config.yaml" in rendered
    assert "```yaml" in rendered
    assert "foo: bar" in rendered
    assert "## Layer Sensitivity Table" in rendered
    assert "|layer0|16.0|0.000e+00|1.230e+02|" in rendered

    out_json = tmp_path / "layer-sensitivity-report.json"
    manifest = {
        "scheme": {},
        "model": {"id": "dummy-model"},
        "quantization": quantization,
        "run_config": run_config,
        "autoquant_state": autoquant_state,
        "layer_sensitivity": layer_sensitivity,
    }
    write_layer_sensitivity_json(manifest=manifest, out_path=out_json)
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["quantization"]["quant_granularity"]["name"] == "per_channel"
    assert payload["run_config"]["composed_yaml_path"] == "composed-config.yaml"
    assert payload["layer_sensitivity"] == [
        {"layer": "layer0", "num_bits": 16.0, "sensitivity": 0.0, "size_cost": 123.0}
    ]
