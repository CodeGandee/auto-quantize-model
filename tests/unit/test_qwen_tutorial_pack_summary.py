from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, cast

import pytest

from auto_quantize_model.qwen.tutorial_pack_summary import (
    TutorialPackScenarioSummary,
    write_summary_json,
    write_summary_md,
)


def _base_manifest() -> dict[str, object]:
    return {
        "scheme": {
            "name": "wint4_afp16_autoquant_all_layers",
            "auto_quantize_score_size": 128,
            "quant_formats": ["INT4_BLOCKWISE_WEIGHT_ONLY_CFG"],
        },
        "dataset": {
            "size": "medium",
            "calib_seq_len": 512,
            "batch_size": 8,
            "num_calib_batches": 16,
            "num_calib_samples": 128,
            "max_calib_samples": 128,
        },
        "autoquant_state": {"keys": ["best"]},
    }


def test_summary_detects_nonzero_sensitivity_from_rows() -> None:
    manifest: dict[str, object] = _base_manifest()
    manifest["layer_sensitivity"] = [
        {"layer": "a", "sensitivity": 0.0, "size_cost": 1.0},
        {"layer": "b", "sensitivity": 1e-9, "size_cost": 1.0},
    ]
    summary = TutorialPackScenarioSummary.from_manifest(
        manifest,
        scenario_id="lm_only/wint4_afp16",
        mode="lm_only",
        quant_pair="wint4_afp16",
        dataset_size="medium",
    )
    assert summary.has_nonzero_sensitivity is True


def test_summary_detects_all_zero_sensitivity_from_rows() -> None:
    manifest: dict[str, object] = _base_manifest()
    manifest["layer_sensitivity"] = [
        {"layer": "a", "sensitivity": 0.0, "size_cost": 1.0},
        {"layer": "b", "sensitivity": 0.0, "size_cost": 1.0},
    ]
    summary = TutorialPackScenarioSummary.from_manifest(
        manifest,
        scenario_id="lm_only/wint4_aint8",
        mode="lm_only",
        quant_pair="wint4_aint8",
        dataset_size="medium",
    )
    assert summary.has_nonzero_sensitivity is False


def test_summary_detects_nonzero_sensitivity_from_candidate_stats_dict() -> None:
    manifest: dict[str, object] = _base_manifest()
    manifest["layer_sensitivity"] = {
        "layer.a": {
            "formats": ["NONE(fp16)", "INT4"],
            "scores": [0.0, 0.0],
            "costs": [1.0, 1.0],
        },
        "layer.b": {
            "formats": ["INT4"],
            "scores": [0.1],
            "costs": [1.0],
        },
    }
    summary = TutorialPackScenarioSummary.from_manifest(
        manifest,
        scenario_id="all_layers/wint4_afp16",
        mode="all_layers",
        quant_pair="wint4_afp16",
        dataset_size="medium",
    )
    assert summary.has_nonzero_sensitivity is True


def test_summary_requires_dataset_metadata() -> None:
    manifest: dict[str, object] = _base_manifest()
    dataset = dict(cast(Mapping[str, Any], manifest["dataset"]))
    dataset.pop("num_calib_samples")
    manifest["dataset"] = dataset

    with pytest.raises(ValueError, match="num_calib_samples"):
        _ = TutorialPackScenarioSummary.from_manifest(
            manifest,
            scenario_id="all_layers/wint4_afp16",
            mode="all_layers",
            quant_pair="wint4_afp16",
            dataset_size="medium",
        )


def test_write_summary_files_are_deterministic(tmp_path: Path) -> None:
    manifest: dict[str, object] = _base_manifest()
    manifest["layer_sensitivity"] = [{"layer": "a", "sensitivity": 0.0, "size_cost": 1.0}]
    summary = TutorialPackScenarioSummary.from_manifest(
        manifest,
        scenario_id="lm_only/wint4_afp16",
        mode="lm_only",
        quant_pair="wint4_afp16",
        dataset_size="medium",
    )

    json_path = tmp_path / "summary.json"
    md_path = tmp_path / "summary.md"
    write_summary_json(json_path, summary)
    write_summary_md(md_path, summary)

    parsed = json.loads(json_path.read_text(encoding="utf-8"))
    assert parsed["scenario_id"] == "lm_only/wint4_afp16"
    assert md_path.read_text(encoding="utf-8").startswith("# Tutorial Pack Scenario Summary")
