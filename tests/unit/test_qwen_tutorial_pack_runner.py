from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from auto_quantize_model.qwen.cli_tutorial_pack_runner import build_arg_parser
from auto_quantize_model.qwen.tutorial_pack_runner import (
    DegenerateSensitivityError,
    ExpectedSnapshotMissingError,
    InvalidModelIdError,
    InvalidScenarioSelectionError,
    ScenarioSpec,
    TutorialPackRunRequest,
    create_workspace_dir,
    enumerate_scenarios,
    parse_modes_csv,
    parse_quant_pairs_csv,
    resolve_dataset_preset,
    resolve_expected_case_dir,
    resolve_output_dir,
    run_tutorial_pack,
    verify_scenario,
)


def _fixture(path: str) -> Path:
    return Path(__file__).parent / "fixtures" / "qwen_tutorial_pack_runner" / path


def test_cli_arg_parser_defaults() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--model-id",
            "qwen3_vl_4b_instruct",
            "--expected-report-dir",
            "/tmp/expected_report",
        ]
    )
    assert args.snapshot_report is False
    assert args.device == "cuda:0"
    assert args.dataset_size == "medium"
    assert args.modes == "all_layers,lm_only"
    assert args.quant_pairs == "wint4_afp16,wint4_aint8"


def test_parse_modes_csv_validates_and_dedupes() -> None:
    assert parse_modes_csv("all_layers,lm_only") == ["all_layers", "lm_only"]
    assert parse_modes_csv("lm_only, lm_only") == ["lm_only"]
    with pytest.raises(InvalidScenarioSelectionError):
        parse_modes_csv("")
    with pytest.raises(InvalidScenarioSelectionError):
        parse_modes_csv("unknown")


def test_parse_quant_pairs_csv_validates_and_dedupes() -> None:
    assert parse_quant_pairs_csv("wint4_afp16,wint4_aint8") == ["wint4_afp16", "wint4_aint8"]
    assert parse_quant_pairs_csv("wint4_afp16, wint4_afp16") == ["wint4_afp16"]
    with pytest.raises(InvalidScenarioSelectionError):
        parse_quant_pairs_csv("")


def test_enumerate_scenarios_is_cartesian_product() -> None:
    scenarios = enumerate_scenarios(["all_layers", "lm_only"], ["a", "b"])
    assert [scenario.scenario_id for scenario in scenarios] == [
        "all_layers/a",
        "all_layers/b",
        "lm_only/a",
        "lm_only/b",
    ]


def test_resolve_dataset_preset_budgets_and_paths() -> None:
    repo_root = Path("/repo-root")
    preset = resolve_dataset_preset(repo_root, "medium")
    assert preset.dataset_size == "medium"
    assert preset.batch_size == 8
    assert preset.num_calib_batches == 16
    assert preset.max_calib_samples == 128
    assert preset.calib_seq_len == 512
    assert preset.auto_quantize_score_size == 128
    assert preset.coco_root == repo_root / "datasets" / "coco2017" / "source-data"
    assert preset.vlm_calib_db == repo_root / "datasets" / "vlm-quantize-calib" / "coco2017_vlm_calib_medium.db"
    assert preset.captions_txt == repo_root / "datasets" / "vlm-quantize-calib" / "coco2017_captions_medium.txt"
    assert preset.num_calib_samples == 128


def test_workspace_and_layout_helpers(tmp_path: Path) -> None:
    repo_root = tmp_path
    workspace = create_workspace_dir(repo_root, "slug", now=datetime(2026, 1, 1, 0, 0, 0, 123456))
    assert workspace.parent == repo_root / "tmp"
    assert (workspace / "outputs").is_dir()
    assert not (workspace / "summaries").exists()

    scenario = ScenarioSpec(mode="all_layers", quant_pair="wint4_afp16")
    assert resolve_output_dir(workspace, scenario) == workspace / "outputs" / "all_layers" / "wint4_afp16"
    assert resolve_expected_case_dir(Path("/expected"), scenario) == Path("/expected") / "outputs" / "all_layers" / "wint4_afp16"


def test_run_tutorial_pack_rejects_unknown_model_id() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    request = TutorialPackRunRequest(
        model_id="unknown_model",
        expected_report_dir=Path("/tmp/expected_report"),
        snapshot_report=False,
        device="cpu",
        dataset_size="medium",
        modes=("all_layers",),
        quant_pairs=("wint4_afp16",),
    )
    with pytest.raises(InvalidModelIdError) as exc_info:
        run_tutorial_pack(request, repo_root=repo_root)
    message = str(exc_info.value)
    assert "qwen3_vl_4b_instruct" in message
    assert "qwen3_vl_8b_instruct" in message


def test_run_tutorial_pack_rejects_invalid_quant_pairs() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    request = TutorialPackRunRequest(
        model_id="qwen3_vl_4b_instruct",
        expected_report_dir=Path("/tmp/expected_report"),
        snapshot_report=False,
        device="cpu",
        dataset_size="medium",
        modes=("all_layers",),
        quant_pairs=("not_allowed",),
    )
    with pytest.raises(InvalidScenarioSelectionError):
        run_tutorial_pack(request, repo_root=repo_root)


def test_verify_fails_when_expected_snapshot_missing(tmp_path: Path) -> None:
    expected_report_dir = tmp_path / "expected_report"
    output_dir = tmp_path / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(_fixture("summary_ok.json").read_text(encoding="utf-8"), encoding="utf-8")

    scenario = ScenarioSpec(mode="all_layers", quant_pair="wint4_afp16")
    with pytest.raises(ExpectedSnapshotMissingError):
        verify_scenario(expected_report_dir, scenario, output_dir)


def test_verify_fails_when_expected_snapshot_incomplete(tmp_path: Path) -> None:
    expected_report_dir = tmp_path / "expected_report"
    scenario = ScenarioSpec(mode="all_layers", quant_pair="wint4_afp16")
    expected_case_dir = resolve_expected_case_dir(expected_report_dir, scenario)
    expected_case_dir.mkdir(parents=True, exist_ok=True)
    # Intentionally missing summary.json

    output_dir = tmp_path / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(_fixture("summary_ok.json").read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(ExpectedSnapshotMissingError):
        verify_scenario(expected_report_dir, scenario, output_dir)


def test_verify_ignores_non_summary_files(tmp_path: Path) -> None:
    expected_report_dir = tmp_path / "expected_report"
    scenario = ScenarioSpec(mode="all_layers", quant_pair="wint4_afp16")
    expected_case_dir = resolve_expected_case_dir(expected_report_dir, scenario)
    expected_case_dir.mkdir(parents=True, exist_ok=True)

    output_dir = tmp_path / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_json = _fixture("summary_ok.json").read_text(encoding="utf-8")

    (expected_case_dir / "summary.json").write_text(summary_json, encoding="utf-8")
    (expected_case_dir / "layer-sensitivity-report.md").write_text("ignored\n", encoding="utf-8")

    (output_dir / "summary.json").write_text(summary_json, encoding="utf-8")
    (output_dir / "other.txt").write_text("ignored\n", encoding="utf-8")

    verify_scenario(expected_report_dir, scenario, output_dir)


def test_verify_enforces_non_degeneracy_gate(tmp_path: Path) -> None:
    expected_report_dir = tmp_path / "expected_report"
    scenario = ScenarioSpec(mode="all_layers", quant_pair="wint4_afp16")
    expected_case_dir = resolve_expected_case_dir(expected_report_dir, scenario)
    expected_case_dir.mkdir(parents=True, exist_ok=True)

    output_dir = tmp_path / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    degenerate_json = _fixture("summary_degenerate.json").read_text(encoding="utf-8")
    (expected_case_dir / "summary.json").write_text(degenerate_json, encoding="utf-8")

    (output_dir / "summary.json").write_text(degenerate_json, encoding="utf-8")

    payload = json.loads(degenerate_json)
    assert payload["has_nonzero_sensitivity"] is False
    with pytest.raises(DegenerateSensitivityError):
        verify_scenario(expected_report_dir, scenario, output_dir)
