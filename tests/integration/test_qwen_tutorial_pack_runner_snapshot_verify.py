from __future__ import annotations

from pathlib import Path

from auto_quantize_model.qwen.tutorial_pack_runner import (
    ScenarioSpec,
    cleanup_expected_report_dir,
    resolve_expected_case_dir,
    snapshot_scenario,
)


def _write_summary_pair(dir_path: Path, *, suffix: str) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "summary.json").write_text(f'{{"has_nonzero_sensitivity": true, "suffix": "{suffix}"}}\n', encoding="utf-8")
    (dir_path / "summary.md").write_text(f"# summary {suffix}\n", encoding="utf-8")


def test_snapshot_cleanup_removes_stale_scenarios_and_files(tmp_path: Path) -> None:
    expected_report_dir = tmp_path / "expected_report"

    stale = ScenarioSpec(mode="all_layers", quant_pair="wint4_aint8")
    stale_dir = resolve_expected_case_dir(expected_report_dir, stale)
    _write_summary_pair(stale_dir, suffix="stale")
    (stale_dir / "layer-sensitivity-report.md").write_text("old\n", encoding="utf-8")

    other_mode = ScenarioSpec(mode="lm_only", quant_pair="wint4_afp16")
    other_mode_dir = resolve_expected_case_dir(expected_report_dir, other_mode)
    _write_summary_pair(other_mode_dir, suffix="other-mode")

    (expected_report_dir / "junk.txt").write_text("junk\n", encoding="utf-8")
    (expected_report_dir / "unknown_dir").mkdir(parents=True, exist_ok=True)

    selected = ScenarioSpec(mode="all_layers", quant_pair="wint4_afp16")
    summary_dir = tmp_path / "summary_out"
    _write_summary_pair(summary_dir, suffix="selected")

    snapshot_scenario(expected_report_dir, selected, summary_dir)
    cleanup_expected_report_dir(expected_report_dir, [selected])

    selected_case_dir = resolve_expected_case_dir(expected_report_dir, selected)
    assert selected_case_dir.is_dir()
    assert sorted(child.name for child in selected_case_dir.iterdir()) == ["summary.json", "summary.md"]

    assert not resolve_expected_case_dir(expected_report_dir, stale).exists()
    assert not resolve_expected_case_dir(expected_report_dir, other_mode).exists()
    assert not (expected_report_dir / "junk.txt").exists()
    assert not (expected_report_dir / "unknown_dir").exists()

