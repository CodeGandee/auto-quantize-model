"""
Shared tutorial-pack runner for Qwen3-VL layer sensitivity demos.

This module consolidates duplicated orchestration logic from the tutorial packs
under `docs/tutorial/howto/` into a single, testable runner:

- validates required model/dataset assets,
- enumerates scenarios as `modes × quant_pairs`,
- creates an isolated workspace under `tmp/`,
- generates schema-locked per-scenario summaries (`summary.json`, `summary.md`),
- supports snapshot mode (refresh expected summaries; remove stale scenarios),
- supports verify mode (diff summary-only; strict + fail-fast; non-degeneracy).

The heavy GPU workflows remain implemented elsewhere (Hydra runner / helper
scripts). This module focuses on deterministic orchestration and filesystem
contracts so it can be unit/integration tested on CPU.
"""

from __future__ import annotations

import difflib
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Optional, Sequence

from auto_quantize_model.qwen.tutorial_pack_registry import ModelConfig, build_model_registry
from auto_quantize_model.qwen.tutorial_pack_summary import (
    DatasetSize,
    Mode,
    TutorialPackScenarioSummary,
    write_summary_json,
    write_summary_md,
)


class TutorialPackRunnerError(RuntimeError):
    """Base error for tutorial pack runner failures."""


class InvalidModelIdError(TutorialPackRunnerError):
    """Raised when an unknown model-id is requested."""


class InvalidScenarioSelectionError(TutorialPackRunnerError):
    """Raised when modes or quant-pairs are invalid."""


class MissingAssetError(TutorialPackRunnerError):
    """Raised when required model or dataset assets are missing."""


class ExpectedSnapshotMissingError(TutorialPackRunnerError):
    """Raised when verify mode is requested but expected snapshots are missing."""


class VerificationMismatchError(TutorialPackRunnerError):
    """Raised when verify mode detects a diff against expected snapshots."""


class DegenerateSensitivityError(TutorialPackRunnerError):
    """Raised when summaries report all-zero sensitivities."""

@dataclass(frozen=True)
class DatasetPreset:
    """Resolved dataset preset (budgets + required asset paths)."""

    dataset_size: DatasetSize
    calib_seq_len: int
    batch_size: int
    num_calib_batches: int
    max_calib_samples: int
    auto_quantize_score_size: int
    coco_root: Path
    vlm_calib_db: Path
    captions_txt: Path

    @property
    def num_calib_samples(self) -> int:
        """Return the effective number of calibration samples."""

        return min(int(self.max_calib_samples), int(self.batch_size) * int(self.num_calib_batches))


@dataclass(frozen=True)
class ScenarioSpec:
    """One selected scenario."""

    mode: Mode
    quant_pair: str

    @property
    def scenario_id(self) -> str:
        """Return the stable scenario identifier."""

        return f"{self.mode}/{self.quant_pair}"


@dataclass(frozen=True)
class TutorialPackRunRequest:
    """Runner request contract (CLI-friendly, deterministic fields)."""

    model_id: str
    expected_report_dir: Path
    snapshot_report: bool
    device: str
    dataset_size: DatasetSize
    modes: tuple[Mode, ...]
    quant_pairs: tuple[str, ...]


@dataclass(frozen=True)
class TutorialPackRunResult:
    """Runner result for successful runs."""

    status: Literal["ok"]
    workspace_dir: Path
    scenarios: tuple[ScenarioSpec, ...]


def _discover_repo_root(start: Optional[Path] = None) -> Path:
    cursor = (start or Path.cwd()).resolve()
    for parent in [cursor, *cursor.parents]:
        if (parent / "pyproject.toml").is_file() and (parent / "src" / "auto_quantize_model").is_dir():
            return parent
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("Could not discover repo root. Run from the repository root or pass an explicit path.")


def _split_csv(value: str) -> list[str]:
    items = [part.strip() for part in str(value).split(",")]
    return [item for item in items if item]


def parse_modes_csv(value: str) -> list[Mode]:
    """Parse comma-separated modes and validate allowed values."""

    raw = _split_csv(value)
    if not raw:
        raise InvalidScenarioSelectionError("`--modes` must contain at least one mode.")

    allowed: set[str] = {"all_layers", "lm_only"}
    modes: list[Mode] = []
    for item in raw:
        if item not in allowed:
            raise InvalidScenarioSelectionError(f"Invalid mode: {item!r}. Allowed: {sorted(allowed)}")
        if item not in modes:
            modes.append(item)  # type: ignore[arg-type]
    return modes


def parse_quant_pairs_csv(value: str) -> list[str]:
    """Parse comma-separated quant-pairs and ensure non-empty + unique."""

    raw = _split_csv(value)
    if not raw:
        raise InvalidScenarioSelectionError("`--quant-pairs` must contain at least one quant-pair.")
    seen: set[str] = set()
    out: list[str] = []
    for item in raw:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def enumerate_scenarios(modes: Sequence[Mode], quant_pairs: Sequence[str]) -> list[ScenarioSpec]:
    """Enumerate scenarios as the cartesian product of modes × quant-pairs."""

    scenarios: list[ScenarioSpec] = []
    for mode in modes:
        for quant_pair in quant_pairs:
            scenarios.append(ScenarioSpec(mode=mode, quant_pair=str(quant_pair)))
    return scenarios


def resolve_dataset_preset(repo_root: Path, dataset_size: DatasetSize) -> DatasetPreset:
    """Resolve dataset preset budgets and required asset paths."""

    batch_size = 8
    if dataset_size == "small":
        max_calib_samples = 16
        num_calib_batches = 2
    elif dataset_size == "medium":
        max_calib_samples = 128
        num_calib_batches = 16
    elif dataset_size == "large":
        max_calib_samples = 512
        num_calib_batches = 64
    else:
        raise InvalidScenarioSelectionError(
            f"Unknown dataset_size: {dataset_size!r} (expected small|medium|large)."
        )

    calib_seq_len = 512
    score_size = 128

    coco_root = repo_root / "datasets" / "coco2017" / "source-data"
    vlm_db = repo_root / "datasets" / "vlm-quantize-calib" / f"coco2017_vlm_calib_{dataset_size}.db"
    captions_txt = repo_root / "datasets" / "vlm-quantize-calib" / f"coco2017_captions_{dataset_size}.txt"

    return DatasetPreset(
        dataset_size=dataset_size,
        calib_seq_len=calib_seq_len,
        batch_size=batch_size,
        num_calib_batches=num_calib_batches,
        max_calib_samples=max_calib_samples,
        auto_quantize_score_size=score_size,
        coco_root=coco_root,
        vlm_calib_db=vlm_db,
        captions_txt=captions_txt,
    )


def _validate_model_id(model_id: str, registry: Mapping[str, ModelConfig]) -> ModelConfig:
    model = registry.get(str(model_id))
    if model is None:
        allowed = ", ".join(sorted(registry.keys()))
        raise InvalidModelIdError(f"Unknown --model-id: {model_id!r}. Allowed: {allowed}")
    return model


def _validate_quant_pairs(selected: Sequence[str], allowed: Sequence[str]) -> None:
    allowed_set = set(allowed)
    invalid = [pair for pair in selected if pair not in allowed_set]
    if invalid:
        raise InvalidScenarioSelectionError(
            f"Invalid quant-pair(s): {invalid}. Allowed: {sorted(allowed_set)}"
        )


def _validate_assets(model: ModelConfig, dataset: DatasetPreset, device: str) -> None:
    missing: list[str] = []

    if not model.checkpoint_dir.is_dir():
        missing.append(str(model.checkpoint_dir))

    if not dataset.coco_root.is_dir():
        missing.append(str(dataset.coco_root))
    if not dataset.vlm_calib_db.is_file():
        missing.append(str(dataset.vlm_calib_db))
    if not dataset.captions_txt.is_file():
        missing.append(str(dataset.captions_txt))

    if missing:
        hint = (
            "Missing required assets. Create the checkpoint link using the model bootstrap script "
            f"({model.bootstrap_script}) and ensure dataset assets exist as documented."
        )
        raise MissingAssetError(f"{hint}\nMissing:\n- " + "\n- ".join(missing))

    if str(device).startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise MissingAssetError("CUDA is not available but a CUDA device was requested.")


def _timestamp_slug(now: Optional[datetime] = None) -> str:
    dt = now or datetime.now()
    return dt.strftime("%Y%m%d_%H%M%S_%f")


def create_workspace_dir(repo_root: Path, workspace_slug: str, *, now: Optional[datetime] = None) -> Path:
    """Create and return a new isolated workspace directory under `tmp/`."""

    root = repo_root / "tmp"
    root.mkdir(parents=True, exist_ok=True)
    workspace = root / f"tutorial_workspace_{workspace_slug}_{_timestamp_slug(now)}"
    workspace.mkdir(parents=True, exist_ok=False)
    (workspace / "outputs").mkdir(parents=True, exist_ok=True)
    (workspace / "summaries").mkdir(parents=True, exist_ok=True)
    return workspace


def resolve_output_dir(workspace_dir: Path, scenario: ScenarioSpec) -> Path:
    return workspace_dir / "outputs" / scenario.mode / scenario.quant_pair


def resolve_summary_dir(workspace_dir: Path, scenario: ScenarioSpec) -> Path:
    return workspace_dir / "summaries" / scenario.mode / scenario.quant_pair


def resolve_expected_case_dir(expected_report_dir: Path, scenario: ScenarioSpec) -> Path:
    return expected_report_dir / scenario.mode / scenario.quant_pair


def find_quant_manifest(output_dir: Path) -> Path:
    """Find `*_quant_manifest.json` under `output_dir`."""

    manifests = sorted(output_dir.glob("*_quant_manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"No *_quant_manifest.json found under: {output_dir}")
    return manifests[0]


def build_and_write_summary(
    manifest_json: Path,
    summary_dir: Path,
    *,
    scenario: ScenarioSpec,
    dataset_size: DatasetSize,
) -> TutorialPackScenarioSummary:
    """Generate `summary.json` + `summary.md` from a scenario manifest."""

    payload = json.loads(manifest_json.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"Manifest JSON must be an object: {manifest_json}")

    summary = TutorialPackScenarioSummary.from_manifest(
        payload,
        scenario_id=scenario.scenario_id,
        mode=scenario.mode,
        quant_pair=scenario.quant_pair,
        dataset_size=dataset_size,
    )
    write_summary_json(summary_dir / "summary.json", summary)
    write_summary_md(summary_dir / "summary.md", summary)
    return summary


def _load_summary_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"Summary JSON must be an object: {path}")
    return payload


def _assert_non_degenerate(summary_json: Path) -> None:
    payload = _load_summary_json(summary_json)
    if payload.get("has_nonzero_sensitivity") is True:
        return
    raise DegenerateSensitivityError(
        "Detected degenerate sensitivity (all sensitivities are exactly 0.0). "
        "Try increasing calibration budget (e.g., --dataset-size medium) and "
        "see the tutorial README for remediation guidance."
    )


def snapshot_scenario(expected_report_dir: Path, scenario: ScenarioSpec, summary_dir: Path) -> None:
    """Refresh the expected snapshot for a single scenario (summary-only)."""

    expected_case_dir = resolve_expected_case_dir(expected_report_dir, scenario)
    expected_case_dir.parent.mkdir(parents=True, exist_ok=True)
    if expected_case_dir.exists():
        if expected_case_dir.is_dir():
            shutil.rmtree(expected_case_dir)
        else:
            expected_case_dir.unlink()
    expected_case_dir.mkdir(parents=True, exist_ok=True)

    src_json = summary_dir / "summary.json"
    src_md = summary_dir / "summary.md"
    if not src_json.is_file() or not src_md.is_file():
        raise FileNotFoundError(f"Missing summary artifacts under: {summary_dir}")

    (expected_case_dir / "summary.json").write_text(src_json.read_text(encoding="utf-8"), encoding="utf-8")
    (expected_case_dir / "summary.md").write_text(src_md.read_text(encoding="utf-8"), encoding="utf-8")


def _diff_text(expected: str, actual: str, *, fromfile: str, tofile: str) -> str:
    diff = difflib.unified_diff(
        expected.splitlines(keepends=True),
        actual.splitlines(keepends=True),
        fromfile=fromfile,
        tofile=tofile,
    )
    return "".join(diff)


def verify_scenario(expected_report_dir: Path, scenario: ScenarioSpec, summary_dir: Path) -> None:
    """Verify a single scenario by diffing summary-only expected snapshots."""

    expected_case_dir = resolve_expected_case_dir(expected_report_dir, scenario)
    if not expected_case_dir.is_dir():
        raise ExpectedSnapshotMissingError(
            f"Expected snapshot case dir is missing: {expected_case_dir}\n"
            "Run with --snapshot-report to create/refresh expected_report/."
        )

    expected_json = expected_case_dir / "summary.json"
    expected_md = expected_case_dir / "summary.md"
    if not expected_json.is_file() or not expected_md.is_file():
        raise ExpectedSnapshotMissingError(
            f"Expected snapshot is incomplete under: {expected_case_dir}\n"
            "Expected: summary.json + summary.md. Run with --snapshot-report to refresh."
        )

    actual_json = summary_dir / "summary.json"
    actual_md = summary_dir / "summary.md"
    if not actual_json.is_file() or not actual_md.is_file():
        raise FileNotFoundError(f"Missing actual summary artifacts under: {summary_dir}")

    _assert_non_degenerate(actual_json)

    json_diff = _diff_text(
        expected_json.read_text(encoding="utf-8"),
        actual_json.read_text(encoding="utf-8"),
        fromfile=str(expected_json),
        tofile=str(actual_json),
    )
    if json_diff:
        raise VerificationMismatchError(f"Verification failed for {scenario.scenario_id}: summary.json differs.\n{json_diff}")

    md_diff = _diff_text(
        expected_md.read_text(encoding="utf-8"),
        actual_md.read_text(encoding="utf-8"),
        fromfile=str(expected_md),
        tofile=str(actual_md),
    )
    if md_diff:
        raise VerificationMismatchError(f"Verification failed for {scenario.scenario_id}: summary.md differs.\n{md_diff}")


def cleanup_expected_report_dir(expected_report_dir: Path, selected: Iterable[ScenarioSpec]) -> None:
    """Remove stale expected scenarios not in the selected set (snapshot mode)."""

    selected_modes: set[str] = {scenario.mode for scenario in selected}
    selected_pairs: set[str] = {scenario.quant_pair for scenario in selected}

    if not expected_report_dir.is_dir():
        return

    for mode_dir in expected_report_dir.iterdir():
        if not mode_dir.is_dir():
            mode_dir.unlink()
            continue
        mode = mode_dir.name
        if mode not in {"all_layers", "lm_only"}:
            shutil.rmtree(mode_dir)
            continue
        if mode not in selected_modes:
            shutil.rmtree(mode_dir)
            continue

        for pair_dir in mode_dir.iterdir():
            if not pair_dir.is_dir():
                pair_dir.unlink()
                continue
            if pair_dir.name not in selected_pairs:
                shutil.rmtree(pair_dir)


def _run_subprocess(argv: Sequence[str], *, cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            list(argv),
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log_file.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {return_code}: {' '.join(argv)}")


def run_scenario(
    *,
    repo_root: Path,
    model: ModelConfig,
    dataset: DatasetPreset,
    scenario: ScenarioSpec,
    device: str,
    output_dir: Path,
) -> None:
    """Execute one scenario (heavy GPU work)."""

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"

    if scenario.mode == "all_layers":
        argv = [
            sys.executable,
            str(model.all_layers_script),
            "--model-dir",
            str(model.checkpoint_dir),
            "--output-dir",
            str(output_dir),
            "--quant-pair",
            scenario.quant_pair,
            "--dataset-size",
            str(dataset.dataset_size),
            "--vlm-calib-db",
            str(dataset.vlm_calib_db),
            "--coco-root",
            str(dataset.coco_root),
            "--max-calib-samples",
            str(dataset.max_calib_samples),
            "--num-calib-batches",
            str(dataset.num_calib_batches),
            "--calib-seq-len",
            str(dataset.calib_seq_len),
            "--batch-size",
            str(dataset.batch_size),
            "--device",
            str(device),
            "--auto-quantize-score-size",
            str(dataset.auto_quantize_score_size),
        ]
        _run_subprocess(argv, cwd=repo_root, log_path=log_path)
        return

    if scenario.mode == "lm_only":
        argv = [
            sys.executable,
            str(model.lm_only_script),
            f"model.path={model.checkpoint_dir}",
            f"model.name={model.lm_only_model_name}",
            f"model.variant={model.lm_only_model_variant}",
            f"dataset.size={dataset.dataset_size}",
            f"autoquant.device={device}",
            f"autoquant.batch_size={dataset.batch_size}",
            f"autoquant.score_size={dataset.auto_quantize_score_size}",
            f"quant_pair={scenario.quant_pair}",
            f"runner.output_dir={output_dir}",
            f"hydra.run.dir={output_dir}",
            "hydra.job.chdir=false",
        ]
        _run_subprocess(argv, cwd=repo_root, log_path=log_path)
        return

    raise InvalidScenarioSelectionError(f"Unknown mode: {scenario.mode!r}")


def run_tutorial_pack(request: TutorialPackRunRequest, *, repo_root: Optional[Path] = None) -> TutorialPackRunResult:
    """Run tutorial pack scenarios in snapshot or verify mode (fail-fast)."""

    resolved_repo_root = _discover_repo_root(repo_root)
    registry = build_model_registry(resolved_repo_root)
    model = _validate_model_id(request.model_id, registry)

    dataset = resolve_dataset_preset(resolved_repo_root, request.dataset_size)
    _validate_quant_pairs(request.quant_pairs, model.allowed_quant_pairs)

    scenarios = enumerate_scenarios(request.modes, request.quant_pairs)
    _validate_assets(model, dataset, request.device)

    workspace_dir = create_workspace_dir(resolved_repo_root, model.workspace_slug)
    for scenario in scenarios:
        output_dir = resolve_output_dir(workspace_dir, scenario)
        summary_dir = resolve_summary_dir(workspace_dir, scenario)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_dir.mkdir(parents=True, exist_ok=True)

        run_scenario(
            repo_root=resolved_repo_root,
            model=model,
            dataset=dataset,
            scenario=scenario,
            device=request.device,
            output_dir=output_dir,
        )
        manifest_json = find_quant_manifest(output_dir)
        _ = build_and_write_summary(
            manifest_json,
            summary_dir,
            scenario=scenario,
            dataset_size=request.dataset_size,
        )
        if request.snapshot_report:
            _assert_non_degenerate(summary_dir / "summary.json")
            snapshot_scenario(request.expected_report_dir, scenario, summary_dir)
        else:
            verify_scenario(request.expected_report_dir, scenario, summary_dir)

    if request.snapshot_report:
        cleanup_expected_report_dir(request.expected_report_dir, scenarios)

    return TutorialPackRunResult(
        status="ok",
        workspace_dir=workspace_dir,
        scenarios=tuple(scenarios),
    )
