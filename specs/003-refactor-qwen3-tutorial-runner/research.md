# Research: Refactor Qwen3-VL Tutorial Pack Runner (Shared Utility)

**Branch**: `003-refactor-qwen3-tutorial-runner`  
**Date**: 2026-01-22  
**Spec**: `/data1/huangzhe/code/auto-quantize-model/specs/003-refactor-qwen3-tutorial-runner/spec.md`

## Context

This feature consolidates duplicated Qwen3-VL tutorial-pack runner logic into a shared runner utility under:

- `/data1/huangzhe/code/auto-quantize-model/src/auto_quantize_model/qwen/`

The current duplication sources are the tutorial packs:

- `/data1/huangzhe/code/auto-quantize-model/docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/`
- `/data1/huangzhe/code/auto-quantize-model/docs/tutorial/howto/tut-qwen3-vl-4b-layer-sensitivity/`

Key constraints (from the feature spec):

- Preserve the existing tutorial flag contract: `--snapshot-report`, `--device`, `--dataset-size`, `--modes`, `--quant-pairs`.
- Verification is strict:
  - Fail if `expected_report/` is missing or incomplete.
  - Diff only per-scenario `summary.json`.
  - Enforce non-degeneracy (at least one non-zero sensitivity value).
- Execution is fail-fast: stop on the first failing scenario.
- Do not auto-create model checkpoint links; fail with instructions instead.
- Snapshot mode writes only per-scenario `summary.json` to `expected_report/`.

## Decisions

### D1: Shared runner lives as library-first code with a stable CLI frontend

- **Decision**: Implement orchestration as importable modules under `/data1/huangzhe/code/auto-quantize-model/src/auto_quantize_model/qwen/` with a stable CLI entrypoint, and reduce each tutorial pack’s `run_demo.sh` to a thin wrapper that delegates to the shared CLI.
- **Rationale**: Moves the “public contract” logic (scenario enumeration, snapshot/verify, layout) into testable code while keeping the user entrypoint (`run_demo.sh`) stable.
- **Alternatives considered**:
  - Keep duplicated bash runners: rejected due to drift risk and poor unit-testability.
  - Keep per-pack Python glue scripts: rejected because it preserves duplication and makes future fixes require multi-pack edits.

### D2: Scenario identity and layout remain stable and minimal

- **Decision**: Use stable `scenario_id = "{mode}/{quant_pair}"` and keep expected snapshot layout:
  - `expected_report/<mode>/<quant_pair>/summary.json`
- **Rationale**: Matches current tutorial semantics and keeps the verification surface small and stable across machines.
- **Alternatives considered**:
  - Include dataset preset and device in the scenario ID: rejected because it expands snapshot churn and makes “golden” outputs less reusable.

### D3: Snapshot/verify semantics are summary-only and strict

- **Decision**:
  - Snapshot mode refreshes only `summary.json` per scenario and removes stale scenarios not selected.
  - Verify mode fails when expected snapshots are missing/incomplete and diffs only `summary.json`.
  - Verify enforces non-degeneracy via `has_nonzero_sensitivity`.
  - All multi-scenario runs stop at the first failure (fail-fast).
- **Rationale**: Produces a deterministic, reviewable contract and ensures “verification complete” means a real regression gate.
- **Alternatives considered**:
  - Warn and skip missing expected snapshots: rejected because it can hide regressions and creates false positives.
  - Snapshot additional sanitized artifacts: rejected per clarified spec to minimize snapshot size and churn.

### D4: Checkpoint handling is explicit (no implicit symlink creation)

- **Decision**: If the checkpoint link/path is missing, fail fast with actionable instructions (including the documented bootstrap path) rather than attempting to create symlinks automatically.
- **Rationale**: Prevents surprising side effects in a tutorial runner and keeps the repo workspace state explicit and reproducible.
- **Alternatives considered**:
  - Auto-create links when a default snapshot root exists: rejected per clarified spec.

### D5: CPU-only unit tests validate the runner’s deterministic logic

- **Decision**: Add unit tests that avoid GPU execution and validate:
  - Scenario enumeration from `--modes` × `--quant-pairs`.
  - Snapshot cleanup rules (stale scenarios removed).
  - Verification diff scope (summary-only) and missing-expected failure behavior.
  - Non-degeneracy gate behavior using synthetic summary JSON fixtures.
- **Rationale**: Keeps CI fast and deterministic while still covering the highest-risk orchestration logic.
- **Alternatives considered**:
  - GPU integration tests for full runs: deferred to manual validation due to cost and environment variability.

## Open Questions

None for planning; remaining work is implementation and test construction consistent with the specification.
