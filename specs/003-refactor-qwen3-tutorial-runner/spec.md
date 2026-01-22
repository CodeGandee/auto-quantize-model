# Feature Specification: Refactor Qwen3-VL Tutorial Pack Runner (Shared Utility)

**Feature Branch**: `[003-refactor-qwen3-tutorial-runner]`  
**Created**: 2026-01-22  
**Status**: Draft  
**Input**: User description: "read context/tasks/refactor/task-refactor-qwen3-tutorial-pack-runner.md first, we are going to implement this, name the new branch according to magic-context/speckit/name-new-branch.md"

## Clarifications

### Session 2026-01-22

- Q: When `expected_report/` is missing or incomplete in verify mode, should verification fail or be treated as a pass? → A: Fail verification if the expected snapshot is missing/incomplete.
- Q: When running multiple selected scenarios, should the runner stop on the first failure or attempt all scenarios and report a combined result? → A: Stop on first failing scenario (fail-fast).
- Q: When the model checkpoint link is missing, should the runner auto-create it or fail with instructions? → A: Never auto-create; fail with instructions.
- Q: In snapshot mode, should `expected_report/` contain only `summary.*` or additional sanitized artifacts? → A: Snapshot only `summary.json` per scenario.

## User Scenarios & Testing *(mandatory)*

### Terminology (plain-language)

- **Tutorial pack**: A self-contained tutorial folder in `docs/` that users run via its `run_demo.sh`.
- **Runner**: The shared command utility that performs the tutorial pack orchestration.
- **Scenario**: One execution defined by (mode, quantization pair, dataset preset, device).
- **Mode**: The type of sensitivity run (e.g., all-layers vs. LM-only).
- **Quantization pair**: The quantization configuration used for a scenario (e.g., weights/activations pairing).
- **Dataset preset**: A named sizing preset (small/medium/large) that trades off runtime vs. stability.
- **Workspace**: A temporary run directory where full outputs are produced (gitignored).
- **Expected report snapshot**: The tracked, per-scenario `summary.json` used for verification.
- **Snapshot mode**: A mode that refreshes the expected report snapshot for selected scenarios.
- **Verify mode**: A mode that compares produced sanitized summaries against the expected snapshot and fails on diffs.
- **Non-degenerate result**: A sensitivity report where at least one sensitivity value is non-zero.

### User Story 1 - Run either tutorial pack end-to-end without interface drift (Priority: P1)

As a tutorial reader, I want each Qwen3-VL tutorial pack to run end-to-end using the same documented flags and produce a verified sensitivity report, so the tutorials stay trustworthy even as the internal implementation evolves.

**Why this priority**: Maintaining the tutorial’s external contract is the primary value; any refactor must preserve the user-facing run experience.

**Independent Test**: From a checkout with prerequisites satisfied, run each pack’s `run_demo.sh` in verify mode and confirm it completes successfully with no diffs against the tracked expected report snapshot.

**Acceptance Scenarios**:

1. **Given** required model and dataset assets are present, **When** a user runs the Qwen3-VL-4B tutorial pack with default settings and verify mode, **Then** the run succeeds and verification reports no diffs against the pack’s expected report snapshot.
2. **Given** required model and dataset assets are present, **When** a user runs the Qwen3-VL-8B tutorial pack with default settings and verify mode, **Then** the run succeeds and verification reports no diffs against the pack’s expected report snapshot.

---

### User Story 2 - Select a subset of scenarios for faster iteration (Priority: P2)

As a user iterating on the workflow, I want to run only selected modes and/or quantization pairs, so I can get faster feedback while preserving the same output layout and summary contract.

**Why this priority**: Scenario selection is a practical necessity for development and debugging; it also reduces the cost of verifying changes.

**Independent Test**: Run a tutorial pack specifying a single mode and single quantization pair, then confirm only that scenario is executed and the expected per-scenario summaries are produced.

**Acceptance Scenarios**:

1. **Given** a user selects a subset of modes and quantization pairs, **When** they run the tutorial pack, **Then** only the selected scenarios run and only those scenarios’ summaries are produced/verified.

---

### User Story 3 - Refresh expected report snapshots and catch regressions safely (Priority: P3)

As a maintainer, I want a safe snapshot workflow to refresh expected report snapshots and a strict verification workflow that detects regressions (including degenerate “all zero” sensitivity outputs), so tutorial stability does not depend on manual inspection.

**Why this priority**: Snapshot/verify is how the repo keeps tutorials reproducible over time; if it is fragile or compares the wrong artifacts, regressions will slip through.

**Independent Test**: Run in snapshot mode for a known set of scenarios, then run again in verify mode and confirm verification passes and only sanitized summaries are compared.

**Acceptance Scenarios**:

1. **Given** a maintainer runs the tutorial pack in snapshot mode for a subset of scenarios, **When** snapshot mode completes, **Then** the expected report snapshot is refreshed for the selected scenarios and any stale scenarios not selected are removed.
2. **Given** a run completes in verify mode, **When** verification is performed, **Then** only sanitized summary artifacts are compared and the run fails if any scenario’s summary indicates a degenerate (all-zero) sensitivity result.

### User Story 4 - Add a new Qwen3-VL tutorial pack without copy/paste runners (Priority: P4)

As a maintainer, I want adding a new Qwen3-VL model tutorial pack to require only model-specific configuration and a thin wrapper, so the tutorial suite can grow without duplicating orchestration logic.

**Why this priority**: Prevents recurring duplication and “drift” bugs as more models/variants are added.

**Independent Test**: Define a new model configuration entry and a wrapper that delegates to the shared runner, then verify the new pack can execute at least one scenario end-to-end and produce the expected summaries.

**Acceptance Scenarios**:

1. **Given** a new model identifier and expected report directory are provided, **When** a new tutorial pack delegates to the shared runner, **Then** the runner can execute at least one scenario and produce the same style of sanitized summaries and verification behavior as existing packs.

### Edge Cases

- Missing model assets: fail fast with a clear list of what is missing and how to provide it.
- Missing dataset assets for the selected preset: fail fast with a clear list of missing assets and required locations.
- Invalid `mode` and/or invalid `quantization pair`: fail with a clear list of allowed values.
- Verify requested but expected report snapshot is missing/incomplete: fail with clear guidance to run snapshot mode.
- Snapshot mode with a subset selection: remove stale scenarios that are not selected to avoid confusing leftovers.
- Partial/failed scenario run: stop immediately and report a failure; verification must not report success unless all requested scenario summaries exist and pass checks.
- Concurrent runs: workspaces must not collide; each run must isolate its outputs.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The repo MUST provide a single shared runner utility that is the “source of truth” for Qwen3-VL tutorial pack orchestration.  
  **Acceptance**: Both existing Qwen3-VL tutorial packs delegate orchestration to the shared runner instead of duplicating it.
- **FR-002**: The shared runner MUST preserve the existing tutorial pack CLI contract (flag names and meaning), including: `--snapshot-report`, `--device`, `--dataset-size`, `--modes`, `--quant-pairs`.  
  **Acceptance**: Running either pack with the same flags as before produces the same scenario selection behavior and the same expected report layout.
- **FR-003**: Users MUST be able to choose which Qwen3-VL model/tutorial configuration to run via an explicit model identifier input to the shared runner.  
  **Acceptance**: One runner supports both Qwen3-VL-4B and Qwen3-VL-8B packs without branching logic duplicated inside each pack.
- **FR-004**: The shared runner MUST resolve a dataset preset into concrete run budgets and required asset paths and validate those assets before any scenario starts.  
  **Acceptance**: If any required asset is missing, the run fails before execution with an actionable error listing missing items (including how to create required checkpoint links).
- **FR-005**: The shared runner MUST enumerate scenarios as the product of the selected modes and selected quantization pairs, with stable scenario identifiers.  
  **Acceptance**: A selected set of 2 modes and 2 quantization pairs produces exactly 4 scenario runs with deterministic scenario IDs.
- **FR-006**: The shared runner MUST create a new isolated workspace per run and write per-scenario outputs and per-scenario summaries using a consistent directory layout.  
  **Acceptance**: Repeated runs do not overwrite each other’s workspaces, and each scenario produces a summary in a predictable location.
- **FR-007**: For each scenario, the runner MUST execute the appropriate sensitivity workflow for the selected mode and produce the inputs required for summarization.  
  **Acceptance**: For every requested scenario, summarization runs and produces the required summary artifacts.
- **FR-008**: The runner MUST produce schema-locked, sanitized summaries for each scenario in machine-readable form.  
  **Acceptance**: For each requested scenario, `summary.json` is generated and remains compatible with the existing summary schema contract.
- **FR-009**: The runner MUST support a snapshot workflow that refreshes the expected report snapshot for selected scenarios and removes stale expected scenarios not selected.  
  **Acceptance**: After snapshot mode, the expected report directory contains exactly the selected scenarios and only `summary.json` for each scenario.
- **FR-010**: The runner MUST support a verification workflow that compares only the sanitized summaries against the expected report snapshot and fails with clear diffs when they differ.  
  **Acceptance**: Verification ignores non-summary artifacts; it fails if the expected snapshot is missing/incomplete; and on mismatch it identifies the scenario and which summary file differs.
- **FR-011**: The runner’s verification MUST enforce a non-degeneracy gate for sensitivity results.  
  **Acceptance**: Verification fails if any scenario’s summary indicates no non-zero sensitivities.
- **FR-012**: The repo MUST include automated, CPU-only tests for the shared runner’s pure logic.  
  **Acceptance**: Tests cover scenario enumeration, snapshot cleanup rules, verification diff scope (summary-only), and non-degeneracy handling without requiring GPU execution.
- **FR-013**: The refactor MUST not require users to change the tutorial packs they run (entrypoints and documentation remain stable).  
  **Acceptance**: `run_demo.sh` remains the user entrypoint for each pack and continues to accept the documented flags.
- **FR-014**: Failures MUST be user-actionable.  
  **Acceptance**: Error messages for missing assets, invalid scenario selectors, or verification diffs include clear next steps (e.g., how to fix inputs or when to use snapshot mode).
- **FR-015**: If any requested scenario fails, the runner MUST stop immediately and report the run as failed.  
  **Acceptance**: When running multiple scenarios, the first failing scenario stops the run, clearly identifies which scenario failed, and does not produce a “verification complete” result.
- **FR-016**: The runner MUST NOT auto-create model checkpoint links.  
  **Acceptance**: If the expected checkpoint path is missing, the runner fails with clear instructions to create the link (or run the documented bootstrap) rather than creating it automatically.

### Key Entities *(include if feature involves data)*

- **Tutorial Pack**: A runnable documentation bundle for a specific Qwen3-VL model that delegates orchestration to the shared runner.
- **Model Configuration**: The pack/model-specific configuration needed by the runner to locate the checkpoint and run the workflows.
- **Dataset Preset**: A named preset that maps to calibration and scoring budgets plus required asset locations.
- **Scenario**: A single run defined by (mode, quantization pair, dataset preset, device) that produces outputs and summaries.
- **Workspace**: A temporary directory used to store full run outputs and summaries.
- **Sanitized Summary**: The stable, reviewable artifact used for snapshotting and verification (`summary.json`).
- **Expected Report Snapshot**: The tracked reference set of per-scenario `summary.json` used as the “golden” verification target.

### Assumptions

- Users can obtain model checkpoints separately and provide them locally (the repo does not ship model weights).
- Dataset assets required by the tutorial packs are available on the user’s machine (or can be bootstrapped using existing repo tooling).
- The existing summary schema contract remains the source of truth; the refactor must not introduce schema changes.

### Dependencies

- Existing tutorial packs for Qwen3-VL-4B and Qwen3-VL-8B (current duplication source).
- Existing summary schema contract and summary-building logic already present in the repo.
- Existing underlying sensitivity workflows (all-layers and LM-only) that the runner orchestrates.

### Out of Scope

- Changing sensitivity algorithms or scoring semantics.
- Changing the summary schema contract.
- Adding new datasets or changing dataset formats.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: With prerequisites satisfied, both Qwen3-VL tutorial packs can be run in verify mode and complete successfully with no diffs against their tracked expected report snapshots.
- **SC-002**: Users can run a subset of scenarios (modes and quantization pairs) and obtain the corresponding per-scenario summaries, without producing or validating unrequested scenarios.
- **SC-003**: A snapshot run for a chosen set of scenarios refreshes expected report summaries and removes stale scenarios; a subsequent verify run passes without manual cleanup.
- **SC-004**: Automated CPU-only tests for runner logic reliably detect regressions in scenario enumeration, snapshot cleanup, summary-only verification, and non-degeneracy gating.
