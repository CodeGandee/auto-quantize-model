# Feature Specification: Revise Qwen3-VL-8B Tutorial Pack (Introduce + Layer Sensitivity)

**Feature Branch**: `[002-revise-qwen3-vl-tutorial]`  
**Created**: 2026-01-21  
**Status**: Draft  
**Input**: User description: "check context/plans/plan-revise-qwen3-vl-8b-tutorial-pack.md, implement it, create 5 user stories, and name new branch according to magic-context/speckit/name-new-branch.md"

## Clarifications

### Session 2026-01-21

- Q: What should the default tutorial run execute (scenario coverage)? → A: Default runs all 4 scenarios (2 modes × 2 worked examples); allow an optional way to run a subset for faster iteration.
- Q: What should happen if required dataset assets are missing? → A: Fail fast with a clear error listing what’s missing and how to bootstrap/locate the dataset assets.
- Q: What is the “non-zero sensitivity” threshold for verification? → A: Treat only exact 0.0 as zero.
- Q: What runtime target should the default medium run have? → A: No runtime limit.
- Q: What should verification compare? → A: Only the sanitized summaries for each scenario.

## User Scenarios & Testing *(mandatory)*

### Terminology (plain-language)

- **All-layers sensitivity**: A per-layer “importance” ranking that covers both the vision and language parts of the model.
- **LM-only sensitivity**: A per-layer “importance” ranking that covers the language-model layers only.
- **Worked example**: A pre-chosen quantization configuration included in the tutorial for reproducible, comparable results.
- **Dataset preset**: A small/medium/large calibration-input choice that trades off runtime vs. stability of the ranking.
- **Expected report snapshot**: A tracked, sanitized set of outputs used as the reference for verification.

### User Story 1 - Run an end-to-end all-layers sensitivity demo (Priority: P1)

As an AI assistant or developer, I want a tutorial pack that can be run end-to-end to generate an **all-layers** per-layer sensitivity report for Qwen3-VL-8B, using meaningful default calibration settings and producing a report that is verifiably non-degenerate.

**Why this priority**: This is the primary value of the tutorial pack: a reproducible, end-to-end reference run that demonstrates the workflow and produces a usable sensitivity ranking.

**Independent Test**: From a fresh checkout with prerequisites satisfied, run the tutorial pack and confirm the all-layers report is generated, includes non-zero sensitivities, and passes verification against the tracked expected report snapshot.

**Acceptance Scenarios**:

1. **Given** a local Qwen3-VL-8B model snapshot is available, **When** the user runs the tutorial pack with default settings, **Then** an all-layers sensitivity report is produced and at least one layer has a non-zero sensitivity value.
2. **Given** the repository’s tracked expected report snapshot exists, **When** the user runs the tutorial pack in verification mode, **Then** the run completes successfully and the verification step reports no diffs for the all-layers scenario.

---

### User Story 2 - Run an end-to-end LM-only sensitivity demo with meaningful results (Priority: P2)

As an AI assistant or developer, I want the tutorial pack to produce a **LM-only** per-layer sensitivity report that is not “all zeros”, so I can trust the workflow and avoid confusion from previously-degenerate outputs.

**Why this priority**: The current LM-only “all zero sensitivities” behavior is a known source of confusion; fixing or clearly addressing it is essential to reader confidence.

**Independent Test**: Run the tutorial pack and confirm the LM-only report is generated, includes non-zero sensitivities, and passes verification against the tracked expected report snapshot.

**Acceptance Scenarios**:

1. **Given** a local Qwen3-VL-8B model snapshot is available, **When** the user runs the tutorial pack with default settings, **Then** an LM-only sensitivity report is produced and at least one layer has a non-zero sensitivity value.
2. **Given** the repository’s tracked expected report snapshot exists, **When** the user runs the tutorial pack in verification mode, **Then** the run completes successfully and the verification step reports no diffs for the LM-only scenario.

---

### User Story 5 - Maintain and refresh the expected report snapshot safely (Priority: P5)

As a maintainer, I want a safe “snapshot” workflow that refreshes the tracked expected report artifacts (with sanitization) and a strict verification workflow that detects regressions, so the tutorial remains stable as the codebase evolves.

**Why this priority**: Tutorial packs only stay valuable if they remain reproducible over time and fail loudly when outputs regress (especially to degenerate all-zero sensitivities).

**Independent Test**: Run the tutorial in snapshot mode to refresh the expected report, then run again in verification mode and confirm verification passes.

**Acceptance Scenarios**:

1. **Given** a maintainer intentionally updates the tutorial outputs, **When** they run the tutorial in snapshot mode, **Then** the expected report snapshot is refreshed with sanitized artifacts.
2. **Given** the expected report snapshot is up to date, **When** a user runs the tutorial in verification mode, **Then** verification passes for all scenarios and fails with a clear message if sensitivities regress to all zeros.

---

### Edge Cases

- Missing local model snapshot or checkpoint link for Qwen3-VL-8B.
- Missing dataset assets required for the selected preset (small/medium/large); the tutorial should fail fast with actionable guidance.
- Running on CPU or without CUDA available (should be allowed with warnings, but may be impractically slow).
- The LM-only scenario produces degenerate “all zeros” sensitivities (must be detected and explained, and/or the run must fail verification with actionable guidance).
- Minor numeric drift between machines (verification should rely on stable, sanitized summaries rather than raw binary artifacts).

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The tutorial pack MUST provide a documented, end-to-end workflow that introduces Qwen3-VL-8B into the repo via a local “checkpoint link” without committing model weights.  
  **Acceptance**: Documentation includes the checkpoint-link steps; the tutorial run fails early with a clear message if the link is missing.
- **FR-002**: The tutorial pack MUST run an all-layers per-layer sensitivity analysis with meaningful default calibration settings and produce a human-readable report.  
  **Acceptance**: The all-layers scenario produces a report and a summary that includes at least one non-zero sensitivity value.
- **FR-003**: The tutorial pack MUST run an LM-only per-layer sensitivity analysis with meaningful default calibration settings and produce a human-readable report.  
  **Acceptance**: The LM-only scenario produces a report and a summary that includes at least one non-zero sensitivity value.
- **FR-004**: The tutorial’s default run MUST use the “medium” dataset preset, with calibration settings equivalent to: sequence length 512, batch size 8, 16 calibration batches, and a scoring budget of 128.  
  **Acceptance**: The default run metadata reflects the medium preset and the effective calibration/scoring budget.
- **FR-005**: The tutorial pack MUST offer dataset sizing presets (small/medium/large) and MUST document when each should be used (smoke testing vs. stable rankings).  
  **Acceptance**: Documentation describes all presets and their intended use; running with at least one non-default preset is supported.
- **FR-006**: For the tutorial’s documented runs, the tutorial pack MUST use two worked quantization examples: (a) INT4 weights + FP16 activations, and (b) INT4 weights + INT8 activations.  
  **Acceptance**: Both worked examples are documented and can be executed to produce their corresponding reports.
- **FR-007**: The tutorial pack MUST produce distinct, clearly-labeled outputs for each combination of (mode: all-layers vs LM-only) × (worked example A vs B).  
  **Acceptance**: Outputs and summaries are separated per scenario and can be unambiguously identified by mode and worked example.
- **FR-008**: The tutorial pack MUST record run metadata alongside outputs, including: selected mode, selected preset, effective sample/batch settings, scoring budget, and which worked example was used.  
  **Acceptance**: Each scenario’s sanitized summary includes all required metadata fields.
- **FR-009**: The tutorial pack MUST include a verification workflow that compares the run’s sanitized summaries against tracked expected summaries and fails with clear guidance when they differ.  
  **Acceptance**: Verification succeeds when the per-scenario sanitized summaries match (e.g., `summary.json` and `summary.md`); on mismatch it fails with actionable guidance on how to proceed (verify vs snapshot).
- **FR-010**: The tutorial pack’s verification summaries MUST explicitly assert that sensitivities are non-degenerate (i.e., not “all zeros”) for both modes in the default medium preset.  
  **Acceptance**: Verification fails if either mode’s default summary indicates an all-zero sensitivity table (where “zero” means exactly 0.0), and the failure message points readers to the documented LM-only explanation/remediation.
- **FR-011**: The tutorial documentation MUST explain, in plain language:
  - what the tutorial runner does step-by-step,
  - how and why the checkpoint link is created,
  - why “4B helper” naming appears even when running 8B,
  - what causes LM-only sensitivity to become degenerate, and how the tutorial avoids or addresses that.
  **Acceptance**: The README contains dedicated sections covering each topic above.
- **FR-012**: The tutorial’s expected report snapshot MUST contain sanitized, real artifacts (not placeholders) for the documented scenarios and remain small enough to be reviewed and versioned.  
  **Acceptance**: The expected report includes sanitized artifacts for all four scenarios and excludes non-sanitized large artifacts that are not needed for verification.
- **FR-013**: The tutorial pack’s default demo/verification run MUST execute and verify all four scenarios (2 modes × 2 worked examples).  
  **Acceptance**: A single default run produces and validates four scenario summaries; an optional way to run a subset exists for faster iteration.
- **FR-014**: If required dataset assets for the selected preset are missing, the tutorial pack MUST fail fast with a clear, actionable error.  
  **Acceptance**: The error explicitly lists missing assets and points to documented steps to provide or bootstrap them.

### Key Entities *(include if feature involves data)*

- **Tutorial Pack**: The self-contained tutorial folder containing inputs, an expected report snapshot, and a runnable demo/verification workflow.
- **Checkpoint Link**: A local link inside the repo that points to a user-provided model snapshot directory.
- **Dataset Preset**: A named sizing preset (small/medium/large) that maps to a specific calibration sample/batch budget.
- **Run Scenario**: A single run defined by (mode, worked example, dataset preset) that produces outputs and a summary.
- **Run Workspace**: A temporary, gitignored directory where full outputs are written during a run.
- **Sanitized Summary**: A stable, reviewable summary of outputs used for verification and expected report snapshots.
- **Expected Report Snapshot**: The tracked set of sanitized artifacts used as the “golden” reference for verification.

### Assumptions

- Users have access to a CUDA-capable machine for practical runtimes (CPU runs are allowed but may be very slow).
- Users can obtain the Qwen3-VL-8B model snapshot separately and provide it locally; the repo continues to avoid committing model weights.
- The repo already provides dataset assets for small/medium/large presets; the tutorial will reuse those assets rather than introducing new dataset formats.
- No maximum runtime is required for the default medium run; the priority is correctness and meaningful (non-degenerate) outputs.

### Dependencies

- Requirements source of truth: `context/tasks/req-revise-qwen3-8b-tutorial.md`
- Known issue reference: `context/issues/known/issue-qwen3-vl-lm-only-tutorial-zero-sensitivity.md`
- Scope location: `docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/`

### Out of Scope

- Publishing or distributing model weights.
- Requiring a specific GPU model; the tutorial should be portable across CUDA-capable machines.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: From a clean checkout with prerequisites satisfied, a user can run the tutorial pack and complete verification with no diffs against the tracked expected report snapshot.
- **SC-002**: The tutorial produces four verified scenarios (2 modes × 2 worked examples), each with a summary that includes at least one non-zero per-layer sensitivity value.
- **SC-003**: The tutorial’s sanitized summary for each scenario includes the required run metadata (mode, preset, effective calibration settings, scoring budget, worked example) and remains stable across repeated runs on the same machine.
- **SC-004**: The documentation explicitly explains the LM-only “all zeros” failure mode and provides a clear, successful path that yields meaningful (non-degenerate) LM-only sensitivities using the default preset.
