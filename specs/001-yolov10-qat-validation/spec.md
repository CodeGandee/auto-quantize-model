# Feature Specification: Validate YOLOv10 W4A16 QAT Stability (EMA + QC)

**Feature Branch**: `001-yolov10-qat-validation`  
**Created**: 2025-12-25  
**Status**: Draft  
**Input**: User description: "Validate whether the research-paper stabilization algorithm works for YOLOv10 W4A16 QAT by focusing on yolo10n and yolo10s first, then attempting yolo10m only if the smaller models validate the approach."

## User Scenarios & Testing *(mandatory)*

<!--
  IMPORTANT: User stories should be PRIORITIZED as user journeys ordered by importance.
  Each user story/journey must be INDEPENDENTLY TESTABLE - meaning if you implement just ONE of them,
  you should still have a viable MVP (Minimum Viable Product) that delivers value.
  
  Assign priorities (P1, P2, P3, etc.) to each story, where P1 is the most critical.
  Think of each story as a standalone slice of functionality that can be:
  - Developed independently
  - Tested independently
  - Deployed independently
  - Demonstrated to users independently
-->

### User Story 1 - Validate the method on yolo10n (Priority: P1)

A quantization engineer validates the research-paper stabilization approach (EMA + post-hoc quantization correction) for W4A16 QAT training on the smallest YOLOv10 variant (yolo10n) to quickly determine whether it prevents the “early peak then collapse” instability.

**Why this priority**: yolo10n is the fastest and cheapest way to verify algorithm correctness before running longer experiments.

**Independent Test**: Run a yolo10n W4A16 QAT training experiment for each method variant (baseline QAT, EMA, EMA+QC) and confirm each run produces a comparable metrics summary and a clear stability outcome.

**Acceptance Scenarios**:

1. **Given** a ready dataset and a yolo10n starting checkpoint, **When** I run a yolo10n W4A16 QAT experiment with method variant = EMA+QC, **Then** training completes and produces a run summary including stability outcome and final validation metrics.
2. **Given** the same dataset and checkpoint, **When** I run baseline QAT and EMA variants with identical experiment settings, **Then** I can compare outcomes across variants in a single consolidated comparison view.

---

### User Story 2 - Confirm results scale to yolo10s (Priority: P2)

A quantization engineer repeats the same validation workflow on yolo10s to ensure the stabilization method continues to work when model capacity increases.

**Why this priority**: A method that only works for yolo10n is not useful for the intended target model sizes; yolo10s is the next cost-effective gate.

**Independent Test**: Run yolo10s W4A16 QAT for each method variant and confirm stability and metrics are reported in the same format as yolo10n.

**Acceptance Scenarios**:

1. **Given** that yolo10n validation has completed, **When** I run yolo10s experiments for the same set of method variants, **Then** the system produces a comparable report and clearly indicates whether the stabilization method still prevents collapse.

---

### User Story 3 - Gate yolo10m until n/s validation passes (Priority: P3)

A quantization engineer uses a pass/fail gate from yolo10n and yolo10s to decide whether it is worth running the expensive yolo10m W4A16 QAT experiments.

**Why this priority**: It prevents wasting time and compute on yolo10m before algorithm correctness is established.

**Independent Test**: After completing yolo10n and yolo10s validation, confirm the system can produce a go/no-go decision for yolo10m based on defined stability and reproducibility criteria.

**Acceptance Scenarios**:

1. **Given** completed yolo10n and yolo10s experiment results, **When** I request a yolo10m run, **Then** the request is accepted only if the defined stage-gate criteria are met (otherwise a clear reason is provided).

---

### Edge Cases

- Missing or inaccessible required inputs (dataset, starting checkpoint, or configuration) prevents a run from starting and yields a clear error message.
- A run terminates early (errors, interruption, resource exhaustion) and is marked “incomplete” so it is excluded from success-gate decisions unless explicitly included.
- Results are not comparable (different dataset slice, different evaluation settings) and are flagged as non-comparable in the comparison view.
- A method variant produces degenerate results (e.g., near-zero detection quality) and is clearly marked as “collapsed/failed” rather than silently included in averages.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST support running W4A16 QAT experiments on YOLOv10 variants yolo10n and yolo10s as first-class targets.
- **FR-002**: The system MUST support method variants required to validate the research paper: baseline QAT, EMA-stabilized QAT, and EMA+QC (post-hoc quantization correction).
- **FR-003**: The system MUST support a fast validation mode suitable for early-stage verification (e.g., deterministic dataset subsetting and shorter training durations), without changing the meaning of “success” reporting.
- **FR-004**: The system MUST produce a run record sufficient to reproduce and compare experiments, including: model variant, method variant, dataset selection/provenance, and all evaluation settings used for reported metrics.
- **FR-005**: The system MUST produce a consolidated comparison summary that allows reviewers to compare baseline vs EMA vs EMA+QC for the same model variant.
- **FR-006**: The system MUST define and apply stage-gate rules that prevent starting yolo10m experiments until yolo10n and yolo10s have met the success criteria.
- **FR-007**: The system MUST classify each run outcome as at least one of: completed-success, completed-failed (collapsed), or incomplete, and MUST use this classification consistently in summaries and gating.
- **FR-008**: The system MUST make it possible to run at least two independent repetitions per (model variant, method variant) and report whether the observed outcome is consistent across repetitions.

### Assumptions

- “W4A16” means weight-only quantization at 4-bit weights with higher-precision activations during training.
- The validation goal is algorithm correctness and stability first; maximizing peak accuracy is explicitly secondary until stability is confirmed.
- The baseline comparison is the same training setup without the stabilization method enabled.
- “Validation quality” refers to a single primary detection-quality metric used consistently for comparisons (for example, a standard mean average precision metric on a fixed validation set).

### Dependencies

- Access to a consistent dataset split suitable for detection evaluation (train and validation).
- Access to baseline pretrained starting checkpoints for yolo10n and yolo10s.
- A stable, repeatable evaluation procedure so “final” vs “best” validation quality comparisons are meaningful.
- A working Pixi environment `cu128` for running and comparing experiments.

### Non-Goals (for this feature)

- Broad hyperparameter sweeps for maximum accuracy.
- Support for additional stabilization techniques beyond EMA and QC (unless they are required to reproduce the research-paper baseline for EMA+QC).

### Key Entities *(include if feature involves data)*

- **Experiment Run**: A single training-and-evaluation attempt for a specific model variant and method variant, with a defined dataset selection and evaluation settings.
- **Method Variant**: The stabilization approach under evaluation (baseline QAT, EMA, EMA+QC).
- **Stage Gate**: A pass/fail decision that determines whether yolo10m experiments are allowed based on yolo10n and yolo10s results.
- **Comparison Summary**: A reviewer-facing summary that compares outcomes and metrics across method variants for the same model variant.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: For yolo10n, at least 2 out of 2 EMA+QC runs complete without “collapse,” where collapse is defined as final validation quality < 50% of the best validation quality achieved during that run.
- **SC-002**: For yolo10s, at least 2 out of 2 EMA+QC runs complete without collapse (same collapse definition as SC-001).
- **SC-003**: For yolo10n and yolo10s, the comparison summary clearly distinguishes baseline QAT vs EMA vs EMA+QC and includes a stability outcome for each run.
- **SC-004**: The yolo10m stage gate is considered “passed” only when SC-001 through SC-003 are met for both yolo10n and yolo10s, and the decision is recorded in the comparison summary.
