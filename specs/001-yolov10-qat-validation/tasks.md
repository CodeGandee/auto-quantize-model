---

description: "Task list for validating EMA+QC W4A16 QAT on YOLOv10n/s with yolo10m gated"
---

# Tasks: Validate YOLOv10 W4A16 QAT Stability (EMA + QC)

**Input**: Design documents from `specs/001-yolov10-qat-validation/`  
**Prerequisites**: `plan.md` (required), `spec.md` (required for user stories), `research.md`, `data-model.md`, `contracts/`, `quickstart.md`

**Tests**: Include the right-sized tests (unit/integration/manual) per the project constitution; only omit tests if explicitly out-of-scope and justified in `spec.md`.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Each task MUST include an exact file path in the description

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create the scaffolding for configs, runner scripts, and manual validation docs (Pixi `cu128`).

- [X] T001 Create Hydra/OmegaConf config scaffold under `conf/cv-models/yolov10_w4a16_validation/` (`config.yaml`, `profile/{smoke,short,full}.yaml`, `variant/{yolo10n,yolo10s,yolo10m}.yaml`, `method/{baseline,ema,ema_qc}.yaml`)
- [X] T002 [P] Create runner CLI skeleton in `scripts/cv-models/run_yolov10_w4a16_qat_validation.py` (argparse contract from `specs/001-yolov10-qat-validation/contracts/cli.md`)
- [X] T003 [P] Create summarizer CLI skeleton in `scripts/cv-models/summarize_yolov10_w4a16_qat_validation.py` (reads multiple `run_summary.json` and writes comparison `summary.md`)
- [X] T004 [P] Create manual-run guide skeleton in `tests/manual/yolov10_w4a16_ema_qc_validation/README.md` (commands must be `pixi run -e cu128 ...`)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Implement shared building blocks: deterministic dataset selection, results parsing, stability classification, and run-summary artifacts.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [X] T005 Implement config models + loader in `src/auto_quantize_model/cv_models/yolov10_w4a16_validation.py` (load defaults from `conf/cv-models/yolov10_w4a16_validation/`)
- [X] T006 [P] Implement deterministic COCO subset dataset builder in `src/auto_quantize_model/cv_models/yolov10_coco_subset_dataset.py` (train selection + val first-N + dataset YAML + provenance JSON)
- [X] T007 [P] Implement Ultralytics `results.csv` parser in `src/auto_quantize_model/cv_models/yolov10_results_csv.py` (extract `metrics/mAP50-95(B)` series robustly)
- [X] T008 [P] Implement collapse detection + run status classification in `src/auto_quantize_model/cv_models/yolov10_stability.py` (collapse if `final < 0.5 * best`)
- [X] T009 Implement `run_summary.json` and `summary.md` writers in `src/auto_quantize_model/cv_models/yolov10_w4a16_validation.py` (align fields with `specs/001-yolov10-qat-validation/contracts/run_summary.schema.json`)
- [X] T010 [P] Export public APIs in `src/auto_quantize_model/cv_models/__init__.py` (new validation helpers + config loader)
- [X] T011 [P] Add unit tests for results parsing in `tests/unit/cv_models/test_yolov10_results_csv.py` (use a tiny fixture results.csv stored under `tests/unit/cv_models/fixtures/yolov10_results.csv`)
- [X] T012 [P] Add unit tests for stability classification in `tests/unit/cv_models/test_yolov10_stability.py`
- [X] T013 [P] Add unit tests for COCO subset selection in `tests/unit/cv_models/test_yolov10_coco_subset_dataset.py` (use tiny COCO JSON fixtures under `tests/unit/cv_models/fixtures/coco_instances_*.json`)
- [X] T014 Add integration test for `run_summary.json` shape/required keys in `tests/integration/test_yolov10_w4a16_run_summary.py` (write to `tmp/` during test)

**Checkpoint**: Foundation ready — user story implementation can now begin.

---

## Phase 3: User Story 1 — Validate the method on yolo10n (Priority: P1) 🎯 MVP

**Goal**: Run yolo10n W4A16 QAT for `baseline`, `ema`, and `ema+qc`, and emit comparable run summaries and a comparison view.

**Independent Test**: Run `scripts/cv-models/run_yolov10_w4a16_qat_validation.py` for yolo10n (smoke profile) for each method and confirm each run writes `run_summary.json` + `summary.md`, then run the summarizer to generate a combined comparison markdown.

### Tests for User Story 1 (recommended) ⚠️

- [X] T015 [P] [US1] Add unit tests for QC wrapper behavior in `tests/unit/cv_models/test_yolov10_qc.py` (toy `nn.Conv2d+nn.BatchNorm2d` model, verify only QC params are trainable)
- [X] T016 [P] [US1] Add integration test for comparison summarizer in `tests/integration/test_yolov10_w4a16_summarize.py` (consume fixture `run_summary.json` files under `tests/integration/fixtures/yolov10_w4a16/`)

### Implementation for User Story 1

- [X] T017 [US1] Add method-aware EMA control in `src/auto_quantize_model/cv_models/yolov10_ultralytics_trainers.py` (baseline uses EMA decay=0; EMA/EMA+QC uses decay≈0.9999)
- [X] T018 [US1] Implement QC modules + 1-epoch QC training loop in `src/auto_quantize_model/cv_models/yolov10_qc.py` (freeze base weights; BN stats fixed; Adam lr=1e-4)
- [X] T019 [US1] Implement yolo10n run execution in `scripts/cv-models/run_yolov10_w4a16_qat_validation.py` (resolve `models/yolo10/checkpoints/yolov10n.pt`, build subset dataset via `src/auto_quantize_model/cv_models/yolov10_coco_subset_dataset.py`, run trainer, write artifacts)
- [X] T020 [US1] Implement method variants (`baseline`, `ema`, `ema+qc`) end-to-end in `scripts/cv-models/run_yolov10_w4a16_qat_validation.py` (QC runs after QAT and updates final metrics in `run_summary.json`)
- [X] T021 [US1] Implement yolo10n comparison summary generator in `scripts/cv-models/summarize_yolov10_w4a16_qat_validation.py` (table of best/final/collapsed per run; writes `summary.md`)
- [X] T022 [US1] Document yolo10n smoke-run procedure in `tests/manual/yolov10_w4a16_ema_qc_validation/README.md` (baseline/ema/ema+qc; seeds 0 and 1; expected outputs under `tmp/`)

**Checkpoint**: yolo10n validation workflow runs end-to-end and produces comparable summaries.

---

## Phase 4: User Story 2 — Confirm results scale to yolo10s (Priority: P2)

**Goal**: Run the same validation workflow on yolo10s and confirm output format and stability classification are unchanged.

**Independent Test**: Run yolo10s `ema+qc` twice (two seeds) and verify both runs are non-collapsed and appear in a combined n/s comparison report.

### Tests for User Story 2 (recommended) ⚠️

- [ ] T023 [P] [US2] Add unit tests for variant resolution in `tests/unit/cv_models/test_yolov10_variant_resolution.py` (yolo10s checkpoint/config resolution)

### Implementation for User Story 2

- [ ] T024 [US2] Add yolo10s defaults in `conf/cv-models/yolov10_w4a16_validation/variant/yolo10s.yaml` (checkpoint path + recommended training defaults for `short`)
- [ ] T025 [US2] Extend runner to support `--variant yolo10s` in `scripts/cv-models/run_yolov10_w4a16_qat_validation.py`
- [ ] T026 [US2] Extend summarizer to produce a combined yolo10n+yolo10s report in `scripts/cv-models/summarize_yolov10_w4a16_qat_validation.py`
- [ ] T027 [US2] Document yolo10s short-run procedure in `tests/manual/yolov10_w4a16_ema_qc_validation/README.md` (two seeds; expected outputs and where to find run summaries)

**Checkpoint**: yolo10s validation runs produce comparable artifacts and summaries consistent with yolo10n.

---

## Phase 5: User Story 3 — Gate yolo10m until n/s validation passes (Priority: P3)

**Goal**: Implement a stage gate that blocks yolo10m runs unless yolo10n and yolo10s EMA+QC meet the stability/repeatability criteria.

**Independent Test**: With fixture run summaries, verify the gate blocks yolo10m when required runs are missing/collapsed and allows yolo10m only when all criteria pass.

### Tests for User Story 3 (recommended) ⚠️

- [ ] T028 [P] [US3] Add unit tests for stage gate logic in `tests/unit/cv_models/test_yolov10_stage_gate.py` (missing runs, non-comparable datasets, collapsed run, happy path)

### Implementation for User Story 3

- [ ] T029 [US3] Implement stage-gate evaluator in `src/auto_quantize_model/cv_models/yolov10_w4a16_validation.py` (inputs: list of run roots; outputs: pass/fail + reason)
- [ ] T030 [US3] Enforce gate for yolo10m in `scripts/cv-models/run_yolov10_w4a16_qat_validation.py` (require `--gate-root` or `--gate-run-summaries`; refuse run with clear reason in `summary.md`)
- [ ] T031 [US3] Extend summarizer to emit a `yolo10m_allowed` decision in `scripts/cv-models/summarize_yolov10_w4a16_qat_validation.py`
- [ ] T032 [US3] Document yolo10m gating workflow in `tests/manual/yolov10_w4a16_ema_qc_validation/README.md` (how to run gate check, how to interpret decision)

**Checkpoint**: yolo10m is blocked until yolo10n+yolo10s EMA+QC runs satisfy success criteria.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Docs alignment, quality gates, and quickstart validation.

- [ ] T033 [P] Align CLI docs with final flags in `specs/001-yolov10-qat-validation/contracts/cli.md`
- [ ] T034 [P] Align runnable examples with final CLI in `specs/001-yolov10-qat-validation/quickstart.md`
- [ ] T035 [P] Add workflow doc in `docs/workflows/yolov10-w4a16-ema-qc-validation.md` (link to `specs/001-yolov10-qat-validation/quickstart.md` and `specs/001-yolov10-qat-validation/contracts/artifacts.md`)
- [ ] T036 Fix `ruff` findings in `src/auto_quantize_model/cv_models/*.py` and `scripts/cv-models/*.py` (run `pixi run -e cu128 ruff check .`)
- [ ] T037 Fix `mypy` findings in `src/auto_quantize_model/cv_models/*.py` and `scripts/cv-models/*.py` (run `pixi run -e cu128 mypy .`)
- [ ] T038 Fix failing tests under `tests/unit/` and `tests/integration/` (run `pixi run -e cu128 pytest`)
- [ ] T039 Validate manual quickstart procedures in `tests/manual/yolov10_w4a16_ema_qc_validation/README.md` and `specs/001-yolov10-qat-validation/quickstart.md`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately.
- **Foundational (Phase 2)**: Depends on Setup completion — BLOCKS all user stories.
- **User Stories (Phase 3+)**: All depend on Foundational completion.
  - US1 (yolo10n) should complete before US2 (yolo10s) to preserve the validation ladder.
  - US3 (yolo10m gate) depends on US1+US2 semantics and artifacts.
- **Polish (Phase 6)**: Depends on all desired user stories being complete.

### User Story Dependencies

- **US1 (P1, yolo10n)**: Can start after Foundational — no dependencies on other stories.
- **US2 (P2, yolo10s)**: Depends on US1 completion (we only scale after yolo10n validates the method).
- **US3 (P3, yolo10m gate)**: Depends on US1 + US2 completion.

### Within Each User Story

- Tests (if included) should be written before implementation tasks they cover.
- Shared library code lives under `src/auto_quantize_model/`; CLI orchestration lives under `scripts/cv-models/`.
- Manual “run on GPU” validation lives under `tests/manual/` and writes artifacts to `tmp/`.

---

## Parallel Example: User Story 1

```bash
# Tests and implementation can be parallelized across different files.
Task: "T015 [US1] Add unit tests for QC wrapper behavior in tests/unit/cv_models/test_yolov10_qc.py"
Task: "T018 [US1] Implement QC modules + training loop in src/auto_quantize_model/cv_models/yolov10_qc.py"

Task: "T016 [US1] Add integration test for comparison summarizer in tests/integration/test_yolov10_w4a16_summarize.py"
Task: "T021 [US1] Implement yolo10n comparison summary generator in scripts/cv-models/summarize_yolov10_w4a16_qat_validation.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3: US1 (yolo10n baseline vs EMA vs EMA+QC)
4. **STOP and VALIDATE** using `tests/manual/yolov10_w4a16_ema_qc_validation/README.md`

### Incremental Delivery

1. Foundation ready → yolo10n validated (US1)
2. Scale to yolo10s (US2)
3. Add yolo10m stage gate enforcement (US3)
4. Polish + docs + quality gates (Phase 6)

---

## Notes

- All runnable commands in docs should use Pixi `cu128`: `pixi run -e cu128 ...`.
- `tmp/` is ignored by Git — do not commit training artifacts.
