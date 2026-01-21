---
description: "Task list: Revise Qwen3-VL-8B tutorial pack (introduce + layer sensitivity)"
---

# Tasks: Revise Qwen3-VL-8B Tutorial Pack (Introduce + Layer Sensitivity)

**Input**: Design documents from `/data1/huangzhe/code/auto-quantize-model/specs/002-revise-qwen3-vl-tutorial/`  
**Scope**: `/data1/huangzhe/code/auto-quantize-model/docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/`

**Tests**: Add unit tests for deterministic summary/verification logic; end-to-end GPU runs remain manual and documented in the tutorial pack.

## Format: `- [ ] T### [P?] [US?] Description (with file path)`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[US#]**: User story label (required for user-story phase tasks only)
- Include exact file paths in descriptions

## Implementation Guides

- `context/tasks/working/002-revise-qwen3-vl-tutorial/impl-phase-1-setup.md`
- `context/tasks/working/002-revise-qwen3-vl-tutorial/impl-phase-2-foundational.md`
- `context/tasks/working/002-revise-qwen3-vl-tutorial/impl-phase-3-us1-all-layers.md`
- `context/tasks/working/002-revise-qwen3-vl-tutorial/impl-phase-4-us2-lm-only.md`
- `context/tasks/working/002-revise-qwen3-vl-tutorial/impl-phase-5-us5-snapshot-verify.md`
- `context/tasks/working/002-revise-qwen3-vl-tutorial/impl-phase-6-polish.md`
- `context/tasks/working/002-revise-qwen3-vl-tutorial/impl-integrate-phases.md`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Add shared config/building blocks needed by the tutorial pack runners.

- [ ] T001 Add Qwen3-VL-8B Hydra model config in `conf/model/qwen3_vl_8b_instruct/arch/qwen3_vl_8b_instruct.default.yaml`
- [ ] T002 [P] Add Qwen3-VL-8B infer defaults in `conf/model/qwen3_vl_8b_instruct/infer/qwen3_vl_8b_instruct.default.yaml`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared scenario/summarization/verification infrastructure that blocks all user stories.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [ ] T003 Implement stable summary builder (schema-locked) in `src/auto_quantize_model/qwen/tutorial_pack_summary.py`
- [ ] T004 Add unit tests for summary builder and non-degeneracy detection in `tests/unit/test_qwen_tutorial_pack_summary.py`
- [ ] T005 Update tutorial summarizer CLI wrapper to emit `summary.json` + `summary.md` using the summary builder in `docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/scripts/summarize_manifest.py`
- [ ] T006 Update `run_demo.sh` CLI parsing to match the contract (`--snapshot-report`, `--device`, `--dataset-size`, `--modes`, `--quant-pairs`) in `docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/run_demo.sh`
- [ ] T007 Add dataset preset resolution + fail-fast missing-asset checks (COCO root, VLM DB, captions) in `docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/run_demo.sh`
- [ ] T008 Implement per-scenario output layout (workspace + outputs + summaries) in `docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/run_demo.sh`
- [ ] T009 Implement verification loop that diffs only sanitized summaries against `expected_report/` (with actionable guidance) in `docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/run_demo.sh`

**Checkpoint**: Foundation ready — user story implementation can begin.

---

## Phase 3: User Story 1 — All-layers sensitivity demo (Priority: P1) 🎯 MVP

**Goal**: Run an end-to-end **all-layers** sensitivity analysis for Qwen3-VL-8B with meaningful defaults and verifiable, non-degenerate outputs.

**Independent Test**: From repo root, run:

- `bash docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/run_demo.sh --modes all_layers --dataset-size medium --quant-pairs wint4_afp16,wint4_aint8`

and confirm:

- Per-scenario outputs exist under `tmp/.../outputs/all_layers/<quant_pair>/`
- Per-scenario summaries exist under `tmp/.../summaries/all_layers/<quant_pair>/`
- Each `summary.json` reports `has_nonzero_sensitivity=true` and correct medium-preset metadata.

### Implementation for User Story 1

- [ ] T010 [US1] Batch VLM calibration dataloader to honor `batch_size` + `num_calib_batches` in `models/qwen3_vl_4b_instruct/helpers/qwen3_vl_4b_autoquant_all_layers/run_qwen3_vl_4b_autoquant_all_layers.py`
- [ ] T011 [US1] Add `--quant-pair` support (load `conf/quant_pair/*.yaml`, resolve `format_name`, set stable scheme/manifest names) in `models/qwen3_vl_4b_instruct/helpers/qwen3_vl_4b_autoquant_all_layers/run_qwen3_vl_4b_autoquant_all_layers.py`
- [ ] T012 [US1] Record complete dataset metadata (`size`, `calib_seq_len`, `batch_size`, `num_calib_batches`, `num/max_calib_samples`) in the manifest in `models/qwen3_vl_4b_instruct/helpers/qwen3_vl_4b_autoquant_all_layers/run_qwen3_vl_4b_autoquant_all_layers.py`
- [ ] T013 [US1] Wire all-layers scenario execution (2 quant pairs) using repo dataset assets + medium defaults in `docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/run_demo.sh`
- [ ] T014 [US1] Generate and commit sanitized expected snapshots for all-layers scenarios under `docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/expected_report/all_layers/`
- [ ] T015 [US1] Update all-layers documentation + output layout examples (and explain “4B helper” naming) in `docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/README.md`

**Checkpoint**: User Story 1 is runnable and independently verifiable (via `--modes all_layers`).

---

## Phase 4: User Story 2 — LM-only sensitivity demo (Priority: P2)

**Goal**: Run an end-to-end **LM-only** sensitivity analysis with meaningful results (not “all zeros”) and verifiable outputs.

**Independent Test**: From repo root, run:

- `bash docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/run_demo.sh --modes lm_only --dataset-size medium --quant-pairs wint4_afp16,wint4_aint8`

and confirm:

- Per-scenario outputs exist under `tmp/.../outputs/lm_only/<quant_pair>/`
- Each `summary.json` reports `has_nonzero_sensitivity=true` for LM-only scenarios.

### Implementation for User Story 2

- [ ] T016 [US2] Run LM-only scenarios via Hydra runner (`scripts/qwen/qwen3_lm_sensitivity.py`) with overrides for model=8B, dataset preset, device, and scenario output dirs in `docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/run_demo.sh`
- [ ] T017 [US2] Ensure LM-only manifests consistently include required dataset metadata fields for summarization in `scripts/qwen/qwen3_lm_sensitivity.py`
- [ ] T018 [US2] Generate and commit sanitized expected snapshots for LM-only scenarios under `docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/expected_report/lm_only/`
- [ ] T019 [US2] Update LM-only documentation to explain the “all-zero sensitivities” failure mode and the tutorial’s remediation path in `docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/README.md`

**Checkpoint**: User Story 2 is runnable and independently verifiable (via `--modes lm_only`).

---

## Phase 5: User Story 5 — Snapshot maintenance workflow (Priority: P5)

**Goal**: Provide a safe snapshot workflow to refresh `expected_report/` (sanitized) and a strict verification workflow to catch regressions.

**Independent Test**:

1. `bash docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/run_demo.sh --snapshot-report`
2. `bash docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/run_demo.sh`

Confirm verification passes (no summary diffs) and summaries assert non-degeneracy for both modes.

### Implementation for User Story 5

- [ ] T020 [US5] Extend snapshot mode to refresh `expected_report/<mode>/<quant_pair>/` for all selected scenarios and delete stale snapshot dirs in `docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/run_demo.sh`
- [ ] T021 [US5] Enforce non-degeneracy at verification time (fail if `has_nonzero_sensitivity=false` with actionable guidance) in `docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/run_demo.sh`
- [ ] T022 [US5] Remove synthetic calibration input generation and delete `docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/inputs/coco2017_captions_small.txt`
- [ ] T023 [US5] Sync contracts with implementation: update `specs/002-revise-qwen3-vl-tutorial/contracts/run_demo_cli.md` (and `specs/002-revise-qwen3-vl-tutorial/contracts/summary.schema.json` if needed)

**Checkpoint**: Snapshot/verify is robust and maintainable for all 4 scenarios.

---

## Phase N: Polish & Cross-Cutting Concerns

**Purpose**: Final cleanup, docs polish, and validation steps.

- [ ] T024 [P] Update validated quickstart commands (including subset-run examples) in `specs/002-revise-qwen3-vl-tutorial/quickstart.md`
- [ ] T025 [P] Add manual GPU validation checklist in `tests/manual/qwen/test_tut_qwen3_vl_8b_layer_sensitivity.md`
- [ ] T026 Update tutorial README troubleshooting and prerequisites (dataset + checkpoint link) in `docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/README.md`
- [ ] T027 Run `pixi run ruff check .`, `pixi run mypy .`, and `pixi run pytest` and record any follow-ups in `specs/002-revise-qwen3-vl-tutorial/research.md`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies
- **Foundational (Phase 2)**: Depends on Setup — **blocks all user stories**
- **User Stories (Phase 3+)**: Depend on Foundational
- **Polish (Final Phase)**: Depends on desired user stories being complete

### User Story Dependencies (suggested)

- **US1 (P1)**: No dependencies beyond Foundational
- **US2 (P2)**: No dependencies beyond Foundational
- **US5 (P5)**: Depends on US1 + US2 (needs all scenarios implemented)

### Dependency Graph (stories)

```text
Setup → Foundational → { US1, US2 } → US5 → Polish
```

---

## Parallel Execution Examples

### Parallel Example: US1

```text
Dev A: T010–T012 (all-layers driver updates)
Dev B: T013 + T015 (run_demo wiring + README updates)
```

### Parallel Example: US2

```text
Dev A: T016–T017 (run_demo + Hydra overrides)
Dev B: T018–T019 (expected_report + README LM-only explanation)
```

### Parallel Example: US5

```text
Dev A: T020–T021 (snapshot/verify hardening)
Dev B: T023 + T024 (contracts + quickstart sync)
```

---

## Implementation Strategy

### MVP First (US1 Only)

1. Complete Phase 1–2
2. Complete Phase 3 (US1)
3. Validate with `--modes all_layers` and the medium preset

### Incremental Delivery

1. US1: all-layers (2 quant pairs)
2. US2: LM-only (2 quant pairs)
3. US5: snapshot/verification hardening for all scenarios
4. Polish: docs + validation tasks
