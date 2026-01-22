---
description: "Task list: refactor Qwen3-VL tutorial-pack runners into a shared Python utility"
---

# Tasks: Refactor Qwen3-VL Tutorial Pack Runner (Shared Utility)

**Input**: Design documents from `/data1/huangzhe/code/auto-quantize-model/specs/003-refactor-qwen3-tutorial-runner/`  
**Prerequisites**: `/data1/huangzhe/code/auto-quantize-model/specs/003-refactor-qwen3-tutorial-runner/plan.md`, `/data1/huangzhe/code/auto-quantize-model/specs/003-refactor-qwen3-tutorial-runner/spec.md`  
**Optional inputs**: `/data1/huangzhe/code/auto-quantize-model/specs/003-refactor-qwen3-tutorial-runner/research.md`, `/data1/huangzhe/code/auto-quantize-model/specs/003-refactor-qwen3-tutorial-runner/data-model.md`, `/data1/huangzhe/code/auto-quantize-model/specs/003-refactor-qwen3-tutorial-runner/contracts/`, `/data1/huangzhe/code/auto-quantize-model/specs/003-refactor-qwen3-tutorial-runner/quickstart.md`

**Tests**: Add CPU-only unit/integration tests for deterministic runner logic (scenario enumeration, snapshot cleanup, verify diff scope, non-degeneracy gate). GPU execution remains manual validation via the tutorial packs’ `run_demo.sh`.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `- [ ] T### [P?] [US?] Description with file path`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[US#]**: Which user story this task belongs to (US1..US4). Only present in story phases.
- All tasks include absolute file paths.

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create the shared runner/CLI scaffolding and test fixtures.

- [X] T001 Create shared runner module skeleton in `/data1/huangzhe/code/auto-quantize-model/src/auto_quantize_model/qwen/tutorial_pack_runner.py`
- [X] T002 [P] Create shared CLI module skeleton in `/data1/huangzhe/code/auto-quantize-model/src/auto_quantize_model/qwen/cli_tutorial_pack_runner.py`
- [X] T003 [P] Export runner/CLI symbols from `/data1/huangzhe/code/auto-quantize-model/src/auto_quantize_model/qwen/__init__.py`
- [X] T004 [P] Create unit test file scaffold in `/data1/huangzhe/code/auto-quantize-model/tests/unit/test_qwen_tutorial_pack_runner.py`
- [X] T005 [P] Add JSON fixtures for summary gate tests under `/data1/huangzhe/code/auto-quantize-model/tests/unit/fixtures/qwen_tutorial_pack_runner/summary_ok.json` and `/data1/huangzhe/code/auto-quantize-model/tests/unit/fixtures/qwen_tutorial_pack_runner/summary_degenerate.json`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core deterministic building blocks needed by all user stories.

**⚠️ CRITICAL**: No user story work should start until this phase is complete.

- [X] T006 Implement core data models (model spec, dataset preset, scenario spec) in `/data1/huangzhe/code/auto-quantize-model/src/auto_quantize_model/qwen/tutorial_pack_runner.py`
- [X] T007 Implement dataset preset resolution (budgets + required asset paths) in `/data1/huangzhe/code/auto-quantize-model/src/auto_quantize_model/qwen/tutorial_pack_runner.py`
- [X] T008 Implement mode/quant-pair parsing + scenario enumeration helpers in `/data1/huangzhe/code/auto-quantize-model/src/auto_quantize_model/qwen/tutorial_pack_runner.py`
- [X] T009 Implement workspace + expected-report path layout helpers in `/data1/huangzhe/code/auto-quantize-model/src/auto_quantize_model/qwen/tutorial_pack_runner.py`
- [X] T010 [P] Add unit tests for dataset preset resolution in `/data1/huangzhe/code/auto-quantize-model/tests/unit/test_qwen_tutorial_pack_runner.py`
- [X] T011 [P] Add unit tests for scenario enumeration and selector validation in `/data1/huangzhe/code/auto-quantize-model/tests/unit/test_qwen_tutorial_pack_runner.py`
- [X] T012 [P] Add unit tests for workspace/expected-report layout helpers in `/data1/huangzhe/code/auto-quantize-model/tests/unit/test_qwen_tutorial_pack_runner.py`
- [X] T013 [P] Create filesystem-level integration test scaffold in `/data1/huangzhe/code/auto-quantize-model/tests/integration/test_qwen_tutorial_pack_runner_snapshot_verify.py`

**Checkpoint**: Foundation ready — user story implementation can now begin.

---

## Phase 3: User Story 1 - Run either tutorial pack end-to-end without interface drift (Priority: P1) 🎯 MVP

**Goal**: Both tutorial packs delegate to the shared runner, keep the same user-facing flags, and run verify mode end-to-end.

**Independent Test**: From `/data1/huangzhe/code/auto-quantize-model/specs/003-refactor-qwen3-tutorial-runner/quickstart.md`, run both packs’ `run_demo.sh` in verify mode and observe success with no diffs.

- [X] T014 [P] [US1] Implement model configuration registry for 4B/8B and model-id validation errors in `/data1/huangzhe/code/auto-quantize-model/src/auto_quantize_model/qwen/tutorial_pack_runner.py`
- [X] T015 [US1] Implement asset gating (checkpoint link + dataset assets + device validation) in `/data1/huangzhe/code/auto-quantize-model/src/auto_quantize_model/qwen/tutorial_pack_runner.py`
- [X] T016 [US1] Implement scenario execution adapters (all-layers + LM-only) in `/data1/huangzhe/code/auto-quantize-model/src/auto_quantize_model/qwen/tutorial_pack_runner.py`
- [X] T017 [US1] Implement manifest discovery (find `*_quant_manifest.json`) in `/data1/huangzhe/code/auto-quantize-model/src/auto_quantize_model/qwen/tutorial_pack_runner.py`
- [X] T018 [US1] Implement summary generation via `/data1/huangzhe/code/auto-quantize-model/src/auto_quantize_model/qwen/tutorial_pack_summary.py` in `/data1/huangzhe/code/auto-quantize-model/src/auto_quantize_model/qwen/tutorial_pack_runner.py`
- [X] T019 [US1] Implement verify-mode behavior (diff only `summary.json` + `summary.md`, fail-fast) in `/data1/huangzhe/code/auto-quantize-model/src/auto_quantize_model/qwen/tutorial_pack_runner.py`
- [X] T020 [P] [US1] Implement CLI arguments + delegation (preserve stable flags, add `--model-id` and `--expected-report-dir`) in `/data1/huangzhe/code/auto-quantize-model/src/auto_quantize_model/qwen/cli_tutorial_pack_runner.py`
- [X] T021 [P] [US1] Replace 4B tutorial runner with a thin wrapper delegating to the shared CLI in `/data1/huangzhe/code/auto-quantize-model/docs/tutorial/howto/tut-qwen3-vl-4b-layer-sensitivity/run_demo.sh`
- [X] T022 [P] [US1] Replace 8B tutorial runner with a thin wrapper delegating to the shared CLI in `/data1/huangzhe/code/auto-quantize-model/docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/run_demo.sh`
- [X] T023 [P] [US1] Remove pack-local summarizer/sanitizer scripts (now owned by shared runner) under `/data1/huangzhe/code/auto-quantize-model/docs/tutorial/howto/tut-qwen3-vl-4b-layer-sensitivity/scripts/` and `/data1/huangzhe/code/auto-quantize-model/docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/scripts/`
- [X] T024 [P] [US1] Update pack READMEs to reference the shared runner and remove pack-local script references in `/data1/huangzhe/code/auto-quantize-model/docs/tutorial/howto/tut-qwen3-vl-4b-layer-sensitivity/README.md` and `/data1/huangzhe/code/auto-quantize-model/docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/README.md`
- [X] T025 [P] [US1] Add unit tests for CLI parsing and model-id validation in `/data1/huangzhe/code/auto-quantize-model/tests/unit/test_qwen_tutorial_pack_runner.py`
- [ ] T026 [US1] Manually run verify-mode commands from `/data1/huangzhe/code/auto-quantize-model/specs/003-refactor-qwen3-tutorial-runner/quickstart.md` and fix issues in `/data1/huangzhe/code/auto-quantize-model/src/auto_quantize_model/qwen/tutorial_pack_runner.py`

**Checkpoint**: User Story 1 is fully functional and independently testable.

---

## Phase 4: User Story 2 - Select a subset of scenarios for faster iteration (Priority: P2)

**Goal**: Users can select subsets via `--modes` and/or `--quant-pairs`, and only requested scenarios run.

**Independent Test**: Run one pack with `--modes lm_only --quant-pairs wint4_afp16` and confirm only `lm_only/wint4_afp16` is executed and verified.

- [X] T027 [US2] Wire `--modes` and `--quant-pairs` from `/data1/huangzhe/code/auto-quantize-model/src/auto_quantize_model/qwen/cli_tutorial_pack_runner.py` into scenario enumeration in `/data1/huangzhe/code/auto-quantize-model/src/auto_quantize_model/qwen/tutorial_pack_runner.py`
- [X] T028 [P] [US2] Add unit tests for subset selection in `/data1/huangzhe/code/auto-quantize-model/tests/unit/test_qwen_tutorial_pack_runner.py`
- [X] T029 [US2] Validate subset examples (and update if needed) in `/data1/huangzhe/code/auto-quantize-model/specs/003-refactor-qwen3-tutorial-runner/quickstart.md`

**Checkpoint**: User Stories 1 and 2 both work and are independently testable.

---

## Phase 5: User Story 3 - Refresh expected report snapshots and catch regressions safely (Priority: P3)

**Goal**: Snapshot mode refreshes expected summaries safely and verification detects regressions (including degeneracy) with strict, actionable failures.

**Independent Test**: Run one pack with `--snapshot-report`, then run again without it and confirm verify passes with no diffs and non-degeneracy enforced.

- [X] T030 [US3] Implement snapshot mode (summary-only expected snapshots + stale scenario cleanup) in `/data1/huangzhe/code/auto-quantize-model/src/auto_quantize_model/qwen/tutorial_pack_runner.py`
- [X] T031 [P] [US3] Extend filesystem integration tests for snapshot/verify behavior in `/data1/huangzhe/code/auto-quantize-model/tests/integration/test_qwen_tutorial_pack_runner_snapshot_verify.py`
- [X] T032 [P] [US3] Add unit tests for strict verify failures (missing/incomplete expected snapshots) in `/data1/huangzhe/code/auto-quantize-model/tests/unit/test_qwen_tutorial_pack_runner.py`
- [X] T033 [P] [US3] Add unit tests for non-degeneracy gate using fixtures under `/data1/huangzhe/code/auto-quantize-model/tests/unit/fixtures/qwen_tutorial_pack_runner/`
- [X] T034 [US3] Regenerate and commit summary-only expected snapshots under `/data1/huangzhe/code/auto-quantize-model/docs/tutorial/howto/tut-qwen3-vl-4b-layer-sensitivity/expected_report/` and `/data1/huangzhe/code/auto-quantize-model/docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/expected_report/`
- [ ] T035 [US3] Manually run verify-mode for both packs after snapshot refresh and fix any issues in `/data1/huangzhe/code/auto-quantize-model/src/auto_quantize_model/qwen/tutorial_pack_runner.py`

**Checkpoint**: User Stories 1–3 work; snapshot and verify provide a reliable regression gate.

---

## Phase 6: User Story 4 - Add a new Qwen3-VL tutorial pack without copy/paste runners (Priority: P4)

**Goal**: Adding a new Qwen3-VL tutorial pack requires only adding a model registry entry and a thin wrapper.

**Independent Test**: Add a dummy/placeholder model registry entry (no GPU run required) and confirm the runner validates model-id selection and produces an actionable “missing checkpoint link” error for that model.

- [X] T036 [US4] Extract the model registry into `/data1/huangzhe/code/auto-quantize-model/src/auto_quantize_model/qwen/tutorial_pack_registry.py` and update `/data1/huangzhe/code/auto-quantize-model/src/auto_quantize_model/qwen/tutorial_pack_runner.py` to use it
- [X] T037 [P] [US4] Document “how to add a new pack” in `/data1/huangzhe/code/auto-quantize-model/src/auto_quantize_model/qwen/README.md` (required registry fields + wrapper snippet)
- [X] T038 [P] [US4] Add unit tests for registry extension behavior and error messages in `/data1/huangzhe/code/auto-quantize-model/tests/unit/test_qwen_tutorial_pack_runner.py`

**Checkpoint**: All user stories are independently functional.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Quality gates, documentation consistency, and repo hygiene.

- [X] T039 [P] Run `pixi run ruff check .` and fix new lint issues in `/data1/huangzhe/code/auto-quantize-model/src/auto_quantize_model/qwen/` and `/data1/huangzhe/code/auto-quantize-model/tests/`
- [X] T040 [P] Run `pixi run mypy .` and fix new typing issues in `/data1/huangzhe/code/auto-quantize-model/src/auto_quantize_model/qwen/` and `/data1/huangzhe/code/auto-quantize-model/tests/`
- [X] T041 [P] Run `pixi run pytest` and fix new test failures in `/data1/huangzhe/code/auto-quantize-model/tests/unit/` and `/data1/huangzhe/code/auto-quantize-model/tests/integration/`
- [X] T042 [P] Reconcile docs with behavior (shared runner + snapshot semantics) in `/data1/huangzhe/code/auto-quantize-model/specs/003-refactor-qwen3-tutorial-runner/quickstart.md` and the tutorial pack READMEs under `/data1/huangzhe/code/auto-quantize-model/docs/tutorial/howto/`
- [X] T043 Ensure artifact hygiene: no committed files under `/data1/huangzhe/code/auto-quantize-model/tmp/` and expected snapshots contain only `summary.json` + `summary.md` under `/data1/huangzhe/code/auto-quantize-model/docs/tutorial/howto/tut-qwen3-vl-4b-layer-sensitivity/expected_report/` and `/data1/huangzhe/code/auto-quantize-model/docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/expected_report/`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately.
- **Foundational (Phase 2)**: Depends on Setup — blocks all user stories.
- **User Stories (Phase 3–6)**: Depend on Foundational.
- **Polish (Phase 7)**: Depends on all desired user stories.

### User Story Dependency Graph

```text
Phase 2 (Foundational)
  └── US1 (shared runner + wrappers)
        ├── US2 (subset selection)
        ├── US3 (snapshot + strict regression gates)
        └── US4 (new-pack extensibility)
```

---

## Parallel Examples (per User Story)

### US1

```text
In parallel:
- T020 (CLI) in /src/auto_quantize_model/qwen/cli_tutorial_pack_runner.py
- T021 + T022 (wrappers) in the two /docs/tutorial/howto/*/run_demo.sh files
- T024 (README updates) in the two /docs/tutorial/howto/*/README.md files
```

### US2

```text
In parallel:
- T027 (CLI wiring) in /src/auto_quantize_model/qwen/cli_tutorial_pack_runner.py
- T028 (unit tests) in /tests/unit/test_qwen_tutorial_pack_runner.py
```

### US3

```text
In parallel:
- T030 (snapshot logic) in /src/auto_quantize_model/qwen/tutorial_pack_runner.py
- T031 (integration tests) in /tests/integration/test_qwen_tutorial_pack_runner_snapshot_verify.py
- T033 (unit tests + fixtures) in /tests/unit/
```

### US4

```text
In parallel:
- T036 (registry extraction) in /src/auto_quantize_model/qwen/tutorial_pack_registry.py
- T037 (docs) in /src/auto_quantize_model/qwen/README.md
- T038 (tests) in /tests/unit/test_qwen_tutorial_pack_runner.py
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Phase 1: Setup
2. Phase 2: Foundational
3. Phase 3: User Story 1 (MVP)
4. **STOP and VALIDATE** via the verify-mode commands in `/data1/huangzhe/code/auto-quantize-model/specs/003-refactor-qwen3-tutorial-runner/quickstart.md`

### Incremental Delivery

1. Complete Setup + Foundational → shared runner building blocks are ready.
2. Add US1 → validate both packs still run end-to-end.
3. Add US2 → validate subset selection.
4. Add US3 → implement snapshot and strict regression gates; regenerate expected snapshots.
5. Add US4 → make adding new packs a config-only change plus wrapper.
6. Polish → run quality gates and ensure docs/expected snapshots match behavior.
