# Phase Integration Guide: Revise Qwen3-VL-8B Tutorial Pack (Introduce + Layer Sensitivity)

**Feature**: `002-revise-qwen3-vl-tutorial` | **Phases**: 6

## Overview

This feature turns the existing Qwen3-VL-8B tutorial pack into a stable, end-to-end workflow that runs **four scenarios** (2 modes × 2 quant pairs) with meaningful defaults and verifies results by diffing **only schema-locked, sanitized summaries**.

The central integration point is `docs/.../run_demo.sh`, which orchestrates scenario runs, generates summaries, and performs snapshot/verify workflows against `expected_report/`.

## Phase Flow

**MUST HAVE: End-to-End Sequence Diagram**

```mermaid
sequenceDiagram
    participant U as User
    participant RD as run_demo.sh<br/>(bash)
    participant AL as all_layers runner<br/>(Python)
    participant LM as lm_only runner<br/>(Hydra)
    participant SM as summarize_manifest.py<br/>(Python)
    participant SA as sanitize_artifacts.py<br/>(Python)
    participant FS as Filesystem

    Note over U,FS: Phase 1: Setup configs
    U->>FS: add Hydra model YAMLs

    Note over U,FS: Phase 2: Foundational runner + summaries
    U->>RD: run_demo.sh<br/>options
    RD->>FS: ensure checkpoint link
    RD->>FS: ensure dataset assets

    Note over U,FS: Phase 3–4: Scenario execution
    RD->>AL: run all_layers<br/>per quant_pair
    AL-->>FS: outputs + manifest
    RD->>LM: run lm_only<br/>per quant_pair
    LM-->>FS: outputs + manifest

    Note over U,FS: Phase 2+: Summaries + verification
    RD->>SM: summarize each manifest
    SM-->>FS: summary.json + summary.md

    alt snapshot mode
        RD->>SA: sanitize artifacts
        SA-->>FS: expected_report updates
        RD-->>U: snapshot complete
    else verify mode
        RD->>FS: diff summaries vs expected_report
        RD-->>U: success or actionable diffs
    end
```

## Artifact Flow Between Phases

```mermaid
graph TD
  subgraph P1["Phase 1: Setup"]
    P1A[T001–T002: Hydra model configs] --> CFG[conf/model/qwen3_vl_8b_instruct/...];
  end

  subgraph P2["Phase 2: Foundational"]
    SB[T003: Summary builder] --> SUM[src/auto_quantize_model/qwen/tutorial_pack_summary.py];
    SD[T005: Summarizer wrapper] --> SM[docs/.../scripts/summarize_manifest.py];
    RDCLI[T006–T009: run_demo core] --> RD[docs/.../run_demo.sh];
  end

  subgraph P3["Phase 3: US1 all_layers"]
    ALR[T010–T012: all_layers runner] --> ALPY[models/.../run_qwen3_vl_4b_autoquant_all_layers.py];
  end

  subgraph P4["Phase 4: US2 lm_only"]
    LMR[T016–T017: Hydra runner wiring] --> LMPY[scripts/qwen/qwen3_lm_sensitivity.py];
  end

  subgraph P5["Phase 5: US5 snapshot/verify"]
    SNAP[T020–T021: snapshot+verify gates] --> EXP[docs/.../expected_report/...];
  end

  CFG -.->|selected by| LMPY;
  SUM -.->|used by| SM;
  RD -.->|calls| ALPY;
  RD -.->|calls| LMPY;
  ALPY --> OUTA[tmp/.../outputs/all_layers/...];
  LMPY --> OUTL[tmp/.../outputs/lm_only/...];
  OUTA -.->|manifest in| SM;
  OUTL -.->|manifest in| SM;
  SM --> SUMA[tmp/.../summaries/...];
  SUMA -.->|diffs against| EXP;
```

## System Architecture

```mermaid
classDiagram
    class RunDemo {
        +parse_args()
        +ensure_assets()
        +run_scenarios()
        +summarize()
        +verify_or_snapshot()
    }

    class AllLayersRunner {
        +main(args) int
        +build_vlm_calib_batches()
        +build_quant_manifest()
    }

    class LmOnlyRunner {
        +main(cfg) void
        +run_qwen3_vl_lm_autoquant_sensitivity()
    }

    class TutorialPackSummary {
        +from_manifest(manifest,scenario_id,mode,quant_pair,dataset_size)
        +to_dict() dict
        +write_summary_json(path)
        +write_summary_md(path)
    }

    RunDemo --> AllLayersRunner: executes
    RunDemo --> LmOnlyRunner: executes
    RunDemo --> TutorialPackSummary: verifies via summaries
```

## Use Cases

```mermaid
graph LR
    Actor((User))
    UC1[Run tutorial verify]
    UC2[Run subset scenarios]
    UC3[Refresh expected snapshot]

    Actor --> UC1
    Actor --> UC2
    Actor --> UC3

    UC3 -.->|enables| UC1
```

## Activity Flow

```mermaid
stateDiagram-v2
    [*] --> Setup
    Setup --> Foundation
    Foundation --> RunScenarios
    RunScenarios --> Summarize

    Summarize --> Verify: default
    Summarize --> Snapshot: --snapshot-report

    Verify --> [*]: match expected
    Verify --> [*]: diff or degenerate
    Snapshot --> [*]: expected updated
```

## Inter-Phase Dependencies

### Phase 1 → Phase 4

**Artifacts**:
- `conf/model/qwen3_vl_8b_instruct/arch/qwen3_vl_8b_instruct.default.yaml` enables selecting/overriding 8B model in Hydra runner.

**Code Dependencies**:

```python
# scripts/qwen/qwen3_lm_sensitivity.py consumes cfg.model.path / cfg.model.name
model_dir = Path(str(cfg.model.path))
```

### Phase 2 → All phases

**Artifacts**:
- `src/auto_quantize_model/qwen/tutorial_pack_summary.py` (schema-locked summaries)
- `docs/.../scripts/summarize_manifest.py` (CLI wrapper)
- `docs/.../run_demo.sh` (orchestrator)

**Critical contract**:
- Summary schema: `specs/002-revise-qwen3-vl-tutorial/contracts/summary.schema.json`

### Phase 3 & 4 → Phase 5

**Artifacts**:
- Per-scenario summaries under `<workspace>/tmp/.../summaries/<mode>/<quant_pair>/summary.{json,md}`
- Golden summaries under `docs/.../expected_report/<mode>/<quant_pair>/summary.{json,md}`

**Requirement**:
- Verification diffs only these summary files and enforces `has_nonzero_sensitivity=true`.

## Integration Testing

```bash
# Deterministic logic
pixi run pytest tests/unit/test_qwen_tutorial_pack_summary.py

# End-to-end (GPU recommended; may be slow)
bash docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/run_demo.sh
```

## Critical Integration Points

1. **Asset gating is correct and actionable**
   - Missing checkpoint link or datasets should fail early with explicit paths and remediation commands.

2. **Scenario identity is stable**
   - `scenario_id` must stay stable across runs (e.g., `all_layers/wint4_afp16`).

3. **Summary schema is locked**
   - `summary.json` must match `specs/002-revise-qwen3-vl-tutorial/contracts/summary.schema.json`.

4. **Non-degeneracy is enforced**
   - Verification must fail when any scenario has all-zero sensitivities (exact `0.0`).

## References

- Individual phase guides:
  - `context/tasks/working/002-revise-qwen3-vl-tutorial/impl-phase-1-setup.md`
  - `context/tasks/working/002-revise-qwen3-vl-tutorial/impl-phase-2-foundational.md`
  - `context/tasks/working/002-revise-qwen3-vl-tutorial/impl-phase-3-us1-all-layers.md`
  - `context/tasks/working/002-revise-qwen3-vl-tutorial/impl-phase-4-us2-lm-only.md`
  - `context/tasks/working/002-revise-qwen3-vl-tutorial/impl-phase-5-us5-snapshot-verify.md`
  - `context/tasks/working/002-revise-qwen3-vl-tutorial/impl-phase-6-polish.md`
- Spec: `specs/002-revise-qwen3-vl-tutorial/spec.md`
- Tasks breakdown: `specs/002-revise-qwen3-vl-tutorial/tasks.md`
- Data model: `specs/002-revise-qwen3-vl-tutorial/data-model.md`
- Contracts: `specs/002-revise-qwen3-vl-tutorial/contracts/`

