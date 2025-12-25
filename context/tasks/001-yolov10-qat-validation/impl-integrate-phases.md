# Phase Integration Guide: Validate YOLOv10 W4A16 QAT Stability (EMA + QC)

**Feature**: `001-yolov10-qat-validation` | **Phases**: 6

## Overview

This feature builds a reproducible validation ladder for the WACV’24 stabilization method (EMA + post-hoc QC) under W4A16 QAT, prioritizing fast correctness checks on `yolo10n` and `yolo10s`. It produces standardized run artifacts (`run_summary.json`, `summary.md`) and a comparison report, and it enforces a stage gate that blocks `yolo10m` until the smaller variants pass stability and repeatability criteria.

The implementation is structured so core logic lives in `src/auto_quantize_model/cv_models/` and user-facing orchestration is via scripts under `scripts/cv-models/`, with all commands executed in Pixi `cu128`.

## Phase Flow

**End-to-End Sequence Diagram**

```mermaid
sequenceDiagram
    participant U as User
    participant RUN as scripts/cv-models<br/>run_yolov10_w4a16_qat_validation.py
    participant CFG as conf/cv-models<br/>yolov10_w4a16_validation/
    participant DS as src/auto_quantize_model/cv_models<br/>yolov10_coco_subset_dataset.py
    participant TR as Ultralytics + Brevitas<br/>W4A16 QAT
    participant QC as src/auto_quantize_model/cv_models<br/>yolov10_qc.py
    participant CSV as src/auto_quantize_model/cv_models<br/>yolov10_results_csv.py
    participant GT as stage gate<br/>(yolo10m)
    participant SUM as scripts/cv-models<br/>summarize_yolov10_w4a16_qat_validation.py
    participant FS as tmp/<run>/

    Note over U,FS: Phase 1–2: Setup<br/>+ Foundation
    U->>RUN: choose variant/method<br/>+ profile
    RUN->>CFG: load config<br/>(profile/variant/method)

    Note over U,FS: Phase 3: US1<br/>(yolo10n)
    RUN->>DS: build COCO subset<br/>YAML + provenance
    DS-->>RUN: dataset_yaml<br/>+ provenance_json
    RUN->>TR: run W4A16 QAT<br/>(baseline/ema)
    TR-->>RUN: results.csv + model<br/>(+ EMA)
    alt method: ema+qc
        RUN->>QC: insert QC modules<br/>+ train QC (1 epoch)<br/>(BN stats fixed)
        QC-->>RUN: QC result
    end
    RUN->>CSV: read metric series<br/>from results.csv
    CSV-->>RUN: best/final + collapsed?
    RUN->>FS: write run_summary.json<br/>+ summary.md

    Note over U,FS: Phase 4: US2<br/>(yolo10s)
    U->>RUN: run yolo10s ema+qc<br/>(seed 0, 1)
    RUN->>FS: write yolo10s summaries

    Note over U,FS: Phase 5: US3<br/>(gate yolo10m)
    U->>SUM: summarize run roots
    Note over SUM,GT: requires 2/2 stable EMA+QC<br/>for yolo10n + yolo10s
    SUM->>GT: evaluate gate
    GT-->>SUM: decision + reason
    SUM-->>U: combined summary.md

    alt gate passed
        U->>RUN: request yolo10m run<br/>(--gate-root)
        RUN->>GT: verify gate
        RUN->>TR: run yolo10m QAT
        RUN->>FS: write yolo10m artifacts
    else gate failed
        RUN-->>U: refuse yolo10m<br/>(reason)
        RUN->>FS: write summary.md<br/>(gate failure)
    end
```

## Artifact Flow Between Phases

```mermaid
graph TD
    subgraph P1["Phase 1: Setup"]
        P1C1[conf/cv-models<br/>yolov10_w4a16_validation/*]:::file
        P1S1[scripts/cv-models<br/>run_yolov10_w4a16_qat_validation.py]:::file
        P1S2[scripts/cv-models<br/>summarize_yolov10_w4a16_qat_validation.py]:::file
    end

    subgraph P2["Phase 2: Foundational"]
        P2L1[src/auto_quantize_model/cv_models<br/>yolov10_coco_subset_dataset.py]:::file
        P2L2[src/auto_quantize_model/cv_models<br/>yolov10_results_csv.py]:::file
        P2L3[src/auto_quantize_model/cv_models<br/>yolov10_stability.py]:::file
        P2L4[src/auto_quantize_model/cv_models<br/>yolov10_w4a16_validation.py]:::file
    end

    subgraph Runs["Run Artifacts (tmp/)"]
        A1[tmp/...<br/>run_summary.json]:::artifact
        A2[tmp/...<br/>summary.md]:::artifact
        A3[tmp/.../ultralytics/...<br/>results.csv]:::artifact
        A4[tmp/.../dataset/...<br/>coco_yolo_subset.yaml]:::artifact
        A5[tmp/.../dataset/...<br/>provenance.json]:::artifact
    end

    subgraph Report["Comparison Report"]
        R1[tmp/.../combined<br/>summary.md]:::artifact
    end

    P1C1 --> P2L4
    P1S1 --> P2L4
    P2L1 --> A4
    P2L1 --> A5
    P1S1 --> A3
    P2L2 --> A1
    P2L3 --> A1
    P2L4 --> A1
    P2L4 --> A2
    A1 --> R1

    classDef file fill:#eef,stroke:#88a,stroke-width:1px;
    classDef artifact fill:#efe,stroke:#8a8,stroke-width:1px;
```

## System Architecture

```mermaid
classDiagram
    class RunnerCLI {
        +main(argv) int
        +run_one(variant, method, profile, run_root) int
    }

    class ValidationConfig {
        +variant: str
        +method: str
        +profile: str
        +coco_root: Path
        +run_root: Path
    }

    class DatasetBuilder {
        +prepare_coco2017_yolo_subset_dataset(...) CocoSubsetYoloDataset
    }

    class ResultsCsvParser {
        +read_metric_series(results_csv, metric_name) list[MetricPoint]
    }

    class StabilityEvaluator {
        +classify_collapse(series, threshold_ratio) (bool, float)
    }

    class QcTrainer {
        +insert_qc_modules(model) int
        +run_qc_training(model, train_batches, device, lr, epochs) QcRunResult
    }

    class StageGate {
        +evaluate_yolo10m_gate(run_summaries) StageGateDecision
    }

    class SummarizerCLI {
        +main(argv) int
        +write_combined_summary(run_roots, out_path) None
    }

    RunnerCLI --> ValidationConfig
    RunnerCLI --> DatasetBuilder
    RunnerCLI --> ResultsCsvParser
    RunnerCLI --> StabilityEvaluator
    RunnerCLI --> QcTrainer
    RunnerCLI --> StageGate
    SummarizerCLI --> StageGate
```

## Use Cases

```mermaid
graph LR
    Actor((Quantization Engineer))

    UC1[Phase 1–2:<br/>Setup + Foundation]
    UC2[US1: Run yolo10n<br/>baseline/ema/ema+qc]
    UC3[US2: Run yolo10s<br/>ema+qc]
    UC4[US3: Evaluate gate<br/>for yolo10m]
    UC5[Optional: Run yolo10m ema+qc<br/>after gate]

    Actor --> UC1
    Actor --> UC2
    Actor --> UC3
    Actor --> UC4
    Actor --> UC5

    UC1 -. prerequisite .-> UC2
    UC2 -. prerequisite .-> UC3
    UC3 -. prerequisite .-> UC4
    UC4 -. prerequisite .-> UC5
```

## Activity Flow

```mermaid
stateDiagram-v2
    [*] --> Setup
    Setup --> Foundation: config + CLIs created
    Foundation --> US1_yolo10n: dataset builder + parsing + run summaries
    US1_yolo10n --> US2_yolo10s: yolo10n validated
    US2_yolo10s --> GateCheck: yolo10s validated
    GateCheck --> yolo10m_allowed: gate passed
    GateCheck --> yolo10m_blocked: gate failed
    yolo10m_allowed --> [*]
    yolo10m_blocked --> [*]
```

## Inter-Phase Dependencies

### Phase 1 → Phase 2

**Artifacts**

- `conf/cv-models/yolov10_w4a16_validation/*` created in Phase 1 are read by Phase 2/3 code.
- `scripts/cv-models/run_yolov10_w4a16_qat_validation.py` exists so Phase 2 can wire config and artifact writers into a real entrypoint.

**Code Dependencies**

```python
# scripts/cv-models/run_yolov10_w4a16_qat_validation.py uses Phase-2 helpers.
from auto_quantize_model.cv_models.yolov10_w4a16_validation import load_validation_config
```

### Phase 2 → Phase 3 (US1)

**Artifacts**

- Dataset builder outputs used by training:
  - `tmp/<run>/dataset/.../coco_yolo_subset.yaml`
  - `tmp/<run>/dataset/.../provenance.json`

**Code Dependencies**

```python
from auto_quantize_model.cv_models.yolov10_coco_subset_dataset import prepare_coco2017_yolo_subset_dataset
from auto_quantize_model.cv_models.yolov10_results_csv import read_metric_series
from auto_quantize_model.cv_models.yolov10_stability import classify_collapse
```

### Phase 3 → Phase 4 (US2)

**Artifacts**

- yolo10n run roots under `tmp/` containing:
  - `run_summary.json`
  - `summary.md`

These are inputs for the combined summarizer used in Phase 4 and the gate used in Phase 5.

### Phase 4 → Phase 5 (US3)

**Artifacts**

- yolo10s EMA+QC run roots (two seeds) with `run_summary.json`.
- Combined report may be written to a chosen `tmp/.../combined/summary.md`.

## Integration Testing

```bash
# Core unit + integration tests (no GPU training).
pixi run -e cu128 pytest tests/unit/
pixi run -e cu128 pytest tests/integration/

# Optional: run the full manual workflow on GPU.
pixi run -e cu128 python tests/manual/yolov10_w4a16_ema_qc_validation/README.md
```

## Critical Integration Points

1. **Dataset comparability**
   - Runs must record dataset selection/provenance in `run_summary.json` so the summarizer and gate can reject non-comparable runs.
2. **Metric parsing correctness**
   - `results.csv` parsing must be robust (duplicate rows, missing values) so collapse decisions are stable and reproducible.
3. **BN handling during QC**
   - QC must keep BN running statistics fixed, otherwise QC can chase moving stats and mask oscillation issues.
4. **Gate enforcement UX**
   - yolo10m must be refused with a clear reason, and that reason must be recorded in `summary.md` for reviewability.

## References

- Individual phase guides: `context/tasks/001-yolov10-qat-validation/impl-phase-*.md`
- Spec: `specs/001-yolov10-qat-validation/spec.md`
- Tasks: `specs/001-yolov10-qat-validation/tasks.md`
- Data model: `specs/001-yolov10-qat-validation/data-model.md`
- Contracts: `specs/001-yolov10-qat-validation/contracts/`
