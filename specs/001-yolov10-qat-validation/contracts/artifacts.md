# Contract: Run Artifact Layout (W4A16 QAT Validation)

This contract defines the minimum file layout that a run MUST produce to be considered “comparable” and eligible for automated gating.

## Minimum layout

```text
<run_root>/
├── run_summary.json          # machine-readable; schema in contracts/run_summary.schema.json
├── summary.md                # human-readable; includes stability outcome
└── ultralytics/              # optional but recommended
    └── <run_name>/
        ├── args.yaml
        ├── results.csv
        └── tensorboard/
```

## Required fields (comparability)

Runs are comparable only if their `run_summary.json` match on:

- `model.variant`
- `quantization.mode`
- `method.variant`
- `dataset.dataset_yaml` (or equivalent provenance fields when YAML differs but is logically identical)
- `training.imgsz`

## Curated reports

If results are promoted to a curated report for review, the report MUST live under:

- `models/yolo10/reports/<run-id>/`

and include:

- `summary.md` (top-level, links to run roots and key plots)
