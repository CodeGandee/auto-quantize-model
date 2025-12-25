# Contract: CLI for YOLOv10 W4A16 QAT Validation

This feature is implemented as local CLI entrypoints (not a network service). This document defines the CLI contract that downstream automation (sweeps, comparison scripts, reports) can rely on.

## Primary command

`scripts/cv-models/run_yolov10_w4a16_qat_validation.py`

Run under the Pixi `cu128` environment:

- `pixi run -e cu128 python scripts/cv-models/run_yolov10_w4a16_qat_validation.py ...`

## Arguments (contract)

- `--variant`: `yolo10n | yolo10s | yolo10m`
- `--method`: `baseline | ema | ema+qc`
- `--profile`: `smoke | short | full`
- `--run-root`: output root directory (expected to be under `tmp/`)
- `--coco-root`: COCO2017 source root (default: `datasets/coco2017/source-data`)
- `--imgsz`: integer, default 640
- `--epochs`: integer (profile default)
- `--batch`: integer (profile default)
- `--device`: Ultralytics device string (e.g. `0`, `0,1`, `cpu`)
- `--workers`: integer
- `--seed`: integer
- `--amp / --no-amp`: enable/disable AMP

## Outputs (contract)

The command MUST write a machine-readable summary file:

- `${run_root}/run_summary.json` (schema: `contracts/run_summary.schema.json`)

The command MUST write a human-readable comparison-friendly summary file:

- `${run_root}/summary.md`

The command SHOULD write Ultralytics-native logs (for debugging):

- `${run_root}/ultralytics/.../results.csv`
- `${run_root}/ultralytics/.../tensorboard/`

## Exit codes (contract)

- `0`: run completed (success or failed-by-collapse; status is recorded in `run_summary.json`)
- `1`: run could not start or crashed (status should be recorded as `incomplete` when possible)
