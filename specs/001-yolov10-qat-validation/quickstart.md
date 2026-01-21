# Quickstart: Validate EMA + QC on YOLOv10 W4A16 QAT (n → s → m)

This quickstart describes how to run the validation ladder once the feature is implemented.

## Prerequisites

- Pixi environment installed: `pixi install`
- Run and development environment: `pixi run -e cu128 ...`
- COCO2017 source data available under `datasets/coco2017/source-data/`
- Pretrained checkpoints present:
  - `models/yolo10/checkpoints/yolov10n.pt`
  - `models/yolo10/checkpoints/yolov10s.pt`
  - `models/yolo10/checkpoints/yolov10m.pt` (only used after gating)

## 1) Run yolo10n validation (smoke)

Run baseline vs EMA vs EMA+QC with at least two seeds.

```bash
# Use the Pixi `cu128` environment and a GPU device.
export RUN_ROOT_BASE=tmp/yolov10-w4a16-ema-qc/$(date +%F_%H-%M-%S)

pixi run -e cu128 python scripts/cv-models/run_yolov10_w4a16_qat_validation.py \
  --variant yolo10n --method baseline --profile smoke --seed 0 --device 0 \
  --run-root ${RUN_ROOT_BASE}/yolo10n/baseline/seed0

pixi run -e cu128 python scripts/cv-models/run_yolov10_w4a16_qat_validation.py \
  --variant yolo10n --method ema --profile smoke --seed 0 --device 0 \
  --run-root ${RUN_ROOT_BASE}/yolo10n/ema/seed0

pixi run -e cu128 python scripts/cv-models/run_yolov10_w4a16_qat_validation.py \
  --variant yolo10n --method ema+qc --profile smoke --seed 0 --device 0 \
  --run-root ${RUN_ROOT_BASE}/yolo10n/ema-qc/seed0

# Repeat seeds (minimum 2).
pixi run -e cu128 python scripts/cv-models/run_yolov10_w4a16_qat_validation.py \
  --variant yolo10n --method ema+qc --profile smoke --seed 1 --device 0 \
  --run-root ${RUN_ROOT_BASE}/yolo10n/ema-qc/seed1
```

Check `run_summary.json` in each run root for:

- `metrics.primary_name` (expected: `metrics/mAP50-95(B)`)
- `metrics.best_value`, `metrics.final_value`
- `stability.collapsed` (expected: `false` for EMA+QC)

## 2) Run yolo10s validation (short)

```bash
pixi run -e cu128 python scripts/cv-models/run_yolov10_w4a16_qat_validation.py \
  --variant yolo10s --method ema+qc --profile short --seed 0 --device 0 \
  --run-root ${RUN_ROOT_BASE}/yolo10s/ema-qc/seed0

pixi run -e cu128 python scripts/cv-models/run_yolov10_w4a16_qat_validation.py \
  --variant yolo10s --method ema+qc --profile short --seed 1 --device 0 \
  --run-root ${RUN_ROOT_BASE}/yolo10s/ema-qc/seed1
```

## 3) Decide whether yolo10m is allowed

The yolo10m stage gate is “passed” only if:

- yolo10n EMA+QC: 2/2 runs not collapsed
- yolo10s EMA+QC: 2/2 runs not collapsed
- runs are comparable (dataset selection + evaluation settings match)

The gating decision is recorded in each run’s `summary.md` and in any consolidated comparison report produced.

## 4) (Only if passed) Run yolo10m

```bash
pixi run -e cu128 python scripts/cv-models/run_yolov10_w4a16_qat_validation.py \
  --variant yolo10m --method ema+qc --profile full --seed 0 --device 0 \
  --run-root ${RUN_ROOT_BASE}/yolo10m/ema-qc/seed0
```
