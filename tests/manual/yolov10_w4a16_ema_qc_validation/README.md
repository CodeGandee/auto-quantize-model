# Manual Validation: YOLOv10 W4A16 QAT Stability (EMA + QC)

This manual test validates the end-to-end workflow on a GPU.

All commands MUST run via Pixi `cu128`:

```bash
pixi run -e cu128 python ...
```

## Prerequisites

- Pixi env installed: `pixi install`
- COCO2017 available under `datasets/coco2017/source-data/`
- Pretrained checkpoints present:
  - `models/yolo10/checkpoints/yolov10n.pt`
  - `models/yolo10/checkpoints/yolov10s.pt` (Phase 4)
  - `models/yolo10/checkpoints/yolov10m.pt` (Phase 5; gated)

## Smoke Run Template (Phase 3 fills details)

```bash
export RUN_ROOT_BASE=tmp/yolov10-w4a16-ema-qc/$(date +%F_%H-%M-%S)

pixi run -e cu128 python scripts/cv-models/run_yolov10_w4a16_qat_validation.py \
  --variant yolo10n --method baseline --profile smoke --seed 0 --device 0 \
  --run-root ${RUN_ROOT_BASE}/yolo10n/baseline/seed0
```

## Phase 3: yolo10n smoke procedure (baseline vs EMA vs EMA+QC)

Run each method for seeds 0 and 1:

```bash
export RUN_ROOT_BASE=tmp/yolov10-w4a16-ema-qc/$(date +%F_%H-%M-%S)

for seed in 0 1; do
  pixi run -e cu128 python scripts/cv-models/run_yolov10_w4a16_qat_validation.py \
    --variant yolo10n --method baseline --profile smoke --seed ${seed} --device 0 \
    --run-root ${RUN_ROOT_BASE}/yolo10n/baseline/seed${seed}

  pixi run -e cu128 python scripts/cv-models/run_yolov10_w4a16_qat_validation.py \
    --variant yolo10n --method ema --profile smoke --seed ${seed} --device 0 \
    --run-root ${RUN_ROOT_BASE}/yolo10n/ema/seed${seed}

  pixi run -e cu128 python scripts/cv-models/run_yolov10_w4a16_qat_validation.py \
    --variant yolo10n --method ema+qc --profile smoke --seed ${seed} --device 0 \
    --run-root ${RUN_ROOT_BASE}/yolo10n/ema-qc/seed${seed}
done
```

Generate a combined comparison report:

```bash
pixi run -e cu128 python scripts/cv-models/summarize_yolov10_w4a16_qat_validation.py \
  --run-roots \
    ${RUN_ROOT_BASE}/yolo10n/baseline/seed0 \
    ${RUN_ROOT_BASE}/yolo10n/ema/seed0 \
    ${RUN_ROOT_BASE}/yolo10n/ema-qc/seed0 \
    ${RUN_ROOT_BASE}/yolo10n/baseline/seed1 \
    ${RUN_ROOT_BASE}/yolo10n/ema/seed1 \
    ${RUN_ROOT_BASE}/yolo10n/ema-qc/seed1 \
  --out-path ${RUN_ROOT_BASE}/yolo10n/comparison/summary.md
```

Expected outputs under each run root:

- `run_summary.json`
- `summary.md`
