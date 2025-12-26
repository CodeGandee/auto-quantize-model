# Phase 3 (US1 yolo10n) — Q&A (Runner, Profiles, and Starting Point)

## How do I run yolo10n with `ema` and `ema+qc`?

Run the Phase-3 runner under Pixi `cu128`, providing `--variant`, `--method`, `--profile`, and a unique `--run-root`:

```bash
pixi install
export RUN_ROOT_BASE=tmp/yolov10-w4a16-ema-qc/$(date +%F_%H-%M-%S)

pixi run -e cu128 python scripts/cv-models/run_yolov10_w4a16_qat_validation.py \
  --variant yolo10n --method ema --profile smoke --seed 0 --device 0 \
  --run-root ${RUN_ROOT_BASE}/yolo10n/ema/seed0

pixi run -e cu128 python scripts/cv-models/run_yolov10_w4a16_qat_validation.py \
  --variant yolo10n --method ema+qc --profile smoke --seed 0 --device 0 \
  --run-root ${RUN_ROOT_BASE}/yolo10n/ema-qc/seed0
```

Each run writes:
- `<run_root>/run_summary.json`
- `<run_root>/summary.md`
- Ultralytics logs under `<run_root>/ultralytics/...` (including `results.csv` and `tensorboard/`)

## Is the example run “full scale training”?

No. The example uses `--profile smoke`, which defaults to a small deterministic COCO subset and a short run (1 epoch).

Use `--profile short` (longer but still subset) or `--profile full` (defaults to 100 epochs; uses the full dataset unless overridden).

## What does the `short` profile do?

The `short` profile is a “medium” validation run:
- Dataset: deterministic COCO subset written under `<run_root>/dataset/`
  - `train_max_images=2048` (deterministic via `--seed`)
  - `val_max_images=256` (deterministic first-N by COCO image id)
- Training: `epochs=10`, `batch=16` by default
- Methods:
  - `ema`: EMA enabled during QAT
  - `ema+qc`: EMA enabled, then a post-hoc QC stage runs after QAT (BN wrapped with learnable `gamma/beta`, base weights frozen, BN stats fixed)

## Is this training “from scratch” or “fine-tuning”?

It is fine-tuning from a pretrained checkpoint.

## Where does the base model (starting point) come from?

From pretrained YOLOv10 checkpoint files in this repo, referenced by the config:
- `models/yolo10/checkpoints/yolov10n.pt` (yolo10n)
- `models/yolo10/checkpoints/yolov10s.pt` (yolo10s)
- `models/yolo10/checkpoints/yolov10m.pt` (yolo10m, used later when gated)

The runner passes the checkpoint path to Ultralytics as `model=...`, so it loads those weights as the starting point.

## Do I need to run PTQ before QAT?

Not as a separate step for W4A16 in this workflow.

The trainer loads the pretrained checkpoint and then immediately inserts Brevitas W4 fake-quant modules before running QAT.
Because this is weight-only quantization (activations stay floating), there is no activation calibration step required.

