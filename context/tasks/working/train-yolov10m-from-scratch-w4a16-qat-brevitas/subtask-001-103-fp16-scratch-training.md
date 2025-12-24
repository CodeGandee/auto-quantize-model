# Subtask 1.3: FP16 scratch training (baseline)

## Scope

- Train YOLOv10m from random init on COCO2017 using:
  - model cfg: `models/yolo10/src/ultralytics/cfg/models/v10/yolov10m.yaml`
  - dataset YAML from Subtask 1.1
  - hyperparams from Subtask 1.2
- Ensure:
  - AMP/FP16 baseline training works,
  - TensorBoard event logs are written under the run root,
  - checkpoints are saved **every 5 epochs** under the run root,
  - a loss curve artifact (CSV + PNG) is produced.

## Planned outputs

- `tmp/.../<run-id>/fp16/ultralytics/<run-name>/` containing:
  - `events.out.tfevents...`
  - `results.csv`
  - `weights/` (including `last.pt` + `epoch*.pt` every 5 epochs)
  - `loss/` (CSV + PNG)

## TODOs

- [ ] Job-001-103-001 Implement a baseline training entrypoint that runs in `pixi run -e rtx5090`.
- [ ] Job-001-103-002 Ensure checkpoints are written every 5 epochs (`save_period=5`) under `tmp/.../<run-id>/`.
- [ ] Job-001-103-003 Ensure TensorBoard logs are present (TB can be launched pointing at the run dir).
- [ ] Job-001-103-004 Emit a standalone loss curve graph (PNG) from `results.csv`.

## Notes

- Use YOLOv10-specific trainer/validator classes from the local `models/yolo10/src/` fork so validation does not crash on dict preds.

