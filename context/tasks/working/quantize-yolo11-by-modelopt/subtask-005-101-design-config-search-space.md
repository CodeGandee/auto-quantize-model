# Subtask 5.1: Design configuration search space

## Scope

Identify the key quantization and engine-building knobs for YOLO11 (ModelOpt and TensorRT) and design a small, tractable configuration grid that can be explored within reasonable compute time.

## Planned outputs

- A list of candidate knobs to vary (e.g., calibration size, excluded layers/ops, TensorRT builder flags, precision combinations).
- A concrete experiment grid or plan specifying which configurations will be run.
- Documentation of any constraints or assumptions (e.g., time budget, hardware availability).

## TODOs

- [ ] Job-005-101-001: Enumerate the most impactful ModelOpt and TensorRT configuration knobs for this project, drawing from earlier subtasks and the ModelOpt knowledge base.
- [ ] Job-005-101-002: Define a small grid of configurations (e.g., different calibration sizes and sets of layers left in FP16) that fits within the expected compute budget.
- [ ] Job-005-101-003: Record the planned configuration grid and rationale in this subtask file or a linked context document.

## Notes

- Keep the grid small enough to be practical while still covering interesting corners of the trade-off space.

