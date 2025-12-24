# Subtask 1.5: QAT training + checkpointing (W4A16, scratch)

## Scope

- Integrate the W4A16 Brevitas model (Subtask 1.4) into Ultralytics YOLOv10 training from scratch.
- Ensure training outputs match baseline expectations:
  - TensorBoard logs,
  - loss curve artifacts,
  - checkpoints every 5 epochs.
- Checkpointing must be robust if Ultralytics default pickle checkpoints fail for Brevitas modules.

## Planned outputs

- `tmp/.../<run-id>/qat-w4a16/ultralytics/<run-name>/` containing:
  - TB logs, results.csv, weights/checkpoints, and loss plots.
- A training entrypoint subcommand that runs QAT from scratch with consistent flags/paths.

## TODOs

- [ ] Job-001-105-001 Implement an Ultralytics trainer path that injects the quantized model before optimizer creation.
- [ ] Job-001-105-002 Ensure checkpoints save every 5 epochs under the run root (and add a fallback “state_dict checkpoint” saver if needed).
- [ ] Job-001-105-003 Emit loss curve CSV + PNG for QAT training.

## Notes

- Prefer `save_period=5` for parity with baseline. If Brevitas modules break Ultralytics pickle checkpoints, write `state_dict` checkpoints instead.

