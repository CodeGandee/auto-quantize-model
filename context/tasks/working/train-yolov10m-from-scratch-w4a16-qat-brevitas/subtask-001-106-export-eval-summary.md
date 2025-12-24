# Subtask 1.6: Export + eval + summary (incl. loss graph)

## Scope

- Export baseline and QAT models to ONNX under the run root.
  - Baseline: raw head ONNX (no NMS), consistent with `scripts/cv-models/eval_yolov10m_onnx_coco.py`.
  - QAT: Brevitas QCDQ ONNX.
- Evaluate both ONNX models on COCO2017 val with the same evaluator contract and write metrics JSON.
- Produce a run-local summary and a final loss graph artifact.

## Planned outputs

- `tmp/.../<run-id>/onnx/`:
  - `yolov10m-baseline-*.onnx`
  - `yolov10m-w4a16-qcdq-qat*.onnx`
- `tmp/.../<run-id>/eval/`:
  - `fp16/metrics.json`
  - `qat-w4a16/metrics.json`
- `tmp/.../<run-id>/summary/`:
  - `summary.md`
  - `loss_curve_comparison.png`

## TODOs

- [ ] Job-001-106-001 Export baseline and QAT ONNX artifacts (and record I/O contracts).
- [ ] Job-001-106-002 Run COCO val eval for both ONNX models and write metrics JSON.
- [ ] Job-001-106-003 Generate `summary.md` and a final loss graph artifact from the training logs.

## Notes

- Keep evaluation provider list configurable but default to CUDA EP preferred.

