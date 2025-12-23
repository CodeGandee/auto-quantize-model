# Subtask 1.6: Orchestration + ONNX optimization + summary + quality gates

## Scope

- Provide a single reproducible entrypoint to run:
  - baseline export + smoke + eval,
  - PTQ exports (W4A16-like, W4A8 INT8 activations) + smoke + eval,
  - optional QAT + export + eval,
  - ONNX graph cleanup with `onnxoptimizer` (without removing Q/DQ nodes),
  - a small `summary.md` comparing accuracy + latency across variants.
- Update runbook docs for the public YOLOv10 assets.
- Run repository quality gates (ruff + mypy) for any introduced code.

## Planned outputs

- `scripts/cv-models/run_yolov10m_brevitas_w4_ptq_qat.sh` end-to-end runner writing into `tmp/yolov10m_brevitas_w4a8_w4a16/<run-id>/`.
- Optimized ONNX artifacts in `tmp/.../<run-id>/onnx/` (retain Q/DQ nodes).
- `tmp/.../<run-id>/summary.md` with a small comparison table.
- A runbook update under `models/yolo10/README.md` with commands and caveats for PTQ/QAT Brevitas exports.

## Dataset plan (runner contract)

- The runner should standardize dataset usage across baseline/PTQ/QAT so comparisons are meaningful:
  - **PTQ calibration (W4A8)**: default `datasets/quantize-calib/quant100.txt` (COCO train2017 images).
  - **Evaluation**: COCO2017 val subset via `scripts/cv-models/eval_yolov10m_onnx_coco.py`:
    - either `--max-images 100` (deterministic first-N IDs), or
    - a shared `--image-ids-list` file under the run root used for all evaluated models.

## TODOs

- [ ] Job-001-106-001 Implement the runner script that creates `RUN_ROOT`, captures logs, and calls the baseline/PTQ/QAT steps.
- [ ] Job-001-106-002 Add `onnxoptimizer` cleanup passes and verify Q/DQ nodes remain in the graph (inspect counts or spot-check).
- [ ] Job-001-106-003 Run baseline/PTQ/(optional QAT) COCO subset evaluations and write `summary.md` under the run root.
- [ ] Job-001-106-004 Update `models/yolo10/README.md` with a “Brevitas PTQ/QAT W4A16/W4A8 (INT8)” runbook section.
- [ ] Job-001-106-005 Run `pixi run -e rtx5090 ruff check .` and `pixi run -e rtx5090 mypy .`, fixing only issues introduced by this task.

## Notes

- Keep outputs under `tmp/` (ignored by Git); only scripts/docs go into the repo.

## Summary

- Added an end-to-end runner: `scripts/cv-models/run_yolov10m_brevitas_w4_ptq_qat.sh` (baseline + PTQ, optional QAT) writing into `tmp/yolov10m_brevitas_w4a8_w4a16/<run-id>/`.
- Implemented conservative ONNX cleanup while keeping Q/DQ nodes: `src/auto_quantize_model/cv_models/yolov10_brevitas.py` (`optimize_onnx_keep_qdq`), emitting `*-opt.onnx` variants.
- Updated runbook: `models/yolo10/README.md` (Brevitas PTQ/QAT W4A16/W4A8 commands + caveats).
- Added a dataset-aware summary writer: `scripts/cv-models/write_yolov10m_brevitas_summary.py` (includes evaluation dataset details, calibration lists, and Ultralytics-trainer QAT log paths).
- Produced updated comparison summary with W8 sanity checks + Ultralytics-trainer QAT: `tmp/yolov10m_brevitas_w4a8_w4a16/2025-12-23_16-12-40/summary.md`
  - baseline `mAP_50_95=0.6022` (FP32)
  - PTQ W8A16 `mAP_50_95=0.5983`
  - PTQ W8A8 `mAP_50_95=0.5932`
  - PTQ W4A16 `mAP_50_95=0.1277`
  - PTQ W4A8 `mAP_50_95=0.1150`
- QAT W4A8 `mAP_50_95=0.2912`
- Conclusion from W8 sanity checks: PTQ INT8 does **not** collapse, so the export/eval wiring is likely correct; the large W4 drop is scheme/config related.
- Logs:
  - Runner: `context/logs/quantize-yolov10m-w4a8-w4a16-brevitas/subtask-001-106-runner-2025-12-23_16-12-40.log`
  - Summary regen: `context/logs/quantize-yolov10m-w4a8-w4a16-brevitas/subtask-001-106-summary-v4.log`
  - Targeted `ruff`: `context/logs/quantize-yolov10m-w4a8-w4a16-brevitas/subtask-001-106-ruff-v3.log`
  - Targeted `mypy`: `context/logs/quantize-yolov10m-w4a8-w4a16-brevitas/subtask-001-106-mypy-v3.log`
  - Full quality gates (known pre-existing failures): `context/logs/quantize-yolov10m-w4a8-w4a16-brevitas/subtask-001-106-quality-gates.log`
