# Subtask 1.6: Summarize results, update docs, and optional INC cross-check

## Scope

- Produce a concise summary comparing baseline vs INT8 vs low-bit candidates:
  - COCO mAP (bbox) on the fixed subset
  - ORT (and optional TRT) latency/throughput
  - Notes on calibration method and scheme settings (K, dtypes, exclusions)
- Update local docs so the workflow is easy to reproduce later.
- Optional: run an Intel Neural Compressor INT8 PTQ/tuning flow as a cross-check and compare results to ModelOpt.

## Planned outputs

- `tmp/yolov10m_lowbit/<run-id>/summary.md` with a comparison table and recommended scheme
- Updated quickstart commands in `models/cv-models/yolov10m/README.md` (or another stable doc location)
- Optional INC outputs under `tmp/yolov10m_lowbit/<run-id>/inc/`

## TODOs

- [x] Job-001-106-001 Write `summary.md` with baseline vs candidate comparisons and a clear recommendation.
- [x] Job-001-106-002 Update `models/cv-models/yolov10m/README.md` (and/or the main task file) with the canonical reproduction commands and pointers to scripts.
- [ ] Job-001-106-003 (Optional) Run an INC INT8 PTQ/tuning flow and record accuracy/latency for comparison.

## Outputs (completed)

- Summary:
  - `tmp/yolov10m_lowbit/2025-12-23_04-40-28_gpu/summary.md`
- Repro/docs:
  - `models/cv-models/yolov10m/README.md`

## INC note

- INC cross-check is optional and was not run in this iteration.

## Notes

- Run Python via `pixi run -e rtx5090 python ...` (see `context/instructions/prep-rtx5090.md`).
- INC scaffolding reference: `src/auto_quantize_model/inc_pytorch_mse_patching.py`
