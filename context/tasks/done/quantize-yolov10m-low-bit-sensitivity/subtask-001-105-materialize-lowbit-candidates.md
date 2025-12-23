# Subtask 1.5: Materialize low-bit candidates and benchmark (ORT/TensorRT)

## Scope

- Generate a small set of candidate artifacts beyond the INT8 baseline, based on Subtask 1.4’s scheme definitions and chosen implementation path.
- Evaluate candidate accuracy on the same **medium** COCO subset (100 images) used for baseline/INT8 comparison.
- Benchmark performance:
  - ONNX Runtime (CUDA EP) latency/throughput for baseline and all candidates
  - Optional TensorRT engines built from baseline and QDQ ONNX (if `trtexec` is available and useful)

## Planned outputs

- Candidate quantized artifacts (QDQ ONNX and/or exported ONNX from Torch) under `tmp/yolov10m_lowbit/<run-id>/candidates/`
- COCO metrics JSON per candidate under `tmp/.../results/`
- Benchmark logs and a small consolidated results table under `tmp/.../results/`

## TODOs

- [x] Job-001-105-001 Implement the chosen “mixed/low-bit” materialization approach (ONNX-native exclusions/overrides or Torch quantize→export).
- [x] Job-001-105-002 Generate candidate artifacts for a small K-set (e.g., K ∈ {0, 5, 10, 20}) and record the exact scheme settings per artifact.
- [x] Job-001-105-003 Run COCO evaluation for all candidates using the evaluator from Subtask 1.2 (same medium subset, same thresholds).
- [x] Job-001-105-004 Benchmark ORT latency/throughput for baseline, INT8, and candidates; optionally build TRT engines with `--precisionConstraints=obey` for QDQ models.

## Outputs (completed)

- Candidate materialization scripts:
  - `scripts/cv-models/quantize_yolov10m_fp8_onnx.sh`
  - `scripts/cv-models/materialize_yolov10m_lowbit_candidates.sh`
- GPU run root: `tmp/yolov10m_lowbit/2025-12-23_04-40-28_gpu/`
  - Candidates: `tmp/yolov10m_lowbit/2025-12-23_04-40-28_gpu/candidates/yolov10m-fp8-k*.onnx`
  - Candidate logs: `tmp/yolov10m_lowbit/2025-12-23_04-40-28_gpu/candidates/logs/`
  - COCO100 eval + latency (warmup + skip-latency):
    - `tmp/yolov10m_lowbit/2025-12-23_04-40-28_gpu/results/baseline_cuda_only.json`
    - `tmp/yolov10m_lowbit/2025-12-23_04-40-28_gpu/results/int8_cuda_cpu.json`
    - `tmp/yolov10m_lowbit/2025-12-23_04-40-28_gpu/results/fp8_k*_cuda_cpu.json`

## TensorRT note

- TensorRT engine builds were skipped because this environment did not have `trtexec` nor an ORT build with `TensorrtExecutionProvider`.

## Notes

- Run Python via `pixi run -e rtx5090 python ...` (see `context/instructions/prep-rtx5090.md`).
- TensorRT reference: `context/summaries/modelopt-kb/howto-qdq-onnx-to-mixed-precision-tensorrt.md`
