# Subtask 1.4: Layer sensitivity sweep (Torch) and candidate mixed schemes

## Scope

- Run the existing YOLOv10 Torch layer-sensitivity sweep tooling for `yolov10m` as a methodology/proxy:
  - `tests/manual/yolo10_layer_sensitivity_sweep/scripts/run_layer_sensitivity_sweep.py`
- Produce a small, explicit set of candidate mixed/low-bit schemes from the sensitivity report(s), such as a top‑K policy:
  - Keep top‑K most sensitive layers at FP16/INT8
  - Quantize remaining layers more aggressively (INT4 where supported)
- Decide which implementation path is viable for materializing candidates:
  - ONNX-native exclusions/overrides (preferred if supported and mappable), or
  - Torch quantize → export to ONNX (research artifact unless proven deployable)

## Planned outputs

- Sensitivity run outputs under `tmp/yolov10m_layer_sensitivity/<timestamp>/` (reports + index)
- A short candidate scheme list (K values + intended dtypes) saved under the same `tmp/` run root
- A recorded decision on the mixed-precision materialization path (used by Subtask 1.5)

## TODOs

- [x] Job-001-104-001 Bootstrap the Ultralytics YOLOv10m checkpoint (`bash models/yolo10/bootstrap.sh`) if not already present.
- [x] Job-001-104-002 Run a focused sensitivity sweep for `yolov10m` covering at least:
  - Weights: `int8` and (if supported) `int4`
  - Activations: `fp16` (weight-only baseline) and `fp8` (if supported)
  - Granularity: start with `per_layer`
- [x] Job-001-104-003 Summarize top‑K sensitive layers and draft 3–5 candidate schemes (keep the set small and testable).
- [x] Job-001-104-004 Decide and document whether candidates will be materialized via ONNX-native PTQ controls or via Torch quantize→export.

## Outputs (completed)

- Torch checkpoint present via bootstrap:
  - `models/yolo10/checkpoints/yolov10m.pt`
- Sensitivity runs (Torch proxy):
  - `tmp/yolov10m_layer_sensitivity/2025-12-23_04-46-25/` (weights fp8/int8 × activations fp16/fp8, per-layer)
  - `tmp/yolov10m_layer_sensitivity/2025-12-23_04-46-59/` (weights int4/int8 × activations fp16, per-layer)
- Candidate schemes derived from `fp8-fp16/per_layer` report:
  - `tmp/yolov10m_lowbit/2025-12-23_04-40-28_gpu/schemes-fp8-topk/` (K ∈ {0,5,10,20})
  - Mapping summary: `tmp/yolov10m_lowbit/2025-12-23_04-40-28_gpu/schemes-fp8-topk/candidates.md`

## Decision: candidate materialization path

- Use **ONNX-native PTQ controls** (ModelOpt ONNX FP8 + per-node exclusions) rather than Torch-quantize→export.
- Rationale:
  - The deployable artifact for this task is the existing ONNX checkpoint.
  - ModelOpt ONNX PTQ supports FP8 QDQ generation with per-node exclusion lists.
  - Torch layer sensitivity is treated as a proxy; we map Torch Conv module names → ONNX Conv node names and keep top‑K at higher precision.

## Notes

- Run Python via `pixi run -e rtx5090 python ...` (see `context/instructions/prep-rtx5090.md`).
- Treat Torch sensitivity as a proxy unless the ONNX checkpoint’s origin checkpoint is confirmed.
