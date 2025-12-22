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

- [ ] Job-001-104-001 Bootstrap the Ultralytics YOLOv10m checkpoint (`bash models/yolo10/bootstrap.sh`) if not already present.
- [ ] Job-001-104-002 Run a focused sensitivity sweep for `yolov10m` covering at least:
  - Weights: `int8` and (if supported) `int4`
  - Activations: `fp16` (weight-only baseline) and `fp8` (if supported)
  - Granularity: start with `per_layer`
- [ ] Job-001-104-003 Summarize top‑K sensitive layers and draft 3–5 candidate schemes (keep the set small and testable).
- [ ] Job-001-104-004 Decide and document whether candidates will be materialized via ONNX-native PTQ controls or via Torch quantize→export.

## Notes

- Run Python via `pixi run -e rtx5090 python ...` (see `context/instructions/prep-rtx5090.md`).
- Treat Torch sensitivity as a proxy unless the ONNX checkpoint’s origin checkpoint is confirmed.

