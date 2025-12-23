# Subtask 1.3: PTQ W4A16(-like) export + validation

## Scope

- Implement a **PTQ** (no training) path that:
  - loads `models/yolo10/checkpoints/yolov10m.pt`,
  - applies **4-bit weight quantization** (Brevitas) while leaving activations floating,
  - runs model in FP16 where feasible,
  - exports a QCDQ ONNX artifact (`yolov10m-w4a16-qcdq-ptq.onnx`),
  - validates ORT inference with CUDA EP preferred.

## Planned outputs

- Quantization/export implementation:
  - `scripts/cv-models/quantize_yolov10m_brevitas_w4.py` (or similar) implementing the W4A16(-like) PTQ config.
  - `scripts/cv-models/run_yolov10m_brevitas_w4_ptq_qat.sh` orchestrator (may be stubbed here and completed in Subtask 1.6).
- ONNX artifact under `tmp/yolov10m_brevitas_w4a8_w4a16/<run-id>/onnx/`:
  - `yolov10m-w4a16-qcdq-ptq.onnx`
- Smoke inference logs and optional COCO subset metrics JSON for the PTQ model.

## Dataset plan

- **Calibration**: none required for W4A16(-like) (activations remain floating). If the implementation needs representative inputs for export shape tracing, reuse the same preprocessing as the evaluator (640×640 letterbox).
- **Evaluation (must match baseline)**:
  - Use the same fixed COCO2017 val subset plan as Subtask 1.1 so deltas are comparable:
    - `datasets/coco2017/source-data/val2017/` + `instances_val2017.json`
    - `--max-images 100` (or a shared `--image-ids-list` file under the run root).

## TODOs

- [ ] Job-001-103-001 Prototype loading `yolov10m.pt` as a Torch `nn.Module` suitable for Brevitas wrapping/export (Ultralytics model internals).
- [ ] Job-001-103-002 Define the W4A16(-like) PTQ configuration (weight bit-width=4; activations floating; FP16 where feasible) and implement it without training.
- [ ] Job-001-103-003 Export QCDQ ONNX using Brevitas (`export_onnx_qcdq(..., dynamo=False)`) and the Torch 2.9 compat helper.
- [ ] Job-001-103-004 Validate the exported ONNX runs in ORT with `CUDAExecutionProvider` preferred; capture a smoke inference summary.
- [ ] Job-001-103-005 Run COCO subset evaluation for W4A16(-like) PTQ and record metrics/latency deltas vs baseline.

## Notes

- Keep function/class names intact when writing any Mermaid diagrams or notes (see `magic-context/instructions/mermaid-seq-styling.md`).
