# Subtask 1.1: Baseline export + evaluation (Torch → ONNX)

## Scope

- Confirm `rtx5090` Pixi env can initialize `CUDAExecutionProvider` in ONNX Runtime.
- Export a baseline ONNX from `models/yolo10/checkpoints/yolov10m.pt` (prefer FP16 export when possible; otherwise FP32).
- Run:
  - Random-tensor smoke inference on the exported ONNX (CUDA EP preferred).
  - COCO2017 val evaluation on the fixed 100-image slice using `scripts/cv-models/eval_yolov10m_onnx_coco.py`.
- Record the baseline ONNX I/O contract and ensure it matches the expected YOLOv10m output format (`models/cv-models/yolov10m/README.md`).

## Planned outputs

- Baseline ONNX artifact under `tmp/yolov10m_brevitas_w4a8_w4a16/<run-id>/onnx/`:
  - `yolov10m-baseline-fp16.onnx` (or FP32 equivalent if needed)
- ORT smoke outputs/logs and a COCO subset metrics JSON under the same run root.
- A short `notes.md` capturing exact commands and provider configuration used.

## Dataset plan

- **Evaluation (fixed subset for iteration)**:
  - Use COCO2017 val via the repo symlink:
    - images: `datasets/coco2017/source-data/val2017/`
    - annotations: `datasets/coco2017/source-data/annotations/instances_val2017.json`
  - Default quick loop: `--max-images 100` with `scripts/cv-models/eval_yolov10m_onnx_coco.py` (deterministic: first N image IDs in sorted order).
  - If we need a more representative but still fixed subset, generate an image-id list under the run root (seeded), then pass `--image-ids-list` so all baseline/PTQ/QAT comparisons use identical images.
- **Calibration**: none for baseline export/eval (random-tensor smoke uses synthetic inputs).

## TODOs

- [ ] Job-001-101-001 Confirm ORT providers include `CUDAExecutionProvider` in `pixi run -e rtx5090`.
- [ ] Job-001-101-002 Export baseline ONNX from `models/yolo10/checkpoints/yolov10m.pt` (use `models/yolo10/helpers/convert_to_onnx.py` by default) and record export settings.
- [ ] Job-001-101-003 Run random-tensor ORT smoke on the exported ONNX with CUDA EP preferred and save outputs under `tmp/.../<run-id>/`.
- [ ] Job-001-101-004 Run COCO subset evaluation (100 images) on the baseline ONNX with providers ordered `CUDAExecutionProvider CPUExecutionProvider`.
- [ ] Job-001-101-005 Verify baseline ONNX input/output names and shapes match the expected evaluator contract (and update the eval config if needed).

## Notes

- Keep run artifacts under `tmp/yolov10m_brevitas_w4a8_w4a16/<run-id>/` (do not commit).
- If FP16 export is unavailable via the default exporter, proceed with FP32 and record the limitation.
