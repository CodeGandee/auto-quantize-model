# Subtask 1.1: Bootstrap assets and baseline smoke checks

## Scope

- Verify the target ONNX checkpoint exists and is readable:
  - `models/cv-models/yolov10m/checkpoints/yolov10m.onnx`
- Ensure COCO2017 images are available and that the calibration list points to real files:
  - `bash datasets/coco2017/bootstrap.sh`
  - `datasets/quantize-calib/quant100.txt`
- Run a baseline ONNXRuntime sanity check (random-tensor inference) in the `rtx5090` Pixi environment.
- Optional: bootstrap the Ultralytics YOLOv10m Torch checkpoint for later sensitivity runs (`models/yolo10/bootstrap.sh`).

## Planned outputs

- Baseline ORT inference summary JSON under `tmp/yolov10m_lowbit/<run-id>/baseline-onnx/`
- A short `tmp/.../baseline-notes.md` capturing the exact commands used

## TODOs

- [ ] Job-001-101-001 Verify the YOLOv10m ONNX checkpoint symlink resolves and the file is readable.
- [ ] Job-001-101-002 Bootstrap COCO2017 and confirm `datasets/quantize-calib/quant100.txt` entries exist on disk.
- [ ] Job-001-101-003 Run baseline random inference with ORT (CUDA EP preferred) and save outputs under `tmp/`.
- [ ] Job-001-101-004 (Optional) Run `bash models/yolo10/bootstrap.sh` to fetch `models/yolo10/checkpoints/yolov10m.pt` for Subtask 1.4.

## Notes

- Run Python via `pixi run -e rtx5090 python ...` (see `context/instructions/prep-rtx5090.md`).
- Example baseline ORT check:

```bash
RUN_ROOT="tmp/yolov10m_lowbit/$(date +%Y-%m-%d_%H-%M-%S)"
pixi run -e rtx5090 python models/cv-models/helpers/run_random_onnx_inference.py \
  --model models/cv-models/yolov10m/checkpoints/yolov10m.onnx \
  --output-root "$RUN_ROOT/baseline-onnx"
```

