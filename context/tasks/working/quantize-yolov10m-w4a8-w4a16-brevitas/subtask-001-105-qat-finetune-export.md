# Subtask 1.5: Optional QAT fine-tune + export (accuracy recovery)

## Scope

- If PTQ accuracy loss is too large, run a **short QAT** fine-tune starting from a PTQ-initialized quantized model and export QAT QCDQ ONNX artifacts.
- Keep QAT intentionally lightweight and reproducible:
  - record seeds, hyperparameters, and dataset provenance,
  - prefer a small number of steps/epochs (e.g., 1–3 epochs or a fixed-step budget),
  - store all outputs under `tmp/yolov10m_brevitas_w4a8_w4a16/<run-id>/`.
- Address dataset format needs:
  - Ultralytics training expects YOLO-format labels; if starting from MS-COCO JSON, define a conversion/label-generation approach.

## Planned outputs

- A QAT entrypoint (script or commands) that can run in `pixi run -e rtx5090` and fine-tune either:
  - W4A16(-like) (weight-only quant), and/or
  - W4A8 (INT8 activations).
- QAT outputs under `tmp/.../<run-id>/qat/`:
  - checkpoints,
  - training logs,
  - exported ONNX artifacts:
    - `onnx/yolov10m-w4a16-qcdq-qat.onnx`
    - `onnx/yolov10m-w4a8-qcdq-qat.onnx`
- COCO subset eval metrics for QAT exports.

## Dataset plan

- **Training data (labeled)**:
  - Use COCO2017 train split from the repo dataset symlink:
    - images: `datasets/coco2017/source-data/train2017/`
    - annotations: `datasets/coco2017/source-data/annotations/instances_train2017.json`
  - Ultralytics training expects YOLO-format labels. Plan:
    - Generate YOLO labels under the run root using Ultralytics’ COCO converter (`convert_coco`) and keep everything under `tmp/.../<run-id>/qat/` (do not write into the shared dataset directory).
    - Create an `images/` layout via symlinks:
      - `.../qat/coco_yolo/images/train2017` → `datasets/coco2017/source-data/train2017`
      - `.../qat/coco_yolo/images/val2017` → `datasets/coco2017/source-data/val2017`
    - Create `labels/train2017` and `labels/val2017` under `.../qat/coco_yolo/` from COCO JSON.
  - **Default “short QAT” size**: use Ultralytics `fraction` to train on ~1–2% of train2017 (deterministic subset based on file ordering) for 1–3 epochs; record the exact `fraction`, epochs, and seed.
- **Validation / reporting**:
  - For fast iteration during QAT runs, validation inside Ultralytics can be disabled or reduced (optional).
  - For the official comparison, always evaluate exported QAT ONNX with `scripts/cv-models/eval_yolov10m_onnx_coco.py` on the same fixed COCO2017 val subset used for baseline/PTQ (Subtask 1.1).

## TODOs

- [ ] Job-001-105-001 Decide the QAT dataset path and format (COCO JSON → YOLO labels, or a prepared YOLO-style COCO dataset).
- [ ] Job-001-105-002 Implement a minimal QAT training entrypoint that integrates Brevitas quant layers with Ultralytics training.
- [ ] Job-001-105-003 Run a short QAT fine-tune for the chosen variant(s) and capture training hyperparameters, seed, and run logs.
- [ ] Job-001-105-004 Export QAT QCDQ ONNX artifacts and validate ORT CUDA EP inference.
- [ ] Job-001-105-005 Evaluate QAT exports on the COCO subset and compare to baseline/PTQ.

## Notes

- If dataset conversion is required, prefer Ultralytics’ built-in COCO conversion utilities when possible (avoid bespoke label tooling).
