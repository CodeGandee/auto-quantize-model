# 2025-12-25 — YOLOv10m scratch QAT (Brevitas W4A16)

This report directory contains intermediate checkpoints copied from an in-progress “from scratch” QAT run of **YOLOv10m** on **COCO2017** using **Brevitas weight-only int4 fake-quant** (**W4A16**: weights quantized to 4-bit; activations left floating).

## How it was trained

- **Entrypoint**: `scripts/cv-models/train_yolov10m_scratch_fp16_vs_w4a16_qat_brevitas.py` (`qat-w4a16` subcommand)
- **Trainer**: local Ultralytics YOLOv10 trainer from `models/yolo10/src/` (Pixi env `rtx5090`)
- **Model**: `models/yolo10/src/ultralytics/cfg/models/v10/yolov10m.yaml` (random init; `pretrained=False`)
- **Dataset**: COCO2017 from `datasets/coco2017/source-data/`, converted run-locally to YOLO-format under the run root via `src/auto_quantize_model/cv_models/yolov10_coco_dataset.py`
- **Hyperparameters**: `conf/cv-models/yolov10m/hyp.scratch.yaml` (SGD + “scratch recipe”-style aug/LR)
- **Quantization**: Brevitas layerwise graph transform (`Conv2d -> QuantConv2d`) via `quantize_model_brevitas_ptq(..., weight_bit_width=4, act_bit_width=None)` in `src/auto_quantize_model/cv_models/yolov10_brevitas.py`
- **Checkpoint cadence**: `save_period=5` (plus `best.pt` and continuously-updated `last.pt`)

The run that produced these checkpoints is under:

- `tmp/yolov10m_scratch_fp16_vs_w4a16_qat_brevitas/2025-12-24_08-38-32/`

To launch a similar QAT run:

```bash
pixi run -e rtx5090 python scripts/cv-models/train_yolov10m_scratch_fp16_vs_w4a16_qat_brevitas.py qat-w4a16 \
  --run-root tmp/yolov10m_scratch_fp16_vs_w4a16_qat_brevitas/<run-id> \
  --coco-root datasets/coco2017/source-data \
  --imgsz 640 --epochs 300 --batch 32 --device 0 --workers 8 --save-period 5
```

## What’s in `checkpoints/`

Directory: `models/yolo10/reports/2025-12-25-qat-w4a16/checkpoints/`

- `epochNNN.pt`: snapshot saved at the end of epoch `NNN` (every 5 epochs).
- `last.pt`: latest snapshot (updated when Ultralytics “save model” runs).
- `best.pt`: best snapshot so far (updated when fitness improves).

### Checkpoint format

These QAT checkpoints are **pickling-free** dict checkpoints (not full Ultralytics `.pt` model objects). Example keys (verified on `epoch065.pt`):

- `epoch`: 1-based epoch number
- `model_state_dict`: `state_dict()` of the (de-paralleled) training model
- `ema_state_dict`: `state_dict()` of the EMA model (when EMA is enabled)
- `optimizer`: `optimizer.state_dict()` (when available)
- `train_args`: resolved Ultralytics args/overrides used for the run
- `date`: ISO timestamp string

### Loading notes (PyTorch 2.6+)

PyTorch defaults `torch.load(..., weights_only=True)` in newer versions; these checkpoints include non-tensor metadata. Load with `weights_only=False`:

```python
import torch

ckpt = torch.load(".../epoch065.pt", map_location="cpu", weights_only=False)
state_dict = ckpt["ema_state_dict"] or ckpt["model_state_dict"]
```

## Provenance / mapping

These files were copied from the training output directory:

- Source: `tmp/yolov10m_scratch_fp16_vs_w4a16_qat_brevitas/2025-12-24_08-38-32/qat-w4a16/ultralytics/yolov10m-scratch-qat-w4a16/weights/`
- Destination: `models/yolo10/reports/2025-12-25-qat-w4a16/checkpoints/`

