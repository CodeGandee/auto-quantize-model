# Manual test: YOLO11n FP8/FP8 granularity sweep

Runs a small-calibration ModelOpt AutoQuant **per-layer sensitivity** sweep on
YOLO11n with **FP8 weights + FP8 activations**, varying activation quantization
granularity:

- Per-axis activations: channel/height/width (`axis` 1/2/3 on NCHW tensors)
- Per-block activations: block sizes 8/16/32 on the channel axis (`block_sizes: {1: N}`)

## Prereqs

- Use the Pixi env from `context/instructions/prep-rtx5090-vllm.md`:
  - `pixi run -e rtx5090-vllm ...`
- COCO images available at `datasets/coco2017/source-data/...`
  - The default image list is `datasets/quantize-calib/quant100.txt`

## Run

```bash
bash tests/manual/fp8fp8_yolo11n_granularity_sweep/run.sh
```

This will:

1. Bootstrap YOLO11 assets (`models/yolo11/bootstrap.sh`) if needed.
2. Run the sweep.
3. Write all logs and outputs under `tmp/fp8fp8_yolo11n_granularity_sweep/<timestamp>/`.
