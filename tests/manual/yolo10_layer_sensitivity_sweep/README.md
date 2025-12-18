# Manual test: YOLOv10 layer sensitivity sweep

Runs NVIDIA ModelOpt AutoQuant **per-layer sensitivity** across a grid of:

- **Models**: `yolov10n`, `yolov10s`, `yolov10m`
- **Weight dtype**: `int4`, `int8`, `fp4`, `fp8`
- **Activation dtype**: `int4`, `int8`, `fp4`, `fp8`
- **Granularity**: `per_channel`, `per_layer`

It uses `datasets/quantize-calib/quant100.txt` (COCO image paths) for calibration
and writes all logs + outputs under `tmp/yolo10_layer_sensitivity_sweep/<timestamp>/`.

## Prereqs

- Use the Pixi env from `context/instructions/prep-rtx5090-vllm.md`:
  - `pixi run -e rtx5090-vllm ...`
- YOLOv10 assets:
  - `bash models/yolo10/bootstrap.sh`
- COCO images available at `datasets/coco2017/source-data/...`

## Run

```bash
bash tests/manual/yolo10_layer_sensitivity_sweep/run.sh
```

To run a smaller filtered subset (recommended for iteration):

```bash
bash tests/manual/yolo10_layer_sensitivity_sweep/run.sh \
  --models yolov10n \
  --weight-dtypes fp8 \
  --act-dtypes fp8 \
  --granularities per_layer \
  --max-calib-samples 16 \
  --max-runs 1
```

Note: `--max-calib-samples` must be `>= 10`.

## Outputs

Under `tmp/yolo10_layer_sensitivity_sweep/<timestamp>/`:

- `outputs/<model>/<w>-<a>/<granularity>/`
  - `*_quant_manifest.json`
  - `composed-config.yaml` (run configuration snapshot)
  - `layer-sensitivity-report.md`
  - `layer-sensitivity-report.json`
  - `*_autoquant_state.pt`
- `logs/<model>/<w>-<a>/<granularity>.log`
- `outputs/sweep_index.json` and `outputs/sweep_index.csv` (top-k sensitive layers per run)
- `outputs/failures.json` (skipped/failed runs with reasons)
