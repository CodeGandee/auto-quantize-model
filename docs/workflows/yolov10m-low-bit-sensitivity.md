# YOLOv10m low-bit quantization (ONNX PTQ + sensitivity-guided candidates)

This workflow targets the **ONNX** checkpoint at `models/cv-models/yolov10m/checkpoints/yolov10m.onnx` and provides a reproducible path to:

- Baseline ONNX Runtime smoke + COCO eval.
- Build a YOLO-style calibration tensor (`.npy`).
- Quantize with NVIDIA ModelOpt **ONNX PTQ**:
  - **INT8 Q/DQ** baseline (`quantize_mode=int8`).
  - Optional **FP8 Q/DQ** candidates (`quantize_mode=fp8`) with *per-node exclusion lists* derived from a Torch sensitivity proxy.

Notes:

- This workflow writes artifacts under `tmp/` (not committed).
- The ONNX checkpoint under `models/cv-models/` is **independent** from the Ultralytics YOLOv10 PyTorch assets under `models/yolo10/`. Sensitivity results from Torch are treated as a **proxy**, not a guarantee of transfer.

## Implemented tools

- `scripts/cv-models/eval_yolov10m_onnx_coco.py`: COCO2017 val evaluation (DFL decode + NMS) for YOLOv10m ONNX.
- `scripts/cv-models/make_yolov10m_calib_npy.py`: builds a YOLO-preprocessed calibration tensor (`float32`, NCHW).
- `scripts/cv-models/quantize_yolov10m_int8_onnx.sh`: ModelOpt ONNX PTQ wrapper for `quantize_mode=int8`.
- `scripts/cv-models/make_yolov10m_head_nodes_to_exclude.py`: generates a conservative “head Conv” node exclusion list for INT8.
- `scripts/cv-models/run_yolov10m_layer_sensitivity_sweep.sh`: runs the Torch/AutoQuant sensitivity sweep (proxy).
- `scripts/cv-models/make_yolov10m_candidate_schemes.py`: converts a sensitivity report into top‑K ONNX node exclusion lists.
- `scripts/cv-models/quantize_yolov10m_fp8_onnx.sh`: ModelOpt ONNX PTQ wrapper for `quantize_mode=fp8` with per-node exclusions.
- `scripts/cv-models/materialize_yolov10m_lowbit_candidates.sh`: materializes a small K-set of FP8 candidates from a scheme directory.

## Prerequisites

- Pixi env available:
  - `pixi install`
- COCO2017 present via repo symlink:
  - `datasets/coco2017/source-data/` should contain `annotations/instances_val2017.json` and `val2017/`.
- CUDA-enabled ONNX Runtime in the target env (recommended):
  - run commands through `pixi run -e rtx5090 ...`

## 1) Baseline smoke + COCO eval (ONNX Runtime)

```bash
RUN_ROOT="tmp/yolov10m_lowbit/$(date +%Y-%m-%d_%H-%M-%S)"

# Random-tensor sanity check
pixi run -e rtx5090 python models/cv-models/helpers/run_random_onnx_inference.py \
  --model models/cv-models/yolov10m/checkpoints/yolov10m.onnx \
  --output-root "$RUN_ROOT/baseline-onnx"

# COCO2017 val subset eval (default: 100 images)
pixi run -e rtx5090 python scripts/cv-models/eval_yolov10m_onnx_coco.py \
  --onnx-path models/cv-models/yolov10m/checkpoints/yolov10m.onnx \
  --data-root datasets/coco2017/source-data \
  --max-images 100 \
  --providers CUDAExecutionProvider CPUExecutionProvider \
  --warmup-runs 10 \
  --skip-latency 10 \
  --imgsz 640 \
  --out "$RUN_ROOT/baseline-coco/metrics.json"
```

## 2) Build calibration tensor (float32, NCHW)

The calibration list `datasets/quantize-calib/quant100.txt` is a fixed 100-image subset (repo-relative paths) used across runs.

```bash
RUN_ROOT="tmp/yolov10m_lowbit/$(date +%Y-%m-%d_%H-%M-%S)"

pixi run -e rtx5090 python scripts/cv-models/make_yolov10m_calib_npy.py \
  --list datasets/quantize-calib/quant100.txt \
  --out "$RUN_ROOT/calib/calib_yolov10m_640.npy" \
  --imgsz 640
```

## 3) INT8 PTQ (ModelOpt ONNX) + eval

`scripts/cv-models/quantize_yolov10m_int8_onnx.sh` wraps `python -m modelopt.onnx.quantization` and emits an INT8 Q/DQ ONNX.

```bash
RUN_ROOT="tmp/yolov10m_lowbit/$(date +%Y-%m-%d_%H-%M-%S)"

RUN_ROOT="$RUN_ROOT" \
CALIB_PATH="$RUN_ROOT/calib/calib_yolov10m_640.npy" \
CALIBRATION_METHOD="entropy" \
USE_ZERO_POINT=True \
CALIBRATION_EPS="cuda:0 cpu" \
pixi run -e rtx5090 bash scripts/cv-models/quantize_yolov10m_int8_onnx.sh

pixi run -e rtx5090 python scripts/cv-models/eval_yolov10m_onnx_coco.py \
  --onnx-path "$RUN_ROOT/onnx/yolov10m-int8-qdq.onnx" \
  --data-root datasets/coco2017/source-data \
  --max-images 100 \
  --providers CUDAExecutionProvider CPUExecutionProvider \
  --warmup-runs 10 \
  --skip-latency 10 \
  --imgsz 640 \
  --out "$RUN_ROOT/int8-coco/metrics.json"
```

Optional: generate a conservative *head exclusion list* (keep the detection head in higher precision) and pass it into INT8 PTQ via `NODES_TO_EXCLUDE_FILE`:

```bash
RUN_ROOT="tmp/yolov10m_lowbit/$(date +%Y-%m-%d_%H-%M-%S)"

pixi run -e rtx5090 python scripts/cv-models/make_yolov10m_head_nodes_to_exclude.py \
  --out "$RUN_ROOT/nodes_to_exclude_head.txt"

RUN_ROOT="$RUN_ROOT" \
CALIB_PATH="$RUN_ROOT/calib/calib_yolov10m_640.npy" \
NODES_TO_EXCLUDE_FILE="$RUN_ROOT/nodes_to_exclude_head.txt" \
pixi run -e rtx5090 bash scripts/cv-models/quantize_yolov10m_int8_onnx.sh
```

## 4) Optional: Torch sensitivity → FP8 candidates (node exclusions)

ModelOpt ONNX `quantize_mode=int4` primarily targets Gemm/MatMul and does not materially quantize this Conv-heavy YOLOv10m ONNX. FP8 Q/DQ is the practical “low-bit” ONNX PTQ candidate implemented here.

### 4a) Run Torch layer sensitivity sweep (proxy)

This runs a focused AutoQuant sensitivity sweep for Ultralytics YOLOv10m and writes reports under `tmp/yolov10m_layer_sensitivity/<run-id>/...`:

```bash
pixi run -e rtx5090 bash scripts/cv-models/run_yolov10m_layer_sensitivity_sweep.sh
```

### 4b) Convert sensitivity report → exclusion schemes

Generate top‑K exclusion lists for ONNX Conv nodes (keep top‑K sensitive nodes in high precision):

```bash
RUN_ROOT="tmp/yolov10m_lowbit/$(date +%Y-%m-%d_%H-%M-%S)"
SENS_RUN="tmp/yolov10m_layer_sensitivity/<run-id>"

pixi run -e rtx5090 python scripts/cv-models/make_yolov10m_candidate_schemes.py \
  --report-json "$SENS_RUN/outputs/yolov10m/fp8-fp8/per_layer/layer-sensitivity-report.json" \
  --out-dir "$RUN_ROOT/schemes" \
  --ks 0 5 10 20
```

### 4c) Materialize FP8 candidates

```bash
RUN_ROOT="$RUN_ROOT" \
CALIB_PATH="$RUN_ROOT/calib/calib_yolov10m_640.npy" \
pixi run -e rtx5090 bash scripts/cv-models/materialize_yolov10m_lowbit_candidates.sh
```

Evaluate each candidate with `scripts/cv-models/eval_yolov10m_onnx_coco.py` (same subset and thresholds as baseline/INT8) and compare results under `tmp/yolov10m_lowbit/<run-id>/results/`.

## Outputs

This workflow uses a run root like `tmp/yolov10m_lowbit/<run-id>/` and produces directories such as:

- `baseline-onnx/` (random-tensor inference summaries)
- `baseline-coco/` (baseline eval metrics)
- `calib/` (calibration tensor `.npy`)
- `onnx/` (quantized artifacts like `yolov10m-int8-qdq.onnx`)
- `candidates/` (optional FP8 candidate ONNX files)
- `results/` (JSON summaries for baseline/INT8/candidates)

For the full task breakdown and the exact outputs captured during development, see:

- `context/tasks/done/quantize-yolov10m-low-bit-sensitivity/`
