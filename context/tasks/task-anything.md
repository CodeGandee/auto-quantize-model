# model low-bit quantization based on layer sensitivity analysis

now we want to quantize model models/cv-models/yolov10m using 4-bit and 8-bit quantization, based on layer sensitivity analysis.

## Goal

- quantize the model using int4/int8 weights (mixed precision, some will be fp16), fp8/fp16 activations, per-layer granularity
- compare the accuracy and performance of the quantized models with the original model
- study mixed precision quantization schemes, that is, how to allocate different bit-widths to different layers based on their sensitivity to quantization

## Tools

- Nvidia Modelopt
    - reference source code: `extern/TensorRT-Model-Optimizer`
- Intel Neural Compressor
    - reference source code: `extern/neural-compressor`

## Approach

> Important: `models/cv-models/yolov10m` (ONNX) and `models/yolo10` (Ultralytics PyTorch) are **independent model artifacts**.
> Even if the names match (`yolov10m`), do not assume they share identical weights/graphs or that sensitivity results transfer 1:1.

### 1) Bootstrap assets + baseline smoke checks

- Ensure the **target ONNX checkpoint** exists:
  - `models/cv-models/yolov10m/checkpoints/yolov10m.onnx` (symlink to external storage)
- If you plan to run **PyTorch-based** ModelOpt AutoQuant sensitivity (Step 2), also bootstrap the **separate** Ultralytics YOLOv10 checkpoint:
  - `bash models/yolo10/bootstrap.sh` → `models/yolo10/checkpoints/yolov10m.pt`
- Ensure COCO images are available:
  - `bash datasets/coco2017/bootstrap.sh`
  - Calibration list: `datasets/quantize-calib/quant100.txt`
- Quick ONNXRuntime sanity check (baseline):
  - `pixi run -e rtx5090 python models/cv-models/helpers/run_random_onnx_inference.py --model models/cv-models/yolov10m/checkpoints/yolov10m.onnx --output-root tmp/yolov10m/baseline-onnx`
- Quick PyTorch sanity check (baseline):
  - `pixi run -e rtx5090 python models/yolo10/helpers/infer_and_annotate.py yolov10m <image-path>`

### 2) Run per-layer sensitivity analysis (ModelOpt AutoQuant)

Use the existing YOLOv10 sensitivity sweep tooling (PyTorch/Ultralytics):

- Driver: `tests/manual/yolo10_layer_sensitivity_sweep/scripts/run_layer_sensitivity_sweep.py`
- Outputs (per run): `layer-sensitivity-report.{md,json}` + a sweep summary index `outputs/sweep_index.csv`

Note:

- This sweep operates on `models/yolo10/checkpoints/*.pt`.
- If your target is `models/cv-models/yolov10m/checkpoints/yolov10m.onnx` and you do not have the *exact* PyTorch checkpoint it was exported from, treat this sweep as methodology/proxy only (rankings may not transfer).
  - For sensitivity on the exact ONNX graph, you either need the originating PyTorch checkpoint, or you need an ONNX-native sensitivity workflow (not yet scripted in this repo).

Recommended initial sweep (focused on this task’s goal):

- Model: `yolov10m`
- Weights: `int4`, `int8`
- Activations: `fp8` and **fp16 as a weight-only baseline**
  - For fp16 activations, prefer ModelOpt weight-only presets (e.g., `INT8_WEIGHT_ONLY_CFG` and the INT4 weight-only preset) rather than quantizing activations.
- Granularity: start with `per_layer` (axis=None); try `per_channel` after the first pass.

Example run (write everything under `tmp/`):

```bash
RUN_ROOT="tmp/yolov10m_layer_sensitivity/$(date +%Y-%m-%d_%H-%M-%S)"
pixi run -e rtx5090 python tests/manual/yolo10_layer_sensitivity_sweep/scripts/run_layer_sensitivity_sweep.py \
  --output-root "$RUN_ROOT/outputs" \
  --log-root "$RUN_ROOT/logs" \
  --models yolov10m \
  --weight-dtypes int4 int8 \
  --act-dtypes fp8 \
  --granularities per_layer \
  --max-calib-samples 100 \
  --batch-size 1 \
  --imgsz 640
```

### 3) Propose mixed-precision schemes from the sensitivity reports

Turn the sensitivity ranking (top sensitive layers) into concrete candidate schemes:

- Start with a simple policy: keep the top‑K most sensitive layers at higher precision (FP16 or INT8), quantize the rest more aggressively (INT4).
- Compare a small set of schemes, for example:
  - **W4A16** (INT4 weight-only) with top‑K layers kept FP16/INT8
  - **W4A8** (INT4 weights + FP8 activations) with top‑K layers kept FP16 activations and/or INT8 weights
  - **W8A16** (INT8 weight-only) as a higher-accuracy baseline

When defining candidate formats/configs, prefer the repo’s named ModelOpt configs in `src/auto_quantize_model/modelopt_configs.py` (e.g., `INT8_WEIGHT_ONLY_CFG`, `INT4_WEIGHT_FP8_ACT_CFG`, `INT8_WEIGHT_FP8_ACT_CFG`) so runs are reproducible.

### 4) Materialize quantized artifacts

Two complementary paths:

1. **ONNX (deployable INT8 baseline):**
   - Build a calibration tensor from `datasets/quantize-calib/quant100.txt`:
     - `pixi run -e rtx5090 python scripts/yolo11/make_yolo11_calib_npy.py --list datasets/quantize-calib/quant100.txt --out tmp/yolov10m/calib_yolo10_640.npy`
   - Quantize the YOLOv10m ONNX with ModelOpt’s ONNX PTQ CLI (adapt `scripts/yolo11/quantize_yolo11n_int8_onnx.sh`).
2. **Torch (ModelOpt mixed precision / research):**
   - Apply the chosen per-layer mixed scheme using ModelOpt torch quantization configs and name-pattern overrides.
   - Export to ONNX when needed via `models/yolo10/helpers/convert_to_onnx.py` (this produces an ONNX model for the PyTorch checkpoint under `models/yolo10/`, not the `models/cv-models/` ONNX).

Always save:

- The composed run config (`*.yaml`), quant manifest JSON, and sensitivity report(s)
- Any generated ONNX or engine artifacts
- Under a task-specific directory in `tmp/`

### 5) Evaluate accuracy + performance and iterate

- Accuracy:
  - Run COCO mAP on a fixed val subset and compare against baseline.
  - Reuse the structure of `scripts/yolo11/eval_yolo11_torch_coco.py`, but note YOLOv10 ONNX output decoding differs (output shape is `[1, 144, 8400]` in `models/cv-models/yolov10m`).
- Performance:
  - Measure latency with ONNX Runtime CUDA EP (and optionally TensorRT `trtexec` for FP16/INT8/FP8 engines).
- Iterate:
  - Adjust the top‑K threshold and/or granularity (per-layer vs per-channel) until you find the best latency ↔ accuracy trade-off.

### 6) Cross-check with Intel Neural Compressor (optional)

Use INC as a second implementation/validation:

- Run an INT8 PTQ/tuning flow and compare its per-op sensitivity ranking vs. ModelOpt.
- The repo has scaffolding for capturing INC MSE sensitivity: `src/auto_quantize_model/inc_pytorch_mse_patching.py`.
