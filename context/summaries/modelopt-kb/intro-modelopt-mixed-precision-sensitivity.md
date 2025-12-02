Intro: ModelOpt mixed precision & sensitivity for YOLO-style CNNs

## HEADER
- **Purpose**: Explain how to use NVIDIA ModelOpt to estimate layer sensitivity and derive an optimal mixed-precision scheme (FP16/INT8 or FP32/FP16-style) for CNNs like YOLO11.
- **Status**: Draft, practical usage notes
- **Date**: 2025-12-01
- **Owner**: AI assistant (Codex CLI)
- **Scope**: ModelOpt PyTorch `auto_quantize`, ONNX AutoCast, and how to combine them with TensorRT for mixed-precision deployment.

## 1. Concepts: “sensitivity” and mixed precision

- Layer sensitivity (for quantization) means: “If I lower this layer’s precision, how much does the end-to-end metric (loss/mAP) degrade?”.
- Mixed precision schemes keep **sensitive layers** in higher precision (FP32/FP16) and quantize **less sensitive layers** to lower precision (INT8, FP8, NVFP4, etc.).
- In the ModelOpt ecosystem there are two main mechanisms related to sensitivity:
  - **PyTorch `auto_quantize`** (true per-layer sensitivity search, currently focused on LLM configs but conceptually applicable to CNNs with INT8).
  - **ONNX AutoCast** (heuristic node classification for FP32↔FP16/BF16 based on activation magnitudes and optional calibration data).

For YOLO11 in this repo, the most practical path today is:
- Use **PyTorch-based ModelOpt** to perform sensitivity-driven mixed-precision quantization (e.g., INT8 vs no-quant) on the YOLO11 model.
- Export the resulting mixed-precision model to ONNX.
- Build mixed-precision TensorRT engines from that ONNX and validate latency/accuracy.

## 2. PyTorch path: `auto_quantize` for sensitivity-driven mixed precision

ModelOpt exposes a PTQ algorithm called `auto_quantize`:

- API: `modelopt.torch.quantization.model_quant.auto_quantize`
- Summary:
  - Inserts quantizer modules like `mtq.quantize` would.
  - Runs calibration on a dataloader.
  - Estimates **per-layer sensitivity scores** using gradient-based or KL-divergence methods.
  - Searches over a list of quantization formats (plus “no quantization”) subject to an **`effective_bits`** constraint.
  - Outputs a model where each eligible layer is assigned the “best” format, or left unquantized if necessary.

### 2.1 Minimal usage sketch for a YOLO-style model

High-level flow:

1. Start from a PyTorch YOLO11 model (eval mode, no training).
2. Prepare a calibration dataloader that yields representative images (e.g., COCO subset or your deployment dataset).
3. Define:
   - `forward_step(model, batch)` → returns outputs (used for both calibration and scoring).
   - `loss_func(output, batch)` or `forward_backward_step(model, batch)` → used to compute gradients or losses for sensitivity scoring.
4. Call `mtq.auto_quantize` with:
   - `quantization_formats` that include at least an INT8 config and implicitly “no quantization”.
   - `constraints={"effective_bits": X.Y}` to control how aggressive the mix is.
   - Reasonable `num_calib_steps` and `num_score_steps` (calibration vs sensitivity runtime).
5. Inspect the resulting model to see which layers are quantized and to what format.

Illustrative snippet (INT8 vs no quantization):

```python
import modelopt.torch.quantization as mtq
import modelopt.torch.opt as mto

model = ...  # YOLO11 nn.Module, in eval() mode, on GPU
calib_loader = ...  # DataLoader yielding (images, labels or meta)

def forward_step(model, batch):
    images, _ = batch  # adapt to your dataset
    return model(images)

def loss_func(output, batch):
    # For detection you can plug in a simple distillation-style loss:
    # e.g., MSE between FP32 outputs and quantized outputs, or task loss if labels exist.
    ...
    return loss

model, search_state = mtq.auto_quantize(
    model,
    constraints={"effective_bits": 7.5},          # tune: lower => more INT8, higher => more FP16/FP32
    quantization_formats=[mtq.INT8_DEFAULT_CFG],  # CNN-friendly INT8 config
    data_loader=calib_loader,
    forward_step=forward_step,
    loss_func=loss_func,                          # or forward_backward_step
    num_calib_steps=256,
    num_score_steps=128,
    method="gradient",                            # or "kl"
    verbose=True,
)

mto.save(model, "yolo11_autoquant_int8.pt")
```

Notes:
- `INT8_DEFAULT_CFG` is explicitly documented by ModelOpt as the recommended INT8 configuration for CNNs.
- The **sensitivity scores** are internal, but you can observe the outcome:
  - Layers that remain unquantized are considered too sensitive.
  - Layers that get INT8 are less sensitive under the `effective_bits` budget.
- You can adjust `effective_bits`:
  - Higher (e.g., 8.0): more layers stay in FP16/FP32, accuracy safer but less compression.
  - Lower (e.g., 6.0): more INT8 coverage, higher speedup but potentially more mAP drop.

### 2.2 Export to ONNX and TensorRT

Once you have a mixed-precision PyTorch model:

1. Export to ONNX using ModelOpt’s torch→ONNX helper (or plain `torch.onnx.export` plus ModelOpt’s helpers):

```python
from modelopt.torch._deploy.utils.torch_onnx import get_onnx_model

onnx_bytes, meta = get_onnx_model(
    model,
    dummy_input=(example_images,),   # match your YOLO11 input tensor shape
    model_name="yolo11_autoquant_int8",
    onnx_opset=13,
    dq_only=False,
)

with open("yolo11_autoquant_int8.onnx", "wb") as f:
    f.write(onnx_bytes)
```

2. Build TensorRT engines with mixed precision enabled (FP16 + INT8) and evaluate:

```bash
trtexec \
  --onnx=yolo11_autoquant_int8.onnx \
  --saveEngine=yolo11_autoquant_int8.plan \
  --fp16 --int8 --best
```

- TensorRT will respect explicit quantization/QDQ where present and otherwise pick the fastest kernels subject to allowed precisions.
- Validation script should compute mAP, latency, and throughput for:
  - Baseline FP16 engine (no INT8).
  - AutoQuantized mixed FP16/INT8 engine.

## 3. ONNX path: AutoCast for FP32/FP16 “sensitivity”

If you only have an ONNX model (e.g., from `models/yolo11/helpers/convert_to_onnx.py`), you can’t yet use `auto_quantize` directly. Instead, ModelOpt provides **AutoCast**:

- API: `modelopt.onnx.autocast.convert_to_mixed_precision`
- CLI: `python -m modelopt.onnx.autocast`
- It converts FP32 ONNX to a **mixed FP32–FP16 or FP32–BF16 graph**, and:
  - Uses activation magnitudes (from random or calibration data) to classify nodes.
  - Leaves **sensitive nodes** (high magnitude, large reduction depth, or matching exclude patterns) in FP32.
  - Casts other nodes to FP16/BF16 and injects cast ops.

Example (CLI):

```bash
python -m modelopt.onnx.autocast \
  --onnx_path=models/yolo11/onnx/yolo11n.onnx \
  --low_precision_type fp16 \
  --calibration_data=calib.npy \
  --providers cpu cuda:0 \
  --output_path=models/yolo11/onnx/yolo11n_fp32_fp16_mix.onnx \
  --log_level INFO
```

Example (Python API):

```python
from modelopt.onnx.autocast import convert_to_mixed_precision
import onnx

mixed = convert_to_mixed_precision(
    onnx_path="models/yolo11/onnx/yolo11n.onnx",
    low_precision_type="fp16",
    calibration_data="calib.npy",  # optional but recommended
    providers=["cuda:0", "cpu"],
    nodes_to_exclude=None,
    op_types_to_exclude=None,
    max_depth_of_reduction=None,
)

onnx.save(mixed, "models/yolo11/onnx/yolo11n_fp32_fp16_mix.onnx")
```

Observations:
- AutoCast is essentially a **sensitivity heuristic**:
  - High-magnitude / risky nodes are kept in FP32.
  - Others are cast to FP16.
- It is currently focused on FP16/BF16, not INT8.
- For YOLO11, this can be a quick way to identify “FP32-only” hotspots in the ONNX graph before introducing INT8.

## 4. ONNX INT8 path: manual sensitivity sweeps

ModelOpt’s ONNX PTQ (`modelopt.onnx.quantization.quantize` or `python -m modelopt.onnx.quantization`) supports:

- INT8/FP8/INT4, calibration, Q/DQ insertion, and some per-node/per-op inclusion/exclusion.
- Key arguments for selective quantization:
  - `op_types_to_quantize`, `op_types_to_exclude`
  - `nodes_to_quantize`, `nodes_to_exclude`
  - `calibrate_per_node` for large models

There is **not yet** an ONNX-level `auto_quantize` equivalent, so to approximate sensitivity you can:

1. Start with a fully quantized INT8 model (or quantize only canonical conv/gemm ops).
2. Measure mAP vs baseline.
3. Iteratively “de-quantize” suspected layers (i.e., add them to `nodes_to_exclude` or `op_types_to_exclude`) and re-quantize + rebuild TensorRT, tracking the accuracy improvements and latency impact.

This manual loop can be guided by:
- AutoCast output (FP32 nodes are likely more sensitive).
- Inspection of YOLO11 blocks (e.g., keep last detection head or NMS/plugin ops in FP16).

Example INT8 CLI skeleton:

```bash
python -m modelopt.onnx.quantization \
  --onnx_path=models/yolo11/onnx/yolo11n.onnx \
  --quantize_mode=int8 \
  --calibration_data=calib.npy \
  --calibration_method=max \
  --output_path=models/yolo11/onnx/yolo11n_int8_qdq.onnx \
  --op_types_to_exclude=Resize,Concat \
  --calibration_eps cuda:0 cpu \
  --calibrate_per_node
```

Then adjust `--op_types_to_exclude` / `--nodes_to_exclude` based on mAP results and/or AutoCast’s FP32 node set.

## 5. Practical recommendations for this repo

For YOLO11 mixed FP16/INT8 in this project:

1. **If you control the PyTorch model:**
   - Use `mtq.quantize` + `mtq.auto_quantize` on the YOLO11 PyTorch model with `INT8_DEFAULT_CFG` and a representative dataloader.
   - Choose an `effective_bits` target that gives acceptable mAP loss.
   - Export to ONNX and then build TensorRT engines.

2. **If you must stay on ONNX only:**
   - Use AutoCast to get a safe FP32/FP16 mix and identify sensitive nodes.
   - Use ONNX PTQ with `op_types_to_exclude` / `nodes_to_exclude` to quantize only less-sensitive parts of the graph to INT8.
   - Explore a small grid of configurations (fully quantized vs backbone-only quantized, etc.) and pick the best latency/accuracy trade-off.

3. **Always validate with real data:**
   - Sensitivity and mixed-precision decisions should be evaluated on the same distribution you care about (e.g., subset of COCO or your deployment dataset).
   - Keep evaluation scripts and configuration (effective bits, excluded ops, etc.) under version control so they can be reproduced later.

## 6. References

- NVIDIA TensorRT Model Optimizer (ModelOpt) GitHub: https://github.com/NVIDIA/TensorRT-Model-Optimizer
- ModelOpt ONNX PTQ guide: `extern/TensorRT-Model-Optimizer/docs/source/guides/_onnx_quantization.rst`
- ModelOpt PyTorch quantization & `auto_quantize`: `extern/TensorRT-Model-Optimizer/docs/source/guides/_pytorch_quantization.rst`
- AutoCast ONNX mixed precision: `extern/TensorRT-Model-Optimizer/docs/source/guides/8_autocast.rst`
- TensorRT “Working with Quantized Types” (explicit vs implicit, Q/DQ behavior): https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html

