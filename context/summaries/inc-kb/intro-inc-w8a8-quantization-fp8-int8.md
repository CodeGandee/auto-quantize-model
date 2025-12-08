# Intro: Intel Neural Compressor W8A8 Quantization (FP8 and INT8)

This note summarizes how Intel Neural Compressor (INC) handles W8A8 quantization (weights and activations both in 8-bit formats) and how that interacts with an RTX 5090 workflow.

## FP8 W8A8 with Intel Neural Compressor

- FP8 support in INC is designed around Intel Gaudi2 (HPU) and CPUs, not NVIDIA GPUs. The FP8 documentation explicitly lists support matrix entries only for HPU (full FP8 mode and FP8 QDQ mode) and CPU (FP8 QDQ mode only), with no NVIDIA backend.
- There are two ways INC represents FP8 models:
  - FP8 mode: tensors are stored and computed directly in FP8 (E4M3 or E5M2), with FP8 kernels; this currently targets Gaudi HPUs only.
  - FP8 QDQ mode: activations stay in higher precision and INC inserts Quantize/Dequantize pairs (QDQ) around tensors; this can run on CPU and is meant for frameworks that can later fuse FP8 QDQ into hardware-specific kernels.
- On an RTX 5090, you should treat INC’s FP8 as an **offline planner**, not a runtime backend:
  - Use CPU or Gaudi to run FP8 QDQ quantization and export a model with QDQ nodes and configuration.
  - Inspect the resulting graph (for example, ONNX QDQ) to see which layers were quantized to FP8 and what scales were used.
  - Re-implement or map that scheme into a CUDA-focused stack (for example vLLM + `llm-compressor` FP8, TorchAO FP8, or your own CUDA kernels).

### Using FP8 QDQ to extract a quantization scheme

- Configure INC’s FP8 quantization in QDQ mode, with a `dump_stats_path` so statistics are recorded and can be reused:
  - `fp8_config`: choose `E4M3` (default) or `E5M2`.
  - `mode`: `MEASURE`, `QUANTIZE`, or `AUTO` (AUTO runs measurement then quantization).
  - `dump_stats_path`: directory prefix where measurement files are written.
  - Optional `allowlist` and `blocklist` to control which modules (for example `Linear`, `Conv2d`, `BMM`) are quantized.
- High-level flow for FP8 QDQ quantization on CPU (pseudocode, refer to official docs for exact API names and options):

```python
from neural_compressor import quantization
from neural_compressor.config import PostTrainingQuantConfig

conf = PostTrainingQuantConfig(
    backend="ipex",  # or other supported backend for FP8
    approach="static",
    fp8_config="E4M3",
    mode="AUTO",
    dump_stats_path="./hqt_output/measure",
)

q_model = quantization.fit(
    model,
    conf=conf,
    calib_dataloader=calib_loader,
)

q_model.save("path/to/fp8_qdq_model")  # framework-specific save (Torch/ONNX/etc.)
```

- After quantization:
  - If you export to ONNX QDQ, scan the graph for `QuantizeLinear` / `DequantizeLinear` nodes; the attached `scale` (and `zero_point`, if present) initializers define the FP8 scaling factors for each tensor.
  - Combined with the FP8 format (E4M3 or E5M2), this lets you reconstruct a per-tensor W8A8 FP8 scheme, even if you do not run the model with INC on RTX hardware.
- For actual FP8 W8A8 inference on GPUs:
  - Use tools that explicitly support FP8 on NVIDIA (for example vLLM’s FP8 W8A8 pipeline with `llm-compressor`, or TorchAO FP8), and treat INC’s FP8 output as guidance for layer selection and scaling strategy rather than a directly deployable engine on RTX 5090.

## INT8 W8A8 with Intel Neural Compressor

- INC’s INT8 quantization is hardware-agnostic and designed for deployment on CPUs, GPUs, and Gaudi. It can quantize both weights and activations to INT8 (W8A8) using post-training quantization flows.
- For ONNX Runtime:
  - INC can generate an ONNX **QDQ** graph where `QuantizeLinear` / `DequantizeLinear` nodes wrap quantized tensors.
  - This QDQ-format ONNX model can then be executed on NVIDIA GPUs via ONNX Runtime CUDA, TensorRT, or other INT8-capable runtimes.
- Typical W8A8 post-training quantization flow with INC for an ONNX model (conceptual):

```python
from neural_compressor import quantization
from neural_compressor.config import PostTrainingQuantConfig

conf = PostTrainingQuantConfig(
    backend="onnxrt_qdq",  # ONNX Runtime QDQ backend
    device="cpu",          # quantization itself can run on CPU
    approach="static",     # calibration-based PTQ
)

q_model = quantization.fit(
    model="path/to/fp32_model.onnx",
    conf=conf,
    calib_dataloader=calib_loader,
)

q_model.save("path/to/int8_w8a8_qdq.onnx")
```

- Once you have `int8_w8a8_qdq.onnx`:
  - You can inspect the graph (for example with Netron or a small `onnx` Python script) to see exactly which layers were quantized and read the `scale` and `zero_point` values from `QuantizeLinear` initializers.
  - You can run this model directly on an RTX 5090 via:
    - ONNX Runtime with CUDA EP, or
    - TensorRT (either by importing the ONNX model directly or via ONNX Runtime’s TensorRT EP).
- Because INT8 QDQ is standardized in ONNX, this W8A8 INT8 flow is the practical way to:
  - Use INC to search and tune quantization strategies, and
  - Deploy and debug those strategies on an RTX 5090.

## Practical guidance for Qwen2.5-VL-3B on RTX 5090

- FP8 W8A8:
  - Use INC’s FP8 QDQ mode only if you want to study which layers can tolerate FP8 and what scales are chosen; do quantization on CPU or Gaudi, export to ONNX QDQ, and treat the result as a reference.
  - For actual FP8 W8A8 inference on RTX 5090, build the deployment with a NVIDIA-focused toolchain such as vLLM + `llm-compressor` FP8 or TorchAO FP8.
- INT8 W8A8:
  - Use INC’s ONNX Runtime INT8 PTQ path to produce a W8A8 QDQ ONNX model.
  - Inspect and run that model on RTX 5090 with ONNX Runtime CUDA or TensorRT, using it as the main deployable artifact.

## References

- Intel Neural Compressor overview: https://www.intel.com/content/www/us/en/developer/tools/oneapi/neural-compressor.html
- FP8 Quantization — Intel Neural Compressor docs (3.x): https://intel.github.io/neural-compressor/latest/docs/source/3x/PT_FP8Quant.html
- General quantization docs (INT8 and recipes): https://github.com/intel/neural-compressor/blob/master/docs/source/quantization.md
- vLLM FP8 W8A8 documentation: https://docs.vllm.ai/en/latest/features/quantization/fp8.html

