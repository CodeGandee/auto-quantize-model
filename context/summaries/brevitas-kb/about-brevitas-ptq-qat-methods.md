# About: Brevitas quantization + PTQ/QAT methods (what it provides)

This note summarizes what “quantization methods” and “QAT methods” Brevitas provides, with an emphasis on what is actually implemented (quantizer building blocks, calibration utilities, and named PTQ algorithms).

## TL;DR

- Brevitas is a PyTorch fake-quantization library: it helps you build/transform quantized PyTorch graphs, run PTQ calibration / PTQ algorithms, and run QAT fine-tuning, then export to formats like ONNX QCDQ.
- Brevitas is not an inference engine: it does not itself run low-bit kernels; exports are meant for downstream runtimes/toolchains.
- “Methods” in Brevitas are mostly expressed as (1) quantizer choices (how to compute scale/zero-point/rounding), (2) graph transforms (how/where to insert quant layers and pre-processing transforms), and (3) a set of named PTQ algorithms (especially for LLMs).

## 1) Quantization building blocks (the core of Brevitas)

### 1.1 Quantized layers (`brevitas.nn`)

Brevitas provides quantized replacements of common PyTorch layers (e.g., `QuantConv2d`, `QuantLinear`, `QuantReLU`, `QuantIdentity`) where you can independently configure quantization of weights / activations / bias / output.

Source links:
- Docs (getting started): https://xilinx.github.io/brevitas/dev/getting_started.html
- Vendored source (this repo): extern/brevitas/src/brevitas/nn/

### 1.2 Quantizers (`brevitas.quant.*`) and what “methods” means here

A lot of what people call “quantization methods” in Brevitas is choosing a quantizer class (and its parameters):

- Integer vs minifloat formats (Brevitas supports both integer quantization and custom floating-point/minifloat-style emulation).
- Per-tensor vs per-channel vs per-group granularity.
- Symmetric vs asymmetric quantization (zero-point behavior).
- How scales / zero-points are computed and represented (statistic-based initialization, percentile-based ranges, MSE-based scale/zero-point variants, and in some cases learned parameters).

Example: MSE-initialized scale/zero-point quantizers exist for both weights and activations (useful for PTQ initialization and potentially as trainable parameters in QAT depending on configuration).

Source links:
- Vendored quantizers: extern/brevitas/src/brevitas/quant/scaled_int.py (e.g., `Int8WeightPerTensorFloatMSE`, `Int8ActPerTensorFloatMSE`)
- Vendored quantizers: extern/brevitas/src/brevitas/quant/shifted_scaled_int.py (asymmetric + MSE variants)
- Docs: https://xilinx.github.io/brevitas/v0.12.1/user_guide/index.html

## 2) PTQ (Post-Training Quantization) in Brevitas

Brevitas supports PTQ in two “shapes”:

1) “Simple PTQ” = insert quantized layers + initialize quantizer parameters (often from weights and/or simple activation stats).
2) “Algorithmic PTQ” = apply a named PTQ algorithm that explicitly optimizes rounding/weights (and sometimes applies pre-processing transforms that reduce outliers).

### 2.1 Calibration utilities (activation stats, bias correction)

Brevitas includes explicit calibration context managers and helpers under `brevitas.graph.calibrate`.

- `calibration_mode(model)` is used to collect activation statistics for runtime activation quantizers while typically disabling the actual quantized values from perturbing the stats collection (observer-only behavior).
- `bias_correction_mode(model)` applies a bias correction procedure (bias update/merge) intended to reduce error introduced by quantization.

Source links:
- Vendored implementation: extern/brevitas/src/brevitas/graph/calibrate.py (exports `calibration_mode`, `bias_correction_mode`)

Minimal example (conceptual):

```python
from brevitas.graph.calibrate import calibration_mode, bias_correction_mode

quant_model.eval()
with calibration_mode(quant_model):
    for x in calib_loader:
        _ = quant_model(x)

with bias_correction_mode(quant_model):
    for x in calib_loader:
        _ = quant_model(x)
```

Important nuance: some quantizers are “stateful” and collect stats for a fixed number of steps by default; calibration helpers can extend/override this behavior (see `extend_collect_stats_steps` / `set_collect_stats_to_average` in the same module).

### 2.2 Graph / layer insertion (“PTQ by transformation”)

Brevitas supports programmatic graph transforms to convert an FP model into a quantized model, for example by mapping `torch.nn.Conv2d -> brevitas.nn.QuantConv2d` with a chosen weight quantizer and optional activation quantizer.

The common API you will see is `layerwise_quantize(...)` (used in this repo’s YOLOv10m Brevitas scripts).

Source links:
- Docs: https://xilinx.github.io/brevitas/dev/getting_started.html
- Vendored helpers: extern/brevitas/src/brevitas/graph/quantize.py and extern/brevitas/src/brevitas/graph/quantize_impl.py

Minimal example (conceptual):

```python
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.graph.quantize import layerwise_quantize
from brevitas.quant.scaled_int import Int8WeightPerChannelFloat, Int8ActPerTensorFloat

compute_layer_map = {
  nn.Conv2d: (qnn.QuantConv2d, {
    "weight_quant": Int8WeightPerChannelFloat,
    "weight_bit_width": 8,
    "input_quant": Int8ActPerTensorFloat,
    "return_quant_tensor": False,
  }),
}

qmodel = layerwise_quantize(fp_model, compute_layer_map=compute_layer_map)
```

### 2.3 Named PTQ algorithms included in Brevitas (LLM-heavy)

Brevitas includes implementations of multiple PTQ algorithms, mainly in `brevitas.graph` and in the Brevitas LLM example entrypoint. These are not just “calibration rules”; they are algorithmic procedures (Hessian/Greedy/rounding optimization, outlier mitigation transforms, etc.).

Concrete examples present in the codebase and docs:

- GPTQ / OPTQ-style PTQ (implemented as `brevitas.graph.gptq.GPTQ`, adapted from IST-DASLab GPTQ code).
  - Source: extern/brevitas/src/brevitas/graph/gptq.py
- GPFQ (Greedy Path Following Quantization).
  - Source: extern/brevitas/src/brevitas/graph/gpfq.py
- Qronos (sequential error-correcting rounding/update).
  - Source: extern/brevitas/src/brevitas/graph/qronos.py
  - Docs: https://xilinx.github.io/brevitas/v0.12.1/papers/qronos.html
- MagR (weight magnitude reduction) as PTQ pre-processing.
  - Source: extern/brevitas/src/brevitas/graph/magr.py
- Activation equalization / SmoothQuant-style transforms.
  - Source: extern/brevitas/src/brevitas/graph/equalize.py

Additionally, Brevitas’ GGUF export docs explicitly list PTQ algorithms compatible with weight-only quantization in their LLM entrypoint, including:

- GPTQ
- AWQ
- Learned Round
- QuaRot / SpinQuant (rotation-based transforms; in Brevitas these are represented by Hadamard/rotation utilities and related transforms)
- MagR

Source link:
- Docs (GGUF export): https://xilinx.github.io/brevitas/v0.12.1/user_guide/export_gguf.html

Notes on “RTN (round-to-nearest)” baseline:
- In Brevitas docs (Qronos page) RTN is used as a baseline meaning “directly cast/round weights to the target format with no calibration/optimization”. In practice, this corresponds to using quantized layers/quantizers without applying an additional PTQ algorithm.

## 3) QAT (Quantization-Aware Training) in Brevitas

Brevitas supports QAT by design: quantization is “in the forward pass” (fake quantization), and gradients flow via STE-style approximations depending on the quantizer/rounding operator.

### 3.1 Standard QAT: train/fine-tune with quantized layers

Once you have a quantized model definition (hand-written with `brevitas.nn` or produced via graph transforms), QAT is “just PyTorch training”:

- Keep quant layers enabled in training.
- Optimize the original weights and any learnable quantization parameters (if configured as `nn.Parameter`s).

Source links:
- Docs (getting started): https://xilinx.github.io/brevitas/dev/getting_started.html
- Docs (compile + shared PTQ/QAT infra): https://xilinx.github.io/brevitas/v0.12.1/user_guide/compile.html

### 3.2 Learnable rounding / learnable quantization parameters

Brevitas includes a “Learned Round” implementation (a form of adaptive/learned rounding useful in PTQ fine-tuning and can also be viewed as a QAT-style mechanism).

Source links:
- Vendored implementation: extern/brevitas/src/brevitas/core/function_wrapper/learned_round.py
- Vendored enum/config plumbing: extern/brevitas/src/brevitas/inject/enum.py (see `LearnedRoundImplType`)

Also note from the GGUF export guide: changing scaling implementations to parameter-based variants can enable learning scale factors during QAT (the doc calls out `weight_scaling_impl_type` / `parameter_from_stats` as a way to make scale learnable, though “not tested” in that specific GGUF context).

Source link:
- Docs (GGUF export): https://xilinx.github.io/brevitas/v0.12.1/user_guide/export_gguf.html

### 3.3 PTQ → QAT workflows

Brevitas explicitly supports the common workflow “PTQ initialization, then short QAT fine-tune”, using the same underlying quantized model definition and quantizer infrastructure.

Source link:
- Docs (getting started): https://xilinx.github.io/brevitas/dev/getting_started.html

## 4) Export (ONNX QCDQ, GGUF)

Brevitas can export quantized models to representations expected by other toolchains.

- ONNX QCDQ export via `brevitas.export.export_onnx_qcdq` (Brevitas extends standard ONNX QDQ with an extra `Clip` to represent <=8-bit constraints).
  - Docs (getting started export section): https://xilinx.github.io/brevitas/dev/getting_started.html
  - Source: extern/brevitas/src/brevitas/export/
- GGUF export for LLM workflows (see the LLM entrypoint docs above).

## 5) Practical guidance (how to interpret “methods” for your use case)

- If you want “classic calibrators like KL-divergence histogram calibration”: Brevitas is not primarily organized around “choose KL/entropy calibrator” like some ONNX/TensorRT toolchains; instead you typically choose quantizers/statistics/percentiles/MSE variants and/or pick a named PTQ algorithm (GPTQ/GPFQ/Qronos/AWQ/etc.) depending on the model class.
- For CNN/vision PTQ+QCDQ (like YOLO): the most direct mapping is usually “weight-only PTQ (RTN-like) + activation calibration via `calibration_mode`” and optionally “bias correction”; algorithmic PTQ like GPTQ/GPFQ/Qronos is much more common in LLM weight-only contexts.

## References

- Brevitas repo: https://github.com/Xilinx/brevitas
- Brevitas docs (latest): https://xilinx.github.io/brevitas/dev/
- Getting started (PTQ/QAT overview + ONNX export): https://xilinx.github.io/brevitas/dev/getting_started.html
- GGUF export (lists PTQ algorithms in LLM entrypoint): https://xilinx.github.io/brevitas/v0.12.1/user_guide/export_gguf.html
- Qronos paper page (lists GPTQ/GPFQ/Qronos and shows CLI usage): https://xilinx.github.io/brevitas/v0.12.1/papers/qronos.html
