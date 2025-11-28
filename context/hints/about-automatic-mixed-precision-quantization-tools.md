Here are the *practical*, battle-tested tools that actually **auto-pick mixed precision per layer** (or node) instead of making you hand-tag layers. I’ll start with NVIDIA-native options, then cross-vendor/open-source toolkits, then “power user” research-grade libraries.

---

## 1. NVIDIA ecosystem (best fit for your GPUs)

### 1.1 TensorRT Model Optimizer (ModelOpt) – `auto_quantize` (PyTorch)

**What it is:**
NVIDIA’s official open-source model-optimization library (Apache-2.0) that sits *before* TensorRT / TensorRT-LLM.([GitHub][1])

**Why you care:** it has an **`auto_quantize` PTQ algorithm** that *searches per-layer quantization formats* under a global constraint like “effective bits”. It’s exactly an automatic mixed-precision selector:

> “`auto_quantize` … quantizes a model by **searching for the best quantization format per-layer** while meeting the performance constraint… You may specify `effective_bits` such as 4.8… `AutoQuantize` will automatically quantize highly sensitive layers in `FP8_DEFAULT_CFG` while keeping less sensitive layers in `NVFP4_DEFAULT_CFG` (and even **skip quantization** for extremely sensitive layers)”([NVIDIA GitHub][2])

So you can, for example, tell it “I want ~4.8 effective bits overall” and it will:

* Try NVFP4 (4-bit) on most layers
* Use FP8 on layers it deems sensitive
* Optionally leave a few layers unquantized if they destroy accuracy

All this is driven by a calibration dataloader + your loss function.

**Pros for you**

* Official NVIDIA, open source, actively maintained.([GitHub][1])
* Supports **block-wise INT4 / FP8** etc. for *low-bit weight* and activation formats.([NVIDIA GitHub][2])
* Directly exportable to ONNX → TensorRT for production.

**When to use**

* You’re okay with NVFP4 / FP8 style formats (FP-ish low-bit), and want **automatic per-layer FP4/FP8 mixing** rather than hand-picking.
* Ideal if you’ll eventually deploy via TensorRT / TensorRT-LLM.

---

### 1.2 TensorRT Model Optimizer – **AutoCast (ONNX)**

**What it is:**
ModelOpt’s **AutoCast** tool that converts FP32 ONNX models to mixed FP16/BF16.

> “AutoCast is a tool for converting FP32 ONNX models to mixed precision FP32–FP16 or FP32–BF16… AutoCast **intelligently selects nodes to keep in FP32** precision to maintain model accuracy while benefiting from reduced precision on the rest of the nodes.”([NVIDIA GitHub][3])

It classifies nodes using actual activation magnitudes (optionally from your calibration data) and keeps high-magnitude / reduction-heavy or otherwise risky ops in FP32.([NVIDIA GitHub][3])

**Why it’s relevant**

* It’s basically an **automatic “sensitive node detector” for FP16/BF16**.
* Works on any ONNX model (ViT, SAM, YOLOvX) and uses ONNX Runtime + TensorRT execution provider under the hood.([NVIDIA GitHub][3])

**Pros**

* Very low friction for **w8a16 / fp32-fp16 style mixes** on NVIDIA.
* Node classification is data-driven (optionally using your calibration set).([NVIDIA GitHub][3])

**When to use**

* You mainly want **FP16/BF16 mixed with FP32**, rather than integer 4/8-bit.
* Quick path to “safe FP16” for complex vision graphs.

---

### 1.3 Core TensorRT – built-in per-layer precision + debug-precision tool

TensorRT itself will choose per-layer precision (FP32/FP16/BF16/FP8/INT8) for performance if you enable the relevant `BuilderFlag`s:

> “TensorRT can execute a layer in FP32, FP16, BF16, FP8, or INT8… **By default, TensorRT chooses to run a layer in a precision that results in optimal performance.** Sometimes, this can result in poor accuracy.”([NVIDIA Docs][4])

NVIDIA also documents an **experimental “debug precision” tool**:

> “An experimental *debug precision* tool can help **automatically find layers to run with high precision**.”([NVIDIA Docs][4])

So the workflow is:

1. Build a mixed-precision engine (FP16+INT8 enabled).
2. If accuracy is off, run the debug precision tool → it figures out which layers must be bumped back to higher precision.

**Caveats**

* It’s more “auto-debugging” than a full **bit-width search engine**, and it’s mostly FP16/INT8, not (say) per-layer INT4.
* You control only the allowed precisions; TensorRT chooses the exact mix.

**When to use**

* You’re already heavy on TensorRT and want an **accuracy-driven FP16/INT8 fallback** mechanism with minimal extra tooling.

---

## 2. Cross-vendor / general open-source toolkits

These you can run on NVIDIA GPUs for *search* (PyTorch/ONNX), even if the vendor’s own deployment stack is Intel/Qualcomm.

### 2.1 Intel Neural Compressor (INC)

**What it is:**
Intel’s official **open-source** Python library for compression (quantization, pruning, distillation).([Intel][5])

Key features:

* “Quantize activations and weights to INT8, FP8, or a **mixture of FP32, FP16, FP8, bfloat16, and INT8**”([Intel][5])
* “Automatic accuracy-driven tuning strategies” with an `autotune` API.([PyTorch Documentation][6])
* Detects accelerators including **CUDA** (set `INC_TARGET_DEVICE=cuda`).([PyTorch Documentation][6])

In v3.0 they explicitly highlight **mixed-precision + fallback**:

> INC 2.x “supported BF16 mixed precision … and provided an **accuracy-driven tuning function to reduce accuracy loss by fallback to FP32 when needed**.”([Medium][7])

Practically, INC’s tuning process explores configs like:

* Layer-wise int8 vs fp32/fp16
* Work with different quantization recipes (SmoothQuant, WOQ, FP8, etc.)
* Stop when accuracy drop ≤ your threshold.

**Pros**

* Mature, used as *the* official quant tool for Intel, but **backend-agnostic for PyTorch/ONNX** so you can run the tuning on NVIDIA hardware.([Intel][5])
* Good for **automatic selection of which layers stay high precision** when doing INT8 / FP8 PTQ.

**When to use**

* You want an **accuracy-driven “try many configs until accuracy is OK”** tool that works with PyTorch/ONNX.
* You’re fine driving NVIDIA inference in PyTorch / ONNXRuntime / TensorRT after INC decides per-op precision.

---

### 2.2 OpenVINO NNCF (Neural Network Compression Framework)

**What it is:**
Intel’s open-source compression framework (PyTorch / ONNX / OpenVINO). It directly supports **mixed-precision weight quantization** and sensitivity-based assignment.([GitHub][8])

The `compress_weights` API includes modes like `INT4_SYM` that are explicitly **mixed-precision**:

> “`INT4_SYM` stands for a **mixed-precision weights quantization** with 4-bit integer as a primary precision… All embeddings and the last layer are always compressed to a backup precision [INT8]… others are quantized to 4-bit or to backup precision **depending on criteria and the given ratio**.”([OpenVINO Toolkit][9])

You can also provide:

* A `dataset` used to detect activation outliers, and
* A `sensitivity_metric` (Hessian-based, activation variance, etc.) so that **“more sensitive layers receive a higher precision.”**([OpenVINO Toolkit][9])

OpenVINO’s older POT tool had an explicit **AccuracyAwareQuantization** algorithm that:

* Quantizes everything, measures accuracy
* Ranks layers by their contribution to accuracy drop, and
* Iteratively **reverts the worst layers back to FP32** until `maximal_drop` is satisfied.([OpenVINO Documentation][10])

That is literally an automatic mixed-precision search (FP32 + INT8).

**Pros**

* Very concrete implementation of **automatic per-layer INT4/INT8 vs FP32 fallback** using sensitivity metrics.([OpenVINO Toolkit][9])
* Heavily used in OpenVINO workflows (YOLO, classification, NLP).([GitHub][8])

**Caveats**

* Deployment is really optimized for **OpenVINO on Intel CPUs/GPUs**. For NVIDIA you’d typically:
  PyTorch → NNCF to find per-layer bits → export ONNX → rebuild for TensorRT.

**When to use**

* You want a **ready-made implementation of “metrics + ratio → per-layer INT4/INT8 assignment”**, and you’re okay not deploying via OpenVINO.

---

### 2.3 Qualcomm AIMET (AI Model Efficiency Toolkit)

**What it is:**
Qualcomm’s official quantization toolkit (PyTorch/TF/ONNX).([quic.github.io][11])

The **AutoQuant** API combines several PTQ techniques and includes an *“Automatic Mixed Precision”* step:

> “AutoQuant includes 1) batchnorm folding, 2) cross-layer equalization, 3) Adaround, and **4) Automatic Mixed Precision (if enabled)**… applied in a best-effort manner until the model meets the evaluation goal (allowed_accuracy_drop).([quic.github.io][12])

AIMET also supports **creating mixed-precision models** and exporting them to ONNX.([Radxa Docs][13])

**Pros**

* Production-grade toolkit used across Qualcomm’s Snapdragon deployment.
* Automatic mixed precision is baked into its PTQ pipeline and is **accuracy-targeted** (similar spirit to OpenVINO’s accuracy-aware quant).

**Caveats**

* Deployment docs are heavily focused on Qualcomm SoCs / QNN / SNPE.
* Still useful as an *off-line search engine* for mixed-precision policies on PyTorch models; you can then port to your own runtime.

**When to use**

* You want another **big-vendor, mature implementation** of accuracy-aware mixed precision to learn from or to prototype with, and hardware vendor coupling is acceptable.

---

## 3. Framework-level / “power user” libraries

These are more “research-driven but robust” and give you finer control. Good if you’re okay running Python code, inspecting configs, and wiring into your own deployment stack.

### 3.1 PyTorch TorchAO – BO-based automatic mixed-precision (weight-only)

TorchAO is the “Advanced Optimization” (AO) / efficiency toolkit under the PyTorch org.([GitHub][14])

Recent releases added:

> “**Automatic mixed-precision quantization through Bayesian Optimization**… a BO tool using Ax to **auto search mixed-precision weight-only quantization configuration, i.e., bit-width and group size of `intN_weight_only(bit_width, group_size)` for each layer**. It also includes a sensitivity analysis tool (Hessian/Fisher traces) as an optional step to customize and improve BO search.”([GitHub][15])

This tool is currently demonstrated on Llama3-8B (int4/int8 weight-only) but is generic PyTorch, so you can in principle point it at ViT / YOLO / SAM-style backbones.

**Pros**

* Official PyTorch project, actively maintained, designed to work with `torch.compile` and CUDA.([GitHub][15])
* Fully automatic per-layer **intN bit-width + group size search** under constraints (model size, throughput vs perplexity – you could replace perplexity with your metric).([GitHub][15])

**Caveats**

* Examples and docs are LLM-centric; you’ll need to adapt the scripts for vision models.
* Weight-only quant at the moment; you’d need to design activation quant around it.

**When to use**

* You want a **search engine for per-layer int4/int8 weight-only** that you can hack for ViT/YOLO.
* You’re comfortable reading/adjusting PyTorch scripts and integrating with your own eval metric.

---

### 3.2 HAWQ library (Hessian-Aware Quantization, with TVM integration)

The HAWQ GitHub repo is an **advanced PyTorch quantization library** implementing HAWQ-V2/V3.

> “HAWQ is an advanced quantization library written for PyTorch. HAWQ enables low-precision and **mixed-precision** uniform quantization, with direct hardware implementation through TVM.”([GitHub][16])

Core idea: compute per-layer Hessian metrics → solve an ILP to pick per-layer 4/8-bit under size/latency constraints.([Proceedings of Machine Learning Research][17])

**Pros**

* Very well-cited research, integrated into TVM’s 4-bit/mixed-precision quantization stack.([Proceedings of Machine Learning Research][17])
* Good match if you’re already using TVM or want to simulate on TVM then deploy elsewhere.

**Caveats**

* More “research toolkit” than turnkey: you’ll typically run the provided training/quant scripts for specific models.
* You’ll need to wire the output to your own runtime (e.g., export to ONNX → TensorRT).

**When to use**

* You want **Hessian-based per-layer bit assignment** and are okay with more engineering effort.
* Especially attractive if you’re already in the TVM ecosystem for NPU design / performance modelling.

---

## 4. What I’d actually use for your workflow

Given your constraints (NVIDIA GPUs, vision models, want automatic selection):

1. **For FP16/FP32 (and BF16) mixing on ViT/SAM/YOLO**

   * Use **ModelOpt AutoCast (ONNX)** to automatically keep numerically sensitive nodes in FP32 and cast the rest to FP16/BF16.([NVIDIA GitHub][3])
   * If you’re in TensorRT already, combine it with **TensorRT’s debug-precision tool** to auto-identify layers that must stay high-precision.([NVIDIA Docs][4])

2. **For low-bit (FP4/FP8-style) mixed precision**

   * Use **ModelOpt `auto_quantize`** with NVFP4/FP8 configs and an `effective_bits` constraint – this is NVIDIA’s own per-layer mixed-precision search and plays nicely with TensorRT deployment.([NVIDIA GitHub][2])

3. **For integer mixed precision (INT8/INT4, w8a16/w4a16 style)**

   * If you want something more “production-toolkit-like”, start with **Intel Neural Compressor** or **NNCF** as an *off-line search engine* for “which layers can tolerate lower bits”, then port the bit allocation to your deployment.([Intel][5])
   * If you’re okay with deeper coding, experiment with **TorchAO’s Bayesian mixed-precision search** for weight-only INT4/INT8 and adapt it to ViT/YOLO.([GitHub][15])

4. **If you want to really dig into the theory and custom hardware**

   * Pull **HAWQ** and/or **NNCF’s sensitivity-metric machinery** as reference implementations of Hessian / activation-variance based layer sensitivity, and integrate them into your own search loop or hardware-aware ILP.

If you tell me your current stack (pure PyTorch vs ONNX→TensorRT pipeline) and whether you prefer weight-only vs full (weights+acts) quant, I can sketch a concrete “drop-in” workflow using one of these toolchains.

[1]: https://github.com/NVIDIA/TensorRT-Model-Optimizer "GitHub - NVIDIA/TensorRT-Model-Optimizer: A unified library of state-of-the-art model optimization techniques like quantization, pruning, distillation, speculative decoding, etc. It compresses deep learning models for downstream deployment frameworks like TensorRT-LLM or TensorRT to optimize inference speed."
[2]: https://nvidia.github.io/TensorRT-Model-Optimizer/guides/_pytorch_quantization.html "PyTorch Quantization — Model Optimizer 0.0.1.dev1+g263b2b72f"
[3]: https://nvidia.github.io/TensorRT-Model-Optimizer/guides/8_autocast.html "AutoCast (ONNX) — Model Optimizer 0.0.1.dev1+g263b2b72f"
[4]: https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html "Best Practices — NVIDIA TensorRT Documentation"
[5]: https://www.intel.com/content/www/us/en/developer/tools/oneapi/neural-compressor.html "Perform Model Optimization Using Intel® Neural Compressor"
[6]: https://docs.pytorch.org/tutorials/recipes/intel_neural_compressor_for_pytorch.html "Ease-of-use quantization for PyTorch with Intel® Neural Compressor — PyTorch Tutorials 2.9.0+cu128 documentation"
[7]: https://medium.com/intel-analytics-software/intel-neural-compressor-v3-0-a-quantization-tool-across-intel-hardware-9856adee6f11 "Quantization on Intel Gaudi Series AI Accelerators | by Intel(R) Neural Compressor & AutoRound | Intel Analytics Software | Medium"
[8]: https://github.com/openvinotoolkit/nncf "GitHub - openvinotoolkit/nncf: Neural Network Compression Framework for enhanced OpenVINO™ inference"
[9]: https://openvinotoolkit.github.io/nncf/autoapi/nncf/ "nncf - NNCF"
[10]: https://docs.openvino.ai/2023.3/accuracy_aware_README.html "[Deprecated] AccuracyAwareQuantization Parameters — OpenVINO™  documentation"
[11]: https://quic.github.io/aimet-pages/index.html?utm_source=chatgpt.com "AIMET Documentation - Qualcomm Innovation Center"
[12]: https://quic.github.io/aimet-pages/AimetDocs/apiref/torch/v1/autoquant.html?utm_source=chatgpt.com "aimet_torch.v1.auto_quant - AIMET"
[13]: https://docs.radxa.com/en/fogwise/airbox-q900/ai-dev/aimet?utm_source=chatgpt.com "AIMET Quantization Tool"
[14]: https://github.com/pytorch/ao?utm_source=chatgpt.com "pytorch/ao: PyTorch native quantization and sparsity for ..."
[15]: https://github.com/pytorch/ao/releases "Releases · pytorch/ao · GitHub"
[16]: https://github.com/Zhen-Dong/HAWQ "GitHub - Zhen-Dong/HAWQ: Quantization library for PyTorch. Support low-precision and mixed-precision quantization, with hardware implementation through TVM."
[17]: https://proceedings.mlr.press/v139/yao21a.html?utm_source=chatgpt.com "HAWQ-V3: Dyadic Neural Network Quantization"
