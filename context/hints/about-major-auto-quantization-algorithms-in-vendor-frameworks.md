This note summarizes the **major automatic quantization algorithms** used by three key vendor frameworks for LLMs and other large models, with a focus on:

- What quantization methods they actually expose for LLM usage.
- How they implement **mixed-precision selection** and which knobs control the search effort.
- Where the algorithms live in source code under `extern/`.
- Pointers to the relevant **papers and repositories** for the algorithms themselves.

Frameworks covered:

- NVIDIA TensorRT Model Optimizer (**ModelOpt**)
- Intel Neural Compressor (**INC**)
- OpenVINO + Neural Network Compression Framework (**NNCF**)

---

## 1. NVIDIA ModelOpt (TensorRT Model Optimizer)

### 1.1 Major quantization methods for LLMs

ModelOpt’s LLM quantization stack lives in `extern/TensorRT-Model-Optimizer` and is centered around:

- **Quantization formats** (LLM-focused):
  - INT8, FP8, NVFP4 (FP4-like), INT4, MXFP4, MXFP8, W4A8, etc.
  - Configs defined in `modelopt/torch/quantization/config.py`, e.g. `INT8_SMOOTHQUANT_CFG`, `INT4_AWQ_CFG`, `W4A8_AWQ_BETA_CFG`, `FP8_DEFAULT_CFG`, `NVFP4_DEFAULT_CFG`, `MXFP8_DEFAULT_CFG`.
- **Algorithms used in LLM examples**:
  - **SmoothQuant** (W8A8 / W4A8 activation-weight rebalancing):
    - Implemented in calibration and quantization helpers under `modelopt/torch/quantization/model_calib.py` and `modelopt/torch/quantization/mode.py`.
  - **AWQ / AWQ-lite / AWQ-Clip** (activation-aware weight-only quantization for INT4/NVFP4/W4A8):
    - Implemented via helper classes and functions such as `AWQLiteHelper`, `awq_clip` in `modelopt/torch/quantization/model_calib.py`.
  - **AutoQuantize** (automatic mixed precision over a set of quantization “recipes”):
    - Front-end API: `auto_quantize` in `modelopt/torch/quantization/model_quant.py`.
    - Core searchers: `AutoQuantizeGradientSearcher` and `AutoQuantizeKLDivSearcher` in `modelopt/torch/quantization/algorithms.py`.
  - **ONNX AutoCast** (FP32→mixed FP16/BF16): provided on the ONNX side for mixed-precision floating point, but less LLM-specific than the PyTorch `auto_quantize` flow.

LLM examples (`extern/TensorRT-Model-Optimizer/examples/llm_eval`, `examples/llm_ptq`) explicitly use configurations such as `INT4_AWQ_CFG`, `W4A8_AWQ_BETA_CFG`, `FP8_DEFAULT_CFG`, `NVFP4_DEFAULT_CFG`, often in combination with `mtq.auto_quantize` for mixed NVFP4/FP8/W4A8 models.

### 1.2 Mixed-precision search and search-effort knobs

ModelOpt’s main mixed-precision engine is `auto_quantize` for PyTorch:

- **AutoQuantizeGradientSearcher** (`modelopt/torch/quantization/algorithms.py`)
  - Uses a **Taylor expansion of the loss w.r.t. the quantized layer output**, with **Fisher information substituted for the Hessian**, to estimate the loss increase per layer per quantization recipe.
  - Solves a **linear programming (LP) problem** to pick the combination of quantization recipes (e.g. NVFP4, FP8, NONE) that minimizes total score under:
    - An `effective_bits` (compression) constraint.
    - Optional hardware/format constraints (which recipes are allowed per layer).
  - Supports grouping rules (e.g. group Q/K/V or MLP experts) via quant-grouping patterns to share a format across related modules.
- **AutoQuantizeKLDivSearcher** (`modelopt/torch/quantization/algorithms.py`)
  - Alternative searcher that scores candidate per-layer configurations via **KL divergence** between logits from the original model and the quantized model on a calibration set.
  - Uses the same LP machinery, with per-layer KL scores instead of gradient/Fisher-based scores.

Key knobs controlling **search effort and behavior** (see `AutoQuantizeGradientSearcher.default_search_config` and `auto_quantize` signature in `modelopt/torch/quantization/model_quant.py`):

- `effective_bits`: target average “effective bits” for the model (e.g. `4.8` for a NVFP4/FP8 mix).
- `quant_cfgs`: list of quantization recipes (configs) to consider, e.g. `[NVFP4_DEFAULT_CFG, FP8_DEFAULT_CFG, NONE]`.
- `auto_quantize_method`: `"gradient"` or `"kl_div"` (see examples in `examples/llm_eval/quantization_utils.py`).
- `auto_quantize_score_size`: number of calibration samples used to estimate scores.
- `loss_func` / `forward_step` / `forward_backward_step`: define how to compute the loss and run backprop to obtain activation gradients (for the gradient/Fisher-based mode).
- Gradient computation/memory knobs:
  - Custom support registration: `AutoQuantizeGradientSearcher.register_custom_support`, which lets the framework enable gradient checkpointing and selective parameter gradients to reduce memory/time for LLMs.

LLM usage pattern (simplified from `examples/llm_eval/quantization_utils.py`):

```python
import modelopt.torch.quantization as mtq

model = ...  # HF transformer
quant_cfgs = [mtq.NVFP4_DEFAULT_CFG, mtq.FP8_DEFAULT_CFG, mtq.NONE_CFG]

quantized_model, state_dict = mtq.auto_quantize(
    model,
    quant_cfgs=quant_cfgs,
    effective_bits=4.8,
    method="gradient",            # or "kl_div"
    score_size=128,               # number of samples for scoring
    loss_func=loss_fn,            # loss over model outputs
    forward_step=forward_step,    # how to run a batch
)
```

For ONNX AutoCast (mixed FP16/BF16), the knobs are mainly ONNX-AutoCast classifier thresholds (e.g. `data_max`, `init_max`, `max_depth_of_reduction`) and optional calibration data; sensitive nodes are kept in FP32 based on activation ranges.

---

## 2. Intel Neural Compressor (INC)

### 2.1 Major quantization methods for LLMs

INC’s LLM quantization stack lives under `extern/neural-compressor` and is documented in `docs/source/quantization_weight_only.md` and transformer helpers. The key LLM algorithms are:

- **Weight-only quantization algorithms for LLMs** (`docs/source/quantization_weight_only.md`):
  - **RTN** (Round-to-Nearest).
  - **AWQ** – Activation-Aware Weight Quantization.
  - **GPTQ** – second-order per-row weight-only PTQ.
  - **TEQ** – Trainable Equivalent Transformation, AWQ-inspired with trainable scaling.
  - NF4/FP4 formats and MX formats for very low precision.
- **Transformer-specific configs** (`neural_compressor/transformers/utils/quantization_config.py`):
  - `RtnConfig`, `AwqConfig`, `TeqConfig`, `GPTQConfig`, `AutoRoundConfig`.
  - All expose LLM-centric parameters such as `bits`, `group_size`, `n_samples`, `seq_len`, `quant_lm_head`, `use_layer_wise`.
- **Algorithm implementations** (PyTorch/ONNX backends):
  - `neural_compressor/adaptor/torch_utils/awq.py`:
    - `ActAwareWeightQuant` class implementing AWQ for transformers, with per-block activation capture, scale search and MSE-based clipping search.
  - `neural_compressor/adaptor/torch_utils/gptq.py`:
    - Implements GPTQ for transformer blocks, building Hessian-like statistics from calibration data and calling `fasterquant` per linear layer.
  - `neural_compressor/adaptor/torch_utils/teq.py`:
    - Implements TEQ for LLMs, inspired by AWQ and using trainable equivalent transformations to find optimal scaling factors.
  - `neural_compressor/adaptor/torch_utils/waq.py` and TensorFlow smoother modules:
    - Implement SmoothQuant-style activation smoothing for LLMs.
  - ONNX backends have analogous implementations under `neural_compressor/adaptor/ox_utils`.

For mixed-precision LLM weight compression (4/8-bit mixtures + int8 fallback), INC often integrates with Intel Extension for Transformers (IET) for runtime deployment, but the configuration and algorithm selection happen in INC.

### 2.2 Mixed-precision and search-effort knobs

INC supports two notions of “automatic” selection for LLM quantization:

1. **Algorithm/format tuning (WOQ algorithm search)**
   - Described in `docs/source/quantization_weight_only.md` under “WOQ Algorithms Tuning”.
   - User can set `quant_level="auto"` in `PostTrainingQuantConfig(approach="weight_only")` to let INC search over predefined weight-only configurations such as:
     - `RTN_G32ASYM`, `GPTQ_G32ASYM`, `GPTQ_G128ASYM`, `AWQ_G32ASYM`, etc.
   - The tuning strategy used is documented under `docs/source/tuning_strategies.md` (“Basic” strategy), and the search effort can be controlled via:
     - `tuning_criterion` fields (`max_trials`, `timeout`).
     - Choice of strategy (e.g., `bayesian`, `random`, `mse`, `hawq_v2`, etc.).

   Example (from `docs/source/quantization_weight_only.md`):

   ```python
   conf = PostTrainingQuantConfig(
       approach="weight_only",
       quant_level="auto",  # auto-tune WOQ config (RTN/AWQ/GPTQ, group_size, scheme)
   )
   q_model = quantization.fit(model, conf, eval_func=eval_func, calib_dataloader=dataloader)
   ```

2. **Mixed-precision / fallback tuning (HAWQ-V2 strategy and general tuning)**
   - `neural_compressor/strategy/hawq_v2.py` implements `HAWQ_V2TuneStrategy`, which:
     - Runs an initial quantization configuration.
     - Uses `adaptor.calculate_hessian_trace` to compute **Hessian traces per op**.
     - Ranks ops by Hessian trace and **greedily falls back** high-trace ops to higher precision (e.g., FP32) using a fallback sampler.
   - This strategy is generic but can be applied to LLMs when using full quantization approaches (INT8/FP16/FP32 mixtures).
   - Search effort knobs include:
     - Tuning criterion (`max_trials`, `timeout`, `strategy="hawq_v2"`) in the INC config.
     - Calibration data size.
     - HAWQ-specific parameters (e.g., number of Hessian samples, tolerance) controlled in the INC config/strategy kwargs.

For transformer-specific LLM flows (e.g., GPTQ/AWQ/TEQ/AutoRound), search effort and granularity are controlled primarily via the configs in `quantization_config.py`:

- `GPTQConfig`:
  - `bits`, `group_size`, `n_samples`, `seq_len`, `blocksize`, `damp_percent`, `use_mse_search`, `true_sequential`, etc.
  - These directly control the computational budget for GPTQ calibration and the degree of per-layer/local search.
- `AwqConfig`:
  - `bits`, `group_size`, `n_samples`, `auto_scale`, `auto_clip`, `zero_point`, `use_layer_wise`, etc.
- `AutoRoundConfig`:
  - `bits`, `group_size`, `iters` (also used as `calib_iters`), `n_samples`, `lr`, `minmax_lr`, `gradient_accumulate_steps`, etc.
  - These knobs let the user choose between “heavy search” (more iterations, more samples) and “light search”.

LLM usage example (simplified, from `docs` and `transformers/quantization/utils.py`):

```python
from neural_compressor.transformers import GPTQConfig
from neural_compressor.transformers import quantization as hf_quant

quant_config = GPTQConfig(
    bits=4,
    group_size=128,
    n_samples=128,
    seq_len=2048,
    damp_percent=0.1,
    use_mse_search=True,
)

quantized_model = hf_quant.quantize(
    model,
    quantization_config=quant_config,
    calib_dataset="NeelNanda/pile-10k",
)
```

### 2.3 INC: what counts as “automatic mixed precision”?

For LLMs, INC’s practice is:

- **Algorithm selection + parameter search** via tuning (`quant_level="auto"` for WOQ; general tuning strategies for full quantization).
- **Limited mixed-precision bit-width mixing** at the weight level (INT4 vs INT8 vs NF4/FP4) in combination with algorithms like TEQ and AutoRound.
- **HAWQ-V2** as a generic Hessian-based tuning strategy, but not the primary LLM recipe (which tends to use GPTQ/AWQ/TEQ/AutoRound).

---

## 3. OpenVINO + NNCF

### 3.1 Major quantization methods for LLMs

OpenVINO uses NNCF as its compression engine; the relevant code is in `extern/nncf`. For LLMs, the main flows are:

- **Post-training weight compression for LLMs** (`docs/usage/post_training_compression/weights_compression/Usage.md`):
  - Targeted at large models such as Llama 2 and Phi-3.
  - Supports **INT4 and INT8 weight compression** with:
    - **AWQ** (Activation-aware Weight Quantization).
    - **Scale Estimation** (per-layer scale refinement).
    - **GPTQ** (optional).
    - **LoRA-based correction** via QAT (for further 4-bit accuracy).
  - Mixed INT4/INT8 models are expressed as percentages of layers in INT4 vs INT8 (e.g. “%int4” and “%int8” in LLM tables), with trade-offs documented.
- **FP8 / INT8 activation/weight quantization**:
  - NNCF supports default INT8 PTQ and FP8 quantization for models including LLMs (see `ReleaseNotes.md` and examples under `examples/llm_compression`).
- **Accuracy-aware INT8 quantization**:
  - Implemented as `AccuracyAwareQuantization` in `nncf/quantization/algorithms/accuracy_control/algorithm.py`.
  - Integrates with OpenVINO’s POT and provides **automatic layer-wise fallback to FP32** for accuracy control.
- **Precision initialization (HAWQ and AutoQ)**:
  - Controlled via the `precision_initializer` in `nncf/config/schemata/algo/quantization.py`.
  - Supported types:
    - `"hawq"` – HAWQ-style Hessian-trace-based initializer.
    - `"autoq"` – AutoQ RL-based initializer.
    - `"manual"` – manual bit assignment.

### 3.2 Mixed-precision and search-effort knobs

NNCF/OpenVINO provide two main mechanisms for automatic mixed precision:

1. **AccuracyAwareQuantization (FP32/INT8 mixed precision)**
   - Implementation: `nncf/quantization/algorithms/accuracy_control/algorithm.py`.
   - High-level pipeline:
     1. Start from a model fully quantized to INT8 (DefaultQuantization).
     2. Compute baseline metric on validation dataset.
     3. Rank groups of quantizers by their impact on accuracy (via `Ranker` using a subset of the data).
     4. Iteratively **revert the most impactful quantizers back to floating point** (FP32/FP16) until the allowed accuracy drop is satisfied or a maximum number of iterations is reached.
   - Search-effort knobs include:
     - `max_num_iterations` – maximum number of fallback iterations.
     - `ranking_subset_size` – number of samples used for ranking.
     - `num_ranking_workers` – number of parallel workers (computed based on model size, timing, and system resources).
     - `max_drop`, `drop_type` – allowed accuracy drop and metric type.

   Example usage in OpenVINO (from docs/discussions):

   ```python
   import nncf
   from openvino.runtime import Core

   quantized_model = nncf.quantize_with_accuracy_control(
       model=ov_model,
       calibration_dataset=calib_dataset,
       validation_dataset=val_dataset,
       max_drop=0.01,  # 1% allowed accuracy drop
   )
   ```

2. **Precision initializer: HAWQ / AutoQ (bit-width mixed precision)**
   - Config schema: `nncf/config/schemata/algo/quantization.py`, `PRECISION_INIT_TYPES_VS_DESCRIPTION` and `PRECISION_INITIALIZER_SCHEMA`.
   - `"hawq"` mode:
     - Uses **Hutchinson iterations** to estimate average Hessian trace per quantized module:
       - `num_data_points`: number of data points for estimating Hessian trace.
       - `iter_number`: maximum number of Hutchinson iterations.
       - `tolerance`: stopping criterion on relative change in trace estimates.
     - `compression_ratio`: desired ratio between bit complexity of fully INT8 model and mixed-precision model.
     - Optionally loads precomputed Hessian traces (`traces_per_layer_path`) to accelerate multiple runs.
   - `"autoq"` mode:
     - Uses **reinforcement learning** (RL), inspired by AutoQ/HAQ, to pick bit-width per quantizer under:
       - `compression_ratio`: target model-size ratio.
       - `eval_subset_ratio`: fraction of dataset evaluated per RL iteration.
       - `warmup_iter_number`: number of random policies to seed the replay buffer.
   - Bit-width candidates are specified via `bits` (e.g. `[4, 8]`), and per-scope overrides via `bitwidth_per_scope`.

For LLM weight compression, the mixed INT4/INT8 schemes are largely implemented in the **weight compression API** rather than in the classic `quantize` pipeline, but the pattern is similar:

- **Compression formats** and parameters (INT4, INT8, NF4, MXFP4, etc.) are chosen via `CompressionMode` and compression parameters.
- **Advanced parameters** (in the LLM examples in `Usage.md`):
  - `AdvancedAWQParameters`, `AdvancedScaleEstimationParameters`, `AdvancedLoraCorrectionParameters`.
  - These control granularity (group size), number of optimization iterations, and LoRA rank, trading off search effort vs accuracy.

Example snippet (simplified from NNCF LLM compression docs):

```python
from nncf import compress_weights, CompressionMode
from nncf.parameters import AdvancedCompressionParameters, AdvancedAWQParameters

params = AdvancedCompressionParameters(
    awq_params=AdvancedAWQParameters(group_size=32, ...),
)

compressed_model = compress_weights(
    ov_model,
    mode=CompressionMode.INT4_ASYM,
    advanced_parameters=params,
)
```

### 3.3 OpenVINO/NNCF: what counts as “automatic mixed precision”?

In practice:

- **AccuracyAwareQuantization** provides automatic **FP32/INT8 mixed precision** via greedy layer fallback.
- **HAWQ / AutoQ precision initializers** provide automatic **INT4/INT8/FP16 bit-width assignment** based on Hessian traces or RL.
- **LLM weight-compression flows** use AWQ, GPTQ, Scale Estimation, and LoRA-based correction to build mixed INT4/INT8 models; here, “mixed precision” is an explicit choice per scope (e.g. % of layers in INT4 vs INT8), guided by calibration and advanced parameters.

---

## 4. References: algorithms and repositories

### 4.1 NVIDIA ModelOpt

- **ModelOpt repository**
  - GitHub: https://github.com/NVIDIA/TensorRT-Model-Optimizer
- **AutoQuantize API and docs**
  - API reference (AutoQuantize): https://nvidia.github.io/TensorRT-Model-Optimizer/reference/generated/modelopt.torch.quantization.model_quant.html
  - LLM PTQ example README: https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/main/examples/llm_ptq/README.md
- **SmoothQuant**
  - Guangxuan Xiao et al., “SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models”, arXiv:2211.10438, ICML 2023. https://arxiv.org/abs/2211.10438
- **AWQ**
  - Ji Lin et al., “AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration”, arXiv:2306.00978, MLSys 2024. https://arxiv.org/abs/2306.00978

### 4.2 Intel Neural Compressor (INC) and associated algorithms

- **INC repository**
  - GitHub: https://github.com/intel/neural-compressor
- **Weight-only LLM quantization docs**
  - Online docs: https://intel.github.io/neural-compressor/latest/docs/source/quantization_weight_only.html
- **GPTQ**
  - Elias Frantar et al., “GPTQ: Accurate Post-Training Quantization for Generative Pretrained Transformers”, ICLR 2023, arXiv:2210.17323. https://arxiv.org/abs/2210.17323
- **AWQ**
  - Ji Lin et al., “AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration”, arXiv:2306.00978. https://arxiv.org/abs/2306.00978
- **SmoothQuant**
  - Guangxuan Xiao et al., “SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models”, arXiv:2211.10438. https://arxiv.org/abs/2211.10438
- **TEQ**
  - Cheng et al., “TEQ: Trainable Equivalent Transformation for Quantization of LLMs”, arXiv:2310.10944. https://arxiv.org/abs/2310.10944
- **AutoRound**
  - GitHub: https://github.com/intel/auto-round
  - HF blog: “Introducing AutoRound: Intel’s Advanced Quantization for LLMs and VLMs”, https://huggingface.co/blog/autoround
- **HAWQ-V2 (tuning strategy)**
  - Zhen Dong et al., “HAWQ-V2: Hessian Aware trace-Weighted Quantization of Neural Networks”, NeurIPS 2020, arXiv:1911.03852. https://arxiv.org/abs/1911.03852

### 4.3 OpenVINO + NNCF and associated algorithms

- **NNCF repository**
  - GitHub: https://github.com/openvinotoolkit/nncf
- **OpenVINO Quantization and AccuracyAwareQuantization docs**
  - “Quantizing with Accuracy Control” (OpenVINO docs): https://docs.openvino.ai/2025/openvino-workflow/model-optimization-guide/quantizing-models-post-training/quantizing-with-accuracy-control.html
- **AutoQ / HAWQ precision initializers (NNCF)**
  - NNCF docs (“LegacyQuantization” and “Quantization” sections) and `nncf/config/schemata/algo/quantization.py`.
  - AutoQ algorithm is based on:
    - Qian Lou, Feng Guo, “AutoQ: Automated Kernel-Wise Neural Network Quantization”, ICLR 2020. https://arxiv.org/abs/1902.05690
  - Underlying HAQ idea:
    - Yunji Wang et al., “HAQ: Hardware-Aware Automated Quantization”, CVPR 2019. https://arxiv.org/abs/1811.08886
- **HAWQ-V2 / HAWQ-V3**
  - HAWQ-V2: Zhen Dong et al., NeurIPS 2020, arXiv:1911.03852. https://arxiv.org/abs/1911.03852
  - HAWQ-V3: Zhewei Yao et al., “HAWQ-V3: Dyadic Neural Network Quantization”, ICML 2021. http://proceedings.mlr.press/v139/yao21a.html
- **LLM weight compression**
  - NNCF weight-compression docs (GitHub view): https://github.com/openvinotoolkit/nncf/blob/develop/docs/usage/post_training_compression/weights_compression/Usage.md
  - OpenVINO LLM weight compression docs: https://docs.openvino.ai/2025/openvino-workflow/model-optimization-guide/weight-compression.html

### 4.4 Common algorithms across frameworks

Across ModelOpt, INC, and OpenVINO/NNCF, several algorithms or ideas appear repeatedly (either explicitly or via closely related variants):

- **AWQ (Activation-aware Weight Quantization)**
  - Directly implemented in:
    - ModelOpt (INT4_AWQ, W4A8_AWQ configs + AWQ-Clip/Lite calibration).
    - INC (`ActAwareWeightQuant` for PyTorch and ONNX backends).
    - NNCF LLM weight compression (AWQ parameters in `AdvancedCompressionParameters`).
  - Provides a common activation-aware weight-only quantization baseline for INT4/INT8 LLMs.
- **SmoothQuant-style activation smoothing**
  - ModelOpt: SmoothQuant-based INT8/W4A8 LLM configs.
  - INC: SmoothQuant implementations for PyTorch/TF backends.
  - NNCF/OpenVINO: adoption of SmoothQuant-inspired activation smoothing in some pipelines and docs.
- **Second-order / Hessian-inspired sensitivity**
  - ModelOpt: AutoQuantizeGradientSearcher uses a Fisher-based Hessian approximation to score per-layer quantization recipes.
  - INC: GPTQ (explicit Hessian approximation per weight block) and HAWQ-V2 tuning strategy (Hessian trace-based fallback).
  - NNCF: HAWQ-V2/V3-inspired precision initializer and, more generally, Hutchinson-based Hessian trace estimation.
- **RL/AutoML-based mixed-precision search**
  - NNCF: AutoQ precision initializer is directly based on AutoQ/HAQ RL algorithms for kernel-wise/bit-width selection.
  - INC/ModelOpt: do not use RL for LLM quant in the same way, but their tuning/AutoQuantize APIs play a similar role (automatically searching quantization configurations under accuracy/size constraints).
- **Accuracy-aware fallback**
  - OpenVINO/NNCF: AccuracyAwareQuantization explicitly performs quantizer fallback to FP32/FP16.
  - INC: HAWQ-V2 strategy and general tuning strategies also fallback high-sensitivity ops to higher precision.
  - ModelOpt: AutoQuantize can leave the most sensitive layers unquantized (effectively a per-layer fallback).

These common building blocks mean that, while each framework has its own API and engineering constraints, they are converging on a similar toolbox: **AWQ/SmoothQuant for robustness, GPTQ/second-order ideas for accuracy at 3–4 bits, Hessian/Fisher-based sensitivity for mixed precision, and (optionally) RL/tuning loops for automated bit-width assignment.**

Taken together, these sources show that:

- **ModelOpt**: uses vendor-native AutoQuantize (Fisher-based + LP) plus AWQ/SmoothQuant over NVFP4/FP8/INT4.
- **INC**: focuses on GPTQ, AWQ, TEQ, SmoothQuant, AutoRound, and generic HAWQ-V2 tuning.
- **OpenVINO/NNCF**: uses AccuracyAwareQuantization for FP32/INT8 mixed precision and HAWQ/AutoQ for bit-width selection, plus LLM-specific weight compression with AWQ, GPTQ, Scale Estimation, and LoRA.
