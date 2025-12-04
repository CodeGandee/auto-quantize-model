This note summarizes how Hessian Aware Quantization (HAWQ and its variants) is viewed today in the context of large language model (LLM) quantization, what has superseded it in practice, and why it is not the de facto choice for modern LLM PTQ. It is based on HAWQ’s original papers, later Hessian-aware works, and recent LLM quantization surveys and benchmarks.

Relevant references:
- HAWQ (ICCV 2019): https://arxiv.org/abs/1905.03696
- HAWQ-V2 (trace-weighted Hessian, NeurIPS 2020): https://arxiv.org/pdf/2004.03340.pdf
- HAWQ-V3 (dyadic quantization, ICML 2021): https://proceedings.mlr.press/v139/yao21a.html
- Q-BERT (Hessian-based ultra low precision BERT): https://arxiv.org/abs/1909.05840
- General LLM quantization overviews: e.g. https://haroldbenoit.com/notes/ML/LLMs/Quantization/Post-training-quantization-(PTQ)/Broad-overview-of-PTQ and curated lists like https://github.com/Zhen-Dong/Awesome-Quantization-Papers

---

## 1. What HAWQ actually does

HAWQ is a family of **Hessian-aware mixed-precision quantization methods**:

- Original HAWQ (2019) introduced a layer-wise bit allocation strategy driven by the **top eigenvalue of the Hessian** of the loss w.r.t. weights:

> HAWQ: Hessian AWare Quantization of neural networks with mixed-precision.  
> The motivation is that parameters with higher Hessian spectrum (i.e., larger top eigenvalues) are more sensitive to quantization and thus require higher precision.

- HAWQ-V2 (2020) improved this by using **trace-weighted metrics** instead of only the largest eigenvalue, and tried to automate mixed precision:

> HAWQ uses a heuristic metric based on top Hessian eigenvalue as a measure of sensitivity, and it ignores the rest of the Hessian spectrum; HAWQ only provides relative sensitivity of different layers, and it still requires a manual selection of the mixed-precision setting; and HAWQ does not consider mixed-precision activation quantization.  
> HAWQ-V2 introduces Hessian-trace-based metrics and automatic mixed-precision bit allocation.

- HAWQ-V3 (2021) moved toward **dyadic quantization formats** and more hardware-aware designs.

In all cases, the core idea is:
- Estimate, per layer, how sensitive the loss is to weight perturbations using **second-order information** from the Hessian.
- Assign higher bit-widths to layers with large curvature (large eigenvalues/trace), lower bit-widths to “flat” layers.
- Optionally fine-tune after quantization to recover some accuracy.

HAWQ and Q-BERT were influential early works that made **Hessian-aware mixed precision** practical for CNNs and BERT-scale transformers.

---

## 2. How HAWQ shows up in LLM quantization literature

In modern LLM-focused PTQ papers and surveys, HAWQ is:
- Cited as an **important second-order baseline** and as a conceptual ancestor of sensitivity-aware bit allocation.
- Mentioned mostly in the **“related work” sections**, especially under “second-order methods” or “Hessian-aware quantization”.

Examples:

- A broad PTQ overview for LLMs describes HAWQ as:

> Q-BERT developed Hessian AWare Quantization (HAWQ) for its mixed-precision quantization.  
> The motivation is that parameters with higher Hessian spectrum are more sensitive to quantization and thus require higher precision.

- Recent LLM compression papers (e.g. SliM-LLM, SqueezeLLM, OmniQuant) usually benchmark against:
  - GPTQ
  - AWQ
  - RTN (round-to-nearest)
  - SmoothQuant or its variants
and sometimes more recent 3–4-bit methods, but **do not include HAWQ as a standard LLM baseline**.

General LLM quantization surveys (2023–2025) tend to:
- Highlight GPTQ, AWQ, SmoothQuant, LLM.int8(), ZeroQuant, OmniQuant, SqueezeLLM, etc. as “state-of-the-art LLM PTQ” options.
- Mention HAWQ/HAWQ-V2/HAWQ-V3 as **earlier, more general NN quantization work**, not as the primary choice for current LLM deployments.

In practice, commonly used open-source LLM quantization toolchains (llama.cpp, AutoGPTQ, AWQ repos, bitsandbytes, etc.) do **not** expose a “HAWQ mode”; they focus on GPTQ/AWQ/RTN-style schemes.

---

## 3. What has superseded HAWQ in the LLM context

For large transformer-based LLMs, HAWQ-style Hessian-based mixed precision has largely been superseded by several other classes of methods:

### 3.1 Second-order but more LLM-focused PTQ (GPTQ, related methods)

- GPTQ (ICLR 2022) uses a **second-order approximation** per row of the weight matrix, but in a more scalable, blockwise fashion:
  - Works very well for 4-bit weight-only quantization on very large transformers.
  - Became the de facto **weight-only PTQ** for many open-source LLMs.
- Numerous follow-ups and variants (SpQR, SqueezeLLM, QuIP, etc.) build on GPTQ-like ideas and are specifically tuned for LLM inference workloads.

These are “Hessian-aware” in spirit (using local curvature or covariance) but implemented in ways that scale better to tens of billions of parameters and are easier to integrate into LLM-serving libraries than classical HAWQ.

### 3.2 Activation- and outlier-aware PTQ (AWQ, SmoothQuant, LLM.int8, etc.)

- **AWQ** (NeurIPS 2023) is heavily cited and widely adopted for LLMs:
  - Activation-aware weight quantization that chooses per-channel scales using activation statistics to preserve outlier channels.
  - Designed for W4A16 or W4A8-style quantization on large LLMs, balancing accuracy and deployment simplicity.
- **SmoothQuant** (ICML 2023) and **LLM.int8()**:
  - Focus on handling activation outliers by shifting the difficulty from activations to weights or using mixed-precision activations.
  - Used as standard baselines in many recent LLM PTQ papers.

These methods do not explicitly compute Hessians, but they are:
- Much easier to deploy at scale.
- Highly effective for typical LLM families (LLaMA, GPT-like, Qwen, etc.).

### 3.3 Vendor / ecosystem-specific toolchains (e.g., ModelOpt)

- NVIDIA’s **TensorRT Model Optimizer (ModelOpt)**:
  - Provides FP8/NVFP4/INT4/W4A8 mixed-precision formats and advanced PTQ algorithms (AWQ, SmoothQuant).
  - Its AutoQuantize algorithm performs **per-layer mixed-precision search** using a Hessian-style Fisher approximation rather than explicit eigen-decomposition, and is tightly integrated with TensorRT-LLM and deployment pipelines.
  - For NVIDIA hardware, ModelOpt (with AWQ/SmoothQuant/AutoQuantize) is the practical “go-to” rather than HAWQ.

Similarly, other vendor stacks (e.g. Intel Neural Compressor, NNCF, etc.) implement their own mixed-precision / sensitivity-aware methods rather than shipping HAWQ as-is.

---

## 4. Why HAWQ is not the de facto choice for LLM quantization

From current public reception (papers, surveys, tools) you can reasonably say that **HAWQ is influential but not the default choice** for LLM PTQ. Main reasons:

1. **Original target models and scale**
   - HAWQ and HAWQ-V2 were validated primarily on:
     - CNNs on ImageNet (ResNet, Inception, SqueezeNext).
     - BERT-size models (via Q-BERT).
   - The LLM landscape (tens of billions of parameters, decoder-only transformers, KV cache, etc.) emerged later, and most “LLM quantization” work focused on GPTQ/AWQ/SmoothQuant-style methods tuned for this regime.

2. **Hessian computation complexity**
   - HAWQ relies on **Hessian eigenvalue or trace estimates** per layer, which can be expensive for very large models.
   - Later theoretical work (HAWQ-V2/V3) improved efficiency, but the implementation and engineering complexity for 7B–70B LLMs remains non-trivial.
   - GPTQ / AWQ and other LLM PTQ methods use curvature-aware approximations that are **easier to implement and scale** in existing serving stacks.

3. **LLM-focused baselines have taken over**
   - Modern LLM PTQ papers and benchmarks overwhelmingly compare:
     - GPTQ, AWQ, SmoothQuant, RTN, and newer 3–4-bit methods.
   - HAWQ rarely appears in tables for LLaMA/Qwen/GPT models; when mentioned, it is typically in related work, rather than as a strong LLM baseline.

4. **Ecosystem and tooling**
   - Commonly used tools for LLM deployment (llama.cpp, AutoGPTQ, AWQ libraries, bitsandbytes, vendor stacks like ModelOpt) directly support GPTQ/AWQ/SmoothQuant-style schemes.
   - There are no widely used LLM-serving libraries where **“HAWQ mode” is a first-class option**; using HAWQ usually means custom research code.

5. **Vendor-native alternatives with Hessian-style ideas**
   - Frameworks like ModelOpt implement their own sensitivity-based mixed-precision search (e.g. AutoQuantize) using Fisher/Hessian-style approximations, but with engineering tuned to their deployment environment.
   - This reduces the practical incentive to adopt HAWQ itself for production LLM inference.

---

## 5. Practical takeaway if you are choosing a method today

If your goal is to choose a **practical, modern PTQ method for LLMs**, current practice and academic benchmarking suggest:

- For **weight-only 4-bit quantization**:
  - Start with GPTQ or one of its improved descendants (SqueezeLLM, QuIP, SpQR, etc.).
  - AWQ is also a strong option when you want activation-aware weight scaling.

- For **W4A8 or FP8-style formats on NVIDIA hardware**:
  - Use vendor tools like NVIDIA ModelOpt:
    - AWQ-, SmoothQuant-based configs.
    - AutoQuantize for mixed NVFP4 / FP8 / INT4 formats under an effective-bit constraint.

- For **mixed-precision research with explicit Hessian use**:
  - HAWQ / HAWQ-V2 / HAWQ-V3 remain important references and starting points, especially if you:
    - Want to design new Hessian-aware schemes.
    - Are working with smaller models where explicit Hessian trace/eigen computations are affordable.

But in terms of **current de facto LLM deployment practice**, HAWQ is best thought of as an influential historical method and theoretical reference, not the main tool people reach for when quantizing LLMs in 2024–2025.

---

## 6. What popular frameworks actually use for mixed-precision quantization

Looking at major production-oriented toolkits reinforces that HAWQ-style Hessian eigenvalue methods are *not* what most users are exposed to for LLM mixed precision. Instead, each framework ships its own, often LLM-focused, algorithms.

### 6.1 NVIDIA TensorRT Model Optimizer (ModelOpt)

ModelOpt is explicitly positioned as a **state-of-the-art LLM quantization toolkit** for NVIDIA GPUs. Its documented formats and algorithms include:

- Quantization formats (for LLMs and VLMs):
  - INT8 (SmoothQuant, max)
  - FP8
  - INT4 (INT4_AWQ, W4A8 AWQ, NVFP4, MXFP4, MXFP8, etc.)
  - NVFP4 (FP4-like format for Blackwell GPUs)
- Algorithms:
  - **SmoothQuant**: activation- and weight rebalancing to handle outliers for W8A8/W4A8.
  - **AWQ / AWQ-lite**: activation-aware weight quantization to preserve salient channels.
   - **AutoQuantize**: per-layer mixed-precision search over formats (NVFP4, FP8, W4A8, etc.) under an effective-bits constraint using gradient/Fisher or KL-divergence scores.

Code evidence from `extern/TensorRT-Model-Optimizer`:

- `modelopt/torch/quantization/algorithms.py:AutoQuantizeGradientSearcher`:
  - States that the AutoQuantize score is a **Taylor expansion of the loss** w.r.t. the quantized output with **Fisher information substituted for the Hessian**, and that a **linear programming solver (LPS)** is used to find the optimal per-layer configuration.
  - This is a Hessian-style approximation but not a full HAWQ-style eigen/trace computation.
- `modelopt/torch/quantization/algorithms.py:AutoQuantizeKLDivSearcher`:
  - Implements the alternative **KL-divergence–based searcher**, scoring each candidate format by KL divergence between logits from the original (teacher) model and the quantized model.
- `modelopt/torch/quantization/model_calib.py:awq_clip` and related helpers:
  - Implement **AWQ Clip / Lite variants**, explicitly described as “AWQ-Clip variant” and operating by searching clipping thresholds and per-block scales for weight quantizers.
- `modelopt/torch/quantization/config.py` and `examples/llm_eval/quantization_utils.py`:
  - Expose LLM-ready configs such as `INT4_AWQ_CFG`, `W4A8_AWQ_BETA_CFG`, `FP8_DEFAULT_CFG`, `NVFP4_DEFAULT_CFG` and show them being passed into `mtq.auto_quantize` with `auto_quantize_bits` to perform **mixed NVFP4/FP8/W4A8 searches** over LLMs.

From NVIDIA’s LLM PTQ blog:

> NVIDIA TensorRT Model Optimizer supports a broad range of formats, including NVFP4, FP8, and INT8, and integrates calibration techniques like SmoothQuant, activation-aware weight quantization (AWQ), and AutoQuantize for improved quantization results.

And from the LLM PTQ examples:

> [AutoQuantize (`mtq.auto_quantize`)] is a PTQ algorithm which quantizes a model by searching for the best quantization format per-layer while meeting performance constraints specified by the user.  
> You may specify an `effective_bits` constraint such as 4.8 for mixed precision quantization using `NVFP4_DEFAULT_CFG` & `FP8_DEFAULT_CFG`.

Observation:
- ModelOpt’s “house style” for mixed precision is **AutoQuantize + AWQ/SmoothQuant over NVFP4/FP8/INT4**.  
- HAWQ is not mentioned in ModelOpt docs; they use their own Hessian-style approximation instead of importing HAWQ directly.

Key algorithm references used by ModelOpt:

- **SmoothQuant** – “SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models”, Xiao et al., arXiv:2211.10438 / ICML 2023. https://arxiv.org/abs/2211.10438
- **AWQ** – “AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration”, Lin et al., arXiv:2306.00978 / MLSys 2024. https://arxiv.org/abs/2306.00978
- **AutoQuantize (ModelOpt)** – vendor algorithm; as of 2025 there is **no standalone academic paper**. The method is described in ModelOpt docs and implemented in `modelopt/torch/quantization/algorithms.py` as a Fisher-based loss approximation plus LP solver.

### 6.2 Intel Neural Compressor (INC)

Intel Neural Compressor is another major open-source toolkit for model compression, with explicit support for LLMs and low-bit formats:

> Intel® Neural Compressor aims to provide popular model compression techniques such as quantization, pruning (sparsity), distillation, and neural architecture search … In particular, the tool provides … SOTA low-bit LLM quantization (INT8/FP8/MXFP8/INT4/MXFP4/NVFP4) & sparsity; leading model compression techniques …  
> Topics: … `awq`, `gptq`, `smoothquant`, `sparsegpt`, `fp4`, `mxformat`, …

Key points:
- INC explicitly advertises support for:
  - GPTQ
  - AWQ
  - SmoothQuant
  - TEQ
  - AutoRound
  - FP4/NVFP4/MX formats
  - Sparsity methods like SparseGPT
- Mixed precision:
  - INC supports **mixed FP32/BF16/INT8** inference and “auto-mixed precision” flows.
  - During quantization, it can combine integer formats with BF16 where supported.
- For LLMs, its “SOTA low-bit quantization” story revolves around GPTQ/AWQ/SmoothQuant-style methods, not HAWQ.

Code evidence from `extern/neural-compressor`:

- LLM-specific weight-only quantization entrypoints live under:
  - `neural_compressor/transformers/utils/quantization_config.py` – defines:
    - `GPTQConfig`, `AwqConfig`, `TeqConfig`, `AutoRoundConfig` with fields such as `bits`, `group_size`, `n_samples`, `seq_len`, `use_layer_wise`, etc.
  - `neural_compressor/transformers/quantization/utils.py` – high-level pipeline that dispatches to GPTQ, AWQ, TEQ, AutoRound based on `quant_method`, logging messages such as “Do GPTQ algorithm with config …” and “Do AWQ algorithm with config …”.
- Algorithm implementations for LLM weight-only quantization:
  - `neural_compressor/adaptor/torch_utils/awq.py`:
    - Class `ActAwareWeightQuant` described in the docstring as “Implementation of Activation-aware Weight quantization (AWQ) algo.”
    - Implements per-block activation collection, scale search (`search_scale`) and clipping search (`search_clip`), then applies weight quantization using those scales and clip ranges.
  - `neural_compressor/adaptor/torch_utils/gptq.py`:
    - Implements GPTQ for transformers: collects hidden states, builds Hessian-like statistics per weight block, then calls `fasterquant` to solve the GPTQ quadratic problem for each linear layer.
  - `neural_compressor/adaptor/torch_utils/teq.py`:
    - Implements **TEQ** (trainable equivalent transformation) as an AWQ-inspired, trainable scaling scheme; comments reference GPTQ-style configs and support LLM architectures.
  - `neural_compressor/adaptor/torch_utils/waq.py` and `neural_compressor/tensorflow/algorithms/smoother/core.py`:
    - Implement **SmoothQuant**-style activation smoothing for PyTorch and TensorFlow backends, with dedicated classes `TorchSmoothQuant` and `SmoothQuant`.
- Mixed-precision and MX formats:
  - Mixed FP32/BF16/INT8 quantization and FP8/FP4/MX formats are exposed via general quantization configs (`neural_compressor/config.py`, backend YAMLs) rather than as separate named algorithms.
  - LLM weight compression can use mixed W4/W8 + BF16/INT8, but the **bit allocation is driven by algorithm/format choices (e.g., TEQ, GPTQ) and layer-wise configs**, not by a HAWQ-style Hessian-eigen search.
- HAWQ’s presence in INC:
  - `neural_compressor/strategy/hawq_v2.py:HAWQ_V2TuneStrategy` implements a **Hessian trace–based tuning strategy**, calling `adaptor.calculate_hessian_trace` and then greedily falling back high-trace ops to higher precision.
  - Docs (`docs/source/tuning_strategies.md`) explicitly state that this implements “HAWQ-V2: Hessian Aware trace-Weighted Quantization of Neural Networks”.
  - This strategy is exposed as a generic tuning option (e.g., for CNNs), but **it is not the main user-facing path for LLM-specific quantization**, where GPTQ/AWQ/SmoothQuant/TEQ/AutoRound are the primary recipes.

Key algorithm references used by INC:

- **GPTQ** – “GPTQ: Accurate Post-Training Quantization for Generative Pretrained Transformers”, Frantar et al., ICLR 2023, arXiv:2210.17323. https://arxiv.org/abs/2210.17323
- **AWQ** – “AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration”, Lin et al., arXiv:2306.00978 / MLSys 2024. https://arxiv.org/abs/2306.00978
- **SmoothQuant** – “SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models”, Xiao et al., arXiv:2211.10438 / ICML 2023. https://arxiv.org/abs/2211.10438
- **TEQ** – “TEQ: Trainable Equivalent Transformation for Quantization of LLMs”, Cheng et al., arXiv:2310.10944. https://arxiv.org/abs/2310.10944
- **AutoRound** – Intel’s gradient-based LLM/VLM quantization algorithm; as of 2025 there is **no peer-reviewed paper**, but the design is described in:
  - GitHub: https://github.com/intel/auto-round
  - Hugging Face blog “Introducing AutoRound: Intel’s Advanced Quantization for LLMs and VLMs”: https://huggingface.co/blog/autoround
- **HAWQ-V2 strategy** – “HAWQ-V2: Hessian Aware trace-Weighted Quantization of Neural Networks”, Dong et al., NeurIPS 2020, arXiv:1911.03852. https://arxiv.org/abs/1911.03852

### 6.3 OpenVINO / NNCF

The OpenVINO ecosystem includes:
- The **Post-Training Optimization Tool (POT)** and
- The **Neural Network Compression Framework (NNCF)** for quantization and other compression.

For mixed precision and accuracy-aware quantization, OpenVINO emphasizes:

- **DefaultQuantization** and **AccuracyAwareQuantization**:
  - Default INT8 PTQ over the whole model.
  - Accuracy-aware mode that:
    - Starts from full INT8 quantization.
    - Iteratively identifies layers whose INT8 quantization hurts accuracy the most.
    - Reverts those layers back to higher precision (FP32/FP16), yielding a *mixed FP32/INT8* model.

Example (from a YOLOv5 POT tutorial):

> Accuracy-aware Quantization (AAQ) is an iterative quantization algorithm based on Default Quantization. The model quantified by DQ is used as the baseline. If the baseline model accuracy does not reach the predefined accuracy range, the AAQ will fall back to the layer with the greatest impact on the accuracy from INT8 precision to FP32 precision. It will then re-evaluate the model accuracy and repeat the process until the model reaches the expected accuracy range.

OpenVINO/NNCF also mentions:

- **AutoQ precision initializer**:
  - In `extern/nncf/src/nncf/config/schemata/algo/quantization.py`, the `precision_initializer` supports types `"hawq"` and `"autoq"`.
  - `"hawq"` is documented as a Hessian-trace–based initializer that uses **Hutchinson iterations** to estimate per-layer Hessian traces and then assigns bitwidths under a `compression_ratio` constraint.
  - `"autoq"` is documented as an AutoQ mode that uses **reinforcement learning** to select bitwidths, again under a `compression_ratio` and evaluation-budget constraint (`eval_subset_ratio`, `warmup_iter_number`).
- **AccuracyAwareQuantization implementation**:
  - `extern/nncf/src/nncf/quantization/algorithms/accuracy_control/algorithm.py` implements the `AccuracyAwareQuantization` algorithm.
  - It quantizes the model, ranks groups of quantizers (`Ranker`), and then **greedily reverts the highest-impact quantizers back to floating-point** while monitoring validation metrics, until the allowed accuracy drop is satisfied.
  - The algorithm exposes knobs like `max_num_iterations`, `ranking_subset_size`, and `num_ranking_workers`, which indirectly control the mixed-precision search effort.
- **LLM weight compression and mixed 4/8-bit schemes**:
  - `extern/nncf/docs/usage/post_training_compression/weights_compression/Usage.md` documents NNCF’s weight-compression pipeline for LLMs.
  - For models such as Llama 2 and Phi-3, NNCF uses **INT4/INT8 mixed precision** combined with:
    - **AWQ** (Activation-aware Weight Quantization),
    - **Scale Estimation** (per-layer scale refinement),
    - **GPTQ** (optional),
    - and **LoRA-based correction** for further accuracy recovery.
  - The tables show configurations like “int4 + awq + scale estimation + gptq” and “int4 + awq + scale estimation + lora correction” that achieve accuracy close to FP32 with significant compression.
- **Where HAWQ appears in NNCF**:
  - HAWQ is used as a **precision initializer** for general quantization (especially CNNs) and is described as a Hessian-based mixed-precision assignment algorithm.
  - For LLMs, the flagship flows are the weight compression API and AccuracyAwareQuantization/AutoQ; **HAWQ is not advertised as the main LLM PTQ algorithm**, although the same machinery can, in principle, be applied.

Key algorithm references used by OpenVINO/NNCF:

- **HAWQ-V2** – “HAWQ-V2: Hessian Aware trace-Weighted Quantization of Neural Networks”, Dong et al., NeurIPS 2020. https://arxiv.org/abs/1911.03852
- **HAWQ-V3** – “HAWQ-V3: Dyadic Neural Network Quantization”, Yao et al., ICML 2021. http://proceedings.mlr.press/v139/yao21a.html
- **AutoQ (kernel-wise RL)** – “AutoQ: Automated Kernel-Wise Neural Network Quantization”, Lou et al., ICLR 2020. https://arxiv.org/abs/1902.05690
- **HAQ (underlying RL idea)** – “HAQ: Hardware-Aware Automated Quantization”, Wang et al., CVPR 2019, arXiv:1811.08886. https://arxiv.org/abs/1811.08886
- **GPTQ / AWQ / SmoothQuant** – same references as in the INC section, reused by NNCF’s weight-compression pipeline for LLMs.
- **LoRA + NLS-based 4-bit LLM QAT** – “Low-Rank Adapters Meet Neural Architecture Search for LLM Compression”, Muñoz et al., arXiv:2501.16372. https://arxiv.org/abs/2501.16372

---

## 7. Summary: frameworks’ choices reinforce that HAWQ is not de facto

Across three major ecosystems:

- **ModelOpt (NVIDIA)**:
  - Mixed precision is built around **AutoQuantize** (Fisher-based, per-layer search) plus **AWQ/SmoothQuant** over NVFP4/FP8/INT4/INT8 formats, tightly integrated with TensorRT-LLM.
  - HAWQ is not surfaced in user docs or examples.

- **Intel Neural Compressor**:
  - Advertises “SOTA low-bit LLM quantization” with **GPTQ**, **AWQ**, **SmoothQuant**, FP8/FP4/MX formats, and SparseGPT.
  - Mixed precision is mainly FP32/BF16/INT8 plus specialized low-bit LLM schemes.
  - HAWQ does not appear as a primary algorithm; instead, modern LLM-focused methods are front and center.

- **OpenVINO / NNCF**:
  - Uses **DefaultQuantization + AccuracyAwareQuantization** and **AutoQ** for mixed precision, reverting sensitive layers to higher precision and/or using AutoML-based bit allocation.
  - Recent releases push INT4 and FP16-NF4 for LLMs, again via OpenVINO’s own quantization pipelines.
  - No deployment-facing HAWQ implementation for LLMs is documented.

This ecosystem evidence matches the picture from recent LLM PTQ papers: HAWQ is an important historical Hessian-aware method, but it is **not** what mainstream frameworks expose as the default or recommended algorithm for LLM mixed-precision quantization today. Instead, GPTQ/AWQ/SmoothQuant-style methods and vendor-native search procedures (ModelOpt AutoQuantize, OpenVINO AutoQ/AccuracyAware) have become the practical standards.

---

## 8. Code-level confirmation from `extern/` toolchains

The analyses above are now backed by concrete code-level inspection of the tool repos under `extern/`:

- **ModelOpt (`extern/TensorRT-Model-Optimizer`)**
  - Auto mixed-precision search is implemented in `modelopt/torch/quantization/algorithms.py` as:
    - `AutoQuantizeGradientSearcher` – Fisher-based Hessian approximation + linear program over `QuantRecipe` hparams.
    - `AutoQuantizeKLDivSearcher` – KL-divergence–based sensitivity scoring for candidate formats.
  - Advanced calibration methods `AWQLiteHelper`, `AWQClipHelper`, `awq_clip`, etc. live in `modelopt/torch/quantization/model_calib.py` and are explicitly labeled AWQ variants.
  - LLM examples (`examples/llm_eval/*`, `examples/llm_ptq/*`) wire these into `mtq.auto_quantize` with configs such as `INT4_AWQ_CFG`, `W4A8_AWQ_BETA_CFG`, `FP8_DEFAULT_CFG`, `NVFP4_DEFAULT_CFG`.
  - No additional HAWQ-style eigenvalue/trace solver is present in the quantization modules; Hessian-like machinery (actual explicit Hessian inverses) appears only in **SparseGPT** code under `modelopt/torch/sparsity/weight_sparsity`, not in the AutoQuantize implementation.

- **Intel Neural Compressor (`extern/neural-compressor`)**
  - LLM PTQ is centered around **GPTQ, AWQ, TEQ, SmoothQuant, AutoRound**:
    - Configs: `neural_compressor/transformers/utils/quantization_config.py` (`GPTQConfig`, `AwqConfig`, `TeqConfig`, `AutoRoundConfig`).
    - Implementations: `neural_compressor/adaptor/torch_utils/gptq.py`, `awq.py`, `teq.py`, `waq.py`, plus TensorFlow SmoothQuant in `neural_compressor/tensorflow/algorithms/smoother`.
  - Mixed-precision and MX formats are exposed via general config/YAML (`neural_compressor/config.py`, `adaptor/*yaml`) and weight-only pipelines (`adaptor/torch_utils/weight_only.py`), not via a dedicated HAWQ-style bit allocator.
  - `neural_compressor/strategy/hawq_v2.py` implements a HAWQ-V2-based tuning strategy that computes Hessian traces and falls back ops by sensitivity, but this is a generic strategy and **not the primary LLM workflow**, which defaults to GPTQ/AWQ/TEQ/AutoRound recipes.

- **OpenVINO / NNCF (`extern/openvino`, `extern/nncf`)**
  - Mixed-precision initialization is controlled via the `precision_initializer` in `nncf/config/schemata/algo/quantization.py`, with `"hawq"` (Hessian-trace) and `"autoq"` (RL) options.
  - `nncf/quantization/algorithms/accuracy_control/algorithm.py` provides the concrete AccuracyAwareQuantization implementation, which greedily reverts quantizers to FP precision based on per-group impact ranking.
  - LLM weight compression examples and docs (`docs/usage/post_training_compression/weights_compression/Usage.md`, `examples/llm_compression/*`) show **INT4/INT8 mixed-precision LLMs** built using combinations of AWQ, Scale Estimation, GPTQ, and LoRA-based correction.
  - HAWQ is present as one option for precision initialization (especially for CNNs), but the **main LLM story is weight compression + AccuracyAware/AutoQ**, not HAWQ-eigenvalue bit allocation.

Taken together, these code paths confirm that:

- All three toolchains provide rich **mixed-precision quantization and LLM-specific flows**, but
- **HAWQ appears mainly as a legacy or optional precision initializer/tuning strategy**, while **GPTQ, AWQ, SmoothQuant, TEQ, AutoRound, AutoQ, and vendor-specific AutoQuantize/AccuracyAware algorithms are the practical defaults** for LLM mixed-precision quantization today.
