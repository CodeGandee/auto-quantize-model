# How to Use Intel Neural Compressor for Layer Sensitivity and Mixed-Precision Planning (LLMs / Qwen-style models)

This hint explains which sensitivity-based features exist in Intel Neural Compressor (INC) and how to use them to understand layer sensitivity for large language models (LLMs), so you can design your own mixed-precision schemes (e.g., deciding which layers stay in higher precision).

It focuses on:
- MSE-based per-op sensitivity (`mse_v2` strategy)
- Hessian-based sensitivity (`hawq_v2` strategy)
- Layer-wise / LLM-specific flows (AutoRound, MX mixed precision) as implicit sensitivity signals

INC source in this repo: `extern/neural-compressor/`.

## 1. MSE-based per-op sensitivity (generic PTQ)

INC implements a generic “op sensitivity” mechanism around the `mse_v2` tuning strategy. It works across frameworks (PyTorch, ONNX Runtime, TensorFlow) and is backed by adaptor-specific methods:
- PyTorch adaptor: `calculate_op_sensitivity` in `neural_compressor/adaptor/pytorch.py`
- ONNX Runtime adaptor: `calculate_op_sensitivity` + `_get_mse_order` in `neural_compressor/adaptor/onnxrt.py`
- TF adaptor: same interface pattern

### What the metric does

High-level algorithm (ONNX Runtime example; PyTorch is analogous):
- For each candidate quantizable op:
  - Temporarily change the op’s quantization config:
    - In fallback phase: quantized → fp32
    - In re-quantization phase: fp32 → quantized (using stored quant configs)
  - Quantize the model (with that one-op change)
  - Run `confidence_batches` of the calibration dataloader
  - Compute MSE between fp32 outputs and quantized outputs over those batches
- Return a list of ops sorted by MSE (sensitivity), from lowest to highest.

ONNX code reference:
- `ONNXRUNTIMEAdaptor.calculate_op_sensitivity(...)` and `_get_mse_order(...)` in `neural_compressor/adaptor/onnxrt.py`
- Docs: `extern/neural-compressor/docs/source/tuning_strategies.md` (`MSE_V2` section)

### How to enable it (example: ONNX Runtime, INT8 W8A8)

```python
from neural_compressor import quantization
from neural_compressor.config import PostTrainingQuantConfig, TuningCriterion

conf = PostTrainingQuantConfig(
    backend="onnxrt_qdq",          # ONNX Runtime QDQ backend
    device="cpu",
    approach="static",             # PTQ with calibration
    tuning_criterion=TuningCriterion(
        strategy="mse_v2",
        strategy_kwargs={"confidence_batches": 2},  # number of batches used to score op impact
    ),
)

q_model = quantization.fit(
    model="path/to/fp32_model.onnx",
    conf=conf,
    calib_dataloader=calib_dl,
)
q_model.save("path/to/int8_w8a8_qdq.onnx")
```

With `LOGLEVEL=DEBUG`, INC logs the per-op MSE order:
- In ONNX adaptor, `calculate_op_sensitivity` logs “Dump MSE order:” and lists `op → mse`.
- In PyTorch adaptor, `torch_utils/util.py` logs “Evaluate the sensitivity for each int8 operation” and “Evaluate the sensitivity for each fp32 operation”.

### How to use it for mixed precision

Basic pattern for Qwen or any LLM:
- Run a PTQ process with `strategy="mse_v2"` on a representative calibration set.
- Capture the sorted op list (either from logs or by calling `adaptor.calculate_op_sensitivity(...)` directly if you are embedding INC).
- Interpret that list as:
  - High-MSE ops = more sensitive; keep them in FP16/FP32 or higher-precision scheme.
  - Low-MSE ops = less sensitive; more aggressive quantization is likely acceptable.
- Encode your decisions back into INC configs:
  - Use `op_type_dict` / `op_name_dict` in `PostTrainingQuantConfig` to constrain which ops can be INT8 vs FP16/FP32.
  - Or, if you work at the adaptor level, adjust `tune_cfg["op"]` per op before calling `quantize`.

This gives you a data-driven way to choose where INT8 W8A8 is applied vs where you keep higher precision, even if the final deployment is on an RTX 5090 (e.g. via ONNX Runtime CUDA or TensorRT).

## 2. Hessian-based sensitivity (HAWQ_V2, PyTorch only)

INC also implements a Hessian-aware sensitivity strategy for PyTorch models, based on HAWQ-V2:
- Strategy class: `HAWQ_V2TuneStrategy` in `neural_compressor/strategy/hawq_v2.py`
- Adaptor hook: `PyTorchAdaptor.calculate_hessian_trace(...)` in `neural_compressor/adaptor/pytorch.py`
- Core math: `HessianTrace` and `hawq_top` in `neural_compressor/adaptor/torch_utils/hawq_metric.py`
- Docs: `extern/neural-compressor/docs/source/tuning_strategies.md` (`HAWQ_V2` section)

### What the metric does

Roughly:
- Uses PyHessian-style stochastic trace estimation to approximate the Hessian trace per weight tensor.
- Optionally incorporates weight quantization perturbation and activation traces into a combined “importance” score per op.
- Sorts ops by this score and then performs datatype fallback in that order (high-trace ops are treated as more sensitive).

HAWQ_V2 is more expensive than MSE-based scoring but provides a more principled curvature-aware view of sensitivity.

### How to enable it (PyTorch LLM / Qwen style)

You must provide a loss function that maps `(output, target)` to a scalar loss, passed via `strategy_kwargs["hawq_v2_loss"]`:

```python
from neural_compressor import quantization
from neural_compressor.config import PostTrainingQuantConfig, TuningCriterion

def model_loss(output, target, criterion):
    # example for language modeling; adapt as needed
    return criterion(output.logits.view(-1, output.logits.size(-1)), target.view(-1))

conf = PostTrainingQuantConfig(
    framework="pytorch",
    quant_level=1,
    tuning_criterion=TuningCriterion(
        strategy="hawq_v2",
        strategy_kwargs={"hawq_v2_loss": model_loss},  # required for HAWQ_V2
    ),
)

q_model = quantization.fit(
    model=fp32_qwen_model,
    conf=conf,
    calib_dataloader=calib_dl,
)
```

During tuning, HAWQ_V2:
- Calls `calculate_hessian_trace(fp32_model, dataloader, q_model, criterion, enable_act=False)` to get `op_to_traces`.
- Sorts ops by trace and logs something like: `*** op: <name>, hessian trace: <value>`.
- Uses that ordering to drive fallback (e.g. higher-precision assignment for the most sensitive ops).

### Using HAWQ_V2 scores yourself

You can also use the adaptor function directly to get a sensitivity map without letting HAWQ_V2 control tuning:

```python
op_to_traces = adaptor.calculate_hessian_trace(
    fp32_model=inc_model,          # INC-wrapped PyTorch model
    dataloader=calib_dl,
    q_model=last_q_model,          # or a dummy quantized model
    criterion=model_loss,
    enable_act=False,
)
```

Then:
- Sort `op_to_traces.items()` by trace.
- Map high-trace ops to higher precision (e.g. FP16 / MXFP8) and low-trace ops to lower precision (e.g. INT8 / MXFP4).

This is a good building block if you want to design your own mixed-precision search for a Qwen-like LLM.

## 3. Layer-wise quantization and LLM-specific algorithms (implicit sensitivity)

Besides the explicit MSE / Hessian metrics, several INC LLM flows produce **implicit** sensitivity information you can mine.

### 3.1 Layer-wise quantization (LWQ)

Layer-wise quantization treats each layer (typically each `Linear`) as a unit and quantizes them one by one:
- Docs: `extern/neural-compressor/docs/source/quantization_layer_wise.md`
- PyTorch example:

```python
from neural_compressor import PostTrainingQuantConfig, quantization
from neural_compressor.adaptor.torch_utils.layer_wise_quant import load_empty_model

fp32_model = load_empty_model(model_name_or_path, torchscript=True)
conf = PostTrainingQuantConfig(
    approach="weight_only",
    recipes={
        "layer_wise_quant": True,
        "rtn_args": {"enable_full_range": True},
    },
)

q_model = quantization.fit(
    fp32_model,
    conf,
    calib_dataloader=eval_dataloader,
    eval_func=lambda x: 0.1,  # your metric
)
q_model.save("./saved_model")
```

LWQ itself doesn’t expose a scalar “sensitivity score”, but it gives you a practical granularity:
- You can quantize layer-by-layer and measure perplexity / task metrics after each change.
- That lets you construct your own empirical sensitivity ranking per layer, using your actual workloads.

### 3.2 AutoRound and MX mixed precision for LLMs

INC’s transformers-like API includes:
- `AutoRoundConfig` for weight-only quantization (INT4/INT8 etc.).
- MX quantization (`MXQuantConfig`) with mixed MXFP4 + MXFP8 (see `PT_MXQuant.md`).
- These tools internally allocate bits per layer based on calibration/eval data.

Docs:
- `extern/neural-compressor/docs/source/3x/transformers_like_api.md`
- `extern/neural-compressor/docs/source/3x/PT_MXQuant.md`

Example: AutoRound (simplified):

```python
from transformers import AutoTokenizer
from neural_compressor.transformers import AutoModelForCausalLM, AutoRoundConfig

model_name_or_path = "MODEL_NAME_OR_PATH"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
woq_config = AutoRoundConfig(bits=4, tokenizer=tokenizer)

q_model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    quantization_config=woq_config,
)
```

After quantization:
- The per-layer quantization decisions are stored in `q_model.qconfig` / `q_model.q_config`.
- You can extract them with `collect_weight_info(model, q_config)` (`torch_utils/util.py`) to get a mapping:
  - `layer_name → {dtype, bits, group_size, scheme, module_type, algorithm}`
- For MX mixed precision, `target_bits` and `options=["MXFP4", "MXFP8"]` let INC compute a layer-wise allocation across MXFP4 vs MXFP8:

```python
from neural_compressor.torch.quantization import AutoRoundConfig, autotune, TuningConfig

config = AutoRoundConfig(
    tokenizer=tokenizer,
    nsamples=128,
    seqlen=2048,
    iters=200,
    target_bits=[7.2, 7.5, 7.8],
    options=["MXFP4", "MXFP8"],
    export_format="auto_round",
    output_dir="./llama3.1-8B-MXFP4-MXFP8",
)

tuning_config = TuningConfig(config_set=[config], tolerable_loss=0.01)
q_model = autotune(fp32_model, tuning_config, eval_fn=eval_fn)
```

Even if you don’t use the final quantized model, this layer-wise bit allocation is an **implicit sensitivity map**:
- Layers assigned higher precision (e.g. MXFP8) are likely more sensitive.
- Layers assigned lower precision (e.g. MXFP4 or 4-bit INT) are more robust.

You can reuse that assignment in your own toolchain (e.g. vLLM/TensorRT on RTX 5090) by mapping INC’s layer naming to your runtime.

## 4. Practical recipe for Qwen / generic LLM on RTX 5090

Putting it together:
- For INT8 W8A8 planning:
  - Run ONNX Runtime or PyTorch PTQ with `strategy="mse_v2"` on a representative calibration set.
  - Extract op sensitivity ordering (MSE-based).
  - Decide which ops/layers to keep in FP16/FP32 (most sensitive), which to quantize to INT8.
  - Encode that scheme into:
    - A constrained INC config (so INC produces a W8A8 ONNX model matching your plan), or
    - Your own mixed-precision configuration for a different runtime.
- For Hessian-aware planning:
  - On PyTorch Qwen, run a small HAWQ_V2 session with a suitable `hawq_v2_loss`.
  - Collect `op_to_traces` from `calculate_hessian_trace` and treat it as a curvature-based sensitivity ranking.
  - Use these traces to inform which layers get higher vs lower precision in your own mixed-precision search.
- For weight-only / MX flows:
  - Use AutoRound/MX autotune on CPU or Intel-targeted environments to get a per-layer bit allocation.
  - Export `model.qconfig` / `collect_weight_info(...)` and map that allocation into your RTX 5090 deployment stack.

These mechanisms let INC act not just as a quantization tool, but as a **sensitivity analysis engine** you can reuse to design custom mixed-precision schemes for Qwen and other LLMs, even if final inference is not run via INC on Intel hardware.

## 5. Best-practice patterns (LLM and non-LLM)

### 5.1 PyTorch backend vs ONNX Runtime GPU

- **PyTorch+INC is typically CPU-centric.** With INC 2.x, the PyTorch FX backend is designed first for Intel CPUs (and, with IPEX/ITEX, Intel GPUs). The `PostTrainingQuantConfig` `backend` options for PTQ (`default`, `ipex`, `itex`, etc.) do not expose a dedicated “PyTorch CUDA” backend for NVIDIA; quantized kernels and FX lowering are generally CPU-focused.
- In practice, many tutorials and examples run **all PTQ tuning and sensitivity analysis on CPU**, even when the final deployment is on a different device (or after exporting to another runtime). This matches how INC is usually used for PyTorch today.
- To leverage **NVIDIA GPUs** during sensitivity/PTQ, the recommended path is:
  - Export an ONNX model (for Qwen-style LLMs, typically the language-model subgraph).
  - Use INC with an ONNX Runtime backend such as `onnxrt_cuda_ep` or `onnxrt_trt_ep`.
  - Build calibration/eval dataloaders that feed numpy arrays to ONNX Runtime, so both sensitivity (`mse_v2`) and PTQ runs execute on GPU via ORT/TensorRT.
- For our Qwen2.5-VL-3B tasks in this repo:
  - We use **PyTorch+INC on CPU** for initial sensitivity analysis and mixed-precision planning, keeping the flow close to the HF model.
  - We treat an **ONNX+ONNX Runtime CUDA** pipeline as the natural option when we want GPU-accelerated PTQ or deployable INT8 W8A8 artifacts for RTX 5090.

This section summarizes practical patterns from INC docs and common usage for turning sensitivity information into mixed-precision recipes.

### 5.1 General patterns for op sensitivity

- Prefer `quant_level=1` and set `TuningCriterion.strategy` explicitly (`"mse_v2"` or `"hawq_v2"`) if you care about per-op sensitivity; the default `"basic"` strategy is not sensitivity-focused.
- Use `strategy_kwargs` to control sensitivity scoring:
  - For `mse_v2`, set `{"confidence_batches": N}`; higher N → more stable scores but more tuning time. For large models, start with 2–4 and increase only if needed.
  - For `hawq_v2`, provide a meaningful loss function via `{"hawq_v2_loss": loss_fn}` so Hessian traces reflect your actual objective.
- Always feed representative calibration data:
  - For LLMs, use text that matches your deployment (chat, code, documents, etc.).
  - For ViT/CV, sample real images from your target domain, not just toy datasets.
- Interpret rankings as guidance, not absolute rules:
  - High-sensitivity ops (high MSE or high Hessian trace) are the best candidates to keep at higher precision.
  - Low-sensitivity ops are where you can “spend” your quantization budget (INT8, 4-bit, MXFP4, etc.).
- Narrow the tuning space:
  - Use `op_type_dict` / `op_name_dict` to force known fragile ops (e.g. LayerNorm, Softmax, logits heads) to stay fp32/bf16 so the strategy focuses on the remaining ops.

### 5.2 LLM-specific best practices

- For full W8A8 (weights + activations):
  - Use `mse_v2` for PTQ static on PyTorch or ONNX:
    - Force embedding layers, LM head, LayerNorms, and Softmax to remain higher precision in `op_type_dict` / `op_name_dict`.
    - Let `mse_v2` decide which attention and MLP blocks can be quantized to INT8.
  - Keep `confidence_batches` relatively small at first (2–4), especially for large sequence lengths and models.
  - Combine with SmoothQuant via `recipes={"smooth_quant": True, "smooth_quant_args": {...}}` to reduce activation outliers before running sensitivity analysis.
- For weight-only LLM quantization:
  - Use transformers-like APIs (RTN, AWQ, GPTQ, AutoRound, MX) so INC can do built-in sensitivity-aware rounding:
    - After quantization, treat `collect_weight_info(...)` and MX `target_bits` allocations as an implicit sensitivity map.
  - When deploying on non-Intel hardware (e.g. RTX 5090 + vLLM/TensorRT), map INC’s per-layer bit allocation and “high-precision layers” into your runtime configuration.
- For mixed-precision searches:
  - Use MSE or HAWQ scores to choose a small set of “high-importance” blocks (e.g. top 10–20% of layers by sensitivity).
  - Restrict high-precision formats (fp16/bf16/MXFP8) to that set and try more aggressive formats (int8/MXFP4/4-bit) on the rest.

### 5.3 ViT / non-LLM model best practices

- For ViT and similar models:
  - Keep patch embedding and final classification head in higher precision.
  - Keep LayerNorm, Softmax, and often GELU (or non-linearities around logits) in higher precision.
  - Use `mse_v2` PTQ static to quantify how sensitive attention and MLP projection layers are.
  - Use sensitivity ranking to:
    - Pin a small number of highly sensitive blocks to fp32/fp16.
    - Quantize the rest to INT8 or lower.
- For ResNet / CNNs:
  - Similar pattern: keep first and last convs (and sometimes downsample blocks) higher precision, let `mse_v2` rank intermediate convs.
  - Use accuracy criteria (`AccuracyCriterion`) to enforce a small tolerable loss (e.g. 0.5–1% top-1 drop).

### 5.4 Choosing between MSE_V2 and HAWQ_V2

- Use `mse_v2` when:
  - You want a simple, directly interpretable metric based on output MSE.
  - You primarily do PTQ and care about end-to-end behavior more than curvature analysis.
- Use `hawq_v2` when:
  - You are designing mixed-precision schemes and want Hessian-based sensitivity.
  - You can afford higher compute and have a well-defined loss that matches your deployment task.
- Practical tip:
  - Start with `mse_v2` to get a quick, robust ranking and a working quantized model.
  - Use `hawq_v2` (and/or AutoRound/MX mixed precision) when you are refining a mixed-precision scheme or doing research-level sensitivity analysis.

## References

- Tuning strategies (MSE_V2, HAWQ_V2): `extern/neural-compressor/docs/source/tuning_strategies.md`
- Hessian-based metrics: `extern/neural-compressor/neural_compressor/adaptor/torch_utils/hawq_metric.py`
- Transformers-like LLM API (Rtn, AWQ, GPTQ, AutoRound): `extern/neural-compressor/docs/source/3x/transformers_like_api.md`
- MX quantization and mixed MXFP4/MXFP8: `extern/neural-compressor/docs/source/3x/PT_MXQuant.md`
- Layer-wise quantization docs: `extern/neural-compressor/docs/source/quantization_layer_wise.md`
