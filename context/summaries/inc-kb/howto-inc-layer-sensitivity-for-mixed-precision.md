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

## 5. How INC actually computes per-layer sensitivity (PyTorch FX MSE_V2)

The precise mechanics of per-op sensitivity matter a lot when you try to interpret the results or re-use them in other frameworks. This section summarizes the critical behavior of `mse_v2` on the PyTorch FX backend and gives pseudocode aligned with the actual INC implementation.

### 5.1 Two phases: fallback vs re-quant (what is being quantized?)

For PyTorch FX, MSE-based op sensitivity is implemented by:

- `MSE_V2TuneStrategy` (strategy driver) in `extern/neural-compressor/neural_compressor/strategy/mse_v2.py`
- `PyTorch_FXAdaptor.calculate_op_sensitivity(...)` in `extern/neural-compressor/neural_compressor/adaptor/pytorch.py`
- Two helpers in `extern/neural-compressor/neural_compressor/adaptor/torch_utils/util.py`:
  - `get_mse_order_per_fp32(...)`
  - `get_mse_order_per_int8(...)`

There are *two distinct phases* in MSE_V2:

1. **Fallback sensitivity (`get_mse_order_per_fp32`)**  
   - Start from a config where **all candidate ops are quantized** (INT8).  
   - For each op, temporarily “fallback” that op to fp32 while keeping **all other ops quantized**.  
   - Measure how much the last-layer output MSE improves; high MSE = more sensitive to being quantized.

2. **Re-quant sensitivity (`get_mse_order_per_int8`)**  
   - Start from a config where some ops have already been kept fp32.  
   - For each fp32 op that could be re-quantized, quantize *only that op* while other ops remain in their current (often fp32) state.  
   - Measure how much the last-layer output MSE increases; low MSE = “safe” to re-quantize.

This means:

- In the **fallback phase**, you are measuring:  
  “If all ops are INT8 and I let this one op be fp32, how much error does it fix?”  
  → all others remain quantized.

- In the **re-quant phase**, you are measuring:  
  “If I leave almost everything fp32 and quantize this one op, how much error does it introduce?”  
  → this op is quantized while many others are fp32.

So INC does **not** always run with “only one quantized layer and everything else fp32”; the behavior depends on which phase you are looking at.

### 5.2 Pseudocode for the PyTorch FX fallback phase (get_mse_order_per_fp32)

This is a cleaned-up pseudocode version of `get_mse_order_per_fp32` from `torch_utils/util.py`, with comments marking the important details:

```python
def get_mse_order_per_fp32(adaptor, model, example_inp, tune_cfg):
    # 1) Build op-wise qconfig mapping from INC's tune_cfg
    op_cfgs = _cfg_to_qconfig(tune_cfg, tune_cfg["approach"])
    op_type_dict = {op_name: op_type for (op_name, op_type) in tune_cfg["op"].keys()}

    # 2) Hook last module to capture its output on FP32 model
    last_module_name = list(op_cfgs.keys())[-1]
    last_module = fetch_module(model, last_module_name)
    last_module.register_forward_hook(capture_inner_output)
    fp32_out = simple_inference(model, example_inp)
    inner_output_fp32 = captured_inner_output

    fallback_order = {}

    # 3) MAIN per-op loop (fallback sensitivity)
    #    Start from "all ops quantized" (op_cfgs), and for each op:
    #    - temporarily disable its qconfig (fallback to fp32)
    #    - quantize the rest of the model with FX
    #    - run forward and compare outputs vs FP32 baseline
    for op_name, qconfig in op_cfgs.items():
        if op_name == "bf16_ops_list":
            continue

        tmp_model = deepcopy(model)
        if not qconfig:
            # No quantization config for this op, skip
            continue

        # Temporarily disable this op's qconfig -> treat it as fp32
        op_cfgs[op_name] = None
        fx_op_cfgs = _cfgs_to_fx_cfgs(op_cfgs, tune_cfg["approach"])
        op_cfgs[op_name] = qconfig  # restore

        # *** FX quantization for "all other ops" ***
        # prepare_fx inserts observers on quantizable ops
        tmp_model = prepare_fx(tmp_model, fx_op_cfgs, example_inp)
        simple_inference(tmp_model, example_inp)  # calibration
        # convert_fx replaces ops with quantized modules (fbgemm backend)
        tmp_model = convert_fx(tmp_model)

        # Hook last module and run quantized model
        last_module_q = fetch_module(tmp_model, last_module_name)
        last_module_q.register_forward_hook(capture_inner_output)
        qdq_out = simple_inference(tmp_model, example_inp)
        inner_output_int8 = dequantize_if_needed(captured_inner_output)

        mse_val = (inner_output_fp32 - inner_output_int8).pow(2).sum()
        key = (op_name, op_type_dict[op_name])
        fallback_order[key] = float(mse_val.item())

    # 4) Return ops sorted by MSE (sensitivity)
    ordered_ops = sorted(fallback_order.keys(), key=lambda k: fallback_order[k], reverse=False)
    return ordered_ops
```

Important points:

- The loop **does not isolate one quantized op at a time**. Instead:
  - All ops that have qconfigs in `op_cfgs` are quantized, except the one being “fallbacked”.
  - This measures the error contribution of each op inside a fully-quantized model.
- The expensive part is inside the loop:
  - `prepare_fx` + `convert_fx` called for every candidate op.
  - This rebuilds an FX-quantized model and repacks quantized weights many times.

### 5.3 Pseudocode for the PyTorch FX re-quant phase (get_mse_order_per_int8)

The re-quant phase is similar, but starts from a more fp32-heavy config and quantizes one op at a time:

```python
def get_mse_order_per_int8(adaptor, fp32_model, example_inp, tune_cfg):
    op_cfgs = _cfg_to_qconfig(tune_cfg, tune_cfg["approach"])
    op_type_dict = {op_name: op_type for (op_name, op_type) in tune_cfg["op"].keys()}

    # Capture last-layer FP32 output once
    last_module = fetch_module(fp32_model, list(op_cfgs.keys())[-1])
    last_module.register_forward_hook(capture_inner_output)
    fp32_out = simple_inference(fp32_model, example_inp)
    inner_output_fp32 = captured_inner_output

    # Build list of ops that are currently fp32 but can be quantized
    quant_list = []
    for (op_name, op_type), cfg_state in tune_cfg["op"].items():
        if op_type in ["LayerNorm", "Dropout", "InstanceNorm3d"]:
            continue  # skip fragile ops
        if cfg_state["weight"]["dtype"] == "fp32":
            quant_list.append((op_name, op_type))

    fallback_order = {}

    # MAIN per-op loop (re-quant sensitivity)
    for op_name, op_type in quant_list:
        if op_name not in op_cfg_mapping:
            continue

        tmp_model = deepcopy(fp32_model)

        # Restore quantization config for this op only
        op_cfgs[op_name] = op_cfg_mapping[op_name]
        fx_op_cfgs = _cfgs_to_fx_cfgs(op_cfgs, tune_cfg["approach"])

        # Quantize this op in an otherwise FP32 model
        tmp_model = prepare_fx(tmp_model, fx_op_cfgs, example_inp)
        simple_inference(tmp_model, example_inp)
        tmp_model = convert_fx(tmp_model)

        last_module_q = fetch_module(tmp_model, list(op_cfgs.keys())[-1])
        last_module_q.register_forward_hook(capture_inner_output)
        qdq_out = simple_inference(tmp_model, example_inp)
        inner_output_int8 = dequantize_if_needed(captured_inner_output)

        mse_val = (inner_output_fp32 - inner_output_int8).pow(2).sum()
        key = (op_name, op_type_dict[op_name])
        fallback_order[key] = float(mse_val.item())

    ordered_ops = sorted(fallback_order.keys(), key=lambda k: fallback_order[k], reverse=False)
    return ordered_ops
```

Here:

- We start from a **mostly FP32 model** and quantify the effect of quantizing one op at a time.
- This is closer to “only this op is quantized” (though there can still be other quantized ops depending on the config).

### 5.4 Why this matters for interpretation and performance

Interpretation:

- Fallback MSE (`get_mse_order_per_fp32`) answers:  
  “In a fully quantized model, which ops would most reduce error if kept fp32?”  
  → Good for deciding which ops to **keep in higher precision**.

- Re-quant MSE (`get_mse_order_per_int8`) answers:  
  “Among currently fp32 ops, which can safely be moved back to int8?”  
  → Good for deciding which fp32 fallbacks can be **aggressively re-quantized**.

Performance:

- Both helpers rebuild an FX-quantized model **inside the per-op loop**, which is the major bottleneck we measured for Qwen2.5-VL-3B:
  - `prepare_fx` + `convert_fx` + quantized `Linear` packing (`linear_prepack`) dominate runtime.
  - Each op in the loop pays this cost separately.
- In this repo we:
  - Force these helpers to run even when `quantization.fit` would not (via `run_single_mse_v2_sensitivity_pass` and monkeypatching).
  - Add `INC_MSE_MAX_OPS` to limit how many ops go through the heavy loop for large LLMs.

For mixed-precision planning, it’s important to remember that INC’s MSE_V2 scores are computed under these two specific contexts (all-INT8 plus single fp32 fallback, and vice versa), not under a “one-layer-only quantized” regime.

## 6. Best-practice patterns (LLM and non-LLM)

### 5.1 Forcing INC to emit layer sensitivity even when tuning fails

For large LLMs it is common that no INT8 or mixed-precision configuration meets a strict accuracy goal within a small number of trials; if you rely only on `quantization.fit(..., strategy="mse_v2")`, the MSE-based sensitivity computation may never run and you will not get a layer-wise report, even though the underlying machinery exists.

For the PyTorch FX backend (used for Qwen-style models), the call path for MSE-based sensitivity is:
- `quantization.fit(...)` (in `neural_compressor/quantization.py`) chooses `strategy_name="mse_v2"` and constructs `MSE_V2TuneStrategy` when `framework="pytorch_fx"` and `tuning_criterion.strategy="mse_v2"`.
- `MSE_V2TuneStrategy.next_tune_cfg(...)` (in `strategy/mse_v2.py`) only calls `self.adaptor.calculate_op_sensitivity(...)` in its fallback / re-quant loops, after it has a “best” quantized config that meets the accuracy criterion.
- `PyTorch_FXAdaptor.calculate_op_sensitivity(...)` (in `adaptor/pytorch.py`) delegates to `get_fallback_order(...)` in `adaptor/torch_utils/util.py`.
- `get_fallback_order(...)` calls `get_mse_order_per_fp32(...)` and `get_mse_order_per_int8(...)`, which are the helpers that actually compute per-op MSE and build a `fallback_order` dict that is sorted into a sensitivity ranking.

If no quantized configuration ever meets the accuracy goal and `max_trials` is small, the fallback / re-quant loops are never entered, `calculate_op_sensitivity(...)` is never called, and the MSE helpers are never executed. To treat INC as a **layer analysis oracle** instead of an accuracy gate, you can use two patterns:

- **Pattern A: Monkeypatch the MSE helpers and call `calculate_op_sensitivity(...)` directly**
  - Monkeypatch `get_mse_order_per_fp32` and `get_mse_order_per_int8` in `neural_compressor.adaptor.torch_utils.util` to:
    - Copy the upstream implementations (from the installed INC version) so behavior stays identical.
    - Record the computed MSE per op (e.g., into `Dict[(op_name, op_type), float]` maps) before returning the sorted list.
  - With those patches active, build a `PostTrainingQuantConfig` for `pytorch_fx`, construct an adaptor and `tune_cfg`, and call:
    ```python
    ops_lst = adaptor.calculate_op_sensitivity(
        model=wrapped_model,
        dataloader=calib_dataloader,
        tune_cfg=tune_cfg,
        output_op_names=None,
        confidence_batches=1,
        fallback=True,
    )
    ```
  - This call goes through `get_fallback_order → get_mse_order_per_fp32`, so your patches see all MSE values and you can serialize the per-op maps to JSON/Markdown, even if `quantization.fit` itself never finds an acceptable model.
  - In this repo, `src/auto_quantize_model/inc_pytorch_mse_patching.py` implements this pattern for Qwen2.5-VL-3B and exposes a `capture_mse_v2_sensitivity()` context manager.

- **Pattern B: Ensure `mse_v2` calls `calculate_op_sensitivity(...)` at least once**
  - Loosen or trivialize the `AccuracyCriterion` for the `mse_v2` run so that at least one INT8 configuration is considered “good enough”, giving `MSE_V2TuneStrategy` a non-`None` `cur_best_tuning_cfg` / `last_qmodel`, and therefore entering the fallback loop where `calculate_op_sensitivity(...)` is invoked.
  - If you are comfortable with a deeper monkeypatch, you can wrap `MSE_V2TuneStrategy.next_tune_cfg` to inject a single `calculate_op_sensitivity(...)` call after the initial stage-1 configs, even if no trial meets the original accuracy goal; the important part is that the MSE helpers run on a representative calibration batch.

In both patterns, the key design principle is:
- Treat INC as a **scoring engine**: you care primarily that `get_mse_order_per_fp32` / `get_mse_order_per_int8` run on a representative dataloader and produce a deterministic ranking; whether `quantization.fit` returns a “good” quantized model is secondary. Once you have the per-op MSE maps, you can build your own mixed-precision profiles (e.g., keep the top-N most sensitive ops in fp16/bf16, quantize the rest) and feed them into INC, ModelOpt, TensorRT, or other runtimes.

### 5.2 PyTorch backend vs ONNX Runtime GPU

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

### 6.1 General patterns for op sensitivity

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

### 6.2 LLM-specific best practices

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

### 6.3 ViT / non-LLM model best practices

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

### 6.4 Choosing between MSE_V2 and HAWQ_V2

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
