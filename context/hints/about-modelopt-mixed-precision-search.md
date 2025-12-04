This note explains how NVIDIA TensorRT Model Optimizer (ModelOpt) decides the *mixed precision scheme* for quantization, and which knobs you can adjust. It is based both on the local source under `extern/TensorRT-Model-Optimizer` and the official docs.

Relevant docs:
- AutoQuantize (PyTorch PTQ search): https://nvidia.github.io/TensorRT-Model-Optimizer/guides/_pytorch_quantization.html
- AutoCast (ONNX mixed FP16/BF16): https://nvidia.github.io/TensorRT-Model-Optimizer/guides/8_autocast.html
- ONNX quantization graph utils (mixed 4/8 bit): https://nvidia.github.io/TensorRT-Model-Optimizer/reference/generated/modelopt.onnx.quantization.graph_utils.html

---

## 1. PyTorch `mtq.auto_quantize`: optimal per-layer mixed precision

**Where it lives**
- API: `modelopt.torch.quantization.model_quant.auto_quantize`
- Search logic: `modelopt.torch.quantization.algorithms.AutoQuantizeGradientSearcher` and `AutoQuantizeKLDivSearcher`
- Docs section: “Optimal Partial Quantization using `auto_quantize`”

**What it does**
- `auto_quantize` is a post-training quantization (PTQ) algorithm that searches, per layer, over a *set of quantization formats* (e.g. NVFP4, FP8, w4a8 AWQ, or “no quantize”) and picks the format that best trades off **accuracy vs. model size / effective bits**.
- You pass:
  - A constraint `constraints={"effective_bits": X}` describing the target average effective bits for the model (e.g. 4.8).
  - A list of candidate quantization formats (`quantization_formats=[mtq.NVFP4_AWQ_LITE_CFG, mtq.FP8_DEFAULT_CFG, ...]`).
- The algorithm then:
  1. Calibrates the model for each candidate format (like `mtq.quantize`).
  2. Estimates a **sensitivity score per (layer, format)**.
  3. Solves an optimization problem to choose exactly one format per layer, subject to the effective bits / weight-size constraint.

### 1.1 Gradient-based AutoQuantize (default `method="gradient"`)

Implementation points:
- Code path: `AutoQuantizeGradientSearcher` in `modelopt/torch/quantization/algorithms.py`.
- It approximates the **change in loss** from quantizing a given layer+format using a second-order Taylor expansion with a Fisher information approximation.

High-level algorithm:
1. For each quantizable module and each candidate quantization format (plus a “no quantize” recipe), temporarily attach a “recipe hyperparameter” (`QuantRecipeHparam`) listing all choices.
2. For each calibration batch:
   - Run a forward pass with all affected quantizers effectively “off” (recipe = `QuantRecipe(quant_cfg=None)`) to obtain the baseline output `Y`.
   - For each candidate format for that module:
     - Turn that recipe on, run the module again on the same inputs to get `Y_q`.
     - Store the output difference `output_diff = Y_q - Y` for this module+format.
3. Run a normal training-like **backward pass** once per batch:
   - Let `grad_output = dL/dY` for each module.
   - For each candidate recipe, compute the **auto_quantize score**:
     - `score = sum((grad_output**2) * (output_diff**2))`
   - This is equivalent to a Hessian/Fisher-based approximation of the increase in loss if you apply that quantization format to that layer.
4. Aggregate scores and costs:
   - For each module+format, we accumulate:
     - `scores`: sensitivity to that format.
     - `costs`: weight size under that format (number of parameters × bit-width).
5. Solve a **linear program** (LP):
   - Objective: minimize total sensitivity `sum_i scores[i, chosen_format_i]`.
   - Subject to: `sum_i costs[i, chosen_format_i] <= max_weight_size`, where `max_weight_size` corresponds to your `effective_bits` constraint.
   - If LP has trouble converging without a compression lower bound, it retries with stricter bounds.
6. The LP selects one format per layer. The chosen formats are applied and pre-quant scales (if any) are folded into weights.

The output is:
- A quantized model with a per-layer mixture of formats (e.g. some layers NVFP4, some FP8, some unquantized).
- A `state_dict` containing search history and stats.

### 1.2 KL-divergence-based AutoQuantize (`method="kl_div"`)

Implementation points:
- Code path: `AutoQuantizeKLDivSearcher` in `modelopt/torch/quantization/algorithms.py`.
- Instead of gradients, it uses **KL divergence** between the unquantized and quantized output distributions as the sensitivity metric.

High-level algorithm:
1. For each batch:
   - Set all `QuantRecipeHparam`s to “unquantized” and run `forward_step(model, data)` to get logits `logits_unquant`.
   - Convert to probabilities `p = softmax(logits_unquant)` (possibly using LM head with distributed awareness).
2. For each quantization recipe for each layer:
   - Turn on that recipe, run `forward_step` again to get logits `logits_quant`.
   - Compute `score = KL(p || q)` (implemented as `-p * log q` using log-softmax for numerical stability).
   - Accumulate scores per (layer, format).
3. After all batches, you have per-layer lists of `(formats, scores, costs)`.
4. Instead of LP, it runs a **threshold-based binary search**:
   - Choose a KL threshold `T`.
   - For each layer, pick the *cheapest format whose score <= T* (else fall back to a safe one).
   - Compute total weight size; if it exceeds the target `max_weight_size`, adjust `T` and repeat.
   - This binary search aims to minimize the maximum tolerated KL-divergence per layer while satisfying the weight-size constraint.

Because it relies only on model outputs (logits), `kl_div` mode requires only a `forward_step` returning logits and typically works well for LLMs and classification models.

### 1.3 Key knobs to adjust for `auto_quantize`

API: `modelopt.torch.quantization.model_quant.auto_quantize`

Core arguments:
- `constraints={"effective_bits": 4.8}`: The main mixed-precision *budget*. Lower values push more aggressive formats (e.g. FP4) into more layers; higher values keep more layers in higher-precision formats or unquantized.
- `quantization_formats=[...]`: Set of candidate formats to search over. Examples:
  - `mtq.NVFP4_AWQ_LITE_CFG` (FP4-style weight quantization)
  - `mtq.FP8_DEFAULT_CFG`
  - `mtq.W4A8_AWQ_BETA_CFG` (4-bit weights, 8-bit activations, AWQ calibration)
  - Custom config dicts matching the `mtq.quantize` config format.
- `data_loader`: Calibration data for both quantizer calibration and sensitivity scoring.
- `forward_step(model, batch)`: Must run the model and return outputs; for `gradient` mode the outputs feed into `loss_func`, for `kl_div` mode they should be logits.
- `loss_func(output, batch)` or `forward_backward_step(model, batch)`: Required for `method="gradient"` to define the loss whose Hessian/Fisher is approximated.

Search / speed vs. quality:
- `num_calib_steps` (default 512): Number of batches used to *calibrate* each candidate quantization format. Larger → better calibration at cost of time.
- `num_score_steps` (default 128): Number of batches used to estimate **sensitivity scores**. This dominates runtime; reducing it is the recommended way to speed up `auto_quantize` with relatively small accuracy impact.
- `method`:
  - `"gradient"`: Gradient-based sensitivity, LP solve; most accurate when you have a well-defined scalar loss.
  - `"kl_div"`: KL-divergence sensitivity, threshold-based search; simpler when you just have logits and no custom loss.

Structure / coverage:
- `disabled_layers`: Wildcard patterns (e.g. `"*lm_head*"`, `"*mlp*"`) to force some layers to stay unquantized in the search.
- Internal grouping rules: `AutoQuantizeSearcher.quant_grouping_rules` groups related layers (e.g. Q/K/V projections in the same transformer block) so they always share the same format; this is important for TensorRT-LLM compatibility and reduces search complexity.

Practical example (gradient-based AutoQuantize mixing NVFP4 and FP8):

```python
import modelopt.torch.quantization as mtq

model = get_model()
calib_loader = get_calib_loader()

def forward_step(model, batch):
    return model(**batch)

def loss_func(output, batch):
    return output["loss"]

model_q, search_state = mtq.auto_quantize(
    model,
    constraints={"effective_bits": 4.8},
    quantization_formats=[mtq.NVFP4_AWQ_LITE_CFG, mtq.FP8_DEFAULT_CFG],
    data_loader=calib_loader,
    forward_step=forward_step,
    loss_func=loss_func,
    num_calib_steps=512,
    num_score_steps=128,
    method="gradient",
    verbose=True,
)
```

---

## 2. ONNX AutoCast: automatic FP32 vs FP16/BF16 node selection

**Where it lives**
- CLI: `python -m modelopt.onnx.autocast`
- Code: `modelopt/onnx/autocast/convert.py`, `nodeclassifier.py`, `precisionconverter.py`
- Docs: “AutoCast (ONNX)” guide

**What it does**
- AutoCast starts from an **FP32 ONNX model** and converts it to *mixed* FP32–FP16 or FP32–BF16:
  - Nodes considered “safe” are executed in low precision.
  - Nodes considered “risky” are kept in FP32, with cast nodes inserted automatically.
- It uses a **NodeClassifier** that applies a sequence of rules based on:
  - Node I/O magnitudes from a reference run.
  - Initializer magnitudes.
  - Reduction depth (size of contractions like matmuls and convolutions).
  - Regex and op-type includes/excludes.

### 2.1 AutoCast classification algorithm (NodeClassifier)

Key structures:
- Class: `modelopt.onnx.autocast.nodeclassifier.NodeClassifier`.
- Inputs:
  - `nodes_to_exclude`: regex patterns for node names to keep in high precision.
  - `op_types_to_exclude`: op types to keep in high precision.
  - `nodes_to_include`: regex patterns to force nodes into low precision.
  - `op_types_to_include`: op types to force into low precision.
  - `data_max`: maximum allowed magnitude for node outputs in low precision.
  - `init_max`: maximum allowed magnitude for initializers in low precision.
  - `max_depth_of_reduction`: maximum allowed depth for reduction operations in low precision.
  - Optionally, a `custom_rule` implementing `NodeRuleBase`.

High-level flow:
1. **Reference data collection**:
   - If `calibration_data` is provided, AutoCast runs the original model with ONNX Runtime (optionally using TensorRT as an EP) to record input/output magnitudes for each node, plus initializer stats.
   - If you don’t provide data, it uses random inputs as a fallback.
2. **Rule construction** (`_gen_exclude_node_rules` and `_gen_include_node_rules`):
   - Exclude rules:
     - `DisabledNodeNameRegexRule(nodes_to_exclude)` → blocks nodes by name.
     - `DisabledOpTypes(op_types_to_exclude)` → blocks nodes by type.
     - `InitializerRangeRule(init_max, node_to_init_map)` → blocks nodes whose initializers exceed `init_max`.
     - `IORangeRule(data_max, reference_data, node_to_init_map)` → blocks nodes whose outputs exceed `data_max`.
     - `DepthOfReductionRule(max_depth_of_reduction, ...)` → blocks nodes with too large reduction dimensions.
     - Optional `custom_rule`.
   - Include rules:
     - `IncludeNodeNameRegexRule(nodes_to_include)`.
     - `IncludeOpTypes(op_types_to_include)`.
3. **Classification** (`NodeClassifier.run`):
   - Iterates over nodes in the graph:
     - If a node name is not already in the low-precision set and **any exclude rule fires** and **no include rule fires**, the node is placed in the **high precision group**.
     - Otherwise, it goes into the **low precision group**.
   - Returns two lists: `low_precision_nodes`, `high_precision_nodes`.
4. **Precision conversion**:
   - `PrecisionConverter.convert(high_precision_nodes, low_precision_nodes)`:
     - Inserts `Cast` ops from FP32 to low precision (FP16/BF16) and back where needed.
     - Converts initializers to low precision when safe.
     - Ensures graph validity and maintains I/O types depending on `keep_io_types`.

Intuitively: nodes with very large activations, large-weight initializers, or very large reduction depths are kept in FP32; everything else is aggressively converted to FP16/BF16, unless you override it via name/type lists.

### 2.2 AutoCast knobs you can tune

CLI: `python -m modelopt.onnx.autocast --onnx_path model.onnx ...`

Important arguments:
- `--low_precision_type {fp16,bf16}`: Target low precision.
- `--calibration_data`: Path to NPZ / Polygraphy JSON with sample inputs for more realistic magnitudes.
- `--nodes_to_exclude`: Regex patterns for node names to **force FP32**.
- `--op_types_to_exclude`: Op types (e.g. `Softmax`, `LayerNormalization`) to **force FP32**.
- `--nodes_to_include`: Regex patterns for node names to **force low precision**, even if other rules would exclude them.
- `--op_types_to_include`: Op types to force into low precision.
- `--data_max`: Maximum allowed activation magnitude in low precision. Lower values keep more nodes in FP32.
- `--init_max`: Maximum allowed initializer magnitude in low precision. Values above this risk overflow and are kept in FP32.
- `--max_depth_of_reduction`: Nodes whose reduction depth exceeds this threshold are kept in FP32.
- `--keep_io_types`: If set, keeps model inputs and outputs in original precision (FP32) and inserts casts internally only.
- `--opset`: Target ONNX opset (e.g. 13 for FP16, 22 for BF16) used by `GraphSanitizer`.

Example AutoCast usage:

```bash
python -m modelopt.onnx.autocast \
  --onnx_path vit_b32.fp32.onnx \
  --output_path vit_b32.fp16_mixed.onnx \
  --low_precision_type fp16 \
  --calibration_data calib_inputs.npz \
  --nodes_to_exclude ".*LayerNorm.*" \
  --op_types_to_exclude Softmax \
  --data_max 512 \
  --init_max 6.55e4 \
  --max_depth_of_reduction 4096 \
  --keep_io_types
```

---

## 3. ONNX INT4/INT8 mixed weights (heuristic 4/8-bit selection)

ModelOpt’s ONNX INT4 path has a more **heuristic** mixed-precision mechanism compared to AutoQuantize.

**Where it lives**
- Code: `modelopt/onnx/quantization/int4.py` and `graph_utils.get_layer_precision_mapping`
- Relevant args: `enable_mixed_quant`, `layers_8bit`, `block_size`, `quantize_axis`, `gather_block_size`, `gather_quantize_axis`

### 3.1 Heuristic layer precision mapping (4 vs 8 bits)

`graph_utils.get_layer_precision_mapping`:
- Looks for MatMul/Gemm nodes that match transformer-style patterns like:
  - `/model/layers.<i>/attn/qkv_proj/MatMul`
  - `/model/layers.<i>/attn/v_proj/MatMul`
  - `/model/layers.<i>/mlp/down_proj/MatMul`
- Groups these nodes by function (e.g. qkv-proj group) and sorts them by layer index.
- For each group, it chooses:
  - **First 1/8 of layers** → upgraded to **8-bit**.
  - **Last 1/8 of layers** → also **8-bit**.
  - In the middle 6/8, it marks **every third** layer as 8-bit.
- Everything else defaults to **4-bit** with the provided `block_size` and `axis`.
- It returns a `layer_info` dict mapping each weight initializer name to:
  - `precision`: `4` or `8`.
  - `block_size`: per-layer block size (e.g. `128` or `-1` for per-channel).
  - `axis`: quantization axis (usually 0 for per-output-channel quantization).

The `int4` quantization code (`_quantize_awq_lite`) then uses:
- `layer_info` to pick `num_bits` for each weight.
- `update_block_size(...)` to adjust block size where 8-bit is selected.
- Mixed quantization can be toggled via `enable_mixed_quant` and optionally overridden via `layers_8bit` patterns.

This is not an optimization-based search like AutoQuantize; it is a **fixed policy** that keeps a fraction of sensitive transformer layers at 8 bits and quantizes the rest to 4 bits, with optional user overrides.

### 3.2 Practical knobs

When using the Python API (`modelopt.onnx.quantization.quantize`), you can control:
- `enable_mixed_quant=True`: Turn on per-layer 4/8-bit mixing.
- `layers_8bit="pattern1,pattern2,...“`: Explicitly mark which ONNX node-name patterns should be 8-bit instead of 4-bit.
- `block_size`, `quantize_axis`: Default 4-bit quantization layout.
- `gather_block_size`, `gather_quantize_axis`: Separate settings for `Gather`-based embeddings.

Conceptually:
- Use AutoQuantize (PyTorch) when you want **optimal per-layer search** over formats like NVFP4, FP8, w4a8 (with effective bits as a budget).
- Use AutoCast (ONNX) when you want **data-driven FP16/BF16 vs FP32 node selection** for general FP models.
- Use the INT4/INT8 mixed-precision heuristics when you only need a lightweight 4/8-bit split for ONNX transformer graphs and can accept a fixed policy rather than a search.

---

## 4. Summary: which algorithm does what?

- **PyTorch `auto_quantize`**:
  - Algorithm: sensitivity-based search (gradient/Fisher or KL-div) + LP / binary search over per-layer formats.
  - Goal: pick a per-layer mixture of quantization formats (e.g. FP4, FP8, w4a8, or unquantized) that minimizes a proxy loss while satisfying an effective bits constraint.
  - Main knobs: `effective_bits`, `quantization_formats`, `num_calib_steps`, `num_score_steps`, `method`, `disabled_layers`, calibration data and loss / forward functions.

- **ONNX AutoCast**:
  - Algorithm: rule-based NodeClassifier using activation and initializer magnitudes, reduction depth, and user include/exclude rules to decide FP32 vs FP16/BF16 per node; then PrecisionConverter inserts casts and converts initializers.
  - Goal: mixed FP16/BF16 with minimal accuracy loss and no quantization (no INT4/INT8).
  - Main knobs: `data_max`, `init_max`, `max_depth_of_reduction`, `{nodes,op_types}_to_{exclude,include}`, `keep_io_types`, `low_precision_type`, `calibration_data`.

- **ONNX INT4 mixed 4/8-bit**:
  - Algorithm: heuristic layer selection via `get_layer_precision_mapping`, plus optional explicit `layers_8bit` patterns; no global optimization or constraints.
  - Goal: simple policy-based mix of 4/8-bit weights in transformer-style ONNX graphs.
  - Main knobs: `enable_mixed_quant`, `layers_8bit`, `block_size`, `quantize_axis`, gather-related options.

---

## 5. Qwen LLMs: support and quantization algorithms

ModelOpt has first-class support for the Qwen family (Qwen2, Qwen2.5, Qwen3, Qwen3-MoE, and several DeepSeek-*Qwen* distills) in both the PyTorch LLM PTQ pipeline and the ONNX PTQ pipeline.

- The ONNX PTQ README lists multiple Qwen2 / Qwen2.5 models as supported for FP16, INT4, FP8, and NVFP4: `extern/TensorRT-Model-Optimizer/examples/onnx_ptq/README.md`.
- The LLM PTQ HF script `hf_ptq.py` drives quantization and export for HF checkpoints, including Qwen variants (via generic Hugging Face handling and Qwen-specific export plugins).

### 5.1 PyTorch LLM PTQ for Qwen (HF checkpoints)

For HF Qwen models (e.g. `Qwen/Qwen2-7B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen3-8B`, DeepSeek-R1 Qwen distills), you typically use `examples/llm_ptq/hf_ptq.py`:

- Qwen is treated like other HF causal LMs; quantization is controlled by:
  - `--qformat`: selects a quantization config from `QUANT_CFG_CHOICES`.
  - `--auto_quantize_bits`: optional AutoQuantize mixed-precision search on top of these formats.

The mapping from `--qformat` to quantization formats and their underlying algorithms lives in `hf_ptq.py`:

> QUANT_CFG_CHOICES: dict[str, dict[str, Any]] = {  
>     "int8": mtq.INT8_DEFAULT_CFG,  
>     "int8_sq": mtq.INT8_SMOOTHQUANT_CFG,  
>     "int8_wo": mtq.INT8_WEIGHT_ONLY_CFG,  
>     "fp8": mtq.FP8_DEFAULT_CFG,  
>     "int4_awq": mtq.INT4_AWQ_CFG,  
>     "w4a8_awq": mtq.W4A8_AWQ_BETA_CFG,  
>     "nvfp4": mtq.NVFP4_DEFAULT_CFG,  
>     "nvfp4_awq": mtq.NVFP4_AWQ_LITE_CFG,  
>     "fp8_pb_wo": mtq.FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG,  
>     "fp8_pc_pt": mtq.FP8_PER_CHANNEL_PER_TOKEN_CFG,  
>     "w4a8_nvfp4_fp8": mtq.W4A8_NVFP4_FP8_CFG,  
>     "w4a8_mxfp4_fp8": mtq.W4A8_MXFP4_FP8_CFG,  
>     "nvfp4_mlp_only": mtq.NVFP4_MLP_ONLY_CFG,  
> }

These configs themselves encode the quantization algorithm used for LLMs via the `algorithm` field in `modelopt/torch/quantization/config.py`. For example:

> INT8_SMOOTHQUANT_CFG = {  
>     "quant_cfg": {  
>         "*weight_quantizer": {"num_bits": 8, "axis": 0},  
>         "*input_quantizer": {"num_bits": 8, "axis": None},  
>         **_default_disabled_quantizer_cfg,  
>     },  
>     "algorithm": "smoothquant",  
> }  
>  
> INT4_AWQ_CFG = {  
>     "quant_cfg": {  
>         "*weight_quantizer": {  
>             "num_bits": 4,  
>             "block_sizes": {-1: 128, "type": "static"},  
>             "enable": True,  
>         },  
>         "*input_quantizer": {"enable": False},  
>         **_default_disabled_quantizer_cfg,  
>     },  
>     "algorithm": {"method": "awq_lite", "alpha_step": 0.1},  
> }

So for Qwen LLMs in the HF PTQ flow, the main algorithms are:
- **INT8 / FP8**: standard PTQ with `algorithm="max"` or `algorithm="smoothquant"` (for INT8 SmoothQuant).
- **INT4 / W4A8 / NVFP4**: activation-aware weight quantization (**AWQ-lite**) for 4-bit weight formats, often combined with FP8 activations (e.g. `W4A8_AWQ_BETA_CFG`, `NVFP4_AWQ_LITE_CFG`).
- **KV cache quantization**: separate FP8 / NVFP4 configs with `algorithm="max"` or affine variants for Qwen’s KV cache.

On top of this, you can enable AutoQuantize for Qwen by passing `--auto_quantize_bits`. The helper in `hf_ptq.py` wraps the generic `mtq.auto_quantize` API:

> def auto_quantize(  
>     model,  
>     qformat,  
>     calib_dataloader,  
>     calibrate_loop,  
>     auto_quantize_bits,  
>     batch_size=1,  
>     auto_quantize_method="gradient",  
>     auto_quantize_score_size=128,  
>     auto_quantize_checkpoint=None,  
> ):  
>     qformat_list = qformat.split(",")  
>     ...  
>     if auto_quantize_method == "gradient":  
>         def forward_step(model, batch):  
>             return model(**batch)  
>     elif auto_quantize_method == "kl_div":  
>         def forward_step(model, batch):  
>             return model(**batch).logits  
>     ...  
>     model, _ = mtq.auto_quantize(  
>         model,  
>         constraints={"effective_bits": auto_quantize_bits},  
>         data_loader=calib_dataloader,  
>         forward_step=forward_step,  
>         loss_func=loss_func,  # Only used for gradient-based method  
>         quantization_formats=[QUANT_CFG_CHOICES[format] for format in qformat_list],  
>         num_calib_steps=len(calib_dataloader),  
>         num_score_steps=min(len(calib_dataloader), max(auto_quantize_score_size // batch_size, 1)),  
>         verbose=True,  
>         disabled_layers=["*lm_head*"],  
>         method=auto_quantize_method,  
>         checkpoint=auto_quantize_checkpoint,  
>     )

For Qwen LLMs, AutoQuantize therefore:
- Searches over one or more of the LLM-oriented formats above (INT4-AWQ, W4A8-AWQ, FP8, NVFP4, etc.).
- Uses the gradient/Fisher-based or KL-divergence–based sensitivity scoring described earlier.
- Optimizes a per-layer format assignment under an `effective_bits` constraint (e.g. 4.8).

### 5.2 ONNX PTQ for Qwen LLMs

For ONNX-exported Qwen models (e.g. Qwen2 / Qwen2.5), the supported ONNX quantization modes in the `examples/onnx_ptq` README show:
- FP16 baseline ONNX.
- INT4 / FP8 / NVFP4 ONNX using `modelopt.onnx.quantization.quantize`.

The underlying algorithms here are the same as described in sections 2 and 3:
- **INT8 / FP8 ONNX**: standard PTQ with `calibration_method=max` or `entropy` for activations, plus max-based weight quantization.
- **INT4 ONNX**: AWQ-lite based weight quantization for transformer blocks, with optional mixed 4/8-bit heuristics via `enable_mixed_quant` and `get_layer_precision_mapping`.
- **AutoCast (FP16/BF16 vs FP32)**: can be used on Qwen ONNX graphs when you only want FP16/BF16 mixed precision rather than integer quantization, driven by the NodeClassifier rules.

In short, ModelOpt can be used with Qwen-family LLMs both in PyTorch (HF checkpoints) and ONNX form, and the algorithms applied are the same advanced LLM quantization methods used for Llama/Gemma/etc.: SmoothQuant and max-based PTQ for 8-bit/FP8, AWQ-lite for 4-bit weights and W4A8/NVFP4 mixes, plus optional AutoQuantize mixed-precision search to optimally combine these formats per layer for a given effective-bit budget.

---

## 6. Generic LLM mixed-precision search for new models

AutoQuantize is not hard-wired to specific model classes like Llama or Qwen. As long as your LLM is a regular PyTorch `nn.Module` for which ModelOpt can insert quantizers, you can run the same per-layer mixed-precision search on it.

The core API is the same `mtq.auto_quantize` described earlier:

> def auto_quantize(  
>     model: nn.Module,  
>     constraints: dict[str, float | str] = {"effective_bits": 4.8},  
>     quantization_formats: list[dict[str, Any] | str] = [...],  
>     data_loader: Iterable | None = None,  
>     forward_step: Callable[[nn.Module, Any], Any | torch.Tensor] | None = None,  
>     loss_func: Callable[[Any, Any], torch.Tensor] | None = None,  
>     forward_backward_step: Callable[[nn.Module, Any], Any] | None = None,  
>     disabled_layers: list[str] | str | None = None,  
>     num_calib_steps: int = 512,  
>     num_score_steps: int = 128,  
>     verbose: bool = False,  
>     method: str = "gradient",  
>     checkpoint: str | None = None,  
> ):  
>     r"""Perform optimal per-layer quantization by searching for the best quantization formats per-layer.

For a *new* LLM that ModelOpt does not “recognize” by name, the generic workflow is:

1. **Ensure the model is structurally quantizable**
   - If it is built from standard `nn.Linear`, `nn.Conv*`, etc., ModelOpt’s default quant modules usually work out of the box.
   - If you have custom blocks (e.g. `MyDecoderBlock`), define a quantized subclass and register it via `QuantModuleRegistry`, then verify that a plain PTQ run with `mtq.quantize` works for your model with a simple config (e.g. `INT8_DEFAULT_CFG` or `FP8_DEFAULT_CFG`).

2. **Set up calibration data and scoring functions**
   - `data_loader`: yields tokenized text batches appropriate for your LLM (e.g. `{"input_ids": ..., "attention_mask": ..., "labels": ...}`).
   - `forward_step(model, batch)`: forwards a batch and returns either:
     - a full output with `.loss` for `method="gradient"`, or
     - logits for `method="kl_div"`.
   - `loss_func(output, batch)`: for gradient-based AutoQuantize, turns `output` into a scalar loss (commonly `output.loss` from `CausalLMOutputWithPast`).

3. **Choose candidate quantization formats**
   - Use the same LLM-friendly formats as for Qwen/Llama:
     - `mtq.INT4_AWQ_CFG` (4-bit weights, AWQ-lite).
     - `mtq.W4A8_AWQ_BETA_CFG` (W4A8 AWQ).
     - `mtq.FP8_DEFAULT_CFG` or `mtq.NVFP4_AWQ_LITE_CFG`.
   - Or define your own configs following `config.py` examples.

4. **Run AutoQuantize on the new LLM**

Example for a generic HF-style causal LM:

```python
import modelopt.torch.quantization as mtq

model = get_new_llm()                # any nn.Module
calib_loader = get_calib_loader()    # iterable of dict batches

def forward_step(model, batch):
    return model(**batch)            # returns object with .loss and .logits

def loss_func(output, batch):
    return output.loss

quant_formats = [mtq.INT4_AWQ_CFG, mtq.FP8_DEFAULT_CFG]

model_q, search_state = mtq.auto_quantize(
    model,
    constraints={"effective_bits": 4.8},
    quantization_formats=quant_formats,
    data_loader=calib_loader,
    forward_step=forward_step,
    loss_func=loss_func,
    num_calib_steps=512,
    num_score_steps=128,
    method="gradient",              # or "kl_div" if you only return logits
    disabled_layers=["*lm_head*"],  # optional
)
```

Under the hood, the same logic as for Qwen/Llama applies:
- AutoQuantize calibrates each candidate format per layer.
- It estimates sensitivity scores per (layer, format) using gradient/Fisher or KL-divergence.
- It solves for the best per-layer format combination subject to your `effective_bits` constraint, using LP (gradient mode) or threshold-based search (KL mode).
- Grouping rules (quant_grouping_rules) ensure functionally coupled layers (e.g. attention projections) share a format, even for unfamiliar architectures, as long as their naming matches typical transformer patterns or you adjust the grouping logic.

If this procedure succeeds once on a new LLM, you effectively have a generic mixed-precision quantization scheme searcher for that model, without needing any special-case logic for its class name. ONNX AutoCast and ONNX INT4 mixed-precision remain available as a generic path on the exported ONNX graph when you only need FP16/BF16 vs FP32 decisions or simple 4/8-bit mixes rather than full AutoQuantize search.

---

## 7. Controlling AutoQuantize search effort

ModelOpt exposes several knobs to control how expensive the mixed-precision search is for LLMs. These primarily affect the AutoQuantize sensitivity scoring and solver phases.

The `auto_quantize` docstring explicitly calls out that scoring dominates runtime:

> The sensitivity scoring phase typically dominates the runtime of ``auto_quantize``, so decreasing the number of  
> samples used for scoring (see ``num_score_steps``) is the recommended way for improving overall auto_quantize time  
> with minimal accuracy impact.

The key levers are:

- **Search space size (candidate formats)**  
  - `quantization_formats=[...]` controls how many formats are considered per layer.  
  - Fewer formats → fewer (layer, format) candidates → less calibration and scoring work.  
  - For a new LLM, starting with 2 formats (e.g. INT4_AWQ + FP8) is a good compromise.

- **Calibration vs scoring iterations**  
  - `num_calib_steps`: number of batches used to calibrate each candidate format (shared behavior with plain PTQ). Lowering it reduces calibration cost at some risk to calibration quality.  
  - `num_score_steps`: number of batches used for sensitivity estimation; this is the **main** cost driver. Reducing it is the recommended way to shrink search effort.

- **LLM helper knobs (`hf_ptq.py`)**  
  - The HF LLM helper computes `num_score_steps` based on `auto_quantize_score_size` and `batch_size`:
    > num_score_steps = min(len(calib_dataloader), max(auto_quantize_score_size // batch_size, 1))  
  - Lower `auto_quantize_score_size` or increase `batch_size` to reduce the number of scoring iterations for LLMs.

- **Scoring method choice**  
  - `method="gradient"`: uses forward+backward; more expensive but potentially more accurate sensitivity scores.  
  - `method="kl_div"`: uses forward-only logits and KL-divergence; cheaper (no backprop, simpler search), often sufficient for causal LMs.  
  - Switching to `kl_div` can significantly reduce search time when gradients are expensive or memory-limited.

- **Layer coverage and grouping**  
  - `disabled_layers=[patterns]` (e.g. `"*lm_head*"` or `"*mlp*"`): removes those layers from search and quantization, shrinking the number of decision variables.  
  - Internal `quant_grouping_rules` aggregate related layers (e.g. Q/K/V projections in an attention block) so they share a single format decision, which also reduces search complexity. For exotic architectures, tightening these groupings further (if you extend the rules) can cut effort.

- **Checkpointing / reusing scores**  
  - `checkpoint=path` lets AutoQuantize save its search state (scores, costs, best recipes, etc.). On rerun, if the checkpoint exists, ModelOpt can skip the expensive sensitivity estimation stage and only rerun the solver for new constraints (e.g. different `effective_bits`).  
  - This is the most efficient way to sweep over multiple effective-bit targets for the same model and calibration dataset.

For ONNX paths (AutoCast and ONNX PTQ), there is no combinatorial AutoQuantize-style search; runtime is controlled mainly by:
- How many calibration samples you feed into AutoCast / ONNX PTQ.  
- Whether you use per-node calibration (`--calibrate_per_node`) for large models, which trades extra ONNX graph processing for reduced peak memory.  
So for ONNX, “search effort” is largely “calibration effort,” whereas for PyTorch LLM AutoQuantize you also directly control the cost of the sensitivity scoring and optimization steps via the knobs above.

---

## 8. Relationship to Hessian-based mixed-precision methods

AutoQuantize’s gradient-based mode is explicitly a *second-order* method that approximates Hessian information, but ModelOpt does not (as of current public docs and examples) provide direct, published comparisons against classic Hessian-based mixed-precision schemes like HAWQ / HAWQ-V3 / GPTQ.

From the algorithms reference:

> The auto_quantize score for a layer quantization configuration is an approximation of model loss change due  
> to quantizing the particular layer with the particular configuration.  
> The approximation is based on taylor expansion of the loss function wrt to the quantized output of the layer and  
> substitution of Fisher information for Hessian.  
> This approximation is mathematically correct for models where the loss  
> is a log likelihood loss such as BERT, GPT, etc. However, the auto_quantize score can still be used as a proxy  
> for other models such as ResNet.

Conceptually:
- **Gradient-based AutoQuantize** uses Fisher information (expected outer product of gradients) as a Hessian surrogate to build per-layer sensitivity scores, then solves a constrained optimization over formats. It is second-order in spirit but avoids explicit Hessian eigen decompositions.
- **KL-divergence AutoQuantize** skips gradients entirely and measures sensitivity as KL(p || q) between unquantized and quantized logits; this is not Hessian-based, but still reflects local curvature of the model’s output distribution.
- **Classic Hessian-based methods** (e.g. HAWQ) compute or approximate Hessian eigenvalues per layer and use them to choose bit-widths; these are separate research lines and are not implemented as named algorithms inside ModelOpt’s AutoQuantize at this time.

As of the online material surveyed, there are no official, head-to-head benchmarks published by NVIDIA that compare AutoQuantize (gradient or KL-div modes) directly against external Hessian-aware schemes like HAWQ / HAWQ-V3 / GPTQ on common LLM tasks. To compare them empirically, you would need to:
- Run ModelOpt AutoQuantize on your model with a chosen effective-bits target and formats (INT4/FP8/NVFP4/etc.).
- Run an external Hessian-aware tool (e.g. a HAWQ-based library) on the same model and dataset.
- Compare accuracy/throughput/latency under matched bit budgets.

### 8.1 How “Hessian-like” is AutoQuantize in practice?

The gradient-based AutoQuantize implementation does not explicitly build or invert a full Hessian matrix for each layer. Instead, it implements the standard diagonal-Fisher approximation to the Hessian and combines it with explicit forward “what-if” evaluations of each quantization recipe.

The relevant pieces in `modelopt/torch/quantization/algorithms.py` are:

- The per-recipe score is defined as:

> def _get_auto_quantize_score(grad_output, output_diff):  
>     return ((grad_output.float() ** 2) * (output_diff.float() ** 2)).sum()

- During score estimation, for each quantizable module:
  - The forward hook stores `output` with quantization disabled.
  - For each candidate quantization recipe, it runs the module again, computes `output_diff = Y_q − Y`, and caches it in `module.output_diff_dict`.
  - A backward hook then accumulates, for each recipe:

> hparam._importance_dict[recipe][module] += _get_auto_quantize_score(grad_output[0], output_diff)

- `estimate_sensitivity_scores` is documented as:

> """Estimate sensitivity scores using hessian approximation."""

and simply wraps `_estimate_auto_quantize_scores` with optional gradient-checkpointing and parameter-gradient selection.

Mathematically, this is the usual second-order approximation:
- Use a Taylor expansion of the loss w.r.t. the module output.  
- Approximate the Hessian by the Fisher information (expected outer product of gradients), so that the scalar loss change for a perturbation Δy is roughly proportional to `(∂L/∂y)^2 * (Δy)^2` summed over elements.  
- AutoQuantize’s `grad_output^2 * output_diff^2` is exactly this diagonal-Fisher proxy, accumulated over calibration data.

So for quantization:
- AutoQuantize **does use Hessian-style information**, but in the efficient “diagonal Fisher + forward perturbation” form.  
- It never constructs a full dense Hessian for each layer, nor does it compute eigenvalues like HAWQ; the only matrix solve is in the LP that chooses formats given the per-layer scores and costs.

By contrast, a true Hessian is used in ModelOpt’s **SparseGPT pruning** path (`modelopt/torch/sparsity/weight_sparsity/sparsegpt.py`), where they:
- Collect a Hessian-like matrix `H ≈ X X^T` per block via forward hooks.  
- Regularize and invert `H` using Cholesky (`invert(hessian)` / `prepare(...)`).  
- Use that inverse Hessian to compute pruning masks that minimize weight-removal error.

That “full Hessian” machinery is specific to pruning and is not invoked by AutoQuantize; mixed-precision quantization relies on the lighter Fisher-based approximation described above.
