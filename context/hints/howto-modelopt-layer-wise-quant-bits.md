# How to do layer-wise quantization bit selection with NVIDIA ModelOpt (PyTorch)

This note explains how to use NVIDIA ModelOpt’s PyTorch APIs to control **quantization bits per layer**, with a focus on:

- Pattern-based quantization configs (`QuantizeConfig`, `QuantizerAttributeConfig`).
- The `auto_quantize` search API and its per-layer recipe selection.
- How to turn AutoQuant’s search results into an **exact, layer-wise bit map** you can apply when exporting or evaluating models.

The examples here are based on the ModelOpt sources vendored in this repo under `extern/TensorRT-Model-Optimizer/modelopt/torch/quantization/` and the documented `auto_quantize` interface in `model_quant.py`.

---

## 1. Key ModelOpt concepts

ModelOpt’s PyTorch quantization API revolves around:

- **Quantization formats / configs** (`QuantizeConfig` in `config.py`):
  - A dict like `FP8_DEFAULT_CFG`, `INT8_SMOOTHQUANT_CFG`, `W4A8_AWQ_BETA_CFG`, etc.
  - Each has:
    - `"quant_cfg"`: mapping from wildcard patterns (e.g. `"*weight_quantizer"`, `"*input_quantizer"`, `"*lm_head*"`) or module class names to `QuantizerAttributeConfig` objects.
    - `"algorithm"`: the calibration algorithm to use (`"max"`, `"smoothquant"`, AWQ variants, etc.).
  - Example (from `config.py`):

    ```python
    FP8_DEFAULT_CFG = {
        "quant_cfg": {
            "*weight_quantizer": {"num_bits": (4, 3), "axis": None},
            "*input_quantizer": {"num_bits": (4, 3), "axis": None},
            # plus a bunch of disable patterns under _default_disabled_quantizer_cfg
        },
        "algorithm": "max",
    }
    ```

- **Quantizer attributes** (`QuantizerAttributeConfig`):
  - Fields include `num_bits`, `axis`, `enable`, `block_sizes`, `type`, etc.
  - Pattern-based matching (wildcards or filter functions) controls which quantizers get which attributes.
  - You can set per-pattern bits, e.g. `{"num_bits": 4}` on `"*weight_quantizer"` for a subset of modules.

- **Quant recipes and AutoQuant** (`QuantRecipe`, `QuantRecipeHparam` in `algorithms.py`):
  - A `QuantRecipe` wraps a `QuantizeConfig` (e.g. FP8, INT4, custom) and normalizes it for AutoQuant.
  - `QuantRecipeHparam` manages a **choice set** of recipes per logical layer group:
    - For each recipe choice, it precomputes the corresponding `TensorQuantizer` objects for the quantized modules.
    - During search, ModelOpt switches the active recipe per layer by swapping in those quantizers (`input_quantizer`, `weight_quantizer`, `output_quantizer`) for that group.

- **AutoQuant search** (`auto_quantize` in `model_quant.py`):
  - Entry point:

    ```python
    from modelopt.torch.quantization import auto_quantize

    model, state_dict = auto_quantize(
        model,
        constraints={"effective_bits": 4.8},
        quantization_formats=[mtq.FP8_DEFAULT_CFG, mtq.W4A8_AWQ_BETA_CFG],
        data_loader=...,
        forward_step=...,
        loss_func=...,   # or forward_backward_step
        num_calib_steps=512,
        num_score_steps=128,
        method="gradient",  # or "kl_div"
    )
    ```

  - It:
    - Builds `QuantRecipeHparam` objects per “quantization group” (e.g. grouped Q/K/V projections).
    - Scores candidate recipes per group using calibration data.
    - Solves a constrained optimization problem over `effective_bits` to pick **one recipe per group**.
    - Applies the chosen recipe per layer, producing a quantized model and a `state_dict` with detailed per-layer stats.

This is the mechanism you use to let ModelOpt pick **different quantization bits or formats per layer**, under a global budget.

---

## 2. Manual per-layer bits with pattern-based configs

Before using AutoQuant, it’s useful to understand the lower-level mechanism: **pattern-based quantization configs**.

- You can define a custom config by editing the `quant_cfg` dict:

  ```python
  import copy
  import modelopt.torch.quantization as mtq

  # Start from a default INT8 config
  CUSTOM_INT8_CFG = copy.deepcopy(mtq.INT8_DEFAULT_CFG)

  # Make lm_head and some MLP layers higher precision or disable their quantization
  CUSTOM_INT8_CFG["quant_cfg"]["*lm_head*"] = {"enable": False}  # leave lm_head in FP16/BF16

  # Quantize everything else to INT8 as usual
  model = mtq.quantize(model, CUSTOM_INT8_CFG, forward_loop)
  ```

- For **exact per-layer bits**, you can rely on module name patterns:

  ```python
  CUSTOM_MIXED_CFG = {
      "quant_cfg": {
          # Default: FP8 weights+activations
          "*weight_quantizer": {"num_bits": (4, 3), "axis": None},
          "*input_quantizer": {"num_bits": (4, 3), "axis": None},

          # Example: keep some layers at INT8
          "*language_model.layers.0.mlp.*weight_quantizer": {"num_bits": 8, "axis": 0},
          "*language_model.layers.0.mlp.*input_quantizer": {"num_bits": 8, "axis": None},

          # Example: disable quantization entirely for a block
          "*language_model.layers.1.mlp*": {"enable": False},

          # Usual disabled patterns (lm_head, BatchNorm, etc.) can be layered in as needed.
      },
      "algorithm": "max",
  }
  ```

This approach gives you *pattern-level* control; for LLMs, patterns like `"*layers.5.mlp*"` or `"*layers.10.self_attn*"` effectively correspond to “layer-wise” knobs.

However, manual configs can be hard to tune. AutoQuant builds on this by searching over **recipes** per layer.

---

## 3. Using `auto_quantize` for per-layer bit selection

With `auto_quantize`, you define **multiple quantization formats** (each with its own bitwidth and behavior) and let ModelOpt decide, per layer, which one to apply under an `effective_bits` constraint.

### 3.1 Define quantization formats (“recipes”)

You can mix built-in configs and custom ones:

```python
import copy
import modelopt.torch.quantization as mtq

# Built-in configs (from config.py)
fp8_cfg = mtq.FP8_DEFAULT_CFG           # FP8 weights + activations
w4a8_cfg = mtq.W4A8_AWQ_BETA_CFG        # W4 (weights) + A8 (activations)

# Custom INT8 config
INT8_CUSTOM_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {"num_bits": 8, "axis": 0},
        "*input_quantizer": {"num_bits": 8, "axis": None},
    },
    "algorithm": "smoothquant",
}

quant_formats = [fp8_cfg, w4a8_cfg, INT8_CUSTOM_CFG]
```

When you pass this `quantization_formats` list into `auto_quantize`, ModelOpt internally constructs one `QuantRecipe` per format plus an implicit “no quantization” recipe. Each recipe corresponds to a particular `QuantizeConfig` (and thus a particular `num_bits` pattern).

### 3.2 Run AutoQuant with per-layer search

You then call `auto_quantize` with:

```python
from modelopt.torch.quantization import auto_quantize

constraints = {"effective_bits": 4.8}  # global budget for average effective bits

model_q, state_dict = auto_quantize(
    model,
    constraints=constraints,
    quantization_formats=quant_formats,
    data_loader=calib_loader,
    forward_step=forward_step,   # or forward_backward_step + loss_func for "gradient"
    loss_func=loss_func,
    num_calib_steps=512,
    num_score_steps=128,
    method="gradient",           # or "kl_div"
    verbose=True,
)
```

Internally this will:

- Calibrate each candidate recipe on `num_calib_steps` batches.
- Estimate per-layer sensitivity scores for each recipe.
- Solve a small optimization problem to pick the **best recipe per group** while satisfying the `effective_bits` constraint.
- Mutate `model` in-place to use the chosen recipes.

In other words, after `auto_quantize`, each layer (or group) has a specific quantization format (bitwidth + algorithm), potentially different from other layers.

---

## 4. Extracting the exact per-layer bit map from AutoQuant

The second return value from `auto_quantize` is a `state_dict` (a `SearchStateDict`), which contains per-layer stats and choices. The `algorithms.py` module wraps this into a more convenient manifest via `QuantRecipeHparam` and related helpers.

In many examples (including this repo’s `qwen2_5_vl_3b_autoquant_fp8_schemes.py`), a helper turns `state_dict` into a manifest like:

```python
manifest = {
    "layer_sensitivity": {
        "language_model.layers.0.mlp.gate_proj.quant_recipe": {
            "formats": [...],  # stringified QuantRecipe configs
            "scores":  [...],  # sensitivity scores per recipe
            "costs":   [...],  # approximate cost / size
        },
        # ...
    },
    "sensitivity_ranking": [
        {"name": "language_model.layers.0.mlp.gate_proj.quant_recipe", "score": ...},
        # sorted list of layers
    ],
    "autoquant_state": {
        "constraints": {"effective_bits": ...},
        "score": ...,
        "is_satisfied": True,
        # plus internal search metadata
    },
}
```

To get the **exact format / bits per layer**:

1. Look at the AutoQuant searcher’s final choices:

   - For each `QuantRecipeHparam`, there is a selected recipe representing the chosen quantization format for that group.
   - In higher-level helpers (like this repo’s `build_quant_manifest`), the chosen recipe is reflected in the manifest’s `layer_sensitivity` and `sensitivity_ranking` fields.

2. Map recipes to bits:

   - Each string in `formats` corresponds to a `QuantRecipe` and can include an `effective-bits: X.Y` annotation.
   - You can parse this annotation (as `build_quant_manifest` does) to get a per-layer `effective_bits` estimate.

3. For a more concrete mapping:

   - You can reconstruct the per-layer `QuantizeConfig` by taking the chosen recipe’s `config.quant_cfg` and applying it via `set_quantizer_by_cfg` (used inside `QuantRecipeHparam`) to the modules you care about.

This gives you a **per-layer map of which quantization format (and thus which bits) AutoQuant picked**.

---

## 5. Applying an explicit layer-wise bit map

Once you’ve decided “layer X should be FP8, layer Y should be INT8, layer Z should be left in BF16”, you have two main options to apply this:

### 5.1 Build a custom `QuantizeConfig` and call `quantize`

If you’re not relying on AutoQuant’s scoring anymore, you can directly construct a combined `QuantizeConfig` that encodes your exact per-layer decisions:

```python
from modelopt.torch.quantization import quantize

CUSTOM_CFG = {
    "quant_cfg": {},
    "algorithm": "max",
}

for layer_name, decision in my_layer_decisions.items():
    pattern = f"*{layer_name}*weight_quantizer"
    if decision == "FP8":
        CUSTOM_CFG["quant_cfg"][pattern] = {"num_bits": (4, 3), "axis": None, "enable": True}
    elif decision == "INT8":
        CUSTOM_CFG["quant_cfg"][pattern] = {"num_bits": 8, "axis": 0, "enable": True}
    elif decision == "NONE":
        CUSTOM_CFG["quant_cfg"][pattern] = {"enable": False}

model_q = quantize(model, CUSTOM_CFG, forward_loop)
```

This approach is straightforward when your layer naming is stable and you can translate your desired bit map into patterns.

### 5.2 Reuse AutoQuant’s quantized model and optionally compress

If you’ve already run `auto_quantize` and are happy with its per-layer decisions:

- The returned `model_q` already has the chosen recipes applied per layer.
- You can:
  - Export it directly via `export_hf_checkpoint` (as this repo does in `qwen2_5_vl_3b_autoquant_fp8_schemes.py`), or
  - Further **compress** it into a “real quantized” model using `mtq.compress` (see `compress.py`):

    ```python
    from modelopt.torch.quantization import compress

    # Compress all supported quantized weights in-place
    model_q, metadata = compress_convert(model_q, CompressConfig())
    ```

In this flow, AutoQuant has already handled the per-layer bit decisions; you just save or deploy the resulting model.

---

## 6. Practical tips and gotchas

- **`quantization_formats` is where per-layer bit choices come from**:
  - To let ModelOpt pick different bits per layer, you must include **multiple** formats (e.g. FP8, INT8, W4A8, and an implicit NONE) in the `quantization_formats` list.
  - If you only pass a single FP8 config (like this repo’s `FP8_ALL_LAYERS_CFG`), AutoQuant’s “choice” per layer is effectively just “FP8 vs no quantization”, not multiple bitwidths.

- **Global `quant_cfg` patterns are ignored for AutoQuant layer disabling** (see `auto_quantize` docstring):
  - Patterns like `"*lm_head*": {"enable": False}` in `FP8_DEFAULT_CFG` are not honored as “don’t search this layer”; AutoQuant treats them as per-layer candidates unless you also pass `disabled_layers` to `auto_quantize`.

- **Grouping rules matter**:
  - AutoQuant groups some layers (e.g. Q, K, V projections in the same transformer block) and assigns them the same recipe for TensorRT-LLM compatibility (`quant_grouping_rules` in `algorithms.AutoQuantizeSearcher`).
  - If you truly need independent bits for individual projections, you’d need to relax those grouping rules (advanced use).

- **Inspect `state_dict` and manifests rather than guessing**:
  - For non-trivial setups, always inspect the AutoQuant `state_dict` or the derived manifest (`layer_sensitivity`, `sensitivity_ranking`) to see which recipe was chosen where.
  - This repo’s helpers (e.g. `build_quant_manifest`) are good patterns to follow when turning search results into human-readable per-layer bit summaries.

By combining `quantization_formats`, `auto_quantize`, and the manifest introspection pattern above, you can implement precise, layer-wise quantization bit selection for your models using ModelOpt’s PyTorch interface.*** End Patch ***!
