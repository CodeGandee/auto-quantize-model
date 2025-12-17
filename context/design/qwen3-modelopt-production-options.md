# Qwen3-VL: ModelOpt options to match the llm-compressor GPTQ recipe

## HEADER
- **Purpose**: Map the Qwen3-VL llm-compressor quantization recipe (GPTQModifier) onto NVIDIA ModelOpt options we can use for (a) sensitivity analysis and (b) a production-leaning PTQ configuration.
- **Status**: Draft
- **Date**: 2025-12-17
- **Source inputs**:
  - `context/design/qwen3-target-recipe.md` (llm-compressor target)
  - ModelOpt source (vendored): `extern/TensorRT-Model-Optimizer/modelopt/torch/quantization/`

## 1) What we’re trying to match (llm-compressor)

From `context/design/qwen3-target-recipe.md`, the intent is:

- **Weights**: `int4`, `strategy: channel`, `dynamic: false`, `symmetric: true`
- **Input activations**: `int8`, `strategy: token`, `dynamic: true`, `symmetric: true`
- **Targets**: attention `q_proj/k_proj/v_proj/o_proj` and MLP `gate_proj/up_proj/down_proj`
- **Ignore**: `lm_head`, `vpm`, `resampler.*`

Important: llm-compressor’s `GPTQModifier` is a specific quantization method (GPTQ). ModelOpt’s `auto_quantize` is not GPTQ; it is a sensitivity-scoring + constrained search over candidate quantization “recipes”. The mapping below is therefore a “format/strategy match” (granularity + dynamic/static behavior), not a method match.

## 2) ModelOpt knobs that correspond to the recipe

### 2.1 Quantization granularity (channel vs token)

ModelOpt expresses these choices at the quantizer attribute level:

- **Per-channel (weights)**: `axis=0` for `nn.Linear.weight` with shape `(out_features, in_features)` (per output channel).
- **Per-token (activations)**: ModelOpt uses `type="dynamic"` and a `block_sizes` pattern with `None` values that gets converted to an `axis` at runtime (see `_block_sizes_to_axis` in `TensorQuantizer`).

If you want the exact details of where “per-channel/per-block/per-layer” is configured in ModelOpt, see `context/hints/about-modelopt-quantization-granularity.md`.

### 2.2 “Symmetric” in ModelOpt (for int quant)

ModelOpt integer fake-quant uses a signed symmetric zero-point (0) by default; to force the “equal magnitude” narrow range (e.g., `[-7, 7]` for int4), set `narrow_range=True` in the quantizer attributes.

### 2.3 Layer selection (targets/ignore)

ModelOpt `auto_quantize(...)` does not apply model-level name-pattern disables inside `quantization_formats` (it only applies per-layer quantizer patterns like `*weight_quantizer` and `*input_quantizer`). To match llm-compressor’s ignore list during `auto_quantize`, use the `disabled_layers` argument (wildcards matched against `model.named_modules()` names).

Practical mapping for Qwen3-VL:

- **Ignore list** (disable quantization): `["*lm_head*", "*vpm*", "*resampler*"]`
- **LM-only sensitivity runs**: also disable vision tower: `["model.visual*"]` (this repo’s default `coverage_mode=lm_only` behavior)

ModelOpt does not have an “enabled_layers/targets-only” selector for `auto_quantize`; it quantizes all quantizable modules except those matching `disabled_layers`. In practice, Qwen-style blocks are dominated by the same projections listed in the recipe, so disabling the ignore list usually gets you close enough.

## 3) Recommended ModelOpt configuration for recipe matching

### 3.1 Sensitivity analysis (closest match to the recipe)

Use ModelOpt AutoQuant scoring with a single candidate recipe that matches the format/strategy, relying on the implicit “NONE” option (unquantized) for comparison:

- `quantization_formats=[CUSTOM_INT4_CHANNEL_W_INT8_TOKEN_A_CFG]` (custom dict; example below)
- `method="gradient"` (default) if you can supply a loss; use `method="kl_div"` if you want label-free logits-only scoring
- `disabled_layers=["*lm_head*", "*vpm*", "*resampler*"]` (plus `"model.visual*"` if LM-only)

Example custom quantization format dict (format/strategy match for the llm-compressor recipe):

```python
CUSTOM_INT4_CHANNEL_W_INT8_TOKEN_A_CFG = {
    "quant_cfg": {
        # W: int4, per-channel (Linear weight is (out, in) so axis=0 is per output channel), static.
        "*weight_quantizer": {"num_bits": 4, "axis": 0, "narrow_range": True, "enable": True},
        # A: int8, dynamic per-token; expressed via `type="dynamic"` and `block_sizes` with `None` values
        # that are converted to an `axis` at runtime by TensorQuantizer._block_sizes_to_axis.
        "*input_quantizer": {
            "num_bits": 8,
            "type": "dynamic",
            "block_sizes": {-1: None},
            "narrow_range": True,
            "enable": True,
        },
    },
    "algorithm": "max",
}
```

### 3.2 Production-leaning PTQ (format match, not GPTQ)

If you want a fixed “apply this recipe everywhere” PTQ config (no per-layer search), use `mtq.quantize(...)` with the same quantizer attributes and explicit model-level disables for `lm_head/vpm/resampler`. This is a practical approximation for “W4A8-int with per-channel weights and dynamic per-token activations”, but it is still not GPTQ.

## 4) GPU/format caveats (RTX 5090 environment)

- **NVFP4** analysis/simulation works on RTX 5090 in the `rtx5090-vllm` Pixi env.
- **MXFP4** analysis currently fails in this stack because ModelOpt’s MX CUDA extension build fails (toolchain + binding errors). Treat MXFP4 as “not available” for automated sensitivity runs on this setup until ModelOpt fixes the extension build for this environment.
