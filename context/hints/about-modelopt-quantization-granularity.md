About: ModelOpt quantization granularity (per-tensor / per-channel / per-block / per-layer)

This hint clarifies where “granularity” is configured in NVIDIA ModelOpt (TensorRT-Model-Optimizer), because “per-layer” and “per-channel/per-block” refer to different layers of the system:

- Quantizer granularity: how a single tensor’s scale(s) are shared (per-tensor, per-channel/per-axis, per-block/group).
- Search granularity (AutoQuant): how ModelOpt chooses different quantization “recipes” across modules (per-module or per-group of modules).

In this repo, quant pairs like `wfp4_afp8` select a built-in ModelOpt preset (e.g. `W4A8_NVFP4_FP8_CFG`) via `conf/quant_pair/*.yaml`, and the preset’s `quant_cfg` controls per-block/per-channel behavior.

## 1) Where granularity is set: `quant_cfg` → `QuantizerAttributeConfig`

ModelOpt quantization presets are dicts like:

```python
CFG = {
  "quant_cfg": {
    "*weight_quantizer": {...},
    "*input_quantizer": {...},
    "default": {...},
  },
  "algorithm": "max" | "smoothquant" | ...
}
```

The inner dict values (the `{...}`) are “quantizer attributes” described by `QuantizerAttributeConfig` in ModelOpt source. The key fields that control granularity are:

- `axis`: static per-channel/per-axis quantization (per-column/per-row/per-channel depending on tensor shape).
- `block_sizes`: block/group quantization (multiple elements share a scale per block); supports static or dynamic blockwise quantization; cannot coexist with `axis`.

Source (ModelOpt): `modelopt/torch/quantization/config.py` (`QuantizerAttributeConfig.axis`, `QuantizerAttributeConfig.block_sizes`) at https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/5a4242faf4147fb0688bb73e10ca30b8ad3aabb3/modelopt/torch/quantization/config.py

## 2) Per-tensor vs per-channel/per-column/per-row: `axis`

`axis` defines the shape of the quantization scale factor(s) for static quantization:

- `axis: None` → per-tensor quantization (single scalar scale).
- `axis: 0` → per-channel along dimension 0 (vector of length `dim0`).
- `axis: (a, b, ...)` → per-axis with a scale tensor shaped like those dimensions.

Important: ModelOpt documents `axis` as “static per-channel quantization” and says it cannot coexist with `block_sizes` (you pick one approach per quantizer).

What “per-column” means depends on tensor layout:

- `nn.Linear.weight` has shape `(out_features, in_features)`; `axis=0` is per-output-channel (rows) and `axis=1` is per-input-channel (columns).
- `nn.Conv2d.weight` has shape `(out_channels, in_channels, kH, kW)`; `axis=0` is per-output-channel.

Source (ModelOpt): `QuantizerAttributeConfig.axis` docstring in https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/5a4242faf4147fb0688bb73e10ca30b8ad3aabb3/modelopt/torch/quantization/config.py

## 3) Per-block / group-wise quantization: `block_sizes`

`block_sizes` is used in two distinct ways in ModelOpt:

- As true “block/group quantization”: integer block sizes (e.g. `{-1: 128}`) mean a fixed number of elements share each scale factor (group quantization).
- As a per-axis/per-token shorthand: `None` values (e.g. `{-1: None}`) are treated as a special representation and are converted into an `axis` at runtime by `TensorQuantizer._block_sizes_to_axis` (this is how some “per-token” configs are expressed).

ModelOpt describes the true block/group quantization version as:

- Keys: axes to block over (e.g. `-1` for last dimension).
- Values: block size along that axis (integer) or `None` for “max possible” (used for some per-token patterns).
- Special keys: `"type"` (`"static"` or `"dynamic"`), `"scale_bits"` (double-quantize the scale), `"scale_block_sizes"` (block sizes for scale quantization).

Examples from the ModelOpt docs in source:

- `{"block_sizes": {-1: 32}}` → static calibrated block quant over the last axis in blocks of 32.
- `{"block_sizes": {-1: 32, "type": "dynamic"}}` → dynamic block quant over the last axis (no calibration stats).

Per-token dynamic quantization in ModelOpt is typically expressed by setting the quantizer attribute `type="dynamic"` and using `block_sizes` with `None` values that get converted to an `axis` during forward, for example:

```python
# Common “per-token” pattern used by ModelOpt presets (shape-dependent, but widely used for Linear inputs).
{"type": "dynamic", "block_sizes": {-1: None}}
```

Source (ModelOpt): `QuantizerAttributeConfig.block_sizes` docstring in https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/5a4242faf4147fb0688bb73e10ca30b8ad3aabb3/modelopt/torch/quantization/config.py

## 4) “Per-layer” in AutoQuant: recipe choice granularity (not `axis`)

If you run `mtq.auto_quantize(...)`, ModelOpt selects a quantization “recipe” per quantized module or per group of modules. This is a different granularity than `axis`/`block_sizes`:

- `axis`/`block_sizes` decide how each quantizer computes scales inside a tensor.
- AutoQuant decides which recipe/config (e.g. NVFP4 vs FP8 vs NONE) each module/group uses.

The “per-layer vs per-group” behavior is controlled by grouping rules in the AutoQuant searcher, not by `axis`:

- `_AutoQuantizeBaseSearcher.quant_grouping_rules` determines which modules share the same recipe (e.g., group `q_proj/k_proj/v_proj`).
- `AutoQuantizeGradientSearcher.score_module_rules` can change where sensitivity is measured (the “score module”), which affects scoring efficiency but not the `axis`/`block_sizes` quantizer granularity.

Source (ModelOpt): `modelopt/torch/quantization/algorithms.py` (`_AutoQuantizeBaseSearcher.quant_grouping_rules`, `AutoQuantizeGradientSearcher.score_module_rules`) at https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/5a4242faf4147fb0688bb73e10ca30b8ad3aabb3/modelopt/torch/quantization/algorithms.py

## 5) Quick inspection: what `wfp4_afp8` implies in this repo

Our `wfp4_afp8` quant pair points at `W4A8_NVFP4_FP8_CFG` (`conf/quant_pair/wfp4_afp8.yaml`). You can inspect the preset to see granularity directly:

```python
import modelopt.torch.quantization as mtq

cfg = mtq.W4A8_NVFP4_FP8_CFG["quant_cfg"]
print(cfg["*weight_quantizer"])  # look for axis vs block_sizes
print(cfg["*input_quantizer"])
```

If `*weight_quantizer` uses `block_sizes` (and `axis: None`), then it is block/group quantization, not per-column scaling.

## 6) How to override granularity (examples)

Make a copy of a preset and edit `axis` / `block_sizes` (never set both):

```python
import copy
import modelopt.torch.quantization as mtq

cfg = copy.deepcopy(mtq.W4A8_NVFP4_FP8_CFG)

# Example: switch weights to static per-output-channel scales for Linear weights
cfg["quant_cfg"]["*weight_quantizer"].pop("block_sizes", None)
cfg["quant_cfg"]["*weight_quantizer"]["axis"] = 0
cfg["quant_cfg"]["*weight_quantizer"]["type"] = "static"

# Example: adjust block size (keep block quantization)
cfg["quant_cfg"]["*weight_quantizer"].pop("axis", None)
cfg["quant_cfg"]["*weight_quantizer"]["block_sizes"] = {-1: 64, "type": "dynamic", "scale_bits": (4, 3)}
```

Then pass `cfg` to `mtq.quantize(...)` or include it in `quantization_formats=[...]` for `mtq.auto_quantize(...)`.

## References

- ModelOpt quantization config overview (source docstring in `config.py`): https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/5a4242faf4147fb0688bb73e10ca30b8ad3aabb3/modelopt/torch/quantization/config.py
- ModelOpt AutoQuantize grouping/scoring rules (source): https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/5a4242faf4147fb0688bb73e10ca30b8ad3aabb3/modelopt/torch/quantization/algorithms.py
- ModelOpt API docs (auto_quantize): https://nvidia.github.io/TensorRT-Model-Optimizer/reference/generated/modelopt.torch.quantization.model_quant.html
