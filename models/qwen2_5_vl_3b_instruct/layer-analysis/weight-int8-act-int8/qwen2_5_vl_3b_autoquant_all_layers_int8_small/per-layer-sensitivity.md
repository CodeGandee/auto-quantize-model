
AutoQuant Layer Sensitivity (int8_autoquant_all_layers_top10_coco2017)
======================================================================


**Scheme:** `int8_autoquant_all_layers_top10_coco2017`

**Model:** `/workspace/code/auto-quantize-model/models/qwen2_5_vl_3b_instruct/checkpoints/Qwen2.5-VL-3B-Instruct`

**Effective bits (from search):** `15.0866`

**Total AutoQuant score:** `4.757315e-02`

**Constraint satisfied:** `False`
## Per-layer sensitivity table


- **Layer**: Name of the quant_recipe handle for a group of quantizable modules (e.g., attention or MLP projections).
- **Num Bits**: Effective number of bits allocated for the quantized recipe(s) considered at this layer.
- **Sensitivity**: AutoQuant sensitivity score for the quantized recipe(s). Higher values indicate that quantizing this layer is more harmful to model quality.
- **Size Cost**: Approximate compressed weight size contribution of the layer under the corresponding recipe(s). Higher values indicate more memory usage.

Note: In the JSON manifest, layer keys end with `.quant_recipe` (e.g., `language_model.layers.0.mlp.gate_proj.quant_recipe`). This suffix is added by ModelOpt to represent the AutoQuant hyperparameter attached to that module. In this table we strip the `.quant_recipe` suffix for readability; the underlying module path is the part before that suffix.

|Layer|Num Bits|Sensitivity|Size Cost|
| :--- | :--- | :--- | :--- |
|model.language_model.layers.5.self_attn.o_proj|8.0|5.333e-03|2.097e+06|
|model.language_model.layers.3.self_attn.o_proj|8.0|3.618e-03|2.097e+06|
|lm_head|8.0|3.565e-03|1.556e+08|
|model.language_model.layers.4.self_attn.o_proj|8.0|3.331e-03|2.097e+06|
|model.language_model.layers.9.self_attn.o_proj|8.0|3.082e-03|2.097e+06|
|model.language_model.layers.14.self_attn.o_proj|8.0|2.765e-03|2.097e+06|
|model.language_model.layers.7.self_attn.o_proj|8.0|2.579e-03|2.097e+06|
|model.language_model.layers.6.self_attn.o_proj|8.0|2.537e-03|2.097e+06|
|model.language_model.layers.15.self_attn.o_proj|8.0|2.286e-03|2.097e+06|
|model.language_model.layers.8.self_attn.o_proj|8.0|2.238e-03|2.097e+06|
|model.language_model.layers.20.self_attn.o_proj|8.0|2.132e-03|2.097e+06|
|model.language_model.layers.13.self_attn.o_proj|8.0|1.982e-03|2.097e+06|
|model.language_model.layers.16.self_attn.o_proj|8.0|1.778e-03|2.097e+06|
|model.language_model.layers.23.self_attn.o_proj|8.0|1.741e-03|2.097e+06|
|model.language_model.layers.11.self_attn.o_proj|8.0|1.737e-03|2.097e+06|
|model.language_model.layers.10.self_attn.o_proj|8.0|1.733e-03|2.097e+06|
|model.language_model.layers.12.self_attn.o_proj|8.0|1.498e-03|2.097e+06|
|model.language_model.layers.22.self_attn.o_proj|8.0|1.177e-03|2.097e+06|
|model.language_model.layers.2.self_attn.o_proj|8.0|7.358e-04|2.097e+06|
|model.language_model.layers.24.self_attn.o_proj|8.0|4.729e-04|2.097e+06|
|model.language_model.layers.25.self_attn.o_proj|8.0|3.383e-04|2.097e+06|
|model.language_model.layers.26.self_attn.o_proj|8.0|2.471e-04|2.097e+06|
|model.language_model.layers.28.self_attn.o_proj|8.0|2.331e-04|2.097e+06|
|model.language_model.layers.30.self_attn.o_proj|8.0|1.765e-04|2.097e+06|
|model.language_model.layers.33.self_attn.o_proj|8.0|9.026e-05|2.097e+06|
|model.language_model.layers.32.self_attn.o_proj|8.0|8.730e-05|2.097e+06|
|model.language_model.layers.29.self_attn.o_proj|8.0|6.968e-05|2.097e+06|
|model.language_model.layers.34.self_attn.o_proj|8.0|6.321e-06|2.097e+06|
|model.language_model.layers.35.self_attn.o_proj|8.0|4.385e-06|2.097e+06|
