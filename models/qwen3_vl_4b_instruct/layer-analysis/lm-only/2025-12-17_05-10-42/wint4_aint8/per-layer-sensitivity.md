
AutoQuant Layer Sensitivity (wint4_aint8_autoquant_lm)
======================================================


**Scheme:** `wint4_aint8_autoquant_lm`

**Model:** `/workspace/code/auto-quantize-model/models/qwen3_vl_4b_instruct/checkpoints/Qwen3-VL-4B-Instruct`
## Dataset


- **Name:** `vlm_coco2017_captions`
- **Size:** `medium`
- **Captions path:** `/workspace/code/auto-quantize-model/datasets/vlm-quantize-calib/coco2017_captions_medium.txt`
- **Root:** `/workspace/code/auto-quantize-model/datasets/vlm-quantize-calib`
- **Calibration samples:** `128` / `128` (used / max)
- **Calibration seq len:** `512`
- **Batch size:** `8`
- **Calibration batches:** `16`

**Effective bits (from search):** `8.7304`

**Total AutoQuant score:** `6.702245e+00`

**Constraint satisfied:** `True`
## Per-layer sensitivity table


- **Layer**: Name of the quant_recipe handle for a group of quantizable modules (e.g., attention or MLP projections).
- **Num Bits**: Effective number of bits allocated for the quantized recipe(s) considered at this layer.
- **Sensitivity**: AutoQuant sensitivity score for the quantized recipe(s). Higher values indicate that quantizing this layer is more harmful to model quality.
- **Size Cost**: Approximate compressed weight size contribution of the layer under the corresponding recipe(s). Higher values indicate more memory usage.

Note: In the JSON manifest, layer keys may end with `.quant_recipe` (e.g., `language_model.layers.0.mlp.gate_proj.quant_recipe`). This suffix is added by ModelOpt to represent the AutoQuant hyperparameter attached to that module. In this table we strip the `.quant_recipe` suffix for readability; the underlying module path is the part before that suffix.

|Layer|Num Bits|Sensitivity|Size Cost|
| :--- | :--- | :--- | :--- |
|model.language_model.layers.6.mlp.gate_proj|4.0|6.096e+01|1.245e+07|
|model.language_model.layers.6.mlp.down_proj|4.0|6.073e+01|6.226e+06|
|model.language_model.layers.12.mlp.down_proj|4.0|4.748e+01|6.226e+06|
|model.language_model.layers.1.mlp.down_proj|4.0|4.170e+01|6.226e+06|
|model.language_model.layers.14.mlp.down_proj|4.0|3.644e+01|6.226e+06|
|model.language_model.layers.16.mlp.down_proj|4.0|2.355e+01|6.226e+06|
|model.language_model.layers.15.mlp.down_proj|4.0|1.558e+01|6.226e+06|
|model.language_model.layers.9.mlp.down_proj|4.0|1.552e+01|6.226e+06|
|model.language_model.layers.3.mlp.down_proj|4.0|1.528e+01|6.226e+06|
|model.language_model.layers.4.mlp.down_proj|4.0|1.421e+01|6.226e+06|
|model.language_model.layers.10.mlp.down_proj|4.0|1.403e+01|6.226e+06|
|model.language_model.layers.0.mlp.down_proj|4.0|1.381e+01|6.226e+06|
|model.language_model.layers.13.mlp.down_proj|4.0|1.161e+01|6.226e+06|
|model.language_model.layers.2.mlp.down_proj|4.0|9.445e+00|6.226e+06|
|model.language_model.layers.4.mlp.gate_proj|4.0|9.336e+00|1.245e+07|
|model.language_model.layers.18.mlp.down_proj|4.0|8.366e+00|6.226e+06|
|model.language_model.layers.11.mlp.down_proj|4.0|7.681e+00|6.226e+06|
|model.language_model.layers.3.mlp.gate_proj|4.0|7.565e+00|1.245e+07|
|model.language_model.layers.2.mlp.gate_proj|4.0|7.053e+00|1.245e+07|
|model.language_model.layers.5.mlp.down_proj|4.0|6.830e+00|6.226e+06|
|model.language_model.layers.7.mlp.down_proj|4.0|5.433e+00|6.226e+06|
|model.language_model.layers.8.mlp.down_proj|4.0|5.404e+00|6.226e+06|
|model.language_model.layers.7.mlp.gate_proj|4.0|3.989e+00|1.245e+07|
|model.language_model.layers.5.mlp.gate_proj|4.0|3.456e+00|1.245e+07|
|model.language_model.layers.1.mlp.gate_proj|4.0|2.481e+00|1.245e+07|
|model.language_model.layers.16.mlp.gate_proj|4.0|2.228e+00|1.245e+07|
|model.language_model.layers.15.mlp.gate_proj|4.0|1.982e+00|1.245e+07|
|model.language_model.layers.14.mlp.gate_proj|4.0|1.968e+00|1.245e+07|
|model.language_model.layers.13.mlp.gate_proj|4.0|1.871e+00|1.245e+07|
|model.language_model.layers.12.mlp.gate_proj|4.0|1.812e+00|1.245e+07|
|model.language_model.layers.10.mlp.gate_proj|4.0|1.692e+00|1.245e+07|
|model.language_model.layers.11.mlp.gate_proj|4.0|1.425e+00|1.245e+07|
|model.language_model.layers.8.mlp.gate_proj|4.0|1.424e+00|1.245e+07|
|model.language_model.layers.9.mlp.gate_proj|4.0|1.374e+00|1.245e+07|
|model.language_model.layers.0.mlp.gate_proj|4.0|1.168e+00|1.245e+07|
|model.language_model.layers.17.mlp.gate_proj|4.0|7.551e-01|1.245e+07|
|model.language_model.layers.17.mlp.down_proj|4.0|6.563e-01|6.226e+06|
|model.language_model.layers.24.mlp.down_proj|4.0|6.277e-01|6.226e+06|
|model.language_model.layers.35.mlp.gate_proj|4.0|5.514e-01|1.245e+07|
|model.language_model.layers.18.mlp.gate_proj|4.0|5.042e-01|1.245e+07|
|model.language_model.layers.23.mlp.down_proj|4.0|4.673e-01|6.226e+06|
|model.language_model.layers.19.mlp.down_proj|4.0|4.429e-01|6.226e+06|
|model.language_model.layers.25.mlp.down_proj|4.0|4.070e-01|6.226e+06|
|model.language_model.layers.19.mlp.gate_proj|4.0|3.416e-01|1.245e+07|
|model.language_model.layers.22.mlp.down_proj|4.0|3.217e-01|6.226e+06|
|model.language_model.layers.20.mlp.down_proj|4.0|2.317e-01|6.226e+06|
|model.language_model.layers.20.mlp.gate_proj|4.0|2.314e-01|1.245e+07|
|model.language_model.layers.21.mlp.down_proj|4.0|2.303e-01|6.226e+06|
|model.language_model.layers.26.mlp.down_proj|4.0|2.143e-01|6.226e+06|
|model.language_model.layers.21.mlp.gate_proj|4.0|1.792e-01|1.245e+07|
|model.language_model.layers.22.mlp.gate_proj|4.0|1.614e-01|1.245e+07|
|model.language_model.layers.23.mlp.gate_proj|4.0|1.405e-01|1.245e+07|
|model.language_model.layers.28.mlp.down_proj|4.0|1.279e-01|6.226e+06|
|model.language_model.layers.24.mlp.gate_proj|4.0|1.063e-01|1.245e+07|
|model.language_model.layers.27.mlp.down_proj|4.0|1.056e-01|6.226e+06|
|model.language_model.layers.25.mlp.gate_proj|4.0|9.664e-02|1.245e+07|
|model.language_model.layers.26.mlp.gate_proj|4.0|6.287e-02|1.245e+07|
|lm_head|4.0|5.412e-02|9.724e+07|
|model.language_model.layers.35.mlp.down_proj|4.0|5.307e-02|6.226e+06|
|model.language_model.layers.34.mlp.down_proj|4.0|3.986e-02|6.226e+06|
|model.language_model.layers.30.mlp.down_proj|4.0|3.796e-02|6.226e+06|
|model.language_model.layers.29.mlp.down_proj|4.0|3.636e-02|6.226e+06|
|model.language_model.layers.34.mlp.gate_proj|4.0|3.478e-02|1.245e+07|
|model.language_model.layers.27.mlp.gate_proj|4.0|3.391e-02|1.245e+07|
|model.language_model.layers.28.mlp.gate_proj|4.0|2.135e-02|1.245e+07|
|model.language_model.layers.29.mlp.gate_proj|4.0|1.515e-02|1.245e+07|
|model.language_model.layers.31.mlp.down_proj|4.0|1.257e-02|6.226e+06|
|model.language_model.layers.30.mlp.gate_proj|4.0|9.284e-03|1.245e+07|
|model.language_model.layers.32.mlp.down_proj|4.0|9.070e-03|6.226e+06|
|model.language_model.layers.31.mlp.gate_proj|4.0|7.431e-03|1.245e+07|
|model.language_model.layers.32.mlp.gate_proj|4.0|6.305e-03|1.245e+07|
|model.language_model.layers.33.mlp.gate_proj|4.0|4.543e-03|1.245e+07|
|model.language_model.layers.33.mlp.down_proj|4.0|4.314e-03|6.226e+06|
|model.language_model.layers.35.self_attn.q_proj|4.0|9.480e-04|3.932e+06|
|model.language_model.layers.6.self_attn.q_proj|4.0|8.028e-04|3.932e+06|
|model.language_model.layers.0.self_attn.q_proj|4.0|7.973e-04|3.932e+06|
|model.language_model.layers.22.self_attn.q_proj|4.0|5.710e-04|3.932e+06|
|model.language_model.layers.23.self_attn.q_proj|4.0|5.426e-04|3.932e+06|
|model.language_model.layers.7.self_attn.q_proj|4.0|5.392e-04|3.932e+06|
|model.language_model.layers.21.self_attn.q_proj|4.0|5.156e-04|3.932e+06|
|model.language_model.layers.15.self_attn.o_proj|4.0|4.594e-04|2.621e+06|
|model.language_model.layers.34.self_attn.q_proj|4.0|4.400e-04|3.932e+06|
|model.language_model.layers.24.self_attn.q_proj|4.0|4.064e-04|3.932e+06|
|model.language_model.layers.9.self_attn.q_proj|4.0|3.796e-04|3.932e+06|
|model.language_model.layers.8.self_attn.q_proj|4.0|3.712e-04|3.932e+06|
|model.language_model.layers.10.self_attn.q_proj|4.0|3.508e-04|3.932e+06|
|model.language_model.layers.28.self_attn.q_proj|4.0|3.256e-04|3.932e+06|
|model.language_model.layers.32.self_attn.q_proj|4.0|3.206e-04|3.932e+06|
|model.language_model.layers.30.self_attn.q_proj|4.0|3.164e-04|3.932e+06|
|model.language_model.layers.3.self_attn.q_proj|4.0|2.963e-04|3.932e+06|
|model.language_model.layers.5.self_attn.q_proj|4.0|2.955e-04|3.932e+06|
|model.language_model.layers.14.self_attn.q_proj|4.0|2.917e-04|3.932e+06|
|model.language_model.layers.31.self_attn.q_proj|4.0|2.701e-04|3.932e+06|
|model.language_model.layers.26.self_attn.q_proj|4.0|2.692e-04|3.932e+06|
|model.language_model.layers.16.self_attn.o_proj|4.0|2.683e-04|2.621e+06|
|model.language_model.layers.4.self_attn.q_proj|4.0|2.656e-04|3.932e+06|
|model.language_model.layers.18.self_attn.q_proj|4.0|2.615e-04|3.932e+06|
|model.language_model.layers.16.self_attn.q_proj|4.0|2.553e-04|3.932e+06|
|model.language_model.layers.15.self_attn.q_proj|4.0|2.500e-04|3.932e+06|
|model.language_model.layers.25.self_attn.q_proj|4.0|2.450e-04|3.932e+06|
|model.language_model.layers.19.self_attn.q_proj|4.0|2.408e-04|3.932e+06|
|model.language_model.layers.17.self_attn.q_proj|4.0|2.392e-04|3.932e+06|
|model.language_model.layers.27.self_attn.q_proj|4.0|2.356e-04|3.932e+06|
|model.language_model.layers.0.self_attn.o_proj|4.0|2.251e-04|2.621e+06|
|model.language_model.layers.20.self_attn.q_proj|4.0|2.235e-04|3.932e+06|
|model.language_model.layers.14.self_attn.o_proj|4.0|2.102e-04|2.621e+06|
|model.language_model.layers.11.self_attn.q_proj|4.0|1.998e-04|3.932e+06|
|model.language_model.layers.12.self_attn.q_proj|4.0|1.978e-04|3.932e+06|
|model.language_model.layers.29.self_attn.q_proj|4.0|1.939e-04|3.932e+06|
|model.language_model.layers.13.self_attn.o_proj|4.0|1.937e-04|2.621e+06|
|model.language_model.layers.33.self_attn.q_proj|4.0|1.921e-04|3.932e+06|
|model.language_model.layers.6.self_attn.o_proj|4.0|1.768e-04|2.621e+06|
|model.language_model.layers.13.self_attn.q_proj|4.0|1.620e-04|3.932e+06|
|model.language_model.layers.22.self_attn.o_proj|4.0|1.597e-04|2.621e+06|
|model.language_model.layers.2.self_attn.q_proj|4.0|1.535e-04|3.932e+06|
|model.language_model.layers.34.self_attn.o_proj|4.0|1.475e-04|2.621e+06|
|model.language_model.layers.1.self_attn.q_proj|4.0|1.239e-04|3.932e+06|
|model.language_model.layers.23.self_attn.o_proj|4.0|1.159e-04|2.621e+06|
|model.language_model.layers.8.self_attn.o_proj|4.0|1.099e-04|2.621e+06|
|model.language_model.layers.35.self_attn.o_proj|4.0|1.070e-04|2.621e+06|
|model.language_model.layers.10.self_attn.o_proj|4.0|9.487e-05|2.621e+06|
|model.language_model.layers.24.self_attn.o_proj|4.0|7.518e-05|2.621e+06|
|model.language_model.layers.12.self_attn.o_proj|4.0|7.488e-05|2.621e+06|
|model.language_model.layers.7.self_attn.o_proj|4.0|7.315e-05|2.621e+06|
|model.language_model.layers.9.self_attn.o_proj|4.0|7.078e-05|2.621e+06|
|model.language_model.layers.5.self_attn.o_proj|4.0|6.969e-05|2.621e+06|
|model.language_model.layers.1.self_attn.o_proj|4.0|6.466e-05|2.621e+06|
|model.language_model.layers.19.self_attn.o_proj|4.0|6.172e-05|2.621e+06|
|model.language_model.layers.18.self_attn.o_proj|4.0|6.070e-05|2.621e+06|
|model.language_model.layers.28.self_attn.o_proj|4.0|5.816e-05|2.621e+06|
|model.language_model.layers.11.self_attn.o_proj|4.0|5.712e-05|2.621e+06|
|model.language_model.layers.17.self_attn.o_proj|4.0|5.687e-05|2.621e+06|
|model.language_model.layers.21.self_attn.o_proj|4.0|5.646e-05|2.621e+06|
|model.language_model.layers.26.self_attn.o_proj|4.0|5.576e-05|2.621e+06|
|model.language_model.layers.4.self_attn.o_proj|4.0|5.573e-05|2.621e+06|
|model.language_model.layers.27.self_attn.o_proj|4.0|5.502e-05|2.621e+06|
|model.language_model.layers.20.self_attn.o_proj|4.0|5.339e-05|2.621e+06|
|model.language_model.layers.30.self_attn.o_proj|4.0|5.199e-05|2.621e+06|
|model.language_model.layers.33.self_attn.o_proj|4.0|4.895e-05|2.621e+06|
|model.language_model.layers.32.self_attn.o_proj|4.0|4.722e-05|2.621e+06|
|model.language_model.layers.29.self_attn.o_proj|4.0|4.582e-05|2.621e+06|
|model.language_model.layers.31.self_attn.o_proj|4.0|4.529e-05|2.621e+06|
|model.language_model.layers.25.self_attn.o_proj|4.0|4.528e-05|2.621e+06|
|model.language_model.layers.3.self_attn.o_proj|4.0|3.607e-05|2.621e+06|
|model.language_model.layers.2.self_attn.o_proj|4.0|2.921e-05|2.621e+06|
|model.visual.patch_embed.proj|16.0|0.000e+00|1.573e+06|
|model.visual.blocks.0.attn.qkv|16.0|0.000e+00|3.146e+06|
|model.visual.blocks.0.attn.proj|16.0|0.000e+00|1.049e+06|
|model.visual.blocks.0.mlp.linear_fc1|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.0.mlp.linear_fc2|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.1.attn.qkv|16.0|0.000e+00|3.146e+06|
|model.visual.blocks.1.attn.proj|16.0|0.000e+00|1.049e+06|
|model.visual.blocks.1.mlp.linear_fc1|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.1.mlp.linear_fc2|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.2.attn.qkv|16.0|0.000e+00|3.146e+06|
|model.visual.blocks.2.attn.proj|16.0|0.000e+00|1.049e+06|
|model.visual.blocks.2.mlp.linear_fc1|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.2.mlp.linear_fc2|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.3.attn.qkv|16.0|0.000e+00|3.146e+06|
|model.visual.blocks.3.attn.proj|16.0|0.000e+00|1.049e+06|
|model.visual.blocks.3.mlp.linear_fc1|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.3.mlp.linear_fc2|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.4.attn.qkv|16.0|0.000e+00|3.146e+06|
|model.visual.blocks.4.attn.proj|16.0|0.000e+00|1.049e+06|
|model.visual.blocks.4.mlp.linear_fc1|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.4.mlp.linear_fc2|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.5.attn.qkv|16.0|0.000e+00|3.146e+06|
|model.visual.blocks.5.attn.proj|16.0|0.000e+00|1.049e+06|
|model.visual.blocks.5.mlp.linear_fc1|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.5.mlp.linear_fc2|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.6.attn.qkv|16.0|0.000e+00|3.146e+06|
|model.visual.blocks.6.attn.proj|16.0|0.000e+00|1.049e+06|
|model.visual.blocks.6.mlp.linear_fc1|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.6.mlp.linear_fc2|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.7.attn.qkv|16.0|0.000e+00|3.146e+06|
|model.visual.blocks.7.attn.proj|16.0|0.000e+00|1.049e+06|
|model.visual.blocks.7.mlp.linear_fc1|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.7.mlp.linear_fc2|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.8.attn.qkv|16.0|0.000e+00|3.146e+06|
|model.visual.blocks.8.attn.proj|16.0|0.000e+00|1.049e+06|
|model.visual.blocks.8.mlp.linear_fc1|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.8.mlp.linear_fc2|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.9.attn.qkv|16.0|0.000e+00|3.146e+06|
|model.visual.blocks.9.attn.proj|16.0|0.000e+00|1.049e+06|
|model.visual.blocks.9.mlp.linear_fc1|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.9.mlp.linear_fc2|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.10.attn.qkv|16.0|0.000e+00|3.146e+06|
|model.visual.blocks.10.attn.proj|16.0|0.000e+00|1.049e+06|
|model.visual.blocks.10.mlp.linear_fc1|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.10.mlp.linear_fc2|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.11.attn.qkv|16.0|0.000e+00|3.146e+06|
|model.visual.blocks.11.attn.proj|16.0|0.000e+00|1.049e+06|
|model.visual.blocks.11.mlp.linear_fc1|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.11.mlp.linear_fc2|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.12.attn.qkv|16.0|0.000e+00|3.146e+06|
|model.visual.blocks.12.attn.proj|16.0|0.000e+00|1.049e+06|
|model.visual.blocks.12.mlp.linear_fc1|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.12.mlp.linear_fc2|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.13.attn.qkv|16.0|0.000e+00|3.146e+06|
|model.visual.blocks.13.attn.proj|16.0|0.000e+00|1.049e+06|
|model.visual.blocks.13.mlp.linear_fc1|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.13.mlp.linear_fc2|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.14.attn.qkv|16.0|0.000e+00|3.146e+06|
|model.visual.blocks.14.attn.proj|16.0|0.000e+00|1.049e+06|
|model.visual.blocks.14.mlp.linear_fc1|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.14.mlp.linear_fc2|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.15.attn.qkv|16.0|0.000e+00|3.146e+06|
|model.visual.blocks.15.attn.proj|16.0|0.000e+00|1.049e+06|
|model.visual.blocks.15.mlp.linear_fc1|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.15.mlp.linear_fc2|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.16.attn.qkv|16.0|0.000e+00|3.146e+06|
|model.visual.blocks.16.attn.proj|16.0|0.000e+00|1.049e+06|
|model.visual.blocks.16.mlp.linear_fc1|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.16.mlp.linear_fc2|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.17.attn.qkv|16.0|0.000e+00|3.146e+06|
|model.visual.blocks.17.attn.proj|16.0|0.000e+00|1.049e+06|
|model.visual.blocks.17.mlp.linear_fc1|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.17.mlp.linear_fc2|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.18.attn.qkv|16.0|0.000e+00|3.146e+06|
|model.visual.blocks.18.attn.proj|16.0|0.000e+00|1.049e+06|
|model.visual.blocks.18.mlp.linear_fc1|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.18.mlp.linear_fc2|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.19.attn.qkv|16.0|0.000e+00|3.146e+06|
|model.visual.blocks.19.attn.proj|16.0|0.000e+00|1.049e+06|
|model.visual.blocks.19.mlp.linear_fc1|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.19.mlp.linear_fc2|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.20.attn.qkv|16.0|0.000e+00|3.146e+06|
|model.visual.blocks.20.attn.proj|16.0|0.000e+00|1.049e+06|
|model.visual.blocks.20.mlp.linear_fc1|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.20.mlp.linear_fc2|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.21.attn.qkv|16.0|0.000e+00|3.146e+06|
|model.visual.blocks.21.attn.proj|16.0|0.000e+00|1.049e+06|
|model.visual.blocks.21.mlp.linear_fc1|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.21.mlp.linear_fc2|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.22.attn.qkv|16.0|0.000e+00|3.146e+06|
|model.visual.blocks.22.attn.proj|16.0|0.000e+00|1.049e+06|
|model.visual.blocks.22.mlp.linear_fc1|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.22.mlp.linear_fc2|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.23.attn.qkv|16.0|0.000e+00|3.146e+06|
|model.visual.blocks.23.attn.proj|16.0|0.000e+00|1.049e+06|
|model.visual.blocks.23.mlp.linear_fc1|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.23.mlp.linear_fc2|16.0|0.000e+00|4.194e+06|
|model.visual.merger.linear_fc1|16.0|0.000e+00|1.678e+07|
|model.visual.merger.linear_fc2|16.0|0.000e+00|1.049e+07|
|model.visual.deepstack_merger_list.0.linear_fc1|16.0|0.000e+00|1.678e+07|
|model.visual.deepstack_merger_list.0.linear_fc2|16.0|0.000e+00|1.049e+07|
|model.visual.deepstack_merger_list.1.linear_fc1|16.0|0.000e+00|1.678e+07|
|model.visual.deepstack_merger_list.1.linear_fc2|16.0|0.000e+00|1.049e+07|
|model.visual.deepstack_merger_list.2.linear_fc1|16.0|0.000e+00|1.678e+07|
|model.visual.deepstack_merger_list.2.linear_fc2|16.0|0.000e+00|1.049e+07|
