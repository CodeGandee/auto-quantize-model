
AutoQuant Layer Sensitivity (wfp8_aint4_autoquant_lm)
=====================================================


**Scheme:** `wfp8_aint4_autoquant_lm`

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

**Total AutoQuant score:** `1.537737e+02`

**Constraint satisfied:** `True`
## Per-layer sensitivity table


- **Layer**: Name of the quant_recipe handle for a group of quantizable modules (e.g., attention or MLP projections).
- **Num Bits**: Effective number of bits allocated for the quantized recipe(s) considered at this layer.
- **Sensitivity**: AutoQuant sensitivity score for the quantized recipe(s). Higher values indicate that quantizing this layer is more harmful to model quality.
- **Size Cost**: Approximate compressed weight size contribution of the layer under the corresponding recipe(s). Higher values indicate more memory usage.

Note: In the JSON manifest, layer keys may end with `.quant_recipe` (e.g., `language_model.layers.0.mlp.gate_proj.quant_recipe`). This suffix is added by ModelOpt to represent the AutoQuant hyperparameter attached to that module. In this table we strip the `.quant_recipe` suffix for readability; the underlying module path is the part before that suffix.

|Layer|Num Bits|Sensitivity|Size Cost|
| :--- | :--- | :--- | :--- |
|model.language_model.layers.4.mlp.gate_proj|4.0|4.321e+02|1.245e+07|
|model.language_model.layers.0.mlp.down_proj|4.0|2.020e+02|6.226e+06|
|model.language_model.layers.4.mlp.down_proj|4.0|1.746e+02|6.226e+06|
|model.language_model.layers.5.mlp.gate_proj|4.0|1.626e+02|1.245e+07|
|model.language_model.layers.13.mlp.gate_proj|4.0|1.506e+02|1.245e+07|
|model.language_model.layers.6.mlp.gate_proj|4.0|1.352e+02|1.245e+07|
|model.language_model.layers.3.mlp.gate_proj|4.0|1.086e+02|1.245e+07|
|model.language_model.layers.14.mlp.gate_proj|4.0|1.085e+02|1.245e+07|
|model.language_model.layers.3.mlp.down_proj|4.0|9.669e+01|6.226e+06|
|model.language_model.layers.11.mlp.gate_proj|4.0|8.033e+01|1.245e+07|
|model.language_model.layers.15.mlp.gate_proj|4.0|7.926e+01|1.245e+07|
|model.language_model.layers.12.mlp.gate_proj|4.0|7.862e+01|1.245e+07|
|model.language_model.layers.5.mlp.down_proj|4.0|7.741e+01|6.226e+06|
|model.language_model.layers.8.mlp.gate_proj|4.0|7.548e+01|1.245e+07|
|model.language_model.layers.9.mlp.gate_proj|4.0|7.380e+01|1.245e+07|
|model.language_model.layers.10.mlp.gate_proj|4.0|7.254e+01|1.245e+07|
|model.language_model.layers.7.mlp.gate_proj|4.0|7.120e+01|1.245e+07|
|model.language_model.layers.2.mlp.gate_proj|4.0|6.972e+01|1.245e+07|
|model.language_model.layers.9.mlp.down_proj|4.0|6.929e+01|6.226e+06|
|model.language_model.layers.16.mlp.gate_proj|4.0|6.721e+01|1.245e+07|
|model.language_model.layers.10.mlp.down_proj|4.0|6.558e+01|6.226e+06|
|model.language_model.layers.12.mlp.down_proj|4.0|6.496e+01|6.226e+06|
|model.language_model.layers.14.mlp.down_proj|4.0|6.144e+01|6.226e+06|
|model.language_model.layers.6.mlp.down_proj|4.0|6.074e+01|6.226e+06|
|model.language_model.layers.1.mlp.down_proj|4.0|5.412e+01|6.226e+06|
|model.language_model.layers.2.mlp.down_proj|4.0|5.272e+01|6.226e+06|
|model.language_model.layers.11.mlp.down_proj|4.0|4.989e+01|6.226e+06|
|model.language_model.layers.7.mlp.down_proj|4.0|4.921e+01|6.226e+06|
|model.language_model.layers.15.mlp.down_proj|4.0|4.877e+01|6.226e+06|
|model.language_model.layers.17.mlp.gate_proj|4.0|4.689e+01|1.245e+07|
|model.language_model.layers.8.mlp.down_proj|4.0|4.278e+01|6.226e+06|
|model.language_model.layers.13.mlp.down_proj|4.0|4.189e+01|6.226e+06|
|model.language_model.layers.1.mlp.gate_proj|4.0|4.105e+01|1.245e+07|
|model.language_model.layers.16.mlp.down_proj|4.0|3.395e+01|6.226e+06|
|model.language_model.layers.18.mlp.gate_proj|4.0|2.537e+01|1.245e+07|
|model.language_model.layers.19.mlp.gate_proj|4.0|1.719e+01|1.245e+07|
|model.language_model.layers.18.mlp.down_proj|4.0|1.679e+01|6.226e+06|
|model.language_model.layers.17.mlp.down_proj|4.0|1.661e+01|6.226e+06|
|lm_head|4.0|1.318e+01|9.724e+07|
|model.language_model.layers.19.mlp.down_proj|4.0|1.094e+01|6.226e+06|
|model.language_model.layers.20.mlp.gate_proj|4.0|1.002e+01|1.245e+07|
|model.language_model.layers.22.mlp.gate_proj|4.0|7.698e+00|1.245e+07|
|model.language_model.layers.21.mlp.gate_proj|4.0|7.684e+00|1.245e+07|
|model.language_model.layers.24.mlp.gate_proj|4.0|6.627e+00|1.245e+07|
|model.language_model.layers.0.mlp.gate_proj|4.0|6.428e+00|1.245e+07|
|model.language_model.layers.23.mlp.gate_proj|4.0|6.114e+00|1.245e+07|
|model.language_model.layers.25.mlp.gate_proj|4.0|5.642e+00|1.245e+07|
|model.language_model.layers.20.mlp.down_proj|4.0|5.502e+00|6.226e+06|
|model.language_model.layers.22.mlp.down_proj|4.0|4.584e+00|6.226e+06|
|model.language_model.layers.21.mlp.down_proj|4.0|4.557e+00|6.226e+06|
|model.language_model.layers.26.mlp.gate_proj|4.0|3.546e+00|1.245e+07|
|model.language_model.layers.23.mlp.down_proj|4.0|3.452e+00|6.226e+06|
|model.language_model.layers.24.mlp.down_proj|4.0|2.612e+00|6.226e+06|
|model.language_model.layers.27.mlp.gate_proj|4.0|1.890e+00|1.245e+07|
|model.language_model.layers.25.mlp.down_proj|4.0|1.702e+00|6.226e+06|
|model.language_model.layers.26.mlp.down_proj|4.0|1.424e+00|6.226e+06|
|model.language_model.layers.28.mlp.gate_proj|4.0|1.066e+00|1.245e+07|
|model.language_model.layers.27.mlp.down_proj|4.0|8.479e-01|6.226e+06|
|model.language_model.layers.34.mlp.down_proj|4.0|8.064e-01|6.226e+06|
|model.language_model.layers.29.mlp.gate_proj|4.0|7.588e-01|1.245e+07|
|model.language_model.layers.35.mlp.gate_proj|4.0|6.813e-01|1.245e+07|
|model.language_model.layers.28.mlp.down_proj|4.0|5.668e-01|6.226e+06|
|model.language_model.layers.30.mlp.gate_proj|4.0|3.948e-01|1.245e+07|
|model.language_model.layers.29.mlp.down_proj|4.0|3.483e-01|6.226e+06|
|model.language_model.layers.35.mlp.down_proj|4.0|3.129e-01|6.226e+06|
|model.language_model.layers.34.mlp.gate_proj|4.0|2.870e-01|1.245e+07|
|model.language_model.layers.30.mlp.down_proj|4.0|2.787e-01|6.226e+06|
|model.language_model.layers.31.mlp.gate_proj|4.0|2.410e-01|1.245e+07|
|model.language_model.layers.33.mlp.gate_proj|4.0|1.899e-01|1.245e+07|
|model.language_model.layers.31.mlp.down_proj|4.0|1.714e-01|6.226e+06|
|model.language_model.layers.32.mlp.gate_proj|4.0|1.675e-01|1.245e+07|
|model.language_model.layers.32.mlp.down_proj|4.0|1.317e-01|6.226e+06|
|model.language_model.layers.33.mlp.down_proj|4.0|6.980e-02|6.226e+06|
|model.language_model.layers.0.self_attn.q_proj|4.0|3.334e-02|3.932e+06|
|model.language_model.layers.35.self_attn.q_proj|4.0|1.712e-02|3.932e+06|
|model.language_model.layers.6.self_attn.q_proj|4.0|1.696e-02|3.932e+06|
|model.language_model.layers.10.self_attn.q_proj|4.0|1.451e-02|3.932e+06|
|model.language_model.layers.8.self_attn.q_proj|4.0|1.190e-02|3.932e+06|
|model.language_model.layers.7.self_attn.q_proj|4.0|1.138e-02|3.932e+06|
|model.language_model.layers.9.self_attn.q_proj|4.0|1.072e-02|3.932e+06|
|model.language_model.layers.14.self_attn.q_proj|4.0|8.987e-03|3.932e+06|
|model.language_model.layers.23.self_attn.q_proj|4.0|8.581e-03|3.932e+06|
|model.language_model.layers.11.self_attn.q_proj|4.0|7.585e-03|3.932e+06|
|model.language_model.layers.5.self_attn.q_proj|4.0|7.553e-03|3.932e+06|
|model.language_model.layers.13.self_attn.q_proj|4.0|7.409e-03|3.932e+06|
|model.language_model.layers.22.self_attn.q_proj|4.0|7.146e-03|3.932e+06|
|model.language_model.layers.34.self_attn.q_proj|4.0|6.888e-03|3.932e+06|
|model.language_model.layers.4.self_attn.q_proj|4.0|6.788e-03|3.932e+06|
|model.language_model.layers.12.self_attn.q_proj|4.0|6.670e-03|3.932e+06|
|model.language_model.layers.0.self_attn.o_proj|4.0|6.497e-03|2.621e+06|
|model.language_model.layers.1.self_attn.q_proj|4.0|6.117e-03|3.932e+06|
|model.language_model.layers.15.self_attn.q_proj|4.0|6.097e-03|3.932e+06|
|model.language_model.layers.24.self_attn.q_proj|4.0|5.900e-03|3.932e+06|
|model.language_model.layers.3.self_attn.q_proj|4.0|5.752e-03|3.932e+06|
|model.language_model.layers.28.self_attn.q_proj|4.0|5.341e-03|3.932e+06|
|model.language_model.layers.21.self_attn.q_proj|4.0|4.988e-03|3.932e+06|
|model.language_model.layers.16.self_attn.q_proj|4.0|4.847e-03|3.932e+06|
|model.language_model.layers.26.self_attn.q_proj|4.0|4.843e-03|3.932e+06|
|model.language_model.layers.30.self_attn.q_proj|4.0|4.546e-03|3.932e+06|
|model.language_model.layers.17.self_attn.q_proj|4.0|4.178e-03|3.932e+06|
|model.language_model.layers.18.self_attn.q_proj|4.0|4.121e-03|3.932e+06|
|model.language_model.layers.32.self_attn.q_proj|4.0|4.092e-03|3.932e+06|
|model.language_model.layers.25.self_attn.q_proj|4.0|3.929e-03|3.932e+06|
|model.language_model.layers.2.self_attn.q_proj|4.0|3.886e-03|3.932e+06|
|model.language_model.layers.27.self_attn.q_proj|4.0|3.801e-03|3.932e+06|
|model.language_model.layers.31.self_attn.q_proj|4.0|3.738e-03|3.932e+06|
|model.language_model.layers.14.self_attn.o_proj|4.0|3.414e-03|2.621e+06|
|model.language_model.layers.19.self_attn.q_proj|4.0|3.359e-03|3.932e+06|
|model.language_model.layers.34.self_attn.o_proj|4.0|3.158e-03|2.621e+06|
|model.language_model.layers.20.self_attn.q_proj|4.0|2.959e-03|3.932e+06|
|model.language_model.layers.29.self_attn.q_proj|4.0|2.909e-03|3.932e+06|
|model.language_model.layers.22.self_attn.o_proj|4.0|2.860e-03|2.621e+06|
|model.language_model.layers.15.self_attn.o_proj|4.0|2.825e-03|2.621e+06|
|model.language_model.layers.13.self_attn.o_proj|4.0|2.820e-03|2.621e+06|
|model.language_model.layers.16.self_attn.o_proj|4.0|2.356e-03|2.621e+06|
|model.language_model.layers.10.self_attn.o_proj|4.0|2.306e-03|2.621e+06|
|model.language_model.layers.6.self_attn.o_proj|4.0|2.222e-03|2.621e+06|
|model.language_model.layers.35.self_attn.o_proj|4.0|2.220e-03|2.621e+06|
|model.language_model.layers.23.self_attn.o_proj|4.0|2.100e-03|2.621e+06|
|model.language_model.layers.5.self_attn.o_proj|4.0|1.871e-03|2.621e+06|
|model.language_model.layers.8.self_attn.o_proj|4.0|1.597e-03|2.621e+06|
|model.language_model.layers.33.self_attn.q_proj|4.0|1.429e-03|3.932e+06|
|model.language_model.layers.24.self_attn.o_proj|4.0|1.205e-03|2.621e+06|
|model.language_model.layers.28.self_attn.o_proj|4.0|1.186e-03|2.621e+06|
|model.language_model.layers.4.self_attn.o_proj|4.0|1.127e-03|2.621e+06|
|model.language_model.layers.7.self_attn.o_proj|4.0|1.119e-03|2.621e+06|
|model.language_model.layers.1.self_attn.o_proj|4.0|1.074e-03|2.621e+06|
|model.language_model.layers.17.self_attn.o_proj|4.0|1.036e-03|2.621e+06|
|model.language_model.layers.9.self_attn.o_proj|4.0|1.028e-03|2.621e+06|
|model.language_model.layers.26.self_attn.o_proj|4.0|1.001e-03|2.621e+06|
|model.language_model.layers.27.self_attn.o_proj|4.0|9.834e-04|2.621e+06|
|model.language_model.layers.12.self_attn.o_proj|4.0|9.380e-04|2.621e+06|
|model.language_model.layers.18.self_attn.o_proj|4.0|9.336e-04|2.621e+06|
|model.language_model.layers.19.self_attn.o_proj|4.0|8.296e-04|2.621e+06|
|model.language_model.layers.30.self_attn.o_proj|4.0|8.237e-04|2.621e+06|
|model.language_model.layers.25.self_attn.o_proj|4.0|8.227e-04|2.621e+06|
|model.language_model.layers.11.self_attn.o_proj|4.0|7.992e-04|2.621e+06|
|model.language_model.layers.32.self_attn.o_proj|4.0|7.826e-04|2.621e+06|
|model.language_model.layers.3.self_attn.o_proj|4.0|7.645e-04|2.621e+06|
|model.language_model.layers.29.self_attn.o_proj|4.0|7.416e-04|2.621e+06|
|model.language_model.layers.21.self_attn.o_proj|4.0|6.709e-04|2.621e+06|
|model.language_model.layers.31.self_attn.o_proj|4.0|6.377e-04|2.621e+06|
|model.language_model.layers.20.self_attn.o_proj|4.0|6.268e-04|2.621e+06|
|model.language_model.layers.2.self_attn.o_proj|4.0|5.788e-04|2.621e+06|
|model.language_model.layers.33.self_attn.o_proj|4.0|5.471e-04|2.621e+06|
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
