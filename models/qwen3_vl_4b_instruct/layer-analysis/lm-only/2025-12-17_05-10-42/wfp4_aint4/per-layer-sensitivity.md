
AutoQuant Layer Sensitivity (wfp4_aint4_autoquant_lm)
=====================================================


**Scheme:** `wfp4_aint4_autoquant_lm`

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

**Total AutoQuant score:** `1.547231e+02`

**Constraint satisfied:** `True`
## Per-layer sensitivity table


- **Layer**: Name of the quant_recipe handle for a group of quantizable modules (e.g., attention or MLP projections).
- **Num Bits**: Effective number of bits allocated for the quantized recipe(s) considered at this layer.
- **Sensitivity**: AutoQuant sensitivity score for the quantized recipe(s). Higher values indicate that quantizing this layer is more harmful to model quality.
- **Size Cost**: Approximate compressed weight size contribution of the layer under the corresponding recipe(s). Higher values indicate more memory usage.

Note: In the JSON manifest, layer keys may end with `.quant_recipe` (e.g., `language_model.layers.0.mlp.gate_proj.quant_recipe`). This suffix is added by ModelOpt to represent the AutoQuant hyperparameter attached to that module. In this table we strip the `.quant_recipe` suffix for readability; the underlying module path is the part before that suffix.

|Layer|Num Bits|Sensitivity|Size Cost|
| :--- | :--- | :--- | :--- |
|model.language_model.layers.4.mlp.gate_proj|4.0|4.220e+02|1.245e+07|
|model.language_model.layers.0.mlp.down_proj|4.0|2.036e+02|6.226e+06|
|model.language_model.layers.4.mlp.down_proj|4.0|1.743e+02|6.226e+06|
|model.language_model.layers.5.mlp.gate_proj|4.0|1.642e+02|1.245e+07|
|model.language_model.layers.13.mlp.gate_proj|4.0|1.493e+02|1.245e+07|
|model.language_model.layers.6.mlp.gate_proj|4.0|1.349e+02|1.245e+07|
|model.language_model.layers.3.mlp.gate_proj|4.0|1.113e+02|1.245e+07|
|model.language_model.layers.14.mlp.gate_proj|4.0|1.083e+02|1.245e+07|
|model.language_model.layers.3.mlp.down_proj|4.0|9.669e+01|6.226e+06|
|model.language_model.layers.11.mlp.gate_proj|4.0|8.081e+01|1.245e+07|
|model.language_model.layers.15.mlp.gate_proj|4.0|7.947e+01|1.245e+07|
|model.language_model.layers.12.mlp.gate_proj|4.0|7.864e+01|1.245e+07|
|model.language_model.layers.5.mlp.down_proj|4.0|7.733e+01|6.226e+06|
|model.language_model.layers.8.mlp.gate_proj|4.0|7.602e+01|1.245e+07|
|model.language_model.layers.10.mlp.gate_proj|4.0|7.307e+01|1.245e+07|
|model.language_model.layers.9.mlp.gate_proj|4.0|7.285e+01|1.245e+07|
|model.language_model.layers.7.mlp.gate_proj|4.0|7.177e+01|1.245e+07|
|model.language_model.layers.2.mlp.gate_proj|4.0|7.042e+01|1.245e+07|
|model.language_model.layers.9.mlp.down_proj|4.0|6.929e+01|6.226e+06|
|model.language_model.layers.16.mlp.gate_proj|4.0|6.732e+01|1.245e+07|
|model.language_model.layers.10.mlp.down_proj|4.0|6.558e+01|6.226e+06|
|model.language_model.layers.12.mlp.down_proj|4.0|6.496e+01|6.226e+06|
|model.language_model.layers.14.mlp.down_proj|4.0|6.144e+01|6.226e+06|
|model.language_model.layers.6.mlp.down_proj|4.0|6.075e+01|6.226e+06|
|model.language_model.layers.1.mlp.down_proj|4.0|5.412e+01|6.226e+06|
|model.language_model.layers.2.mlp.down_proj|4.0|5.272e+01|6.226e+06|
|model.language_model.layers.11.mlp.down_proj|4.0|4.989e+01|6.226e+06|
|model.language_model.layers.7.mlp.down_proj|4.0|4.921e+01|6.226e+06|
|model.language_model.layers.15.mlp.down_proj|4.0|4.877e+01|6.226e+06|
|model.language_model.layers.17.mlp.gate_proj|4.0|4.686e+01|1.245e+07|
|model.language_model.layers.8.mlp.down_proj|4.0|4.282e+01|6.226e+06|
|model.language_model.layers.13.mlp.down_proj|4.0|4.192e+01|6.226e+06|
|model.language_model.layers.1.mlp.gate_proj|4.0|4.157e+01|1.245e+07|
|model.language_model.layers.16.mlp.down_proj|4.0|3.395e+01|6.226e+06|
|model.language_model.layers.18.mlp.gate_proj|4.0|2.538e+01|1.245e+07|
|model.language_model.layers.19.mlp.gate_proj|4.0|1.729e+01|1.245e+07|
|model.language_model.layers.18.mlp.down_proj|4.0|1.679e+01|6.226e+06|
|model.language_model.layers.17.mlp.down_proj|4.0|1.670e+01|6.226e+06|
|lm_head|4.0|1.335e+01|9.724e+07|
|model.language_model.layers.19.mlp.down_proj|4.0|1.092e+01|6.226e+06|
|model.language_model.layers.20.mlp.gate_proj|4.0|9.984e+00|1.245e+07|
|model.language_model.layers.21.mlp.gate_proj|4.0|7.621e+00|1.245e+07|
|model.language_model.layers.22.mlp.gate_proj|4.0|7.609e+00|1.245e+07|
|model.language_model.layers.0.mlp.gate_proj|4.0|7.245e+00|1.245e+07|
|model.language_model.layers.24.mlp.gate_proj|4.0|6.591e+00|1.245e+07|
|model.language_model.layers.23.mlp.gate_proj|4.0|6.135e+00|1.245e+07|
|model.language_model.layers.25.mlp.gate_proj|4.0|5.605e+00|1.245e+07|
|model.language_model.layers.20.mlp.down_proj|4.0|5.504e+00|6.226e+06|
|model.language_model.layers.22.mlp.down_proj|4.0|4.585e+00|6.226e+06|
|model.language_model.layers.21.mlp.down_proj|4.0|4.549e+00|6.226e+06|
|model.language_model.layers.26.mlp.gate_proj|4.0|3.530e+00|1.245e+07|
|model.language_model.layers.23.mlp.down_proj|4.0|3.455e+00|6.226e+06|
|model.language_model.layers.24.mlp.down_proj|4.0|2.608e+00|6.226e+06|
|model.language_model.layers.27.mlp.gate_proj|4.0|1.916e+00|1.245e+07|
|model.language_model.layers.25.mlp.down_proj|4.0|1.706e+00|6.226e+06|
|model.language_model.layers.26.mlp.down_proj|4.0|1.427e+00|6.226e+06|
|model.language_model.layers.28.mlp.gate_proj|4.0|1.087e+00|1.245e+07|
|model.language_model.layers.27.mlp.down_proj|4.0|8.500e-01|6.226e+06|
|model.language_model.layers.34.mlp.down_proj|4.0|8.068e-01|6.226e+06|
|model.language_model.layers.29.mlp.gate_proj|4.0|7.567e-01|1.245e+07|
|model.language_model.layers.35.mlp.gate_proj|4.0|7.239e-01|1.245e+07|
|model.language_model.layers.28.mlp.down_proj|4.0|5.671e-01|6.226e+06|
|model.language_model.layers.30.mlp.gate_proj|4.0|3.963e-01|1.245e+07|
|model.language_model.layers.29.mlp.down_proj|4.0|3.480e-01|6.226e+06|
|model.language_model.layers.35.mlp.down_proj|4.0|3.107e-01|6.226e+06|
|model.language_model.layers.34.mlp.gate_proj|4.0|2.901e-01|1.245e+07|
|model.language_model.layers.30.mlp.down_proj|4.0|2.788e-01|6.226e+06|
|model.language_model.layers.31.mlp.gate_proj|4.0|2.468e-01|1.245e+07|
|model.language_model.layers.33.mlp.gate_proj|4.0|1.914e-01|1.245e+07|
|model.language_model.layers.32.mlp.gate_proj|4.0|1.810e-01|1.245e+07|
|model.language_model.layers.31.mlp.down_proj|4.0|1.713e-01|6.226e+06|
|model.language_model.layers.32.mlp.down_proj|4.0|1.335e-01|6.226e+06|
|model.language_model.layers.33.mlp.down_proj|4.0|7.031e-02|6.226e+06|
|model.language_model.layers.0.self_attn.q_proj|4.0|3.350e-02|3.932e+06|
|model.language_model.layers.35.self_attn.q_proj|4.0|1.752e-02|3.932e+06|
|model.language_model.layers.6.self_attn.q_proj|4.0|1.700e-02|3.932e+06|
|model.language_model.layers.10.self_attn.q_proj|4.0|1.451e-02|3.932e+06|
|model.language_model.layers.8.self_attn.q_proj|4.0|1.201e-02|3.932e+06|
|model.language_model.layers.7.self_attn.q_proj|4.0|1.147e-02|3.932e+06|
|model.language_model.layers.9.self_attn.q_proj|4.0|1.100e-02|3.932e+06|
|model.language_model.layers.14.self_attn.q_proj|4.0|9.172e-03|3.932e+06|
|model.language_model.layers.23.self_attn.q_proj|4.0|8.954e-03|3.932e+06|
|model.language_model.layers.5.self_attn.q_proj|4.0|7.678e-03|3.932e+06|
|model.language_model.layers.11.self_attn.q_proj|4.0|7.579e-03|3.932e+06|
|model.language_model.layers.13.self_attn.q_proj|4.0|7.522e-03|3.932e+06|
|model.language_model.layers.22.self_attn.q_proj|4.0|7.347e-03|3.932e+06|
|model.language_model.layers.34.self_attn.q_proj|4.0|7.099e-03|3.932e+06|
|model.language_model.layers.4.self_attn.q_proj|4.0|6.843e-03|3.932e+06|
|model.language_model.layers.12.self_attn.q_proj|4.0|6.696e-03|3.932e+06|
|model.language_model.layers.0.self_attn.o_proj|4.0|6.578e-03|2.621e+06|
|model.language_model.layers.15.self_attn.q_proj|4.0|6.197e-03|3.932e+06|
|model.language_model.layers.24.self_attn.q_proj|4.0|6.164e-03|3.932e+06|
|model.language_model.layers.1.self_attn.q_proj|4.0|6.133e-03|3.932e+06|
|model.language_model.layers.3.self_attn.q_proj|4.0|5.705e-03|3.932e+06|
|model.language_model.layers.28.self_attn.q_proj|4.0|5.410e-03|3.932e+06|
|model.language_model.layers.21.self_attn.q_proj|4.0|5.227e-03|3.932e+06|
|model.language_model.layers.16.self_attn.q_proj|4.0|5.004e-03|3.932e+06|
|model.language_model.layers.26.self_attn.q_proj|4.0|4.929e-03|3.932e+06|
|model.language_model.layers.30.self_attn.q_proj|4.0|4.601e-03|3.932e+06|
|model.language_model.layers.32.self_attn.q_proj|4.0|4.352e-03|3.932e+06|
|model.language_model.layers.17.self_attn.q_proj|4.0|4.229e-03|3.932e+06|
|model.language_model.layers.18.self_attn.q_proj|4.0|4.160e-03|3.932e+06|
|model.language_model.layers.25.self_attn.q_proj|4.0|3.988e-03|3.932e+06|
|model.language_model.layers.2.self_attn.q_proj|4.0|3.923e-03|3.932e+06|
|model.language_model.layers.31.self_attn.q_proj|4.0|3.923e-03|3.932e+06|
|model.language_model.layers.27.self_attn.q_proj|4.0|3.906e-03|3.932e+06|
|model.language_model.layers.19.self_attn.q_proj|4.0|3.462e-03|3.932e+06|
|model.language_model.layers.14.self_attn.o_proj|4.0|3.428e-03|2.621e+06|
|model.language_model.layers.34.self_attn.o_proj|4.0|3.175e-03|2.621e+06|
|model.language_model.layers.20.self_attn.q_proj|4.0|3.067e-03|3.932e+06|
|model.language_model.layers.29.self_attn.q_proj|4.0|3.054e-03|3.932e+06|
|model.language_model.layers.22.self_attn.o_proj|4.0|2.888e-03|2.621e+06|
|model.language_model.layers.15.self_attn.o_proj|4.0|2.836e-03|2.621e+06|
|model.language_model.layers.13.self_attn.o_proj|4.0|2.825e-03|2.621e+06|
|model.language_model.layers.6.self_attn.o_proj|4.0|2.386e-03|2.621e+06|
|model.language_model.layers.16.self_attn.o_proj|4.0|2.364e-03|2.621e+06|
|model.language_model.layers.10.self_attn.o_proj|4.0|2.332e-03|2.621e+06|
|model.language_model.layers.35.self_attn.o_proj|4.0|2.270e-03|2.621e+06|
|model.language_model.layers.23.self_attn.o_proj|4.0|2.144e-03|2.621e+06|
|model.language_model.layers.5.self_attn.o_proj|4.0|1.912e-03|2.621e+06|
|model.language_model.layers.8.self_attn.o_proj|4.0|1.645e-03|2.621e+06|
|model.language_model.layers.33.self_attn.q_proj|4.0|1.559e-03|3.932e+06|
|model.language_model.layers.24.self_attn.o_proj|4.0|1.234e-03|2.621e+06|
|model.language_model.layers.28.self_attn.o_proj|4.0|1.203e-03|2.621e+06|
|model.language_model.layers.7.self_attn.o_proj|4.0|1.151e-03|2.621e+06|
|model.language_model.layers.4.self_attn.o_proj|4.0|1.148e-03|2.621e+06|
|model.language_model.layers.1.self_attn.o_proj|4.0|1.113e-03|2.621e+06|
|model.language_model.layers.9.self_attn.o_proj|4.0|1.063e-03|2.621e+06|
|model.language_model.layers.17.self_attn.o_proj|4.0|1.055e-03|2.621e+06|
|model.language_model.layers.26.self_attn.o_proj|4.0|1.018e-03|2.621e+06|
|model.language_model.layers.27.self_attn.o_proj|4.0|9.977e-04|2.621e+06|
|model.language_model.layers.12.self_attn.o_proj|4.0|9.699e-04|2.621e+06|
|model.language_model.layers.18.self_attn.o_proj|4.0|9.517e-04|2.621e+06|
|model.language_model.layers.19.self_attn.o_proj|4.0|8.498e-04|2.621e+06|
|model.language_model.layers.30.self_attn.o_proj|4.0|8.385e-04|2.621e+06|
|model.language_model.layers.25.self_attn.o_proj|4.0|8.381e-04|2.621e+06|
|model.language_model.layers.11.self_attn.o_proj|4.0|8.263e-04|2.621e+06|
|model.language_model.layers.32.self_attn.o_proj|4.0|7.941e-04|2.621e+06|
|model.language_model.layers.3.self_attn.o_proj|4.0|7.800e-04|2.621e+06|
|model.language_model.layers.29.self_attn.o_proj|4.0|7.540e-04|2.621e+06|
|model.language_model.layers.21.self_attn.o_proj|4.0|6.923e-04|2.621e+06|
|model.language_model.layers.31.self_attn.o_proj|4.0|6.478e-04|2.621e+06|
|model.language_model.layers.20.self_attn.o_proj|4.0|6.459e-04|2.621e+06|
|model.language_model.layers.2.self_attn.o_proj|4.0|5.917e-04|2.621e+06|
|model.language_model.layers.33.self_attn.o_proj|4.0|5.500e-04|2.621e+06|
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
