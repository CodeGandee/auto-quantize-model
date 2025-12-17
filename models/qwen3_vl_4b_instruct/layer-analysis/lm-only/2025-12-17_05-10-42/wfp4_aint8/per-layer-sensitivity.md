
AutoQuant Layer Sensitivity (wfp4_aint8_autoquant_lm)
=====================================================


**Scheme:** `wfp4_aint8_autoquant_lm`

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

**Total AutoQuant score:** `5.193711e+00`

**Constraint satisfied:** `True`
## Per-layer sensitivity table


- **Layer**: Name of the quant_recipe handle for a group of quantizable modules (e.g., attention or MLP projections).
- **Num Bits**: Effective number of bits allocated for the quantized recipe(s) considered at this layer.
- **Sensitivity**: AutoQuant sensitivity score for the quantized recipe(s). Higher values indicate that quantizing this layer is more harmful to model quality.
- **Size Cost**: Approximate compressed weight size contribution of the layer under the corresponding recipe(s). Higher values indicate more memory usage.

Note: In the JSON manifest, layer keys may end with `.quant_recipe` (e.g., `language_model.layers.0.mlp.gate_proj.quant_recipe`). This suffix is added by ModelOpt to represent the AutoQuant hyperparameter attached to that module. In this table we strip the `.quant_recipe` suffix for readability; the underlying module path is the part before that suffix.

|Layer|Num Bits|Sensitivity|Size Cost|
| :--- | :--- | :--- | :--- |
|model.language_model.layers.6.mlp.down_proj|4.0|6.072e+01|6.226e+06|
|model.language_model.layers.6.mlp.gate_proj|4.0|6.040e+01|1.245e+07|
|model.language_model.layers.12.mlp.down_proj|4.0|4.656e+01|6.226e+06|
|model.language_model.layers.1.mlp.down_proj|4.0|4.094e+01|6.226e+06|
|model.language_model.layers.14.mlp.down_proj|4.0|3.661e+01|6.226e+06|
|model.language_model.layers.16.mlp.down_proj|4.0|2.354e+01|6.226e+06|
|model.language_model.layers.9.mlp.down_proj|4.0|1.528e+01|6.226e+06|
|model.language_model.layers.15.mlp.down_proj|4.0|1.508e+01|6.226e+06|
|model.language_model.layers.3.mlp.down_proj|4.0|1.454e+01|6.226e+06|
|model.language_model.layers.4.mlp.down_proj|4.0|1.387e+01|6.226e+06|
|model.language_model.layers.10.mlp.down_proj|4.0|1.384e+01|6.226e+06|
|model.language_model.layers.13.mlp.down_proj|4.0|1.126e+01|6.226e+06|
|model.language_model.layers.0.mlp.down_proj|4.0|1.048e+01|6.226e+06|
|model.language_model.layers.2.mlp.down_proj|4.0|9.167e+00|6.226e+06|
|model.language_model.layers.18.mlp.down_proj|4.0|8.340e+00|6.226e+06|
|model.language_model.layers.11.mlp.down_proj|4.0|7.366e+00|6.226e+06|
|model.language_model.layers.5.mlp.down_proj|4.0|6.239e+00|6.226e+06|
|model.language_model.layers.4.mlp.gate_proj|4.0|5.790e+00|1.245e+07|
|model.language_model.layers.3.mlp.gate_proj|4.0|5.080e+00|1.245e+07|
|model.language_model.layers.7.mlp.down_proj|4.0|4.996e+00|6.226e+06|
|model.language_model.layers.8.mlp.down_proj|4.0|4.949e+00|6.226e+06|
|model.language_model.layers.2.mlp.gate_proj|4.0|3.753e+00|1.245e+07|
|model.language_model.layers.7.mlp.gate_proj|4.0|2.622e+00|1.245e+07|
|model.language_model.layers.5.mlp.gate_proj|4.0|2.561e+00|1.245e+07|
|model.language_model.layers.16.mlp.gate_proj|4.0|1.943e+00|1.245e+07|
|model.language_model.layers.14.mlp.gate_proj|4.0|1.645e+00|1.245e+07|
|model.language_model.layers.15.mlp.gate_proj|4.0|1.556e+00|1.245e+07|
|model.language_model.layers.1.mlp.gate_proj|4.0|1.470e+00|1.245e+07|
|model.language_model.layers.12.mlp.gate_proj|4.0|1.366e+00|1.245e+07|
|model.language_model.layers.13.mlp.gate_proj|4.0|1.297e+00|1.245e+07|
|model.language_model.layers.11.mlp.gate_proj|4.0|1.042e+00|1.245e+07|
|model.language_model.layers.10.mlp.gate_proj|4.0|9.991e-01|1.245e+07|
|model.language_model.layers.8.mlp.gate_proj|4.0|9.936e-01|1.245e+07|
|model.language_model.layers.9.mlp.gate_proj|4.0|9.864e-01|1.245e+07|
|model.language_model.layers.0.mlp.gate_proj|4.0|7.403e-01|1.245e+07|
|model.language_model.layers.24.mlp.down_proj|4.0|6.089e-01|6.226e+06|
|model.language_model.layers.17.mlp.gate_proj|4.0|5.832e-01|1.245e+07|
|model.language_model.layers.17.mlp.down_proj|4.0|5.311e-01|6.226e+06|
|model.language_model.layers.35.mlp.gate_proj|4.0|4.872e-01|1.245e+07|
|model.language_model.layers.23.mlp.down_proj|4.0|4.428e-01|6.226e+06|
|model.language_model.layers.25.mlp.down_proj|4.0|3.927e-01|6.226e+06|
|model.language_model.layers.19.mlp.down_proj|4.0|3.606e-01|6.226e+06|
|model.language_model.layers.18.mlp.gate_proj|4.0|3.309e-01|1.245e+07|
|model.language_model.layers.22.mlp.down_proj|4.0|2.923e-01|6.226e+06|
|model.language_model.layers.19.mlp.gate_proj|4.0|2.174e-01|1.245e+07|
|model.language_model.layers.26.mlp.down_proj|4.0|1.998e-01|6.226e+06|
|model.language_model.layers.20.mlp.down_proj|4.0|1.932e-01|6.226e+06|
|model.language_model.layers.21.mlp.down_proj|4.0|1.812e-01|6.226e+06|
|model.language_model.layers.20.mlp.gate_proj|4.0|1.639e-01|1.245e+07|
|model.language_model.layers.21.mlp.gate_proj|4.0|1.249e-01|1.245e+07|
|model.language_model.layers.28.mlp.down_proj|4.0|1.223e-01|6.226e+06|
|model.language_model.layers.22.mlp.gate_proj|4.0|1.001e-01|1.245e+07|
|model.language_model.layers.23.mlp.gate_proj|4.0|9.677e-02|1.245e+07|
|model.language_model.layers.27.mlp.down_proj|4.0|9.485e-02|6.226e+06|
|model.language_model.layers.24.mlp.gate_proj|4.0|7.930e-02|1.245e+07|
|model.language_model.layers.25.mlp.gate_proj|4.0|6.501e-02|1.245e+07|
|model.language_model.layers.26.mlp.gate_proj|4.0|4.585e-02|1.245e+07|
|model.language_model.layers.35.mlp.down_proj|4.0|4.549e-02|6.226e+06|
|lm_head|4.0|4.300e-02|9.724e+07|
|model.language_model.layers.34.mlp.down_proj|4.0|3.681e-02|6.226e+06|
|model.language_model.layers.30.mlp.down_proj|4.0|3.674e-02|6.226e+06|
|model.language_model.layers.29.mlp.down_proj|4.0|3.305e-02|6.226e+06|
|model.language_model.layers.27.mlp.gate_proj|4.0|3.053e-02|1.245e+07|
|model.language_model.layers.34.mlp.gate_proj|4.0|1.919e-02|1.245e+07|
|model.language_model.layers.28.mlp.gate_proj|4.0|1.563e-02|1.245e+07|
|model.language_model.layers.31.mlp.down_proj|4.0|1.097e-02|6.226e+06|
|model.language_model.layers.29.mlp.gate_proj|4.0|1.067e-02|1.245e+07|
|model.language_model.layers.32.mlp.down_proj|4.0|7.923e-03|6.226e+06|
|model.language_model.layers.30.mlp.gate_proj|4.0|6.299e-03|1.245e+07|
|model.language_model.layers.31.mlp.gate_proj|4.0|4.924e-03|1.245e+07|
|model.language_model.layers.32.mlp.gate_proj|4.0|3.966e-03|1.245e+07|
|model.language_model.layers.33.mlp.down_proj|4.0|3.609e-03|6.226e+06|
|model.language_model.layers.33.mlp.gate_proj|4.0|2.970e-03|1.245e+07|
|model.language_model.layers.35.self_attn.q_proj|4.0|6.600e-04|3.932e+06|
|model.language_model.layers.0.self_attn.q_proj|4.0|4.825e-04|3.932e+06|
|model.language_model.layers.6.self_attn.q_proj|4.0|4.752e-04|3.932e+06|
|model.language_model.layers.15.self_attn.o_proj|4.0|4.110e-04|2.621e+06|
|model.language_model.layers.22.self_attn.q_proj|4.0|3.619e-04|3.932e+06|
|model.language_model.layers.23.self_attn.q_proj|4.0|3.448e-04|3.932e+06|
|model.language_model.layers.21.self_attn.q_proj|4.0|3.307e-04|3.932e+06|
|model.language_model.layers.34.self_attn.q_proj|4.0|3.074e-04|3.932e+06|
|model.language_model.layers.10.self_attn.q_proj|4.0|2.742e-04|3.932e+06|
|model.language_model.layers.9.self_attn.q_proj|4.0|2.726e-04|3.932e+06|
|model.language_model.layers.24.self_attn.q_proj|4.0|2.556e-04|3.932e+06|
|model.language_model.layers.8.self_attn.q_proj|4.0|2.423e-04|3.932e+06|
|model.language_model.layers.16.self_attn.o_proj|4.0|2.379e-04|2.621e+06|
|model.language_model.layers.7.self_attn.q_proj|4.0|2.372e-04|3.932e+06|
|model.language_model.layers.14.self_attn.q_proj|4.0|2.344e-04|3.932e+06|
|model.language_model.layers.26.self_attn.q_proj|4.0|2.154e-04|3.932e+06|
|model.language_model.layers.30.self_attn.q_proj|4.0|2.055e-04|3.932e+06|
|model.language_model.layers.3.self_attn.q_proj|4.0|2.050e-04|3.932e+06|
|model.language_model.layers.5.self_attn.q_proj|4.0|1.963e-04|3.932e+06|
|model.language_model.layers.28.self_attn.q_proj|4.0|1.941e-04|3.932e+06|
|model.language_model.layers.32.self_attn.q_proj|4.0|1.923e-04|3.932e+06|
|model.language_model.layers.25.self_attn.q_proj|4.0|1.858e-04|3.932e+06|
|model.language_model.layers.4.self_attn.q_proj|4.0|1.850e-04|3.932e+06|
|model.language_model.layers.14.self_attn.o_proj|4.0|1.778e-04|2.621e+06|
|model.language_model.layers.19.self_attn.q_proj|4.0|1.777e-04|3.932e+06|
|model.language_model.layers.16.self_attn.q_proj|4.0|1.771e-04|3.932e+06|
|model.language_model.layers.13.self_attn.o_proj|4.0|1.753e-04|2.621e+06|
|model.language_model.layers.15.self_attn.q_proj|4.0|1.736e-04|3.932e+06|
|model.language_model.layers.31.self_attn.q_proj|4.0|1.667e-04|3.932e+06|
|model.language_model.layers.18.self_attn.q_proj|4.0|1.658e-04|3.932e+06|
|model.language_model.layers.33.self_attn.q_proj|4.0|1.607e-04|3.932e+06|
|model.language_model.layers.11.self_attn.q_proj|4.0|1.582e-04|3.932e+06|
|model.language_model.layers.27.self_attn.q_proj|4.0|1.542e-04|3.932e+06|
|model.language_model.layers.17.self_attn.q_proj|4.0|1.542e-04|3.932e+06|
|model.language_model.layers.20.self_attn.q_proj|4.0|1.460e-04|3.932e+06|
|model.language_model.layers.0.self_attn.o_proj|4.0|1.413e-04|2.621e+06|
|model.language_model.layers.29.self_attn.q_proj|4.0|1.311e-04|3.932e+06|
|model.language_model.layers.6.self_attn.o_proj|4.0|1.289e-04|2.621e+06|
|model.language_model.layers.13.self_attn.q_proj|4.0|1.263e-04|3.932e+06|
|model.language_model.layers.22.self_attn.o_proj|4.0|1.252e-04|2.621e+06|
|model.language_model.layers.34.self_attn.o_proj|4.0|1.167e-04|2.621e+06|
|model.language_model.layers.12.self_attn.q_proj|4.0|1.096e-04|3.932e+06|
|model.language_model.layers.2.self_attn.q_proj|4.0|1.025e-04|3.932e+06|
|model.language_model.layers.1.self_attn.q_proj|4.0|7.901e-05|3.932e+06|
|model.language_model.layers.23.self_attn.o_proj|4.0|7.570e-05|2.621e+06|
|model.language_model.layers.8.self_attn.o_proj|4.0|7.363e-05|2.621e+06|
|model.language_model.layers.35.self_attn.o_proj|4.0|6.964e-05|2.621e+06|
|model.language_model.layers.10.self_attn.o_proj|4.0|6.900e-05|2.621e+06|
|model.language_model.layers.7.self_attn.o_proj|4.0|4.936e-05|2.621e+06|
|model.language_model.layers.24.self_attn.o_proj|4.0|4.858e-05|2.621e+06|
|model.language_model.layers.12.self_attn.o_proj|4.0|4.715e-05|2.621e+06|
|model.language_model.layers.9.self_attn.o_proj|4.0|4.703e-05|2.621e+06|
|model.language_model.layers.5.self_attn.o_proj|4.0|4.668e-05|2.621e+06|
|model.language_model.layers.1.self_attn.o_proj|4.0|4.160e-05|2.621e+06|
|model.language_model.layers.26.self_attn.o_proj|4.0|4.133e-05|2.621e+06|
|model.language_model.layers.28.self_attn.o_proj|4.0|4.107e-05|2.621e+06|
|model.language_model.layers.33.self_attn.o_proj|4.0|4.023e-05|2.621e+06|
|model.language_model.layers.27.self_attn.o_proj|4.0|4.011e-05|2.621e+06|
|model.language_model.layers.19.self_attn.o_proj|4.0|3.878e-05|2.621e+06|
|model.language_model.layers.4.self_attn.o_proj|4.0|3.804e-05|2.621e+06|
|model.language_model.layers.11.self_attn.o_proj|4.0|3.769e-05|2.621e+06|
|model.language_model.layers.18.self_attn.o_proj|4.0|3.766e-05|2.621e+06|
|model.language_model.layers.17.self_attn.o_proj|4.0|3.727e-05|2.621e+06|
|model.language_model.layers.32.self_attn.o_proj|4.0|3.692e-05|2.621e+06|
|model.language_model.layers.29.self_attn.o_proj|4.0|3.551e-05|2.621e+06|
|model.language_model.layers.30.self_attn.o_proj|4.0|3.525e-05|2.621e+06|
|model.language_model.layers.31.self_attn.o_proj|4.0|3.370e-05|2.621e+06|
|model.language_model.layers.21.self_attn.o_proj|4.0|3.354e-05|2.621e+06|
|model.language_model.layers.20.self_attn.o_proj|4.0|3.143e-05|2.621e+06|
|model.language_model.layers.25.self_attn.o_proj|4.0|3.137e-05|2.621e+06|
|model.language_model.layers.3.self_attn.o_proj|4.0|2.460e-05|2.621e+06|
|model.language_model.layers.2.self_attn.o_proj|4.0|1.937e-05|2.621e+06|
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
