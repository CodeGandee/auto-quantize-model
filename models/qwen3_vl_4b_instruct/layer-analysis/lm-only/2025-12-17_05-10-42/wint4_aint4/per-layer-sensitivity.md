
AutoQuant Layer Sensitivity (wint4_aint4_autoquant_lm)
======================================================


**Scheme:** `wint4_aint4_autoquant_lm`

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

**Total AutoQuant score:** `1.561253e+02`

**Constraint satisfied:** `True`
## Per-layer sensitivity table


- **Layer**: Name of the quant_recipe handle for a group of quantizable modules (e.g., attention or MLP projections).
- **Num Bits**: Effective number of bits allocated for the quantized recipe(s) considered at this layer.
- **Sensitivity**: AutoQuant sensitivity score for the quantized recipe(s). Higher values indicate that quantizing this layer is more harmful to model quality.
- **Size Cost**: Approximate compressed weight size contribution of the layer under the corresponding recipe(s). Higher values indicate more memory usage.

Note: In the JSON manifest, layer keys may end with `.quant_recipe` (e.g., `language_model.layers.0.mlp.gate_proj.quant_recipe`). This suffix is added by ModelOpt to represent the AutoQuant hyperparameter attached to that module. In this table we strip the `.quant_recipe` suffix for readability; the underlying module path is the part before that suffix.

|Layer|Num Bits|Sensitivity|Size Cost|
| :--- | :--- | :--- | :--- |
|model.language_model.layers.4.mlp.gate_proj|4.0|4.607e+02|1.245e+07|
|model.language_model.layers.0.mlp.down_proj|4.0|2.030e+02|6.226e+06|
|model.language_model.layers.4.mlp.down_proj|4.0|1.767e+02|6.226e+06|
|model.language_model.layers.5.mlp.gate_proj|4.0|1.607e+02|1.245e+07|
|model.language_model.layers.13.mlp.gate_proj|4.0|1.542e+02|1.245e+07|
|model.language_model.layers.6.mlp.gate_proj|4.0|1.324e+02|1.245e+07|
|model.language_model.layers.3.mlp.gate_proj|4.0|1.127e+02|1.245e+07|
|model.language_model.layers.14.mlp.gate_proj|4.0|1.084e+02|1.245e+07|
|model.language_model.layers.3.mlp.down_proj|4.0|9.670e+01|6.226e+06|
|model.language_model.layers.11.mlp.gate_proj|4.0|8.107e+01|1.245e+07|
|model.language_model.layers.15.mlp.gate_proj|4.0|7.927e+01|1.245e+07|
|model.language_model.layers.12.mlp.gate_proj|4.0|7.909e+01|1.245e+07|
|model.language_model.layers.5.mlp.down_proj|4.0|7.764e+01|6.226e+06|
|model.language_model.layers.8.mlp.gate_proj|4.0|7.654e+01|1.245e+07|
|model.language_model.layers.9.mlp.gate_proj|4.0|7.448e+01|1.245e+07|
|model.language_model.layers.2.mlp.gate_proj|4.0|7.374e+01|1.245e+07|
|model.language_model.layers.10.mlp.gate_proj|4.0|7.326e+01|1.245e+07|
|model.language_model.layers.7.mlp.gate_proj|4.0|7.215e+01|1.245e+07|
|model.language_model.layers.9.mlp.down_proj|4.0|6.929e+01|6.226e+06|
|model.language_model.layers.16.mlp.gate_proj|4.0|6.693e+01|1.245e+07|
|model.language_model.layers.10.mlp.down_proj|4.0|6.558e+01|6.226e+06|
|model.language_model.layers.12.mlp.down_proj|4.0|6.496e+01|6.226e+06|
|model.language_model.layers.14.mlp.down_proj|4.0|6.144e+01|6.226e+06|
|model.language_model.layers.6.mlp.down_proj|4.0|6.076e+01|6.226e+06|
|model.language_model.layers.1.mlp.down_proj|4.0|5.415e+01|6.226e+06|
|model.language_model.layers.2.mlp.down_proj|4.0|5.272e+01|6.226e+06|
|model.language_model.layers.11.mlp.down_proj|4.0|4.989e+01|6.226e+06|
|model.language_model.layers.7.mlp.down_proj|4.0|4.921e+01|6.226e+06|
|model.language_model.layers.15.mlp.down_proj|4.0|4.877e+01|6.226e+06|
|model.language_model.layers.17.mlp.gate_proj|4.0|4.684e+01|1.245e+07|
|model.language_model.layers.8.mlp.down_proj|4.0|4.299e+01|6.226e+06|
|model.language_model.layers.13.mlp.down_proj|4.0|4.220e+01|6.226e+06|
|model.language_model.layers.1.mlp.gate_proj|4.0|4.175e+01|1.245e+07|
|model.language_model.layers.16.mlp.down_proj|4.0|3.395e+01|6.226e+06|
|model.language_model.layers.18.mlp.gate_proj|4.0|2.563e+01|1.245e+07|
|model.language_model.layers.19.mlp.gate_proj|4.0|1.737e+01|1.245e+07|
|model.language_model.layers.18.mlp.down_proj|4.0|1.679e+01|6.226e+06|
|model.language_model.layers.17.mlp.down_proj|4.0|1.677e+01|6.226e+06|
|lm_head|4.0|1.307e+01|9.724e+07|
|model.language_model.layers.19.mlp.down_proj|4.0|1.086e+01|6.226e+06|
|model.language_model.layers.20.mlp.gate_proj|4.0|1.016e+01|1.245e+07|
|model.language_model.layers.22.mlp.gate_proj|4.0|7.767e+00|1.245e+07|
|model.language_model.layers.0.mlp.gate_proj|4.0|7.727e+00|1.245e+07|
|model.language_model.layers.21.mlp.gate_proj|4.0|7.668e+00|1.245e+07|
|model.language_model.layers.24.mlp.gate_proj|4.0|6.639e+00|1.245e+07|
|model.language_model.layers.23.mlp.gate_proj|4.0|6.240e+00|1.245e+07|
|model.language_model.layers.25.mlp.gate_proj|4.0|5.714e+00|1.245e+07|
|model.language_model.layers.20.mlp.down_proj|4.0|5.517e+00|6.226e+06|
|model.language_model.layers.22.mlp.down_proj|4.0|4.610e+00|6.226e+06|
|model.language_model.layers.21.mlp.down_proj|4.0|4.597e+00|6.226e+06|
|model.language_model.layers.26.mlp.gate_proj|4.0|3.595e+00|1.245e+07|
|model.language_model.layers.23.mlp.down_proj|4.0|3.460e+00|6.226e+06|
|model.language_model.layers.24.mlp.down_proj|4.0|2.601e+00|6.226e+06|
|model.language_model.layers.27.mlp.gate_proj|4.0|1.923e+00|1.245e+07|
|model.language_model.layers.25.mlp.down_proj|4.0|1.718e+00|6.226e+06|
|model.language_model.layers.26.mlp.down_proj|4.0|1.430e+00|6.226e+06|
|model.language_model.layers.28.mlp.gate_proj|4.0|1.092e+00|1.245e+07|
|model.language_model.layers.27.mlp.down_proj|4.0|8.618e-01|6.226e+06|
|model.language_model.layers.34.mlp.down_proj|4.0|8.031e-01|6.226e+06|
|model.language_model.layers.35.mlp.gate_proj|4.0|7.772e-01|1.245e+07|
|model.language_model.layers.29.mlp.gate_proj|4.0|7.748e-01|1.245e+07|
|model.language_model.layers.28.mlp.down_proj|4.0|5.676e-01|6.226e+06|
|model.language_model.layers.30.mlp.gate_proj|4.0|4.030e-01|1.245e+07|
|model.language_model.layers.29.mlp.down_proj|4.0|3.504e-01|6.226e+06|
|model.language_model.layers.35.mlp.down_proj|4.0|3.187e-01|6.226e+06|
|model.language_model.layers.34.mlp.gate_proj|4.0|3.084e-01|1.245e+07|
|model.language_model.layers.30.mlp.down_proj|4.0|2.797e-01|6.226e+06|
|model.language_model.layers.31.mlp.gate_proj|4.0|2.517e-01|1.245e+07|
|model.language_model.layers.33.mlp.gate_proj|4.0|1.966e-01|1.245e+07|
|model.language_model.layers.32.mlp.gate_proj|4.0|1.765e-01|1.245e+07|
|model.language_model.layers.31.mlp.down_proj|4.0|1.734e-01|6.226e+06|
|model.language_model.layers.32.mlp.down_proj|4.0|1.314e-01|6.226e+06|
|model.language_model.layers.33.mlp.down_proj|4.0|7.320e-02|6.226e+06|
|model.language_model.layers.0.self_attn.q_proj|4.0|3.373e-02|3.932e+06|
|model.language_model.layers.35.self_attn.q_proj|4.0|1.776e-02|3.932e+06|
|model.language_model.layers.6.self_attn.q_proj|4.0|1.739e-02|3.932e+06|
|model.language_model.layers.10.self_attn.q_proj|4.0|1.486e-02|3.932e+06|
|model.language_model.layers.8.self_attn.q_proj|4.0|1.212e-02|3.932e+06|
|model.language_model.layers.7.self_attn.q_proj|4.0|1.146e-02|3.932e+06|
|model.language_model.layers.9.self_attn.q_proj|4.0|1.100e-02|3.932e+06|
|model.language_model.layers.14.self_attn.q_proj|4.0|9.193e-03|3.932e+06|
|model.language_model.layers.23.self_attn.q_proj|4.0|8.868e-03|3.932e+06|
|model.language_model.layers.5.self_attn.q_proj|4.0|7.754e-03|3.932e+06|
|model.language_model.layers.11.self_attn.q_proj|4.0|7.628e-03|3.932e+06|
|model.language_model.layers.22.self_attn.q_proj|4.0|7.456e-03|3.932e+06|
|model.language_model.layers.13.self_attn.q_proj|4.0|7.427e-03|3.932e+06|
|model.language_model.layers.34.self_attn.q_proj|4.0|7.002e-03|3.932e+06|
|model.language_model.layers.4.self_attn.q_proj|4.0|6.892e-03|3.932e+06|
|model.language_model.layers.12.self_attn.q_proj|4.0|6.822e-03|3.932e+06|
|model.language_model.layers.0.self_attn.o_proj|4.0|6.723e-03|2.621e+06|
|model.language_model.layers.15.self_attn.q_proj|4.0|6.306e-03|3.932e+06|
|model.language_model.layers.1.self_attn.q_proj|4.0|6.271e-03|3.932e+06|
|model.language_model.layers.24.self_attn.q_proj|4.0|6.133e-03|3.932e+06|
|model.language_model.layers.3.self_attn.q_proj|4.0|5.936e-03|3.932e+06|
|model.language_model.layers.28.self_attn.q_proj|4.0|5.487e-03|3.932e+06|
|model.language_model.layers.21.self_attn.q_proj|4.0|5.297e-03|3.932e+06|
|model.language_model.layers.26.self_attn.q_proj|4.0|4.969e-03|3.932e+06|
|model.language_model.layers.16.self_attn.q_proj|4.0|4.962e-03|3.932e+06|
|model.language_model.layers.30.self_attn.q_proj|4.0|4.835e-03|3.932e+06|
|model.language_model.layers.32.self_attn.q_proj|4.0|4.468e-03|3.932e+06|
|model.language_model.layers.17.self_attn.q_proj|4.0|4.321e-03|3.932e+06|
|model.language_model.layers.18.self_attn.q_proj|4.0|4.252e-03|3.932e+06|
|model.language_model.layers.25.self_attn.q_proj|4.0|4.051e-03|3.932e+06|
|model.language_model.layers.2.self_attn.q_proj|4.0|3.976e-03|3.932e+06|
|model.language_model.layers.31.self_attn.q_proj|4.0|3.931e-03|3.932e+06|
|model.language_model.layers.27.self_attn.q_proj|4.0|3.924e-03|3.932e+06|
|model.language_model.layers.19.self_attn.q_proj|4.0|3.509e-03|3.932e+06|
|model.language_model.layers.14.self_attn.o_proj|4.0|3.446e-03|2.621e+06|
|model.language_model.layers.34.self_attn.o_proj|4.0|3.195e-03|2.621e+06|
|model.language_model.layers.29.self_attn.q_proj|4.0|3.064e-03|3.932e+06|
|model.language_model.layers.20.self_attn.q_proj|4.0|3.028e-03|3.932e+06|
|model.language_model.layers.22.self_attn.o_proj|4.0|2.926e-03|2.621e+06|
|model.language_model.layers.15.self_attn.o_proj|4.0|2.856e-03|2.621e+06|
|model.language_model.layers.13.self_attn.o_proj|4.0|2.826e-03|2.621e+06|
|model.language_model.layers.6.self_attn.o_proj|4.0|2.443e-03|2.621e+06|
|model.language_model.layers.16.self_attn.o_proj|4.0|2.393e-03|2.621e+06|
|model.language_model.layers.10.self_attn.o_proj|4.0|2.374e-03|2.621e+06|
|model.language_model.layers.35.self_attn.o_proj|4.0|2.293e-03|2.621e+06|
|model.language_model.layers.23.self_attn.o_proj|4.0|2.196e-03|2.621e+06|
|model.language_model.layers.5.self_attn.o_proj|4.0|1.932e-03|2.621e+06|
|model.language_model.layers.8.self_attn.o_proj|4.0|1.700e-03|2.621e+06|
|model.language_model.layers.33.self_attn.q_proj|4.0|1.549e-03|3.932e+06|
|model.language_model.layers.24.self_attn.o_proj|4.0|1.269e-03|2.621e+06|
|model.language_model.layers.28.self_attn.o_proj|4.0|1.229e-03|2.621e+06|
|model.language_model.layers.7.self_attn.o_proj|4.0|1.186e-03|2.621e+06|
|model.language_model.layers.4.self_attn.o_proj|4.0|1.175e-03|2.621e+06|
|model.language_model.layers.1.self_attn.o_proj|4.0|1.140e-03|2.621e+06|
|model.language_model.layers.9.self_attn.o_proj|4.0|1.098e-03|2.621e+06|
|model.language_model.layers.17.self_attn.o_proj|4.0|1.077e-03|2.621e+06|
|model.language_model.layers.26.self_attn.o_proj|4.0|1.039e-03|2.621e+06|
|model.language_model.layers.27.self_attn.o_proj|4.0|1.018e-03|2.621e+06|
|model.language_model.layers.12.self_attn.o_proj|4.0|1.011e-03|2.621e+06|
|model.language_model.layers.18.self_attn.o_proj|4.0|9.804e-04|2.621e+06|
|model.language_model.layers.19.self_attn.o_proj|4.0|8.761e-04|2.621e+06|
|model.language_model.layers.30.self_attn.o_proj|4.0|8.608e-04|2.621e+06|
|model.language_model.layers.25.self_attn.o_proj|4.0|8.574e-04|2.621e+06|
|model.language_model.layers.11.self_attn.o_proj|4.0|8.564e-04|2.621e+06|
|model.language_model.layers.32.self_attn.o_proj|4.0|8.068e-04|2.621e+06|
|model.language_model.layers.3.self_attn.o_proj|4.0|8.041e-04|2.621e+06|
|model.language_model.layers.29.self_attn.o_proj|4.0|7.700e-04|2.621e+06|
|model.language_model.layers.21.self_attn.o_proj|4.0|7.261e-04|2.621e+06|
|model.language_model.layers.20.self_attn.o_proj|4.0|6.693e-04|2.621e+06|
|model.language_model.layers.31.self_attn.o_proj|4.0|6.662e-04|2.621e+06|
|model.language_model.layers.2.self_attn.o_proj|4.0|6.082e-04|2.621e+06|
|model.language_model.layers.33.self_attn.o_proj|4.0|5.666e-04|2.621e+06|
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
