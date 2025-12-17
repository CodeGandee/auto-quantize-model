
AutoQuant Layer Sensitivity (wfp8_aint8_autoquant_lm)
=====================================================


**Scheme:** `wfp8_aint8_autoquant_lm`

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

**Effective bits (from search):** `8.7443`

**Total AutoQuant score:** `4.205751e+02`

**Constraint satisfied:** `True`
## Per-layer sensitivity table


- **Layer**: Name of the quant_recipe handle for a group of quantizable modules (e.g., attention or MLP projections).
- **Num Bits**: Effective number of bits allocated for the quantized recipe(s) considered at this layer.
- **Sensitivity**: AutoQuant sensitivity score for the quantized recipe(s). Higher values indicate that quantizing this layer is more harmful to model quality.
- **Size Cost**: Approximate compressed weight size contribution of the layer under the corresponding recipe(s). Higher values indicate more memory usage.

Note: In the JSON manifest, layer keys may end with `.quant_recipe` (e.g., `language_model.layers.0.mlp.gate_proj.quant_recipe`). This suffix is added by ModelOpt to represent the AutoQuant hyperparameter attached to that module. In this table we strip the `.quant_recipe` suffix for readability; the underlying module path is the part before that suffix.

|Layer|Num Bits|Sensitivity|Size Cost|
| :--- | :--- | :--- | :--- |
|model.language_model.layers.6.mlp.down_proj|8.0|6.071e+01|1.245e+07|
|model.language_model.layers.6.mlp.gate_proj|8.0|5.988e+01|2.490e+07|
|model.language_model.layers.12.mlp.down_proj|8.0|4.670e+01|1.245e+07|
|model.language_model.layers.1.mlp.down_proj|8.0|4.125e+01|1.245e+07|
|model.language_model.layers.14.mlp.down_proj|8.0|3.639e+01|1.245e+07|
|model.language_model.layers.16.mlp.down_proj|8.0|2.352e+01|1.245e+07|
|model.language_model.layers.15.mlp.down_proj|8.0|1.481e+01|1.245e+07|
|model.language_model.layers.9.mlp.down_proj|8.0|1.473e+01|1.245e+07|
|model.language_model.layers.3.mlp.down_proj|8.0|1.417e+01|1.245e+07|
|model.language_model.layers.10.mlp.down_proj|8.0|1.355e+01|1.245e+07|
|model.language_model.layers.4.mlp.down_proj|8.0|1.258e+01|1.245e+07|
|model.language_model.layers.13.mlp.down_proj|8.0|1.065e+01|1.245e+07|
|model.language_model.layers.2.mlp.down_proj|8.0|8.860e+00|1.245e+07|
|model.language_model.layers.18.mlp.down_proj|8.0|8.227e+00|1.245e+07|
|model.language_model.layers.0.mlp.down_proj|8.0|7.663e+00|1.245e+07|
|model.language_model.layers.11.mlp.down_proj|8.0|7.088e+00|1.245e+07|
|model.language_model.layers.5.mlp.down_proj|8.0|5.858e+00|1.245e+07|
|model.language_model.layers.7.mlp.down_proj|8.0|4.719e+00|1.245e+07|
|model.language_model.layers.8.mlp.down_proj|8.0|4.607e+00|1.245e+07|
|model.language_model.layers.4.mlp.gate_proj|8.0|3.530e+00|2.490e+07|
|model.language_model.layers.3.mlp.gate_proj|8.0|3.211e+00|2.490e+07|
|model.language_model.layers.2.mlp.gate_proj|8.0|2.679e+00|2.490e+07|
|model.language_model.layers.7.mlp.gate_proj|8.0|1.916e+00|2.490e+07|
|model.language_model.layers.16.mlp.gate_proj|8.0|1.531e+00|2.490e+07|
|model.language_model.layers.5.mlp.gate_proj|8.0|1.408e+00|2.490e+07|
|model.language_model.layers.1.mlp.gate_proj|8.0|9.543e-01|2.490e+07|
|model.language_model.layers.14.mlp.gate_proj|8.0|9.211e-01|2.490e+07|
|model.language_model.layers.15.mlp.gate_proj|8.0|8.994e-01|2.490e+07|
|model.language_model.layers.13.mlp.gate_proj|8.0|7.407e-01|2.490e+07|
|model.language_model.layers.12.mlp.gate_proj|8.0|7.034e-01|2.490e+07|
|model.language_model.layers.24.mlp.down_proj|8.0|5.922e-01|1.245e+07|
|model.language_model.layers.11.mlp.gate_proj|8.0|5.208e-01|2.490e+07|
|model.language_model.layers.23.mlp.down_proj|8.0|4.135e-01|1.245e+07|
|model.language_model.layers.17.mlp.down_proj|8.0|4.033e-01|1.245e+07|
|model.language_model.layers.35.mlp.gate_proj|8.0|4.009e-01|2.490e+07|
|model.language_model.layers.8.mlp.gate_proj|8.0|3.885e-01|2.490e+07|
|model.language_model.layers.10.mlp.gate_proj|8.0|3.807e-01|2.490e+07|
|model.language_model.layers.9.mlp.gate_proj|8.0|3.739e-01|2.490e+07|
|model.language_model.layers.25.mlp.down_proj|8.0|3.673e-01|1.245e+07|
|model.language_model.layers.19.mlp.down_proj|8.0|2.824e-01|1.245e+07|
|model.language_model.layers.17.mlp.gate_proj|8.0|2.727e-01|2.490e+07|
|model.language_model.layers.22.mlp.down_proj|8.0|2.519e-01|1.245e+07|
|model.language_model.layers.26.mlp.down_proj|8.0|1.863e-01|1.245e+07|
|model.language_model.layers.21.mlp.down_proj|8.0|1.496e-01|1.245e+07|
|model.language_model.layers.20.mlp.down_proj|8.0|1.435e-01|1.245e+07|
|model.language_model.layers.18.mlp.gate_proj|8.0|1.418e-01|2.490e+07|
|model.language_model.layers.28.mlp.down_proj|8.0|1.179e-01|1.245e+07|
|model.language_model.layers.27.mlp.down_proj|8.0|8.962e-02|1.245e+07|
|model.language_model.layers.19.mlp.gate_proj|8.0|8.446e-02|2.490e+07|
|model.language_model.layers.20.mlp.gate_proj|8.0|7.398e-02|2.490e+07|
|model.language_model.layers.0.mlp.gate_proj|8.0|6.939e-02|2.490e+07|
|model.language_model.layers.21.mlp.gate_proj|8.0|4.438e-02|2.490e+07|
|model.language_model.layers.35.mlp.down_proj|8.0|4.433e-02|1.245e+07|
|model.language_model.layers.34.mlp.down_proj|8.0|3.607e-02|1.245e+07|
|model.language_model.layers.23.mlp.gate_proj|8.0|3.518e-02|2.490e+07|
|model.language_model.layers.30.mlp.down_proj|8.0|3.456e-02|1.245e+07|
|model.language_model.layers.22.mlp.gate_proj|8.0|3.285e-02|2.490e+07|
|model.language_model.layers.24.mlp.gate_proj|8.0|3.251e-02|2.490e+07|
|model.language_model.layers.29.mlp.down_proj|8.0|3.010e-02|1.245e+07|
|model.language_model.layers.25.mlp.gate_proj|8.0|2.746e-02|2.490e+07|
|lm_head|8.0|1.789e-02|1.945e+08|
|model.language_model.layers.26.mlp.gate_proj|8.0|1.777e-02|2.490e+07|
|model.language_model.layers.34.mlp.gate_proj|8.0|1.430e-02|2.490e+07|
|model.language_model.layers.27.mlp.gate_proj|8.0|9.593e-03|2.490e+07|
|model.language_model.layers.31.mlp.down_proj|8.0|9.386e-03|1.245e+07|
|model.language_model.layers.32.mlp.down_proj|8.0|6.734e-03|1.245e+07|
|model.language_model.layers.28.mlp.gate_proj|8.0|5.104e-03|2.490e+07|
|model.language_model.layers.29.mlp.gate_proj|8.0|3.367e-03|2.490e+07|
|model.language_model.layers.33.mlp.down_proj|8.0|3.050e-03|1.245e+07|
|model.language_model.layers.30.mlp.gate_proj|8.0|1.801e-03|2.490e+07|
|model.language_model.layers.31.mlp.gate_proj|8.0|1.107e-03|2.490e+07|
|model.language_model.layers.32.mlp.gate_proj|8.0|9.785e-04|2.490e+07|
|model.language_model.layers.33.mlp.gate_proj|8.0|7.725e-04|2.490e+07|
|model.language_model.layers.35.self_attn.q_proj|8.0|5.044e-04|7.864e+06|
|model.language_model.layers.15.self_attn.o_proj|8.0|3.687e-04|5.243e+06|
|model.language_model.layers.6.self_attn.q_proj|8.0|3.550e-04|7.864e+06|
|model.language_model.layers.16.self_attn.o_proj|8.0|2.092e-04|5.243e+06|
|model.language_model.layers.22.self_attn.q_proj|8.0|1.873e-04|7.864e+06|
|model.language_model.layers.34.self_attn.q_proj|8.0|1.797e-04|7.864e+06|
|model.language_model.layers.0.self_attn.q_proj|8.0|1.767e-04|7.864e+06|
|model.language_model.layers.21.self_attn.q_proj|8.0|1.637e-04|7.864e+06|
|model.language_model.layers.23.self_attn.q_proj|8.0|1.636e-04|7.864e+06|
|model.language_model.layers.13.self_attn.o_proj|8.0|1.519e-04|5.243e+06|
|model.language_model.layers.3.self_attn.q_proj|8.0|1.456e-04|7.864e+06|
|model.language_model.layers.14.self_attn.o_proj|8.0|1.389e-04|5.243e+06|
|model.language_model.layers.24.self_attn.q_proj|8.0|1.178e-04|7.864e+06|
|model.language_model.layers.9.self_attn.q_proj|8.0|1.167e-04|7.864e+06|
|model.language_model.layers.4.self_attn.q_proj|8.0|1.070e-04|7.864e+06|
|model.language_model.layers.10.self_attn.q_proj|8.0|1.022e-04|7.864e+06|
|model.language_model.layers.30.self_attn.q_proj|8.0|1.011e-04|7.864e+06|
|model.language_model.layers.16.self_attn.q_proj|8.0|1.001e-04|7.864e+06|
|model.language_model.layers.5.self_attn.q_proj|8.0|1.001e-04|7.864e+06|
|model.language_model.layers.32.self_attn.q_proj|8.0|9.955e-05|7.864e+06|
|model.language_model.layers.7.self_attn.q_proj|8.0|9.873e-05|7.864e+06|
|model.language_model.layers.8.self_attn.q_proj|8.0|9.637e-05|7.864e+06|
|model.language_model.layers.34.self_attn.o_proj|8.0|9.026e-05|5.243e+06|
|model.language_model.layers.28.self_attn.q_proj|8.0|8.762e-05|7.864e+06|
|model.language_model.layers.19.self_attn.q_proj|8.0|8.513e-05|7.864e+06|
|model.language_model.layers.22.self_attn.o_proj|8.0|8.422e-05|5.243e+06|
|model.language_model.layers.18.self_attn.q_proj|8.0|8.332e-05|7.864e+06|
|model.language_model.layers.26.self_attn.q_proj|8.0|8.220e-05|7.864e+06|
|model.language_model.layers.17.self_attn.q_proj|8.0|7.921e-05|7.864e+06|
|model.language_model.layers.20.self_attn.q_proj|8.0|7.836e-05|7.864e+06|
|model.language_model.layers.14.self_attn.q_proj|8.0|7.727e-05|7.864e+06|
|model.language_model.layers.31.self_attn.q_proj|8.0|7.607e-05|7.864e+06|
|model.language_model.layers.25.self_attn.q_proj|8.0|7.558e-05|7.864e+06|
|model.language_model.layers.15.self_attn.q_proj|8.0|7.223e-05|7.864e+06|
|model.language_model.layers.33.self_attn.q_proj|8.0|6.856e-05|7.864e+06|
|model.language_model.layers.11.self_attn.q_proj|8.0|6.494e-05|7.864e+06|
|model.language_model.layers.27.self_attn.q_proj|8.0|6.241e-05|7.864e+06|
|model.language_model.layers.2.self_attn.q_proj|8.0|6.040e-05|7.864e+06|
|model.language_model.layers.29.self_attn.q_proj|8.0|5.889e-05|7.864e+06|
|model.language_model.layers.13.self_attn.q_proj|8.0|4.991e-05|7.864e+06|
|model.language_model.layers.12.self_attn.q_proj|8.0|4.547e-05|7.864e+06|
|model.language_model.layers.0.self_attn.o_proj|8.0|3.645e-05|5.243e+06|
|model.language_model.layers.1.self_attn.q_proj|8.0|3.172e-05|7.864e+06|
|model.language_model.layers.33.self_attn.o_proj|8.0|3.125e-05|5.243e+06|
|model.language_model.layers.10.self_attn.o_proj|8.0|3.043e-05|5.243e+06|
|model.language_model.layers.23.self_attn.o_proj|8.0|2.610e-05|5.243e+06|
|model.language_model.layers.32.self_attn.o_proj|8.0|2.261e-05|5.243e+06|
|model.language_model.layers.35.self_attn.o_proj|8.0|2.217e-05|5.243e+06|
|model.language_model.layers.29.self_attn.o_proj|8.0|2.142e-05|5.243e+06|
|model.language_model.layers.27.self_attn.o_proj|8.0|2.075e-05|5.243e+06|
|model.language_model.layers.26.self_attn.o_proj|8.0|2.050e-05|5.243e+06|
|model.language_model.layers.31.self_attn.o_proj|8.0|1.917e-05|5.243e+06|
|model.language_model.layers.6.self_attn.o_proj|8.0|1.882e-05|5.243e+06|
|model.language_model.layers.8.self_attn.o_proj|8.0|1.842e-05|5.243e+06|
|model.language_model.layers.28.self_attn.o_proj|8.0|1.759e-05|5.243e+06|
|model.language_model.layers.30.self_attn.o_proj|8.0|1.611e-05|5.243e+06|
|model.language_model.layers.24.self_attn.o_proj|8.0|1.554e-05|5.243e+06|
|model.language_model.layers.17.self_attn.o_proj|8.0|1.458e-05|5.243e+06|
|model.language_model.layers.18.self_attn.o_proj|8.0|1.410e-05|5.243e+06|
|model.language_model.layers.19.self_attn.o_proj|8.0|1.366e-05|5.243e+06|
|model.language_model.layers.25.self_attn.o_proj|8.0|1.311e-05|5.243e+06|
|model.language_model.layers.7.self_attn.o_proj|8.0|1.284e-05|5.243e+06|
|model.language_model.layers.9.self_attn.o_proj|8.0|1.226e-05|5.243e+06|
|model.language_model.layers.5.self_attn.o_proj|8.0|1.140e-05|5.243e+06|
|model.language_model.layers.12.self_attn.o_proj|8.0|1.081e-05|5.243e+06|
|model.language_model.layers.4.self_attn.o_proj|8.0|1.005e-05|5.243e+06|
|model.language_model.layers.20.self_attn.o_proj|8.0|9.329e-06|5.243e+06|
|model.language_model.layers.21.self_attn.o_proj|8.0|8.642e-06|5.243e+06|
|model.language_model.layers.11.self_attn.o_proj|8.0|8.096e-06|5.243e+06|
|model.language_model.layers.1.self_attn.o_proj|8.0|6.603e-06|5.243e+06|
|model.language_model.layers.3.self_attn.o_proj|8.0|5.929e-06|5.243e+06|
|model.language_model.layers.2.self_attn.o_proj|8.0|4.129e-06|5.243e+06|
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
