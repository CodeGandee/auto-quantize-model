
AutoQuant Layer Sensitivity (wint8_aint4_autoquant_lm)
======================================================


**Scheme:** `wint8_aint4_autoquant_lm`

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

**Total AutoQuant score:** `1.537329e+02`

**Constraint satisfied:** `True`
## Per-layer sensitivity table


- **Layer**: Name of the quant_recipe handle for a group of quantizable modules (e.g., attention or MLP projections).
- **Num Bits**: Effective number of bits allocated for the quantized recipe(s) considered at this layer.
- **Sensitivity**: AutoQuant sensitivity score for the quantized recipe(s). Higher values indicate that quantizing this layer is more harmful to model quality.
- **Size Cost**: Approximate compressed weight size contribution of the layer under the corresponding recipe(s). Higher values indicate more memory usage.

Note: In the JSON manifest, layer keys may end with `.quant_recipe` (e.g., `language_model.layers.0.mlp.gate_proj.quant_recipe`). This suffix is added by ModelOpt to represent the AutoQuant hyperparameter attached to that module. In this table we strip the `.quant_recipe` suffix for readability; the underlying module path is the part before that suffix.

|Layer|Num Bits|Sensitivity|Size Cost|
| :--- | :--- | :--- | :--- |
|model.language_model.layers.4.mlp.gate_proj|4.0|4.318e+02|1.245e+07|
|model.language_model.layers.0.mlp.down_proj|4.0|1.976e+02|6.226e+06|
|model.language_model.layers.4.mlp.down_proj|4.0|1.743e+02|6.226e+06|
|model.language_model.layers.5.mlp.gate_proj|4.0|1.628e+02|1.245e+07|
|model.language_model.layers.13.mlp.gate_proj|4.0|1.507e+02|1.245e+07|
|model.language_model.layers.6.mlp.gate_proj|4.0|1.351e+02|1.245e+07|
|model.language_model.layers.14.mlp.gate_proj|4.0|1.086e+02|1.245e+07|
|model.language_model.layers.3.mlp.gate_proj|4.0|1.081e+02|1.245e+07|
|model.language_model.layers.3.mlp.down_proj|4.0|9.669e+01|6.226e+06|
|model.language_model.layers.11.mlp.gate_proj|4.0|8.060e+01|1.245e+07|
|model.language_model.layers.15.mlp.gate_proj|4.0|7.925e+01|1.245e+07|
|model.language_model.layers.12.mlp.gate_proj|4.0|7.869e+01|1.245e+07|
|model.language_model.layers.5.mlp.down_proj|4.0|7.746e+01|6.226e+06|
|model.language_model.layers.8.mlp.gate_proj|4.0|7.588e+01|1.245e+07|
|model.language_model.layers.9.mlp.gate_proj|4.0|7.390e+01|1.245e+07|
|model.language_model.layers.10.mlp.gate_proj|4.0|7.235e+01|1.245e+07|
|model.language_model.layers.7.mlp.gate_proj|4.0|7.128e+01|1.245e+07|
|model.language_model.layers.2.mlp.gate_proj|4.0|6.957e+01|1.245e+07|
|model.language_model.layers.9.mlp.down_proj|4.0|6.929e+01|6.226e+06|
|model.language_model.layers.16.mlp.gate_proj|4.0|6.720e+01|1.245e+07|
|model.language_model.layers.10.mlp.down_proj|4.0|6.558e+01|6.226e+06|
|model.language_model.layers.12.mlp.down_proj|4.0|6.496e+01|6.226e+06|
|model.language_model.layers.14.mlp.down_proj|4.0|6.144e+01|6.226e+06|
|model.language_model.layers.6.mlp.down_proj|4.0|6.074e+01|6.226e+06|
|model.language_model.layers.1.mlp.down_proj|4.0|5.412e+01|6.226e+06|
|model.language_model.layers.2.mlp.down_proj|4.0|5.272e+01|6.226e+06|
|model.language_model.layers.11.mlp.down_proj|4.0|4.989e+01|6.226e+06|
|model.language_model.layers.7.mlp.down_proj|4.0|4.919e+01|6.226e+06|
|model.language_model.layers.15.mlp.down_proj|4.0|4.877e+01|6.226e+06|
|model.language_model.layers.17.mlp.gate_proj|4.0|4.694e+01|1.245e+07|
|model.language_model.layers.8.mlp.down_proj|4.0|4.277e+01|6.226e+06|
|model.language_model.layers.13.mlp.down_proj|4.0|4.190e+01|6.226e+06|
|model.language_model.layers.1.mlp.gate_proj|4.0|4.104e+01|1.245e+07|
|model.language_model.layers.16.mlp.down_proj|4.0|3.395e+01|6.226e+06|
|model.language_model.layers.18.mlp.gate_proj|4.0|2.540e+01|1.245e+07|
|model.language_model.layers.19.mlp.gate_proj|4.0|1.718e+01|1.245e+07|
|model.language_model.layers.18.mlp.down_proj|4.0|1.679e+01|6.226e+06|
|model.language_model.layers.17.mlp.down_proj|4.0|1.662e+01|6.226e+06|
|lm_head|4.0|1.317e+01|9.724e+07|
|model.language_model.layers.19.mlp.down_proj|4.0|1.096e+01|6.226e+06|
|model.language_model.layers.20.mlp.gate_proj|4.0|1.003e+01|1.245e+07|
|model.language_model.layers.22.mlp.gate_proj|4.0|7.696e+00|1.245e+07|
|model.language_model.layers.21.mlp.gate_proj|4.0|7.680e+00|1.245e+07|
|model.language_model.layers.24.mlp.gate_proj|4.0|6.616e+00|1.245e+07|
|model.language_model.layers.0.mlp.gate_proj|4.0|6.360e+00|1.245e+07|
|model.language_model.layers.23.mlp.gate_proj|4.0|6.124e+00|1.245e+07|
|model.language_model.layers.25.mlp.gate_proj|4.0|5.609e+00|1.245e+07|
|model.language_model.layers.20.mlp.down_proj|4.0|5.498e+00|6.226e+06|
|model.language_model.layers.22.mlp.down_proj|4.0|4.593e+00|6.226e+06|
|model.language_model.layers.21.mlp.down_proj|4.0|4.555e+00|6.226e+06|
|model.language_model.layers.26.mlp.gate_proj|4.0|3.536e+00|1.245e+07|
|model.language_model.layers.23.mlp.down_proj|4.0|3.450e+00|6.226e+06|
|model.language_model.layers.24.mlp.down_proj|4.0|2.612e+00|6.226e+06|
|model.language_model.layers.27.mlp.gate_proj|4.0|1.886e+00|1.245e+07|
|model.language_model.layers.25.mlp.down_proj|4.0|1.700e+00|6.226e+06|
|model.language_model.layers.26.mlp.down_proj|4.0|1.426e+00|6.226e+06|
|model.language_model.layers.28.mlp.gate_proj|4.0|1.068e+00|1.245e+07|
|model.language_model.layers.27.mlp.down_proj|4.0|8.469e-01|6.226e+06|
|model.language_model.layers.34.mlp.down_proj|4.0|8.038e-01|6.226e+06|
|model.language_model.layers.29.mlp.gate_proj|4.0|7.564e-01|1.245e+07|
|model.language_model.layers.35.mlp.gate_proj|4.0|7.312e-01|1.245e+07|
|model.language_model.layers.28.mlp.down_proj|4.0|5.673e-01|6.226e+06|
|model.language_model.layers.30.mlp.gate_proj|4.0|3.945e-01|1.245e+07|
|model.language_model.layers.29.mlp.down_proj|4.0|3.473e-01|6.226e+06|
|model.language_model.layers.35.mlp.down_proj|4.0|3.126e-01|6.226e+06|
|model.language_model.layers.34.mlp.gate_proj|4.0|2.885e-01|1.245e+07|
|model.language_model.layers.30.mlp.down_proj|4.0|2.784e-01|6.226e+06|
|model.language_model.layers.31.mlp.gate_proj|4.0|2.426e-01|1.245e+07|
|model.language_model.layers.33.mlp.gate_proj|4.0|1.904e-01|1.245e+07|
|model.language_model.layers.31.mlp.down_proj|4.0|1.710e-01|6.226e+06|
|model.language_model.layers.32.mlp.gate_proj|4.0|1.683e-01|1.245e+07|
|model.language_model.layers.32.mlp.down_proj|4.0|1.316e-01|6.226e+06|
|model.language_model.layers.33.mlp.down_proj|4.0|6.990e-02|6.226e+06|
|model.language_model.layers.0.self_attn.q_proj|4.0|3.335e-02|3.932e+06|
|model.language_model.layers.35.self_attn.q_proj|4.0|1.715e-02|3.932e+06|
|model.language_model.layers.6.self_attn.q_proj|4.0|1.691e-02|3.932e+06|
|model.language_model.layers.10.self_attn.q_proj|4.0|1.446e-02|3.932e+06|
|model.language_model.layers.8.self_attn.q_proj|4.0|1.188e-02|3.932e+06|
|model.language_model.layers.7.self_attn.q_proj|4.0|1.123e-02|3.932e+06|
|model.language_model.layers.9.self_attn.q_proj|4.0|1.074e-02|3.932e+06|
|model.language_model.layers.14.self_attn.q_proj|4.0|8.950e-03|3.932e+06|
|model.language_model.layers.23.self_attn.q_proj|4.0|8.611e-03|3.932e+06|
|model.language_model.layers.11.self_attn.q_proj|4.0|7.547e-03|3.932e+06|
|model.language_model.layers.5.self_attn.q_proj|4.0|7.523e-03|3.932e+06|
|model.language_model.layers.13.self_attn.q_proj|4.0|7.350e-03|3.932e+06|
|model.language_model.layers.22.self_attn.q_proj|4.0|7.118e-03|3.932e+06|
|model.language_model.layers.34.self_attn.q_proj|4.0|6.828e-03|3.932e+06|
|model.language_model.layers.4.self_attn.q_proj|4.0|6.782e-03|3.932e+06|
|model.language_model.layers.12.self_attn.q_proj|4.0|6.683e-03|3.932e+06|
|model.language_model.layers.0.self_attn.o_proj|4.0|6.495e-03|2.621e+06|
|model.language_model.layers.1.self_attn.q_proj|4.0|6.154e-03|3.932e+06|
|model.language_model.layers.15.self_attn.q_proj|4.0|6.073e-03|3.932e+06|
|model.language_model.layers.24.self_attn.q_proj|4.0|5.827e-03|3.932e+06|
|model.language_model.layers.3.self_attn.q_proj|4.0|5.752e-03|3.932e+06|
|model.language_model.layers.28.self_attn.q_proj|4.0|5.308e-03|3.932e+06|
|model.language_model.layers.21.self_attn.q_proj|4.0|4.998e-03|3.932e+06|
|model.language_model.layers.16.self_attn.q_proj|4.0|4.822e-03|3.932e+06|
|model.language_model.layers.26.self_attn.q_proj|4.0|4.808e-03|3.932e+06|
|model.language_model.layers.30.self_attn.q_proj|4.0|4.546e-03|3.932e+06|
|model.language_model.layers.17.self_attn.q_proj|4.0|4.166e-03|3.932e+06|
|model.language_model.layers.32.self_attn.q_proj|4.0|4.135e-03|3.932e+06|
|model.language_model.layers.18.self_attn.q_proj|4.0|4.107e-03|3.932e+06|
|model.language_model.layers.25.self_attn.q_proj|4.0|3.894e-03|3.932e+06|
|model.language_model.layers.2.self_attn.q_proj|4.0|3.884e-03|3.932e+06|
|model.language_model.layers.27.self_attn.q_proj|4.0|3.782e-03|3.932e+06|
|model.language_model.layers.31.self_attn.q_proj|4.0|3.721e-03|3.932e+06|
|model.language_model.layers.14.self_attn.o_proj|4.0|3.415e-03|2.621e+06|
|model.language_model.layers.19.self_attn.q_proj|4.0|3.364e-03|3.932e+06|
|model.language_model.layers.34.self_attn.o_proj|4.0|3.152e-03|2.621e+06|
|model.language_model.layers.20.self_attn.q_proj|4.0|2.965e-03|3.932e+06|
|model.language_model.layers.29.self_attn.q_proj|4.0|2.913e-03|3.932e+06|
|model.language_model.layers.22.self_attn.o_proj|4.0|2.860e-03|2.621e+06|
|model.language_model.layers.15.self_attn.o_proj|4.0|2.823e-03|2.621e+06|
|model.language_model.layers.13.self_attn.o_proj|4.0|2.820e-03|2.621e+06|
|model.language_model.layers.16.self_attn.o_proj|4.0|2.356e-03|2.621e+06|
|model.language_model.layers.10.self_attn.o_proj|4.0|2.306e-03|2.621e+06|
|model.language_model.layers.6.self_attn.o_proj|4.0|2.245e-03|2.621e+06|
|model.language_model.layers.35.self_attn.o_proj|4.0|2.218e-03|2.621e+06|
|model.language_model.layers.23.self_attn.o_proj|4.0|2.097e-03|2.621e+06|
|model.language_model.layers.5.self_attn.o_proj|4.0|1.862e-03|2.621e+06|
|model.language_model.layers.8.self_attn.o_proj|4.0|1.592e-03|2.621e+06|
|model.language_model.layers.33.self_attn.q_proj|4.0|1.450e-03|3.932e+06|
|model.language_model.layers.24.self_attn.o_proj|4.0|1.204e-03|2.621e+06|
|model.language_model.layers.28.self_attn.o_proj|4.0|1.186e-03|2.621e+06|
|model.language_model.layers.4.self_attn.o_proj|4.0|1.126e-03|2.621e+06|
|model.language_model.layers.7.self_attn.o_proj|4.0|1.117e-03|2.621e+06|
|model.language_model.layers.1.self_attn.o_proj|4.0|1.072e-03|2.621e+06|
|model.language_model.layers.17.self_attn.o_proj|4.0|1.034e-03|2.621e+06|
|model.language_model.layers.9.self_attn.o_proj|4.0|1.030e-03|2.621e+06|
|model.language_model.layers.26.self_attn.o_proj|4.0|1.002e-03|2.621e+06|
|model.language_model.layers.27.self_attn.o_proj|4.0|9.831e-04|2.621e+06|
|model.language_model.layers.12.self_attn.o_proj|4.0|9.376e-04|2.621e+06|
|model.language_model.layers.18.self_attn.o_proj|4.0|9.338e-04|2.621e+06|
|model.language_model.layers.19.self_attn.o_proj|4.0|8.284e-04|2.621e+06|
|model.language_model.layers.25.self_attn.o_proj|4.0|8.231e-04|2.621e+06|
|model.language_model.layers.30.self_attn.o_proj|4.0|8.211e-04|2.621e+06|
|model.language_model.layers.11.self_attn.o_proj|4.0|7.976e-04|2.621e+06|
|model.language_model.layers.32.self_attn.o_proj|4.0|7.821e-04|2.621e+06|
|model.language_model.layers.3.self_attn.o_proj|4.0|7.625e-04|2.621e+06|
|model.language_model.layers.29.self_attn.o_proj|4.0|7.420e-04|2.621e+06|
|model.language_model.layers.21.self_attn.o_proj|4.0|6.709e-04|2.621e+06|
|model.language_model.layers.31.self_attn.o_proj|4.0|6.376e-04|2.621e+06|
|model.language_model.layers.20.self_attn.o_proj|4.0|6.248e-04|2.621e+06|
|model.language_model.layers.2.self_attn.o_proj|4.0|5.779e-04|2.621e+06|
|model.language_model.layers.33.self_attn.o_proj|4.0|5.477e-04|2.621e+06|
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
