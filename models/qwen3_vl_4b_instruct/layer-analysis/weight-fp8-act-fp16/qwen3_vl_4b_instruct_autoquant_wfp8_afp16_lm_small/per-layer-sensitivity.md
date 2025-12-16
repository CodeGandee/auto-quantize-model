
AutoQuant Layer Sensitivity (wfp8_afp16_autoquant_lm)
=====================================================


**Scheme:** `wfp8_afp16_autoquant_lm`

**Model:** `/workspace/code/auto-quantize-model/models/qwen3_vl_4b_instruct/checkpoints/Qwen3-VL-4B-Instruct`

**Effective bits (from search):** `8.0000`

**Total AutoQuant score:** `2.731442e-01`

**Constraint satisfied:** `True`
## Per-layer sensitivity table


- **Layer**: Name of the quant_recipe handle for a group of quantizable modules (e.g., attention or MLP projections).
- **Num Bits**: Effective number of bits allocated for the quantized recipe(s) considered at this layer.
- **Sensitivity**: AutoQuant sensitivity score for the quantized recipe(s). Higher values indicate that quantizing this layer is more harmful to model quality.
- **Size Cost**: Approximate compressed weight size contribution of the layer under the corresponding recipe(s). Higher values indicate more memory usage.

Note: In the JSON manifest, layer keys may end with `.quant_recipe` (e.g., `language_model.layers.0.mlp.gate_proj.quant_recipe`). This suffix is added by ModelOpt to represent the AutoQuant hyperparameter attached to that module. In this table we strip the `.quant_recipe` suffix for readability; the underlying module path is the part before that suffix.

|Layer|Num Bits|Sensitivity|Size Cost|
| :--- | :--- | :--- | :--- |
|layers.0.mlp.down_proj|8.0|3.231e-02|1.245e+07|
|layers.4.mlp.gate_proj|8.0|2.127e-02|2.490e+07|
|layers.3.mlp.gate_proj|8.0|1.447e-02|2.490e+07|
|layers.5.mlp.gate_proj|8.0|1.419e-02|2.490e+07|
|layers.6.mlp.gate_proj|8.0|1.080e-02|2.490e+07|
|layers.2.mlp.gate_proj|8.0|9.612e-03|2.490e+07|
|layers.7.mlp.gate_proj|8.0|9.445e-03|2.490e+07|
|layers.4.mlp.down_proj|8.0|8.908e-03|1.245e+07|
|layers.5.mlp.down_proj|8.0|7.588e-03|1.245e+07|
|layers.14.mlp.gate_proj|8.0|7.561e-03|2.490e+07|
|layers.0.mlp.gate_proj|8.0|7.329e-03|2.490e+07|
|layers.8.mlp.gate_proj|8.0|7.109e-03|2.490e+07|
|layers.10.mlp.gate_proj|8.0|7.061e-03|2.490e+07|
|layers.13.mlp.gate_proj|8.0|6.895e-03|2.490e+07|
|layers.15.mlp.gate_proj|8.0|6.847e-03|2.490e+07|
|layers.12.mlp.gate_proj|8.0|6.667e-03|2.490e+07|
|layers.9.mlp.gate_proj|8.0|6.601e-03|2.490e+07|
|layers.11.mlp.gate_proj|8.0|5.466e-03|2.490e+07|
|layers.6.mlp.down_proj|8.0|5.000e-03|1.245e+07|
|layers.13.mlp.down_proj|8.0|4.863e-03|1.245e+07|
|layers.16.mlp.gate_proj|8.0|4.851e-03|2.490e+07|
|layers.1.mlp.gate_proj|8.0|4.807e-03|2.490e+07|
|layers.3.mlp.down_proj|8.0|4.749e-03|1.245e+07|
|layers.14.mlp.down_proj|8.0|4.619e-03|1.245e+07|
|layers.10.mlp.down_proj|8.0|4.616e-03|1.245e+07|
|layers.12.mlp.down_proj|8.0|4.463e-03|1.245e+07|
|layers.9.mlp.down_proj|8.0|4.353e-03|1.245e+07|
|layers.7.mlp.down_proj|8.0|3.777e-03|1.245e+07|
|layers.15.mlp.down_proj|8.0|3.737e-03|1.245e+07|
|layers.8.mlp.down_proj|8.0|3.485e-03|1.245e+07|
|layers.11.mlp.down_proj|8.0|3.324e-03|1.245e+07|
|layers.2.mlp.down_proj|8.0|3.278e-03|1.245e+07|
|layers.17.mlp.gate_proj|8.0|3.081e-03|2.490e+07|
|layers.1.mlp.down_proj|8.0|2.660e-03|1.245e+07|
|layers.16.mlp.down_proj|8.0|2.280e-03|1.245e+07|
|layers.18.mlp.gate_proj|8.0|2.236e-03|2.490e+07|
|layers.19.mlp.gate_proj|8.0|1.764e-03|2.490e+07|
|layers.17.mlp.down_proj|8.0|1.402e-03|1.245e+07|
|layers.18.mlp.down_proj|8.0|1.267e-03|1.245e+07|
|layers.20.mlp.gate_proj|8.0|9.544e-04|2.490e+07|
|layers.21.mlp.gate_proj|8.0|8.421e-04|2.490e+07|
|layers.19.mlp.down_proj|8.0|7.888e-04|1.245e+07|
|layers.22.mlp.gate_proj|8.0|7.028e-04|2.490e+07|
|layers.23.mlp.gate_proj|8.0|5.899e-04|2.490e+07|
|layers.22.mlp.down_proj|8.0|5.475e-04|1.245e+07|
|layers.24.mlp.gate_proj|8.0|5.039e-04|2.490e+07|
|layers.20.mlp.down_proj|8.0|4.890e-04|1.245e+07|
|layers.21.mlp.down_proj|8.0|4.552e-04|1.245e+07|
|layers.25.mlp.gate_proj|8.0|3.694e-04|2.490e+07|
|layers.23.mlp.down_proj|8.0|3.282e-04|1.245e+07|
|layers.26.mlp.gate_proj|8.0|2.744e-04|2.490e+07|
|layers.24.mlp.down_proj|8.0|2.359e-04|1.245e+07|
|layers.25.mlp.down_proj|8.0|1.916e-04|1.245e+07|
|layers.27.mlp.gate_proj|8.0|1.899e-04|2.490e+07|
|layers.26.mlp.down_proj|8.0|1.367e-04|1.245e+07|
|layers.28.mlp.gate_proj|8.0|1.203e-04|2.490e+07|
|layers.27.mlp.down_proj|8.0|8.662e-05|1.245e+07|
|layers.29.mlp.gate_proj|8.0|7.920e-05|2.490e+07|
|layers.35.mlp.gate_proj|8.0|6.140e-05|2.490e+07|
|layers.30.mlp.gate_proj|8.0|5.739e-05|2.490e+07|
|layers.28.mlp.down_proj|8.0|5.386e-05|1.245e+07|
|layers.31.mlp.gate_proj|8.0|4.626e-05|2.490e+07|
|layers.29.mlp.down_proj|8.0|3.404e-05|1.245e+07|
|layers.32.mlp.gate_proj|8.0|3.361e-05|2.490e+07|
|layers.32.mlp.down_proj|8.0|3.258e-05|1.245e+07|
|layers.30.mlp.down_proj|8.0|2.625e-05|1.245e+07|
|layers.35.mlp.down_proj|8.0|2.511e-05|1.245e+07|
|layers.34.mlp.gate_proj|8.0|2.463e-05|2.490e+07|
|layers.33.mlp.gate_proj|8.0|2.378e-05|2.490e+07|
|layers.31.mlp.down_proj|8.0|2.281e-05|1.245e+07|
|layers.34.mlp.down_proj|8.0|9.821e-06|1.245e+07|
|layers.33.mlp.down_proj|8.0|8.878e-06|1.245e+07|
|layers.0.self_attn.q_proj|8.0|5.679e-06|7.864e+06|
|layers.9.self_attn.q_proj|8.0|2.375e-06|7.864e+06|
|layers.10.self_attn.q_proj|8.0|2.344e-06|7.864e+06|
|layers.7.self_attn.q_proj|8.0|2.329e-06|7.864e+06|
|layers.21.self_attn.q_proj|8.0|2.160e-06|7.864e+06|
|layers.11.self_attn.q_proj|8.0|2.152e-06|7.864e+06|
|layers.8.self_attn.q_proj|8.0|2.077e-06|7.864e+06|
|layers.22.self_attn.q_proj|8.0|1.730e-06|7.864e+06|
|layers.6.self_attn.q_proj|8.0|1.655e-06|7.864e+06|
|layers.23.self_attn.q_proj|8.0|1.650e-06|7.864e+06|
|layers.14.self_attn.q_proj|8.0|1.625e-06|7.864e+06|
|layers.24.self_attn.q_proj|8.0|1.389e-06|7.864e+06|
|layers.13.self_attn.q_proj|8.0|1.381e-06|7.864e+06|
|layers.0.self_attn.o_proj|8.0|1.325e-06|5.243e+06|
|layers.17.self_attn.q_proj|8.0|1.306e-06|7.864e+06|
|layers.6.self_attn.o_proj|8.0|1.230e-06|5.243e+06|
|layers.5.self_attn.q_proj|8.0|1.216e-06|7.864e+06|
|layers.16.self_attn.q_proj|8.0|1.195e-06|7.864e+06|
|layers.35.self_attn.q_proj|8.0|1.162e-06|7.864e+06|
|layers.26.self_attn.q_proj|8.0|1.091e-06|7.864e+06|
|layers.25.self_attn.q_proj|8.0|1.073e-06|7.864e+06|
|layers.4.self_attn.q_proj|8.0|1.021e-06|7.864e+06|
|layers.30.self_attn.q_proj|8.0|1.011e-06|7.864e+06|
|layers.28.self_attn.q_proj|8.0|9.939e-07|7.864e+06|
|layers.15.self_attn.q_proj|8.0|9.385e-07|7.864e+06|
|layers.34.self_attn.q_proj|8.0|9.250e-07|7.864e+06|
|layers.18.self_attn.q_proj|8.0|9.029e-07|7.864e+06|
|layers.32.self_attn.q_proj|8.0|8.784e-07|7.864e+06|
|layers.3.self_attn.q_proj|8.0|8.725e-07|7.864e+06|
|layers.31.self_attn.q_proj|8.0|8.544e-07|7.864e+06|
|layers.12.self_attn.q_proj|8.0|8.274e-07|7.864e+06|
|layers.29.self_attn.q_proj|8.0|8.264e-07|7.864e+06|
|layers.27.self_attn.q_proj|8.0|7.856e-07|7.864e+06|
|layers.19.self_attn.q_proj|8.0|7.622e-07|7.864e+06|
|layers.8.self_attn.o_proj|8.0|6.927e-07|5.243e+06|
|layers.1.self_attn.q_proj|8.0|6.070e-07|7.864e+06|
|layers.20.self_attn.q_proj|8.0|5.788e-07|7.864e+06|
|layers.10.self_attn.o_proj|8.0|5.238e-07|5.243e+06|
|layers.23.self_attn.o_proj|8.0|5.232e-07|5.243e+06|
|layers.33.self_attn.q_proj|8.0|5.226e-07|7.864e+06|
|layers.12.self_attn.o_proj|8.0|4.732e-07|5.243e+06|
|layers.35.self_attn.o_proj|8.0|4.718e-07|5.243e+06|
|layers.2.self_attn.q_proj|8.0|4.668e-07|7.864e+06|
|layers.14.self_attn.o_proj|8.0|4.515e-07|5.243e+06|
|layers.7.self_attn.o_proj|8.0|4.503e-07|5.243e+06|
|layers.9.self_attn.o_proj|8.0|4.383e-07|5.243e+06|
|layers.1.self_attn.o_proj|8.0|4.342e-07|5.243e+06|
|layers.11.self_attn.o_proj|8.0|4.188e-07|5.243e+06|
|layers.15.self_attn.o_proj|8.0|4.014e-07|5.243e+06|
|layers.22.self_attn.o_proj|8.0|3.976e-07|5.243e+06|
|layers.5.self_attn.o_proj|8.0|3.866e-07|5.243e+06|
|layers.24.self_attn.o_proj|8.0|3.304e-07|5.243e+06|
|layers.4.self_attn.o_proj|8.0|3.107e-07|5.243e+06|
|layers.16.self_attn.o_proj|8.0|2.974e-07|5.243e+06|
|layers.13.self_attn.o_proj|8.0|2.890e-07|5.243e+06|
|layers.18.self_attn.o_proj|8.0|2.651e-07|5.243e+06|
|layers.34.self_attn.o_proj|8.0|2.419e-07|5.243e+06|
|layers.19.self_attn.o_proj|8.0|2.412e-07|5.243e+06|
|layers.17.self_attn.o_proj|8.0|2.402e-07|5.243e+06|
|layers.21.self_attn.o_proj|8.0|2.348e-07|5.243e+06|
|layers.28.self_attn.o_proj|8.0|2.339e-07|5.243e+06|
|layers.20.self_attn.o_proj|8.0|2.090e-07|5.243e+06|
|layers.27.self_attn.o_proj|8.0|1.950e-07|5.243e+06|
|layers.25.self_attn.o_proj|8.0|1.926e-07|5.243e+06|
|layers.26.self_attn.o_proj|8.0|1.925e-07|5.243e+06|
|layers.3.self_attn.o_proj|8.0|1.924e-07|5.243e+06|
|layers.31.self_attn.o_proj|8.0|1.770e-07|5.243e+06|
|layers.30.self_attn.o_proj|8.0|1.766e-07|5.243e+06|
|layers.29.self_attn.o_proj|8.0|1.671e-07|5.243e+06|
|layers.2.self_attn.o_proj|8.0|1.577e-07|5.243e+06|
|layers.32.self_attn.o_proj|8.0|1.362e-07|5.243e+06|
|layers.33.self_attn.o_proj|8.0|8.626e-08|5.243e+06|
