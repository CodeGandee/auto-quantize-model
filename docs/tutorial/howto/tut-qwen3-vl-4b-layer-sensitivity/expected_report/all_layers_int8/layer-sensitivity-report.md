
AutoQuant Layer Sensitivity (int8_autoquant_all_layers_int8)
============================================================

## Summary


|Key|Value|
| :--- | :--- |
|Scheme|`int8_autoquant_all_layers_int8`|
|Model|`<ABSOLUTE_PATH>-quantize-model<ABSOLUTE_PATH>-VL-4B-Instruct`|
|Effective bits (from search)|`8.0000`|
|Total AutoQuant score|`1.050409e+03`|
|Constraint satisfied|`True`|

## Quantization


|Key|Value|
| :--- | :--- |
|Base format|`INT8_ALL_LAYERS_CFG`|

## Layer Sensitivity Table


Sorted by sensitivity (descending). Layer names are AutoQuant recipe handles; a trailing `.quant_recipe` suffix (if present) is stripped for readability.

|Layer|Num Bits|Sensitivity|Size Cost|
| :--- | :--- | :--- | :--- |
|model.visual.blocks.0.mlp.linear_fc2|8.0|8.126e+02|2.097e+06|
|model.visual.blocks.1.mlp.linear_fc2|8.0|1.036e+02|2.097e+06|
|model.visual.blocks.11.mlp.linear_fc2|8.0|4.447e+01|2.097e+06|
|model.visual.blocks.0.mlp.linear_fc1|8.0|2.935e+01|2.097e+06|
|model.visual.blocks.2.mlp.linear_fc2|8.0|1.610e+01|2.097e+06|
|model.visual.patch_embed.proj|8.0|5.393e+00|7.864e+05|
|model.visual.blocks.0.attn.proj|8.0|5.256e+00|5.243e+05|
|model.visual.blocks.10.mlp.linear_fc2|8.0|4.223e+00|2.097e+06|
|model.language_model.layers.4.mlp.down_proj|8.0|3.724e+00|1.245e+07|
|model.language_model.layers.6.mlp.gate_proj|8.0|3.236e+00|2.490e+07|
|model.language_model.layers.6.mlp.down_proj|8.0|2.500e+00|1.245e+07|
|model.visual.blocks.1.attn.proj|8.0|2.480e+00|5.243e+05|
|model.visual.blocks.1.mlp.linear_fc1|8.0|1.831e+00|2.097e+06|
|model.visual.blocks.12.mlp.linear_fc2|8.0|1.662e+00|2.097e+06|
|model.language_model.layers.1.mlp.down_proj|8.0|1.040e+00|1.245e+07|
|model.visual.blocks.11.mlp.linear_fc1|8.0|9.091e-01|2.097e+06|
|model.visual.blocks.6.mlp.linear_fc2|8.0|8.163e-01|2.097e+06|
|model.visual.blocks.3.mlp.linear_fc2|8.0|7.653e-01|2.097e+06|
|model.language_model.layers.2.mlp.down_proj|8.0|6.339e-01|1.245e+07|
|model.visual.blocks.4.mlp.linear_fc2|8.0|6.176e-01|2.097e+06|
|model.visual.blocks.7.mlp.linear_fc2|8.0|5.558e-01|2.097e+06|
|model.visual.blocks.5.mlp.linear_fc1|8.0|4.870e-01|2.097e+06|
|model.language_model.layers.3.mlp.down_proj|8.0|4.340e-01|1.245e+07|
|model.visual.blocks.4.mlp.linear_fc1|8.0|4.264e-01|2.097e+06|
|model.visual.blocks.3.mlp.linear_fc1|8.0|3.787e-01|2.097e+06|
|model.visual.blocks.2.mlp.linear_fc1|8.0|3.632e-01|2.097e+06|
|model.visual.blocks.13.mlp.linear_fc2|8.0|3.181e-01|2.097e+06|
|model.visual.blocks.6.mlp.linear_fc1|8.0|2.630e-01|2.097e+06|
|model.language_model.layers.34.mlp.down_proj|8.0|2.620e-01|1.245e+07|
|model.visual.blocks.8.mlp.linear_fc1|8.0|2.573e-01|2.097e+06|
|model.visual.blocks.7.mlp.linear_fc1|8.0|2.351e-01|2.097e+06|
|model.visual.blocks.15.mlp.linear_fc1|8.0|2.167e-01|2.097e+06|
|model.language_model.layers.8.mlp.down_proj|8.0|2.114e-01|1.245e+07|
|model.visual.blocks.10.mlp.linear_fc1|8.0|2.061e-01|2.097e+06|
|model.language_model.layers.12.mlp.down_proj|8.0|1.839e-01|1.245e+07|
|model.visual.blocks.9.mlp.linear_fc2|8.0|1.703e-01|2.097e+06|
|model.visual.blocks.8.mlp.linear_fc2|8.0|1.660e-01|2.097e+06|
|model.visual.deepstack_merger_list.2.linear_fc1|8.0|1.591e-01|8.389e+06|
|model.visual.blocks.5.mlp.linear_fc2|8.0|1.579e-01|2.097e+06|
|model.language_model.layers.16.mlp.down_proj|8.0|1.410e-01|1.245e+07|
|model.visual.blocks.9.mlp.linear_fc1|8.0|1.332e-01|2.097e+06|
|model.visual.blocks.12.mlp.linear_fc1|8.0|1.256e-01|2.097e+06|
|model.visual.blocks.1.attn.qkv|8.0|1.255e-01|1.573e+06|
|model.visual.blocks.2.attn.proj|8.0|1.189e-01|5.243e+05|
|model.visual.blocks.14.mlp.linear_fc2|8.0|1.170e-01|2.097e+06|
|model.language_model.layers.5.mlp.down_proj|8.0|1.145e-01|1.245e+07|
|model.language_model.layers.10.mlp.down_proj|8.0|1.079e-01|1.245e+07|
|model.visual.blocks.16.mlp.linear_fc1|8.0|1.054e-01|2.097e+06|
|model.language_model.layers.14.mlp.down_proj|8.0|1.016e-01|1.245e+07|
|model.visual.blocks.5.attn.qkv|8.0|9.995e-02|1.573e+06|
|model.visual.blocks.0.attn.qkv|8.0|7.858e-02|1.573e+06|
|model.language_model.layers.11.mlp.down_proj|8.0|7.735e-02|1.245e+07|
|model.visual.blocks.4.attn.qkv|8.0|7.690e-02|1.573e+06|
|model.language_model.layers.7.mlp.gate_proj|8.0|7.542e-02|2.490e+07|
|model.visual.blocks.14.mlp.linear_fc1|8.0|7.511e-02|2.097e+06|
|model.visual.blocks.17.mlp.linear_fc1|8.0|7.097e-02|2.097e+06|
|model.visual.blocks.13.mlp.linear_fc1|8.0|7.094e-02|2.097e+06|
|model.visual.deepstack_merger_list.1.linear_fc1|8.0|6.509e-02|8.389e+06|
|model.visual.blocks.2.attn.qkv|8.0|6.475e-02|1.573e+06|
|model.visual.merger.linear_fc1|8.0|6.045e-02|8.389e+06|
|model.language_model.layers.0.mlp.down_proj|8.0|5.178e-02|1.245e+07|
|model.language_model.layers.7.mlp.down_proj|8.0|4.853e-02|1.245e+07|
|model.visual.blocks.23.mlp.linear_fc2|8.0|4.728e-02|2.097e+06|
|model.language_model.layers.15.mlp.down_proj|8.0|4.360e-02|1.245e+07|
|model.visual.blocks.7.attn.qkv|8.0|3.992e-02|1.573e+06|
|model.language_model.layers.9.mlp.down_proj|8.0|3.867e-02|1.245e+07|
|model.visual.blocks.6.attn.qkv|8.0|3.835e-02|1.573e+06|
|model.language_model.layers.35.mlp.gate_proj|8.0|3.820e-02|2.490e+07|
|model.visual.blocks.23.mlp.linear_fc1|8.0|3.737e-02|2.097e+06|
|model.visual.blocks.3.attn.qkv|8.0|3.682e-02|1.573e+06|
|model.visual.blocks.3.attn.proj|8.0|3.469e-02|5.243e+05|
|model.language_model.layers.9.mlp.gate_proj|8.0|3.246e-02|2.490e+07|
|model.language_model.layers.35.mlp.down_proj|8.0|3.225e-02|1.245e+07|
|model.visual.blocks.15.mlp.linear_fc2|8.0|3.204e-02|2.097e+06|
|model.language_model.layers.18.mlp.down_proj|8.0|3.186e-02|1.245e+07|
|model.visual.merger.linear_fc2|8.0|3.145e-02|5.243e+06|
|model.language_model.layers.5.mlp.gate_proj|8.0|3.051e-02|2.490e+07|
|model.language_model.layers.6.self_attn.q_proj|8.0|2.996e-02|7.864e+06|
|model.visual.blocks.12.attn.qkv|8.0|2.972e-02|1.573e+06|
|model.language_model.layers.8.mlp.gate_proj|8.0|2.957e-02|2.490e+07|
|model.visual.blocks.11.attn.proj|8.0|2.734e-02|5.243e+05|
|model.language_model.layers.10.mlp.gate_proj|8.0|2.644e-02|2.490e+07|
|model.language_model.layers.2.mlp.gate_proj|8.0|2.486e-02|2.490e+07|
|model.visual.blocks.8.attn.qkv|8.0|2.450e-02|1.573e+06|
|model.visual.blocks.18.mlp.linear_fc1|8.0|2.368e-02|2.097e+06|
|model.language_model.layers.4.mlp.gate_proj|8.0|2.344e-02|2.490e+07|
|model.visual.blocks.19.mlp.linear_fc1|8.0|2.325e-02|2.097e+06|
|model.language_model.layers.11.mlp.gate_proj|8.0|2.229e-02|2.490e+07|
|model.language_model.layers.8.self_attn.q_proj|8.0|2.174e-02|7.864e+06|
|model.language_model.layers.13.self_attn.q_proj|8.0|2.147e-02|7.864e+06|
|model.visual.blocks.11.attn.qkv|8.0|2.082e-02|1.573e+06|
|model.visual.blocks.21.mlp.linear_fc1|8.0|2.000e-02|2.097e+06|
|model.language_model.layers.3.mlp.gate_proj|8.0|1.979e-02|2.490e+07|
|model.visual.blocks.9.attn.qkv|8.0|1.918e-02|1.573e+06|
|model.visual.blocks.6.attn.proj|8.0|1.905e-02|5.243e+05|
|model.language_model.layers.12.self_attn.q_proj|8.0|1.897e-02|7.864e+06|
|model.visual.deepstack_merger_list.2.linear_fc2|8.0|1.845e-02|5.243e+06|
|model.visual.blocks.7.attn.proj|8.0|1.807e-02|5.243e+05|
|model.language_model.layers.10.self_attn.q_proj|8.0|1.752e-02|7.864e+06|
|model.language_model.layers.7.self_attn.q_proj|8.0|1.732e-02|7.864e+06|
|model.visual.blocks.4.attn.proj|8.0|1.717e-02|5.243e+05|
|model.visual.blocks.10.attn.qkv|8.0|1.613e-02|1.573e+06|
|model.visual.blocks.9.attn.proj|8.0|1.599e-02|5.243e+05|
|model.language_model.layers.9.self_attn.q_proj|8.0|1.585e-02|7.864e+06|
|model.language_model.layers.14.self_attn.q_proj|8.0|1.578e-02|7.864e+06|
|model.visual.blocks.13.attn.qkv|8.0|1.498e-02|1.573e+06|
|model.language_model.layers.11.self_attn.q_proj|8.0|1.495e-02|7.864e+06|
|model.visual.blocks.22.mlp.linear_fc1|8.0|1.413e-02|2.097e+06|
|model.visual.blocks.12.attn.proj|8.0|1.358e-02|5.243e+05|
|model.visual.blocks.10.attn.proj|8.0|1.322e-02|5.243e+05|
|model.language_model.layers.1.mlp.gate_proj|8.0|1.257e-02|2.490e+07|
|model.language_model.layers.12.mlp.gate_proj|8.0|1.212e-02|2.490e+07|
|model.visual.blocks.5.attn.proj|8.0|1.168e-02|5.243e+05|
|model.language_model.layers.15.self_attn.o_proj|8.0|1.116e-02|5.243e+06|
|model.visual.blocks.13.attn.proj|8.0|1.097e-02|5.243e+05|
|model.language_model.layers.33.mlp.down_proj|8.0|1.090e-02|1.245e+07|
|model.visual.deepstack_merger_list.0.linear_fc1|8.0|1.062e-02|8.389e+06|
|model.language_model.layers.13.mlp.down_proj|8.0|9.763e-03|1.245e+07|
|model.language_model.layers.3.self_attn.q_proj|8.0|9.687e-03|7.864e+06|
|model.language_model.layers.13.self_attn.o_proj|8.0|9.475e-03|5.243e+06|
|model.visual.blocks.8.attn.proj|8.0|9.359e-03|5.243e+05|
|model.language_model.layers.13.mlp.gate_proj|8.0|9.127e-03|2.490e+07|
|model.visual.deepstack_merger_list.1.linear_fc2|8.0|8.690e-03|5.243e+06|
|model.visual.blocks.14.attn.qkv|8.0|8.667e-03|1.573e+06|
|model.language_model.layers.16.mlp.gate_proj|8.0|8.529e-03|2.490e+07|
|model.language_model.layers.34.mlp.gate_proj|8.0|8.482e-03|2.490e+07|
|model.visual.blocks.15.attn.qkv|8.0|7.861e-03|1.573e+06|
|model.language_model.layers.15.self_attn.q_proj|8.0|7.673e-03|7.864e+06|
|model.visual.blocks.16.mlp.linear_fc2|8.0|7.402e-03|2.097e+06|
|model.visual.blocks.20.mlp.linear_fc1|8.0|7.332e-03|2.097e+06|
|model.language_model.layers.35.self_attn.q_proj|8.0|6.774e-03|7.864e+06|
|model.language_model.layers.16.self_attn.q_proj|8.0|6.755e-03|7.864e+06|
|model.visual.blocks.22.attn.qkv|8.0|6.002e-03|1.573e+06|
|model.visual.blocks.21.attn.qkv|8.0|5.993e-03|1.573e+06|
|model.language_model.layers.34.self_attn.q_proj|8.0|5.925e-03|7.864e+06|
|model.language_model.layers.17.self_attn.o_proj|8.0|5.923e-03|5.243e+06|
|model.visual.blocks.16.attn.qkv|8.0|5.480e-03|1.573e+06|
|model.language_model.layers.15.mlp.gate_proj|8.0|4.939e-03|2.490e+07|
|model.language_model.layers.14.mlp.gate_proj|8.0|4.644e-03|2.490e+07|
|model.visual.blocks.17.attn.qkv|8.0|4.572e-03|1.573e+06|
|model.language_model.layers.16.self_attn.o_proj|8.0|4.441e-03|5.243e+06|
|model.visual.blocks.23.attn.qkv|8.0|4.242e-03|1.573e+06|
|model.language_model.layers.28.mlp.down_proj|8.0|4.077e-03|1.245e+07|
|model.language_model.layers.14.self_attn.o_proj|8.0|3.870e-03|5.243e+06|
|model.language_model.layers.17.self_attn.q_proj|8.0|3.863e-03|7.864e+06|
|model.language_model.layers.7.self_attn.o_proj|8.0|3.787e-03|5.243e+06|
|model.language_model.layers.25.mlp.down_proj|8.0|3.502e-03|1.245e+07|
|model.language_model.layers.8.self_attn.o_proj|8.0|3.385e-03|5.243e+06|
|model.visual.blocks.17.mlp.linear_fc2|8.0|3.248e-03|2.097e+06|
|model.language_model.layers.24.mlp.down_proj|8.0|3.180e-03|1.245e+07|
|model.language_model.layers.5.self_attn.q_proj|8.0|3.150e-03|7.864e+06|
|model.language_model.layers.0.self_attn.q_proj|8.0|2.909e-03|7.864e+06|
|model.language_model.layers.4.self_attn.q_proj|8.0|2.822e-03|7.864e+06|
|model.language_model.layers.27.mlp.down_proj|8.0|2.611e-03|1.245e+07|
|model.language_model.layers.33.mlp.gate_proj|8.0|2.605e-03|2.490e+07|
|model.language_model.layers.10.self_attn.o_proj|8.0|2.573e-03|5.243e+06|
|model.visual.deepstack_merger_list.0.linear_fc2|8.0|2.505e-03|5.243e+06|
|model.language_model.layers.33.self_attn.q_proj|8.0|2.467e-03|7.864e+06|
|model.language_model.layers.32.mlp.down_proj|8.0|2.443e-03|1.245e+07|
|model.language_model.layers.6.self_attn.o_proj|8.0|2.423e-03|5.243e+06|
|model.language_model.layers.32.self_attn.q_proj|8.0|2.300e-03|7.864e+06|
|model.language_model.layers.31.mlp.down_proj|8.0|2.181e-03|1.245e+07|
|model.visual.blocks.15.attn.proj|8.0|2.173e-03|5.243e+05|
|model.language_model.layers.9.self_attn.o_proj|8.0|2.119e-03|5.243e+06|
|model.language_model.layers.29.mlp.down_proj|8.0|2.097e-03|1.245e+07|
|model.visual.blocks.21.mlp.linear_fc2|8.0|1.976e-03|2.097e+06|
|model.visual.blocks.19.attn.qkv|8.0|1.943e-03|1.573e+06|
|model.language_model.layers.4.self_attn.o_proj|8.0|1.921e-03|5.243e+06|
|model.language_model.layers.22.self_attn.q_proj|8.0|1.876e-03|7.864e+06|
|model.visual.blocks.20.attn.qkv|8.0|1.818e-03|1.573e+06|
|model.language_model.layers.21.self_attn.q_proj|8.0|1.781e-03|7.864e+06|
|model.language_model.layers.18.self_attn.q_proj|8.0|1.684e-03|7.864e+06|
|model.language_model.layers.23.self_attn.q_proj|8.0|1.542e-03|7.864e+06|
|model.language_model.layers.2.self_attn.q_proj|8.0|1.517e-03|7.864e+06|
|model.language_model.layers.19.mlp.down_proj|8.0|1.510e-03|1.245e+07|
|model.language_model.layers.23.mlp.down_proj|8.0|1.444e-03|1.245e+07|
|model.visual.blocks.17.attn.proj|8.0|1.435e-03|5.243e+05|
|model.language_model.layers.19.self_attn.q_proj|8.0|1.311e-03|7.864e+06|
|model.language_model.layers.30.mlp.down_proj|8.0|1.301e-03|1.245e+07|
|model.language_model.layers.17.mlp.down_proj|8.0|1.248e-03|1.245e+07|
|model.language_model.layers.11.self_attn.o_proj|8.0|1.238e-03|5.243e+06|
|model.language_model.layers.17.mlp.gate_proj|8.0|1.235e-03|2.490e+07|
|model.visual.blocks.14.attn.proj|8.0|1.227e-03|5.243e+05|
|model.language_model.layers.21.mlp.down_proj|8.0|1.204e-03|1.245e+07|
|model.visual.blocks.18.attn.qkv|8.0|1.203e-03|1.573e+06|
|model.language_model.layers.20.self_attn.q_proj|8.0|1.193e-03|7.864e+06|
|model.visual.blocks.21.attn.proj|8.0|1.183e-03|5.243e+05|
|model.language_model.layers.5.self_attn.o_proj|8.0|1.156e-03|5.243e+06|
|model.language_model.layers.30.self_attn.q_proj|8.0|1.074e-03|7.864e+06|
|model.language_model.layers.24.self_attn.q_proj|8.0|1.003e-03|7.864e+06|
|model.visual.blocks.18.attn.proj|8.0|9.866e-04|5.243e+05|
|model.language_model.layers.22.mlp.down_proj|8.0|9.786e-04|1.245e+07|
|model.language_model.layers.31.self_attn.q_proj|8.0|9.377e-04|7.864e+06|
|model.language_model.layers.0.self_attn.o_proj|8.0|8.983e-04|5.243e+06|
|model.language_model.layers.20.mlp.gate_proj|8.0|8.917e-04|2.490e+07|
|model.language_model.layers.18.mlp.gate_proj|8.0|8.880e-04|2.490e+07|
|model.language_model.layers.27.self_attn.q_proj|8.0|8.516e-04|7.864e+06|
|model.visual.blocks.19.mlp.linear_fc2|8.0|8.462e-04|2.097e+06|
|model.visual.blocks.16.attn.proj|8.0|8.271e-04|5.243e+05|
|model.language_model.layers.12.self_attn.o_proj|8.0|8.239e-04|5.243e+06|
|model.language_model.layers.26.self_attn.q_proj|8.0|7.882e-04|7.864e+06|
|model.language_model.layers.19.mlp.gate_proj|8.0|7.574e-04|2.490e+07|
|model.language_model.layers.1.self_attn.q_proj|8.0|7.310e-04|7.864e+06|
|model.language_model.layers.29.self_attn.q_proj|8.0|7.063e-04|7.864e+06|
|model.language_model.layers.25.self_attn.q_proj|8.0|6.568e-04|7.864e+06|
|model.language_model.layers.21.mlp.gate_proj|8.0|6.365e-04|2.490e+07|
|model.visual.blocks.22.mlp.linear_fc2|8.0|6.042e-04|2.097e+06|
|model.visual.blocks.19.attn.proj|8.0|6.016e-04|5.243e+05|
|model.language_model.layers.28.self_attn.q_proj|8.0|5.826e-04|7.864e+06|
|model.language_model.layers.22.mlp.gate_proj|8.0|5.321e-04|2.490e+07|
|model.visual.blocks.18.mlp.linear_fc2|8.0|4.990e-04|2.097e+06|
|model.language_model.layers.26.mlp.down_proj|8.0|4.944e-04|1.245e+07|
|model.language_model.layers.3.self_attn.o_proj|8.0|4.851e-04|5.243e+06|
|model.visual.blocks.20.attn.proj|8.0|4.757e-04|5.243e+05|
|model.language_model.layers.20.mlp.down_proj|8.0|4.750e-04|1.245e+07|
|lm_head|8.0|4.593e-04|1.945e+08|
|model.language_model.layers.25.mlp.gate_proj|8.0|4.554e-04|2.490e+07|
|model.visual.blocks.20.mlp.linear_fc2|8.0|4.387e-04|2.097e+06|
|model.language_model.layers.23.mlp.gate_proj|8.0|4.059e-04|2.490e+07|
|model.visual.blocks.22.attn.proj|8.0|3.895e-04|5.243e+05|
|model.language_model.layers.24.mlp.gate_proj|8.0|3.886e-04|2.490e+07|
|model.language_model.layers.26.mlp.gate_proj|8.0|3.442e-04|2.490e+07|
|model.language_model.layers.0.mlp.gate_proj|8.0|3.278e-04|2.490e+07|
|model.language_model.layers.27.mlp.gate_proj|8.0|3.059e-04|2.490e+07|
|model.visual.blocks.23.attn.proj|8.0|2.701e-04|5.243e+05|
|model.language_model.layers.28.mlp.gate_proj|8.0|2.264e-04|2.490e+07|
|model.language_model.layers.29.mlp.gate_proj|8.0|1.944e-04|2.490e+07|
|model.language_model.layers.33.self_attn.o_proj|8.0|1.625e-04|5.243e+06|
|model.language_model.layers.31.mlp.gate_proj|8.0|1.599e-04|2.490e+07|
|model.language_model.layers.18.self_attn.o_proj|8.0|1.481e-04|5.243e+06|
|model.language_model.layers.30.mlp.gate_proj|8.0|1.374e-04|2.490e+07|
|model.language_model.layers.32.mlp.gate_proj|8.0|1.361e-04|2.490e+07|
|model.language_model.layers.19.self_attn.o_proj|8.0|1.324e-04|5.243e+06|
|model.language_model.layers.2.self_attn.o_proj|8.0|1.226e-04|5.243e+06|
|model.language_model.layers.1.self_attn.o_proj|8.0|1.198e-04|5.243e+06|
|model.language_model.layers.20.self_attn.o_proj|8.0|1.169e-04|5.243e+06|
|model.language_model.layers.30.self_attn.o_proj|8.0|1.114e-04|5.243e+06|
|model.language_model.layers.34.self_attn.o_proj|8.0|1.078e-04|5.243e+06|
|model.language_model.layers.27.self_attn.o_proj|8.0|1.060e-04|5.243e+06|
|model.language_model.layers.28.self_attn.o_proj|8.0|1.035e-04|5.243e+06|
|model.language_model.layers.31.self_attn.o_proj|8.0|7.341e-05|5.243e+06|
|model.language_model.layers.35.self_attn.o_proj|8.0|6.860e-05|5.243e+06|
|model.language_model.layers.29.self_attn.o_proj|8.0|6.701e-05|5.243e+06|
|model.language_model.layers.22.self_attn.o_proj|8.0|6.498e-05|5.243e+06|
|model.language_model.layers.21.self_attn.o_proj|8.0|6.426e-05|5.243e+06|
|model.language_model.layers.24.self_attn.o_proj|8.0|6.138e-05|5.243e+06|
|model.language_model.layers.23.self_attn.o_proj|8.0|5.576e-05|5.243e+06|
|model.language_model.layers.32.self_attn.o_proj|8.0|5.310e-05|5.243e+06|
|model.language_model.layers.26.self_attn.o_proj|8.0|4.351e-05|5.243e+06|
|model.language_model.layers.25.self_attn.o_proj|8.0|3.071e-05|5.243e+06|

## Composed Config (`composed-config.yaml`)


```yaml
script: run_qwen3_vl_4b_autoquant_all_layers.py
scheme:
  name: int8_autoquant_all_layers_int8
  auto_quantize_bits: 8.0
  auto_quantize_method: gradient
  auto_quantize_score_size: 1
  coverage_mode: full
  coverage_fraction: 1.0
  quant_formats:
  - INT8_ALL_LAYERS_CFG
args:
  model_dir: <ABSOLUTE_PATH>-quantize-model<ABSOLUTE_PATH>-VL-4B-Instruct
  output_dir: <ABSOLUTE_PATH>-quantize-model<ABSOLUTE_PATH>
  vlm_calib_db: <ABSOLUTE_PATH>-quantize-model<ABSOLUTE_PATH>
  coco_root: <ABSOLUTE_PATH>-quantize-model<ABSOLUTE_PATH>-data
  max_calib_samples: 1
  calib_seq_len: 64
  batch_size: 1
  device: cuda:0
  quant_format: int8
  effective_bits: null
  auto_quantize_score_size: 1
  report_only: false
dataset:
  vlm_calib_db: <ABSOLUTE_PATH>-quantize-model<ABSOLUTE_PATH>
  coco_root: <ABSOLUTE_PATH>-quantize-model<ABSOLUTE_PATH>-data
  calib_seq_len: 64
  batch_size: 1
  num_calib_samples: null
  max_calib_samples: 1
quantization:
  base_format_name: INT8_ALL_LAYERS_CFG
  format_names:
  - INT8_ALL_LAYERS_CFG
  quant_format: int8
```