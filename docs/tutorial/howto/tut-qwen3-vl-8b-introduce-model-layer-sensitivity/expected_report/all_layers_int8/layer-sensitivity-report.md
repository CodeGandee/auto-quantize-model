
AutoQuant Layer Sensitivity (int8_autoquant_all_layers_int8)
============================================================

## Summary


|Key|Value|
| :--- | :--- |
|Scheme|`int8_autoquant_all_layers_int8`|
|Model|`<ABSOLUTE_PATH>-quantize-model<ABSOLUTE_PATH>-VL-8B-Instruct`|
|Effective bits (from search)|`8.0000`|
|Total AutoQuant score|`2.298313e+01`|
|Constraint satisfied|`False`|

## Quantization


|Key|Value|
| :--- | :--- |
|Base format|`INT8_ALL_LAYERS_CFG`|

## Layer Sensitivity Table


Sorted by sensitivity (descending). Layer names are AutoQuant recipe handles; a trailing `.quant_recipe` suffix (if present) is stripped for readability.

|Layer|Num Bits|Sensitivity|Size Cost|
| :--- | :--- | :--- | :--- |
|model.visual.blocks.0.mlp.linear_fc2|8.0|1.216e+01|2.479e+06|
|model.visual.blocks.9.mlp.linear_fc2|8.0|2.486e+00|2.479e+06|
|model.visual.blocks.0.mlp.linear_fc1|8.0|2.117e+00|2.479e+06|
|model.visual.patch_embed.proj|8.0|1.410e+00|8.847e+05|
|model.language_model.layers.6.mlp.gate_proj|8.0|8.451e-01|5.033e+07|
|model.visual.blocks.2.mlp.linear_fc2|8.0|5.577e-01|2.479e+06|
|model.visual.blocks.0.attn.proj|8.0|4.411e-01|6.636e+05|
|model.language_model.layers.6.mlp.down_proj|8.0|4.176e-01|2.517e+07|
|model.language_model.layers.34.mlp.down_proj|8.0|3.820e-01|2.517e+07|
|model.language_model.layers.16.mlp.down_proj|8.0|1.963e-01|2.517e+07|
|model.language_model.layers.34.mlp.gate_proj|8.0|1.504e-01|5.033e+07|
|model.visual.blocks.1.mlp.linear_fc2|8.0|1.104e-01|2.479e+06|
|model.language_model.layers.14.mlp.down_proj|8.0|1.102e-01|2.517e+07|
|model.language_model.layers.34.self_attn.q_proj|8.0|1.054e-01|1.258e+07|
|model.visual.blocks.1.attn.proj|8.0|9.852e-02|6.636e+05|
|model.language_model.layers.35.mlp.gate_proj|8.0|9.261e-02|5.033e+07|
|model.language_model.layers.12.mlp.down_proj|8.0|8.928e-02|2.517e+07|
|model.visual.blocks.3.mlp.linear_fc2|8.0|7.501e-02|2.479e+06|
|model.language_model.layers.18.mlp.down_proj|8.0|6.065e-02|2.517e+07|
|model.language_model.layers.35.mlp.down_proj|8.0|5.238e-02|2.517e+07|
|model.language_model.layers.15.mlp.down_proj|8.0|4.728e-02|2.517e+07|
|model.language_model.layers.1.mlp.down_proj|8.0|4.534e-02|2.517e+07|
|model.language_model.layers.7.mlp.gate_proj|8.0|4.336e-02|5.033e+07|
|model.visual.blocks.4.mlp.linear_fc2|8.0|3.470e-02|2.479e+06|
|model.language_model.layers.33.self_attn.q_proj|8.0|2.835e-02|1.258e+07|
|model.visual.blocks.10.mlp.linear_fc2|8.0|2.690e-02|2.479e+06|
|model.language_model.layers.35.self_attn.q_proj|8.0|2.409e-02|1.258e+07|
|model.language_model.layers.5.mlp.gate_proj|8.0|2.119e-02|5.033e+07|
|model.language_model.layers.31.mlp.down_proj|8.0|2.088e-02|2.517e+07|
|model.language_model.layers.16.mlp.gate_proj|8.0|1.953e-02|5.033e+07|
|model.visual.blocks.1.mlp.linear_fc1|8.0|1.868e-02|2.479e+06|
|model.language_model.layers.3.mlp.down_proj|8.0|1.788e-02|2.517e+07|
|model.visual.blocks.7.mlp.linear_fc2|8.0|1.780e-02|2.479e+06|
|model.language_model.layers.4.mlp.down_proj|8.0|1.715e-02|2.517e+07|
|model.language_model.layers.2.mlp.down_proj|8.0|1.623e-02|2.517e+07|
|model.language_model.layers.16.self_attn.q_proj|8.0|1.499e-02|1.258e+07|
|model.language_model.layers.5.mlp.down_proj|8.0|1.473e-02|2.517e+07|
|model.visual.deepstack_merger_list.1.linear_fc1|8.0|1.420e-02|1.062e+07|
|model.visual.blocks.5.mlp.linear_fc2|8.0|1.386e-02|2.479e+06|
|model.visual.blocks.7.mlp.linear_fc1|8.0|1.370e-02|2.479e+06|
|model.language_model.layers.22.self_attn.q_proj|8.0|1.366e-02|1.258e+07|
|model.language_model.layers.17.self_attn.q_proj|8.0|1.292e-02|1.258e+07|
|model.visual.blocks.6.mlp.linear_fc2|8.0|1.291e-02|2.479e+06|
|model.language_model.layers.2.mlp.gate_proj|8.0|1.283e-02|5.033e+07|
|model.language_model.layers.10.mlp.down_proj|8.0|1.235e-02|2.517e+07|
|model.language_model.layers.0.mlp.down_proj|8.0|1.175e-02|2.517e+07|
|model.visual.blocks.1.attn.qkv|8.0|1.112e-02|1.991e+06|
|model.language_model.layers.8.mlp.down_proj|8.0|1.076e-02|2.517e+07|
|model.language_model.layers.3.mlp.gate_proj|8.0|1.076e-02|5.033e+07|
|model.language_model.layers.4.mlp.gate_proj|8.0|1.032e-02|5.033e+07|
|model.language_model.layers.13.mlp.down_proj|8.0|1.028e-02|2.517e+07|
|model.visual.blocks.3.mlp.linear_fc1|8.0|1.026e-02|2.479e+06|
|model.language_model.layers.15.self_attn.o_proj|8.0|9.515e-03|8.389e+06|
|model.language_model.layers.9.mlp.down_proj|8.0|9.032e-03|2.517e+07|
|model.language_model.layers.28.mlp.down_proj|8.0|8.860e-03|2.517e+07|
|model.language_model.layers.32.self_attn.q_proj|8.0|8.564e-03|1.258e+07|
|model.language_model.layers.27.mlp.down_proj|8.0|8.259e-03|2.517e+07|
|model.visual.blocks.2.attn.qkv|8.0|8.092e-03|1.991e+06|
|model.visual.deepstack_merger_list.2.linear_fc1|8.0|8.061e-03|1.062e+07|
|model.language_model.layers.30.self_attn.q_proj|8.0|7.886e-03|1.258e+07|
|model.visual.blocks.8.mlp.linear_fc1|8.0|7.759e-03|2.479e+06|
|model.visual.blocks.4.mlp.linear_fc1|8.0|7.748e-03|2.479e+06|
|model.visual.blocks.5.mlp.linear_fc1|8.0|7.720e-03|2.479e+06|
|model.visual.blocks.6.mlp.linear_fc1|8.0|6.886e-03|2.479e+06|
|model.language_model.layers.17.self_attn.o_proj|8.0|6.743e-03|8.389e+06|
|model.visual.blocks.10.mlp.linear_fc1|8.0|6.393e-03|2.479e+06|
|model.visual.blocks.2.mlp.linear_fc1|8.0|6.064e-03|2.479e+06|
|model.language_model.layers.15.mlp.gate_proj|8.0|6.049e-03|5.033e+07|
|model.language_model.layers.32.mlp.down_proj|8.0|5.599e-03|2.517e+07|
|model.language_model.layers.24.mlp.down_proj|8.0|5.571e-03|2.517e+07|
|model.language_model.layers.33.mlp.down_proj|8.0|5.402e-03|2.517e+07|
|model.language_model.layers.23.mlp.down_proj|8.0|5.337e-03|2.517e+07|
|model.language_model.layers.19.mlp.down_proj|8.0|5.257e-03|2.517e+07|
|model.language_model.layers.29.self_attn.q_proj|8.0|5.179e-03|1.258e+07|
|model.language_model.layers.18.self_attn.q_proj|8.0|5.033e-03|1.258e+07|
|model.language_model.layers.7.mlp.down_proj|8.0|4.948e-03|2.517e+07|
|model.language_model.layers.1.mlp.gate_proj|8.0|4.870e-03|5.033e+07|
|model.visual.blocks.2.attn.proj|8.0|4.746e-03|6.636e+05|
|model.language_model.layers.21.self_attn.q_proj|8.0|4.616e-03|1.258e+07|
|model.language_model.layers.30.mlp.down_proj|8.0|4.437e-03|2.517e+07|
|model.language_model.layers.15.self_attn.q_proj|8.0|4.436e-03|1.258e+07|
|model.language_model.layers.14.self_attn.q_proj|8.0|4.378e-03|1.258e+07|
|model.language_model.layers.33.mlp.gate_proj|8.0|4.343e-03|5.033e+07|
|model.visual.blocks.9.mlp.linear_fc1|8.0|4.334e-03|2.479e+06|
|model.language_model.layers.25.mlp.down_proj|8.0|4.230e-03|2.517e+07|
|model.language_model.layers.29.mlp.down_proj|8.0|4.068e-03|2.517e+07|
|model.visual.blocks.8.mlp.linear_fc2|8.0|4.061e-03|2.479e+06|
|model.language_model.layers.6.self_attn.q_proj|8.0|4.053e-03|1.258e+07|
|model.visual.blocks.22.mlp.linear_fc2|8.0|4.030e-03|2.479e+06|
|model.visual.blocks.0.attn.qkv|8.0|4.010e-03|1.991e+06|
|model.language_model.layers.28.self_attn.q_proj|8.0|3.999e-03|1.258e+07|
|model.language_model.layers.14.mlp.gate_proj|8.0|3.999e-03|5.033e+07|
|model.language_model.layers.14.self_attn.o_proj|8.0|3.855e-03|8.389e+06|
|model.visual.blocks.26.mlp.linear_fc2|8.0|3.802e-03|2.479e+06|
|model.language_model.layers.23.self_attn.q_proj|8.0|3.757e-03|1.258e+07|
|model.language_model.layers.13.self_attn.q_proj|8.0|3.706e-03|1.258e+07|
|model.language_model.layers.31.self_attn.q_proj|8.0|3.599e-03|1.258e+07|
|model.visual.blocks.21.mlp.linear_fc2|8.0|3.540e-03|2.479e+06|
|model.visual.blocks.14.mlp.linear_fc2|8.0|3.407e-03|2.479e+06|
|model.language_model.layers.20.self_attn.q_proj|8.0|3.363e-03|1.258e+07|
|model.language_model.layers.26.self_attn.q_proj|8.0|3.342e-03|1.258e+07|
|model.language_model.layers.24.self_attn.q_proj|8.0|3.291e-03|1.258e+07|
|model.visual.blocks.3.attn.proj|8.0|3.280e-03|6.636e+05|
|model.visual.blocks.12.attn.qkv|8.0|3.219e-03|1.991e+06|
|model.language_model.layers.21.mlp.down_proj|8.0|3.081e-03|2.517e+07|
|model.language_model.layers.25.self_attn.q_proj|8.0|3.059e-03|1.258e+07|
|model.language_model.layers.11.mlp.down_proj|8.0|2.984e-03|2.517e+07|
|model.visual.blocks.15.mlp.linear_fc2|8.0|2.933e-03|2.479e+06|
|model.language_model.layers.19.self_attn.q_proj|8.0|2.887e-03|1.258e+07|
|model.language_model.layers.8.mlp.gate_proj|8.0|2.857e-03|5.033e+07|
|model.language_model.layers.27.self_attn.q_proj|8.0|2.846e-03|1.258e+07|
|model.language_model.layers.26.mlp.down_proj|8.0|2.844e-03|2.517e+07|
|model.language_model.layers.13.mlp.gate_proj|8.0|2.787e-03|5.033e+07|
|model.visual.blocks.23.mlp.linear_fc2|8.0|2.740e-03|2.479e+06|
|model.language_model.layers.10.self_attn.q_proj|8.0|2.719e-03|1.258e+07|
|model.language_model.layers.9.mlp.gate_proj|8.0|2.371e-03|5.033e+07|
|model.visual.blocks.10.attn.qkv|8.0|2.359e-03|1.991e+06|
|model.language_model.layers.17.mlp.down_proj|8.0|2.356e-03|2.517e+07|
|model.language_model.layers.16.self_attn.o_proj|8.0|2.334e-03|8.389e+06|
|model.language_model.layers.10.mlp.gate_proj|8.0|2.324e-03|5.033e+07|
|model.language_model.layers.22.mlp.down_proj|8.0|2.317e-03|2.517e+07|
|model.language_model.layers.12.mlp.gate_proj|8.0|2.256e-03|5.033e+07|
|model.language_model.layers.17.mlp.gate_proj|8.0|2.208e-03|5.033e+07|
|model.visual.blocks.11.mlp.linear_fc2|8.0|2.179e-03|2.479e+06|
|model.visual.blocks.11.mlp.linear_fc1|8.0|2.150e-03|2.479e+06|
|model.language_model.layers.13.self_attn.o_proj|8.0|2.008e-03|8.389e+06|
|model.language_model.layers.11.mlp.gate_proj|8.0|2.008e-03|5.033e+07|
|model.visual.blocks.3.attn.qkv|8.0|2.002e-03|1.991e+06|
|model.language_model.layers.8.self_attn.q_proj|8.0|1.968e-03|1.258e+07|
|model.language_model.layers.20.mlp.down_proj|8.0|1.964e-03|2.517e+07|
|model.visual.blocks.26.mlp.linear_fc1|8.0|1.840e-03|2.479e+06|
|model.visual.blocks.12.mlp.linear_fc2|8.0|1.818e-03|2.479e+06|
|model.language_model.layers.12.self_attn.q_proj|8.0|1.728e-03|1.258e+07|
|model.language_model.layers.18.mlp.gate_proj|8.0|1.723e-03|5.033e+07|
|model.visual.blocks.20.mlp.linear_fc2|8.0|1.712e-03|2.479e+06|
|model.language_model.layers.9.self_attn.q_proj|8.0|1.657e-03|1.258e+07|
|lm_head|8.0|1.640e-03|3.112e+08|
|model.visual.blocks.5.attn.proj|8.0|1.639e-03|6.636e+05|
|model.language_model.layers.20.mlp.gate_proj|8.0|1.561e-03|5.033e+07|
|model.language_model.layers.7.self_attn.q_proj|8.0|1.553e-03|1.258e+07|
|model.language_model.layers.19.mlp.gate_proj|8.0|1.533e-03|5.033e+07|
|model.visual.blocks.25.attn.qkv|8.0|1.413e-03|1.991e+06|
|model.visual.blocks.13.mlp.linear_fc1|8.0|1.353e-03|2.479e+06|
|model.visual.blocks.24.mlp.linear_fc2|8.0|1.322e-03|2.479e+06|
|model.visual.blocks.13.mlp.linear_fc2|8.0|1.301e-03|2.479e+06|
|model.visual.blocks.17.mlp.linear_fc1|8.0|1.229e-03|2.479e+06|
|model.visual.blocks.9.attn.proj|8.0|1.211e-03|6.636e+05|
|model.visual.blocks.8.attn.qkv|8.0|1.206e-03|1.991e+06|
|model.visual.blocks.11.attn.qkv|8.0|1.201e-03|1.991e+06|
|model.language_model.layers.21.mlp.gate_proj|8.0|1.194e-03|5.033e+07|
|model.visual.blocks.4.attn.qkv|8.0|1.144e-03|1.991e+06|
|model.visual.blocks.7.attn.proj|8.0|1.119e-03|6.636e+05|
|model.visual.blocks.13.attn.qkv|8.0|1.115e-03|1.991e+06|
|model.visual.blocks.4.attn.proj|8.0|1.071e-03|6.636e+05|
|model.visual.blocks.6.attn.proj|8.0|1.048e-03|6.636e+05|
|model.language_model.layers.11.self_attn.q_proj|8.0|1.017e-03|1.258e+07|
|model.visual.blocks.5.attn.qkv|8.0|9.903e-04|1.991e+06|
|model.language_model.layers.3.self_attn.q_proj|8.0|9.514e-04|1.258e+07|
|model.language_model.layers.22.mlp.gate_proj|8.0|8.829e-04|5.033e+07|
|model.visual.blocks.18.mlp.linear_fc2|8.0|8.780e-04|2.479e+06|
|model.visual.blocks.8.attn.proj|8.0|8.393e-04|6.636e+05|
|model.visual.blocks.15.attn.qkv|8.0|8.093e-04|1.991e+06|
|model.language_model.layers.5.self_attn.q_proj|8.0|8.009e-04|1.258e+07|
|model.visual.blocks.16.mlp.linear_fc2|8.0|7.820e-04|2.479e+06|
|model.visual.blocks.12.mlp.linear_fc1|8.0|7.651e-04|2.479e+06|
|model.visual.blocks.14.mlp.linear_fc1|8.0|7.476e-04|2.479e+06|
|model.visual.blocks.24.mlp.linear_fc1|8.0|7.313e-04|2.479e+06|
|model.visual.blocks.15.mlp.linear_fc1|8.0|6.811e-04|2.479e+06|
|model.visual.blocks.10.attn.proj|8.0|6.619e-04|6.636e+05|
|model.visual.blocks.17.mlp.linear_fc2|8.0|6.537e-04|2.479e+06|
|model.visual.blocks.20.mlp.linear_fc1|8.0|6.445e-04|2.479e+06|
|model.visual.blocks.9.attn.qkv|8.0|6.205e-04|1.991e+06|
|model.visual.blocks.14.attn.qkv|8.0|6.165e-04|1.991e+06|
|model.visual.blocks.7.attn.qkv|8.0|6.153e-04|1.991e+06|
|model.visual.merger.linear_fc1|8.0|6.129e-04|1.062e+07|
|model.visual.deepstack_merger_list.1.linear_fc2|8.0|6.072e-04|9.437e+06|
|model.visual.blocks.18.mlp.linear_fc1|8.0|5.818e-04|2.479e+06|
|model.language_model.layers.23.mlp.gate_proj|8.0|5.719e-04|5.033e+07|
|model.visual.blocks.23.mlp.linear_fc1|8.0|5.699e-04|2.479e+06|
|model.language_model.layers.1.self_attn.q_proj|8.0|5.602e-04|1.258e+07|
|model.visual.merger.linear_fc2|8.0|5.569e-04|9.437e+06|
|model.visual.deepstack_merger_list.2.linear_fc2|8.0|5.518e-04|9.437e+06|
|model.visual.deepstack_merger_list.0.linear_fc1|8.0|5.228e-04|1.062e+07|
|model.visual.blocks.16.mlp.linear_fc1|8.0|5.098e-04|2.479e+06|
|model.visual.deepstack_merger_list.0.linear_fc2|8.0|5.089e-04|9.437e+06|
|model.visual.blocks.19.mlp.linear_fc1|8.0|5.072e-04|2.479e+06|
|model.language_model.layers.24.mlp.gate_proj|8.0|4.994e-04|5.033e+07|
|model.language_model.layers.25.mlp.gate_proj|8.0|4.855e-04|5.033e+07|
|model.visual.blocks.19.mlp.linear_fc2|8.0|4.703e-04|2.479e+06|
|model.language_model.layers.0.self_attn.q_proj|8.0|4.624e-04|1.258e+07|
|model.visual.blocks.15.attn.proj|8.0|4.542e-04|6.636e+05|
|model.visual.blocks.16.attn.qkv|8.0|4.458e-04|1.991e+06|
|model.visual.blocks.26.attn.qkv|8.0|4.294e-04|1.991e+06|
|model.visual.blocks.25.mlp.linear_fc1|8.0|4.197e-04|2.479e+06|
|model.language_model.layers.27.mlp.gate_proj|8.0|4.147e-04|5.033e+07|
|model.visual.blocks.21.mlp.linear_fc1|8.0|4.073e-04|2.479e+06|
|model.language_model.layers.4.self_attn.q_proj|8.0|3.897e-04|1.258e+07|
|model.language_model.layers.28.mlp.gate_proj|8.0|3.862e-04|5.033e+07|
|model.language_model.layers.2.self_attn.q_proj|8.0|3.837e-04|1.258e+07|
|model.language_model.layers.26.mlp.gate_proj|8.0|3.811e-04|5.033e+07|
|model.language_model.layers.32.mlp.gate_proj|8.0|3.569e-04|5.033e+07|
|model.visual.blocks.11.attn.proj|8.0|3.549e-04|6.636e+05|
|model.visual.blocks.22.mlp.linear_fc1|8.0|3.509e-04|2.479e+06|
|model.visual.blocks.18.attn.qkv|8.0|3.426e-04|1.991e+06|
|model.language_model.layers.31.mlp.gate_proj|8.0|3.328e-04|5.033e+07|
|model.visual.blocks.6.attn.qkv|8.0|3.239e-04|1.991e+06|
|model.visual.blocks.20.attn.qkv|8.0|3.197e-04|1.991e+06|
|model.visual.blocks.12.attn.proj|8.0|3.041e-04|6.636e+05|
|model.language_model.layers.29.mlp.gate_proj|8.0|2.988e-04|5.033e+07|
|model.visual.blocks.17.attn.qkv|8.0|2.975e-04|1.991e+06|
|model.language_model.layers.6.self_attn.o_proj|8.0|2.901e-04|8.389e+06|
|model.language_model.layers.30.mlp.gate_proj|8.0|2.861e-04|5.033e+07|
|model.visual.blocks.24.attn.qkv|8.0|2.757e-04|1.991e+06|
|model.visual.blocks.19.attn.qkv|8.0|2.649e-04|1.991e+06|
|model.language_model.layers.34.self_attn.o_proj|8.0|2.469e-04|8.389e+06|
|model.visual.blocks.21.attn.qkv|8.0|2.375e-04|1.991e+06|
|model.visual.blocks.25.mlp.linear_fc2|8.0|2.307e-04|2.479e+06|
|model.visual.blocks.13.attn.proj|8.0|2.248e-04|6.636e+05|
|model.visual.blocks.23.attn.qkv|8.0|2.109e-04|1.991e+06|
|model.visual.blocks.22.attn.proj|8.0|2.053e-04|6.636e+05|
|model.visual.blocks.21.attn.proj|8.0|1.860e-04|6.636e+05|
|model.visual.blocks.22.attn.qkv|8.0|1.849e-04|1.991e+06|
|model.language_model.layers.33.self_attn.o_proj|8.0|1.771e-04|8.389e+06|
|model.visual.blocks.23.attn.proj|8.0|1.748e-04|6.636e+05|
|model.language_model.layers.21.self_attn.o_proj|8.0|1.684e-04|8.389e+06|
|model.language_model.layers.22.self_attn.o_proj|8.0|1.649e-04|8.389e+06|
|model.visual.blocks.20.attn.proj|8.0|1.499e-04|6.636e+05|
|model.language_model.layers.0.mlp.gate_proj|8.0|1.231e-04|5.033e+07|
|model.visual.blocks.18.attn.proj|8.0|1.172e-04|6.636e+05|
|model.language_model.layers.18.self_attn.o_proj|8.0|1.111e-04|8.389e+06|
|model.language_model.layers.32.self_attn.o_proj|8.0|1.103e-04|8.389e+06|
|model.language_model.layers.8.self_attn.o_proj|8.0|1.102e-04|8.389e+06|
|model.language_model.layers.11.self_attn.o_proj|8.0|1.057e-04|8.389e+06|
|model.language_model.layers.10.self_attn.o_proj|8.0|1.049e-04|8.389e+06|
|model.language_model.layers.12.self_attn.o_proj|8.0|1.048e-04|8.389e+06|
|model.language_model.layers.20.self_attn.o_proj|8.0|9.788e-05|8.389e+06|
|model.visual.blocks.14.attn.proj|8.0|9.467e-05|6.636e+05|
|model.language_model.layers.0.self_attn.o_proj|8.0|9.098e-05|8.389e+06|
|model.language_model.layers.31.self_attn.o_proj|8.0|9.007e-05|8.389e+06|
|model.language_model.layers.19.self_attn.o_proj|8.0|8.991e-05|8.389e+06|
|model.language_model.layers.7.self_attn.o_proj|8.0|8.866e-05|8.389e+06|
|model.visual.blocks.16.attn.proj|8.0|8.284e-05|6.636e+05|
|model.visual.blocks.19.attn.proj|8.0|7.959e-05|6.636e+05|
|model.visual.blocks.24.attn.proj|8.0|7.916e-05|6.636e+05|
|model.language_model.layers.24.self_attn.o_proj|8.0|7.836e-05|8.389e+06|
|model.language_model.layers.26.self_attn.o_proj|8.0|7.335e-05|8.389e+06|
|model.language_model.layers.25.self_attn.o_proj|8.0|6.761e-05|8.389e+06|
|model.language_model.layers.30.self_attn.o_proj|8.0|6.467e-05|8.389e+06|
|model.language_model.layers.23.self_attn.o_proj|8.0|6.418e-05|8.389e+06|
|model.language_model.layers.9.self_attn.o_proj|8.0|5.919e-05|8.389e+06|
|model.language_model.layers.28.self_attn.o_proj|8.0|5.632e-05|8.389e+06|
|model.language_model.layers.27.self_attn.o_proj|8.0|5.485e-05|8.389e+06|
|model.visual.blocks.26.attn.proj|8.0|5.355e-05|6.636e+05|
|model.language_model.layers.35.self_attn.o_proj|8.0|4.893e-05|8.389e+06|
|model.language_model.layers.29.self_attn.o_proj|8.0|4.280e-05|8.389e+06|
|model.language_model.layers.5.self_attn.o_proj|8.0|4.248e-05|8.389e+06|
|model.visual.blocks.17.attn.proj|8.0|3.941e-05|6.636e+05|
|model.language_model.layers.1.self_attn.o_proj|8.0|3.881e-05|8.389e+06|
|model.language_model.layers.3.self_attn.o_proj|8.0|3.634e-05|8.389e+06|
|model.visual.blocks.25.attn.proj|8.0|2.872e-05|6.636e+05|
|model.language_model.layers.4.self_attn.o_proj|8.0|2.433e-05|8.389e+06|
|model.language_model.layers.2.self_attn.o_proj|8.0|1.706e-05|8.389e+06|

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
  model_dir: <ABSOLUTE_PATH>-quantize-model<ABSOLUTE_PATH>-VL-8B-Instruct
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