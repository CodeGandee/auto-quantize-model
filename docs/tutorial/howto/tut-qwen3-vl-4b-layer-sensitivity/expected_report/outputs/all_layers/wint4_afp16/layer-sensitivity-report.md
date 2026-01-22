# Tutorial Pack Scenario Summary

| Key | Value |
|---|---|
| scenario_id | `all_layers<ABSOLUTE_PATH> |
| mode | `all_layers` |
| quant_pair | `wint4_afp16` |
| dataset_size | `medium` |
| dataset_calib_seq_len | `512` |
| dataset_batch_size | `8` |
| dataset_num_calib_batches | `16` |
| dataset_num_calib_samples | `128` |
| dataset_max_calib_samples | `128` |
| auto_quantize_score_size | `128` |
| scheme_name | `wint4_afp16_autoquant_all_layers` |
| quant_formats | `["INT4_BLOCKWISE_WEIGHT_ONLY_CFG"]` |
| has_layer_sensitivity | `True` |
| has_autoquant_state | `True` |
| has_nonzero_sensitivity | `True` |
| manifest_keys | `["autoquant_state", "dataset", "layer_sensitivity", "layers", "model", "num_quantized_layers", "quantization", "run_config", "scheme", "sensitivity_ranking"]` |


---


AutoQuant Layer Sensitivity (wint4_afp16_autoquant_all_layers)
==============================================================

## Summary


|Key|Value|
| :--- | :--- |
|Scheme|`wint4_afp16_autoquant_all_layers`|
|Model|`<ABSOLUTE_PATH>-quantize-model<ABSOLUTE_PATH>-VL-4B-Instruct`|
|Effective bits (from search)|`7.9927`|
|Total AutoQuant score|`4.831816e-01`|
|Constraint satisfied|`True`|

## Quantization


|Key|Value|
| :--- | :--- |
|Base format|`INT4_BLOCKWISE_WEIGHT_ONLY_CFG`|
|Dtypes|`W=int4` / `A=fp16`|

## Layer Sensitivity Table


Sorted by sensitivity (descending). Layer names are AutoQuant recipe handles; a trailing `.quant_recipe` suffix (if present) is stripped for readability.

|Layer|Num Bits|Sensitivity|Size Cost|
| :--- | :--- | :--- | :--- |
|model.visual.patch_embed.proj|4.0|2.476e+02|3.932e+05|
|model.visual.blocks.0.mlp.linear_fc2|4.0|1.140e+02|1.049e+06|
|model.visual.blocks.0.mlp.linear_fc1|4.0|3.096e+01|1.049e+06|
|model.visual.blocks.1.mlp.linear_fc2|4.0|2.579e+01|1.049e+06|
|model.visual.blocks.1.attn.proj|4.0|2.139e+01|2.621e+05|
|model.visual.blocks.2.mlp.linear_fc2|4.0|1.502e+01|1.049e+06|
|model.visual.blocks.0.attn.proj|4.0|1.125e+01|2.621e+05|
|model.visual.blocks.3.mlp.linear_fc2|4.0|9.862e+00|1.049e+06|
|model.visual.blocks.1.mlp.linear_fc1|4.0|8.564e+00|1.049e+06|
|model.visual.blocks.4.mlp.linear_fc2|4.0|6.863e+00|1.049e+06|
|model.visual.blocks.5.mlp.linear_fc2|4.0|6.143e+00|1.049e+06|
|model.visual.blocks.2.attn.proj|4.0|4.918e+00|2.621e+05|
|model.visual.blocks.6.mlp.linear_fc2|4.0|4.894e+00|1.049e+06|
|model.visual.blocks.2.mlp.linear_fc1|4.0|4.866e+00|1.049e+06|
|model.visual.blocks.3.mlp.linear_fc1|4.0|3.690e+00|1.049e+06|
|model.visual.blocks.4.mlp.linear_fc1|4.0|3.446e+00|1.049e+06|
|model.visual.blocks.6.mlp.linear_fc1|4.0|3.242e+00|1.049e+06|
|model.visual.blocks.5.attn.qkv|4.0|3.167e+00|7.864e+05|
|model.visual.blocks.5.mlp.linear_fc1|4.0|3.085e+00|1.049e+06|
|model.visual.blocks.10.mlp.linear_fc2|4.0|3.007e+00|1.049e+06|
|model.visual.blocks.2.attn.qkv|4.0|2.737e+00|7.864e+05|
|model.visual.blocks.7.mlp.linear_fc2|4.0|2.535e+00|1.049e+06|
|model.visual.blocks.8.mlp.linear_fc1|4.0|2.428e+00|1.049e+06|
|model.visual.blocks.7.mlp.linear_fc1|4.0|2.307e+00|1.049e+06|
|model.visual.blocks.8.mlp.linear_fc2|4.0|2.198e+00|1.049e+06|
|model.visual.blocks.4.attn.proj|4.0|2.118e+00|2.621e+05|
|model.visual.blocks.4.attn.qkv|4.0|2.106e+00|7.864e+05|
|model.visual.blocks.1.attn.qkv|4.0|2.095e+00|7.864e+05|
|model.visual.blocks.6.attn.proj|4.0|2.080e+00|2.621e+05|
|model.visual.blocks.23.mlp.linear_fc2|4.0|1.874e+00|1.049e+06|
|model.visual.blocks.3.attn.proj|4.0|1.865e+00|2.621e+05|
|model.visual.blocks.6.attn.qkv|4.0|1.838e+00|7.864e+05|
|model.visual.blocks.7.attn.proj|4.0|1.838e+00|2.621e+05|
|model.visual.blocks.5.attn.proj|4.0|1.830e+00|2.621e+05|
|model.visual.blocks.8.attn.qkv|4.0|1.816e+00|7.864e+05|
|model.visual.blocks.9.mlp.linear_fc2|4.0|1.767e+00|1.049e+06|
|model.visual.blocks.8.attn.proj|4.0|1.674e+00|2.621e+05|
|model.visual.blocks.9.mlp.linear_fc1|4.0|1.621e+00|1.049e+06|
|model.visual.blocks.7.attn.qkv|4.0|1.615e+00|7.864e+05|
|model.visual.blocks.9.attn.proj|4.0|1.547e+00|2.621e+05|
|model.visual.blocks.21.mlp.linear_fc1|4.0|1.368e+00|1.049e+06|
|model.visual.blocks.10.mlp.linear_fc1|4.0|1.293e+00|1.049e+06|
|model.visual.blocks.3.attn.qkv|4.0|1.246e+00|7.864e+05|
|model.visual.blocks.16.mlp.linear_fc1|4.0|9.343e-01|1.049e+06|
|model.visual.blocks.9.attn.qkv|4.0|9.221e-01|7.864e+05|
|model.visual.blocks.11.mlp.linear_fc2|4.0|8.832e-01|1.049e+06|
|model.visual.blocks.19.mlp.linear_fc1|4.0|8.682e-01|1.049e+06|
|model.visual.blocks.10.attn.qkv|4.0|8.567e-01|7.864e+05|
|model.visual.blocks.11.mlp.linear_fc1|4.0|8.162e-01|1.049e+06|
|model.visual.blocks.10.attn.proj|4.0|7.790e-01|2.621e+05|
|model.visual.blocks.18.mlp.linear_fc1|4.0|6.299e-01|1.049e+06|
|model.visual.blocks.11.attn.proj|4.0|5.957e-01|2.621e+05|
|model.visual.blocks.11.attn.qkv|4.0|5.826e-01|7.864e+05|
|model.visual.blocks.15.mlp.linear_fc1|4.0|5.759e-01|1.049e+06|
|model.visual.blocks.12.attn.qkv|4.0|5.511e-01|7.864e+05|
|model.visual.blocks.13.mlp.linear_fc1|4.0|5.388e-01|1.049e+06|
|model.visual.blocks.17.mlp.linear_fc1|4.0|5.275e-01|1.049e+06|
|model.visual.blocks.13.mlp.linear_fc2|4.0|5.177e-01|1.049e+06|
|model.visual.blocks.12.mlp.linear_fc2|4.0|5.122e-01|1.049e+06|
|model.visual.blocks.12.mlp.linear_fc1|4.0|4.932e-01|1.049e+06|
|model.visual.blocks.12.attn.proj|4.0|4.605e-01|2.621e+05|
|model.visual.blocks.22.mlp.linear_fc1|4.0|4.588e-01|1.049e+06|
|model.visual.blocks.0.attn.qkv|4.0|4.282e-01|7.864e+05|
|model.visual.blocks.14.mlp.linear_fc1|4.0|4.256e-01|1.049e+06|
|model.visual.blocks.20.mlp.linear_fc1|4.0|3.516e-01|1.049e+06|
|model.visual.blocks.21.attn.qkv|4.0|3.144e-01|7.864e+05|
|model.visual.blocks.14.mlp.linear_fc2|4.0|2.957e-01|1.049e+06|
|model.visual.blocks.13.attn.proj|4.0|2.831e-01|2.621e+05|
|model.visual.blocks.13.attn.qkv|4.0|2.588e-01|7.864e+05|
|model.visual.blocks.15.mlp.linear_fc2|4.0|2.538e-01|1.049e+06|
|model.visual.blocks.21.mlp.linear_fc2|4.0|2.091e-01|1.049e+06|
|model.visual.blocks.16.mlp.linear_fc2|4.0|1.925e-01|1.049e+06|
|model.visual.blocks.22.attn.qkv|4.0|1.915e-01|7.864e+05|
|model.visual.merger.linear_fc2|4.0|1.888e-01|2.621e+06|
|model.visual.blocks.17.mlp.linear_fc2|4.0|1.856e-01|1.049e+06|
|model.visual.blocks.14.attn.proj|4.0|1.621e-01|2.621e+05|
|model.visual.deepstack_merger_list.1.linear_fc1|4.0|1.587e-01|4.194e+06|
|model.visual.blocks.19.mlp.linear_fc2|4.0|1.517e-01|1.049e+06|
|model.visual.blocks.23.mlp.linear_fc1|4.0|1.508e-01|1.049e+06|
|model.visual.blocks.15.attn.proj|4.0|1.419e-01|2.621e+05|
|model.visual.blocks.17.attn.proj|4.0|1.377e-01|2.621e+05|
|model.language_model.layers.2.mlp.gate_proj|4.0|1.364e-01|1.245e+07|
|model.visual.blocks.18.mlp.linear_fc2|4.0|1.250e-01|1.049e+06|
|model.visual.blocks.22.mlp.linear_fc2|4.0|1.216e-01|1.049e+06|
|model.visual.blocks.16.attn.proj|4.0|1.147e-01|2.621e+05|
|model.visual.blocks.14.attn.qkv|4.0|1.104e-01|7.864e+05|
|model.visual.blocks.16.attn.qkv|4.0|1.083e-01|7.864e+05|
|model.visual.merger.linear_fc1|4.0|1.055e-01|4.194e+06|
|model.language_model.layers.6.mlp.down_proj|4.0|1.049e-01|6.226e+06|
|model.visual.blocks.20.mlp.linear_fc2|4.0|1.045e-01|1.049e+06|
|model.visual.blocks.23.attn.qkv|4.0|9.610e-02|7.864e+05|
|model.visual.blocks.15.attn.qkv|4.0|9.094e-02|7.864e+05|
|model.visual.deepstack_merger_list.2.linear_fc1|4.0|8.757e-02|4.194e+06|
|model.language_model.layers.12.self_attn.q_proj|4.0|8.468e-02|3.932e+06|
|model.visual.blocks.18.attn.proj|4.0|8.434e-02|2.621e+05|
|model.visual.blocks.17.attn.qkv|4.0|8.346e-02|7.864e+05|
|model.visual.blocks.20.attn.qkv|4.0|7.928e-02|7.864e+05|
|model.visual.blocks.19.attn.qkv|4.0|7.618e-02|7.864e+05|
|model.visual.blocks.20.attn.proj|4.0|7.114e-02|2.621e+05|
|model.visual.blocks.19.attn.proj|4.0|7.017e-02|2.621e+05|
|model.visual.blocks.21.attn.proj|4.0|6.596e-02|2.621e+05|
|model.language_model.layers.10.self_attn.q_proj|4.0|6.357e-02|3.932e+06|
|model.visual.blocks.18.attn.qkv|4.0|6.271e-02|7.864e+05|
|model.language_model.layers.9.self_attn.q_proj|4.0|5.720e-02|3.932e+06|
|model.language_model.layers.14.self_attn.q_proj|4.0|5.661e-02|3.932e+06|
|model.visual.deepstack_merger_list.1.linear_fc2|4.0|4.580e-02|2.621e+06|
|model.language_model.layers.4.mlp.gate_proj|4.0|4.393e-02|1.245e+07|
|model.language_model.layers.1.mlp.gate_proj|4.0|4.196e-02|1.245e+07|
|model.visual.deepstack_merger_list.2.linear_fc2|4.0|4.177e-02|2.621e+06|
|model.visual.blocks.22.attn.proj|4.0|4.145e-02|2.621e+05|
|model.language_model.layers.15.self_attn.q_proj|4.0|4.012e-02|3.932e+06|
|model.visual.deepstack_merger_list.0.linear_fc1|4.0|4.008e-02|4.194e+06|
|model.language_model.layers.8.self_attn.q_proj|4.0|3.813e-02|3.932e+06|
|model.language_model.layers.1.mlp.down_proj|4.0|3.793e-02|6.226e+06|
|model.language_model.layers.11.self_attn.q_proj|4.0|3.283e-02|3.932e+06|
|model.language_model.layers.3.mlp.gate_proj|4.0|3.264e-02|1.245e+07|
|model.language_model.layers.16.self_attn.q_proj|4.0|3.219e-02|3.932e+06|
|model.language_model.layers.0.mlp.gate_proj|4.0|3.152e-02|1.245e+07|
|model.language_model.layers.20.self_attn.q_proj|4.0|3.064e-02|3.932e+06|
|model.language_model.layers.5.mlp.gate_proj|4.0|3.044e-02|1.245e+07|
|model.language_model.layers.18.self_attn.q_proj|4.0|2.913e-02|3.932e+06|
|model.language_model.layers.0.mlp.down_proj|4.0|2.858e-02|6.226e+06|
|model.language_model.layers.6.self_attn.q_proj|4.0|2.794e-02|3.932e+06|
|model.language_model.layers.34.self_attn.q_proj|4.0|2.773e-02|3.932e+06|
|model.language_model.layers.2.self_attn.q_proj|4.0|2.727e-02|3.932e+06|
|model.language_model.layers.17.self_attn.q_proj|4.0|2.573e-02|3.932e+06|
|model.language_model.layers.13.self_attn.q_proj|4.0|2.551e-02|3.932e+06|
|model.visual.blocks.23.attn.proj|4.0|2.533e-02|2.621e+05|
|model.language_model.layers.19.self_attn.q_proj|4.0|2.502e-02|3.932e+06|
|model.language_model.layers.9.mlp.gate_proj|4.0|2.360e-02|1.245e+07|
|model.language_model.layers.7.self_attn.q_proj|4.0|2.277e-02|3.932e+06|
|model.language_model.layers.7.mlp.gate_proj|4.0|2.226e-02|1.245e+07|
|model.visual.deepstack_merger_list.0.linear_fc2|4.0|2.217e-02|2.621e+06|
|model.language_model.layers.14.mlp.gate_proj|4.0|2.089e-02|1.245e+07|
|model.language_model.layers.15.mlp.gate_proj|4.0|1.938e-02|1.245e+07|
|model.language_model.layers.6.mlp.gate_proj|4.0|1.885e-02|1.245e+07|
|model.language_model.layers.12.mlp.gate_proj|4.0|1.874e-02|1.245e+07|
|model.language_model.layers.13.mlp.gate_proj|4.0|1.856e-02|1.245e+07|
|model.language_model.layers.10.mlp.gate_proj|4.0|1.720e-02|1.245e+07|
|model.language_model.layers.32.self_attn.q_proj|4.0|1.540e-02|3.932e+06|
|model.language_model.layers.21.self_attn.q_proj|4.0|1.519e-02|3.932e+06|
|model.language_model.layers.8.mlp.gate_proj|4.0|1.515e-02|1.245e+07|
|model.language_model.layers.9.mlp.down_proj|4.0|1.500e-02|6.226e+06|
|model.language_model.layers.11.mlp.gate_proj|4.0|1.476e-02|1.245e+07|
|model.language_model.layers.35.self_attn.q_proj|4.0|1.420e-02|3.932e+06|
|model.language_model.layers.22.self_attn.q_proj|4.0|1.402e-02|3.932e+06|
|model.language_model.layers.4.mlp.down_proj|4.0|1.397e-02|6.226e+06|
|model.language_model.layers.16.mlp.down_proj|4.0|1.351e-02|6.226e+06|
|model.language_model.layers.16.mlp.gate_proj|4.0|1.255e-02|1.245e+07|
|model.language_model.layers.33.self_attn.q_proj|4.0|1.238e-02|3.932e+06|
|model.language_model.layers.15.mlp.down_proj|4.0|1.148e-02|6.226e+06|
|model.language_model.layers.14.mlp.down_proj|4.0|1.112e-02|6.226e+06|
|model.language_model.layers.10.mlp.down_proj|4.0|1.084e-02|6.226e+06|
|model.language_model.layers.12.mlp.down_proj|4.0|1.056e-02|6.226e+06|
|model.language_model.layers.5.self_attn.q_proj|4.0|1.043e-02|3.932e+06|
|model.language_model.layers.0.self_attn.q_proj|4.0|1.032e-02|3.932e+06|
|model.language_model.layers.13.mlp.down_proj|4.0|1.020e-02|6.226e+06|
|model.language_model.layers.15.self_attn.o_proj|4.0|9.940e-03|2.621e+06|
|model.language_model.layers.5.mlp.down_proj|4.0|9.767e-03|6.226e+06|
|model.language_model.layers.18.mlp.gate_proj|4.0|9.476e-03|1.245e+07|
|model.language_model.layers.17.mlp.gate_proj|4.0|9.186e-03|1.245e+07|
|model.language_model.layers.7.mlp.down_proj|4.0|8.840e-03|6.226e+06|
|model.language_model.layers.10.self_attn.o_proj|4.0|8.703e-03|2.621e+06|
|model.language_model.layers.1.self_attn.q_proj|4.0|8.661e-03|3.932e+06|
|model.language_model.layers.8.mlp.down_proj|4.0|8.550e-03|6.226e+06|
|model.language_model.layers.11.mlp.down_proj|4.0|8.231e-03|6.226e+06|
|model.language_model.layers.30.self_attn.q_proj|4.0|7.659e-03|3.932e+06|
|model.language_model.layers.19.mlp.gate_proj|4.0|7.654e-03|1.245e+07|
|model.language_model.layers.23.self_attn.q_proj|4.0|7.534e-03|3.932e+06|
|model.language_model.layers.3.mlp.down_proj|4.0|7.500e-03|6.226e+06|
|model.language_model.layers.14.self_attn.o_proj|4.0|7.300e-03|2.621e+06|
|model.language_model.layers.9.self_attn.o_proj|4.0|7.263e-03|2.621e+06|
|model.language_model.layers.18.mlp.down_proj|4.0|7.180e-03|6.226e+06|
|model.language_model.layers.12.self_attn.o_proj|4.0|6.987e-03|2.621e+06|
|model.language_model.layers.2.mlp.down_proj|4.0|6.827e-03|6.226e+06|
|model.language_model.layers.3.self_attn.q_proj|4.0|6.753e-03|3.932e+06|
|model.language_model.layers.8.self_attn.o_proj|4.0|6.423e-03|2.621e+06|
|model.language_model.layers.24.self_attn.q_proj|4.0|6.305e-03|3.932e+06|
|model.language_model.layers.17.self_attn.o_proj|4.0|6.083e-03|2.621e+06|
|model.language_model.layers.6.self_attn.o_proj|4.0|5.840e-03|2.621e+06|
|model.language_model.layers.18.self_attn.o_proj|4.0|5.833e-03|2.621e+06|
|model.language_model.layers.16.self_attn.o_proj|4.0|5.665e-03|2.621e+06|
|model.language_model.layers.17.mlp.down_proj|4.0|5.418e-03|6.226e+06|
|model.language_model.layers.28.self_attn.q_proj|4.0|5.355e-03|3.932e+06|
|model.language_model.layers.4.self_attn.q_proj|4.0|5.086e-03|3.932e+06|
|model.language_model.layers.31.self_attn.q_proj|4.0|5.073e-03|3.932e+06|
|model.language_model.layers.11.self_attn.o_proj|4.0|4.820e-03|2.621e+06|
|model.language_model.layers.29.self_attn.q_proj|4.0|4.807e-03|3.932e+06|
|model.language_model.layers.27.self_attn.q_proj|4.0|4.791e-03|3.932e+06|
|model.language_model.layers.34.mlp.gate_proj|4.0|4.790e-03|1.245e+07|
|model.language_model.layers.22.mlp.gate_proj|4.0|4.735e-03|1.245e+07|
|model.language_model.layers.19.self_attn.o_proj|4.0|4.655e-03|2.621e+06|
|model.language_model.layers.2.self_attn.o_proj|4.0|4.542e-03|2.621e+06|
|model.language_model.layers.23.mlp.gate_proj|4.0|4.481e-03|1.245e+07|
|model.language_model.layers.13.self_attn.o_proj|4.0|4.230e-03|2.621e+06|
|model.language_model.layers.19.mlp.down_proj|4.0|4.216e-03|6.226e+06|
|model.language_model.layers.24.mlp.gate_proj|4.0|4.104e-03|1.245e+07|
|model.language_model.layers.20.mlp.gate_proj|4.0|4.058e-03|1.245e+07|
|model.language_model.layers.25.mlp.gate_proj|4.0|3.972e-03|1.245e+07|
|model.language_model.layers.25.self_attn.q_proj|4.0|3.688e-03|3.932e+06|
|model.language_model.layers.26.mlp.gate_proj|4.0|3.549e-03|1.245e+07|
|model.language_model.layers.21.mlp.gate_proj|4.0|3.509e-03|1.245e+07|
|model.language_model.layers.22.mlp.down_proj|4.0|3.295e-03|6.226e+06|
|model.language_model.layers.27.mlp.gate_proj|4.0|3.184e-03|1.245e+07|
|model.language_model.layers.23.mlp.down_proj|4.0|3.083e-03|6.226e+06|
|model.language_model.layers.26.self_attn.q_proj|4.0|2.937e-03|3.932e+06|
|model.language_model.layers.35.mlp.gate_proj|4.0|2.931e-03|1.245e+07|
|model.language_model.layers.1.self_attn.o_proj|4.0|2.900e-03|2.621e+06|
|model.language_model.layers.28.mlp.gate_proj|4.0|2.828e-03|1.245e+07|
|model.language_model.layers.34.mlp.down_proj|4.0|2.808e-03|6.226e+06|
|model.language_model.layers.0.self_attn.o_proj|4.0|2.805e-03|2.621e+06|
|model.language_model.layers.33.mlp.gate_proj|4.0|2.616e-03|1.245e+07|
|model.language_model.layers.24.mlp.down_proj|4.0|2.466e-03|6.226e+06|
|model.language_model.layers.29.mlp.gate_proj|4.0|2.350e-03|1.245e+07|
|model.language_model.layers.7.self_attn.o_proj|4.0|2.281e-03|2.621e+06|
|model.language_model.layers.20.mlp.down_proj|4.0|2.270e-03|6.226e+06|
|model.language_model.layers.21.mlp.down_proj|4.0|2.243e-03|6.226e+06|
|model.language_model.layers.25.mlp.down_proj|4.0|2.242e-03|6.226e+06|
|model.language_model.layers.32.mlp.gate_proj|4.0|2.195e-03|1.245e+07|
|model.language_model.layers.31.mlp.gate_proj|4.0|2.085e-03|1.245e+07|
|model.language_model.layers.30.mlp.gate_proj|4.0|2.078e-03|1.245e+07|
|model.language_model.layers.5.self_attn.o_proj|4.0|2.028e-03|2.621e+06|
|model.language_model.layers.26.mlp.down_proj|4.0|1.998e-03|6.226e+06|
|model.language_model.layers.20.self_attn.o_proj|4.0|1.975e-03|2.621e+06|
|model.language_model.layers.27.mlp.down_proj|4.0|1.899e-03|6.226e+06|
|model.language_model.layers.33.mlp.down_proj|4.0|1.869e-03|6.226e+06|
|model.language_model.layers.3.self_attn.o_proj|4.0|1.805e-03|2.621e+06|
|model.language_model.layers.4.self_attn.o_proj|4.0|1.762e-03|2.621e+06|
|model.language_model.layers.22.self_attn.o_proj|4.0|1.761e-03|2.621e+06|
|lm_head|4.0|1.721e-03|9.724e+07|
|model.language_model.layers.28.mlp.down_proj|4.0|1.713e-03|6.226e+06|
|model.language_model.layers.23.self_attn.o_proj|4.0|1.541e-03|2.621e+06|
|model.language_model.layers.29.mlp.down_proj|4.0|1.503e-03|6.226e+06|
|model.language_model.layers.32.mlp.down_proj|4.0|1.301e-03|6.226e+06|
|model.language_model.layers.31.mlp.down_proj|4.0|1.243e-03|6.226e+06|
|model.language_model.layers.30.mlp.down_proj|4.0|1.216e-03|6.226e+06|
|model.language_model.layers.35.mlp.down_proj|4.0|1.056e-03|6.226e+06|
|model.language_model.layers.21.self_attn.o_proj|4.0|8.657e-04|2.621e+06|
|model.language_model.layers.24.self_attn.o_proj|4.0|5.964e-04|2.621e+06|
|model.language_model.layers.34.self_attn.o_proj|4.0|5.283e-04|2.621e+06|
|model.language_model.layers.32.self_attn.o_proj|4.0|4.286e-04|2.621e+06|
|model.language_model.layers.33.self_attn.o_proj|4.0|4.089e-04|2.621e+06|
|model.language_model.layers.31.self_attn.o_proj|4.0|3.333e-04|2.621e+06|
|model.language_model.layers.35.self_attn.o_proj|4.0|2.950e-04|2.621e+06|
|model.language_model.layers.25.self_attn.o_proj|4.0|2.779e-04|2.621e+06|
|model.language_model.layers.30.self_attn.o_proj|4.0|2.659e-04|2.621e+06|
|model.language_model.layers.28.self_attn.o_proj|4.0|2.487e-04|2.621e+06|
|model.language_model.layers.26.self_attn.o_proj|4.0|1.899e-04|2.621e+06|
|model.language_model.layers.29.self_attn.o_proj|4.0|1.849e-04|2.621e+06|
|model.language_model.layers.27.self_attn.o_proj|4.0|1.601e-04|2.621e+06|

## Composed Config (`composed-config.yaml`)


```yaml
script: run_qwen3_vl_4b_autoquant_all_layers.py
scheme:
  name: wint4_afp16_autoquant_all_layers
  auto_quantize_bits: 8.0
  auto_quantize_method: gradient
  auto_quantize_score_size: 128
  coverage_mode: full
  coverage_fraction: 1.0
  quant_formats:
  - INT4_BLOCKWISE_WEIGHT_ONLY_CFG
args:
  model_dir: <ABSOLUTE_PATH>-quantize-model<ABSOLUTE_PATH>-VL-4B-Instruct
  output_dir: <ABSOLUTE_PATH>-quantize-model<ABSOLUTE_PATH>
  vlm_calib_db: <ABSOLUTE_PATH>-quantize-model<ABSOLUTE_PATH>-quantize-calib<ABSOLUTE_PATH>
  coco_root: <ABSOLUTE_PATH>-quantize-model<ABSOLUTE_PATH>-data
  max_calib_samples: 128
  num_calib_batches: 16
  calib_seq_len: 512
  batch_size: 8
  device: cuda:0
  dataset_size: medium
  quant_pair: wint4_afp16
  quant_format: int8
  effective_bits: null
  auto_quantize_score_size: 128
  report_only: false
dataset:
  size: medium
  vlm_calib_db: <ABSOLUTE_PATH>-quantize-model<ABSOLUTE_PATH>-quantize-calib<ABSOLUTE_PATH>
  coco_root: <ABSOLUTE_PATH>-quantize-model<ABSOLUTE_PATH>-data
  calib_seq_len: 512
  batch_size: 8
  num_calib_batches: 16
  num_calib_samples: 128
  max_calib_samples: 128
quantization:
  base_format_name: INT4_BLOCKWISE_WEIGHT_ONLY_CFG
  format_names:
  - INT4_BLOCKWISE_WEIGHT_ONLY_CFG
  quant_format: int8
  quant_pair:
    name: wint4_afp16
    weight: int4
    activation: fp16
    format_name: INT4_BLOCKWISE_WEIGHT_ONLY_CFG
```