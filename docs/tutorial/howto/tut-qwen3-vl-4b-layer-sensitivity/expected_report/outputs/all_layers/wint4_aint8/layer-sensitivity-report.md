# Tutorial Pack Scenario Summary

| Key | Value |
|---|---|
| scenario_id | `all_layers<ABSOLUTE_PATH> |
| mode | `all_layers` |
| quant_pair | `wint4_aint8` |
| dataset_size | `medium` |
| dataset_calib_seq_len | `512` |
| dataset_batch_size | `8` |
| dataset_num_calib_batches | `16` |
| dataset_num_calib_samples | `128` |
| dataset_max_calib_samples | `128` |
| auto_quantize_score_size | `128` |
| scheme_name | `wint4_aint8_autoquant_all_layers` |
| quant_formats | `["INT4_WEIGHT_INT8_ACT_CFG"]` |
| has_layer_sensitivity | `True` |
| has_autoquant_state | `True` |
| has_nonzero_sensitivity | `True` |
| manifest_keys | `["autoquant_state", "dataset", "layer_sensitivity", "layers", "model", "num_quantized_layers", "quantization", "run_config", "scheme", "sensitivity_ranking"]` |


---


AutoQuant Layer Sensitivity (wint4_aint8_autoquant_all_layers)
==============================================================

## Summary


|Key|Value|
| :--- | :--- |
|Scheme|`wint4_aint8_autoquant_all_layers`|
|Model|`<ABSOLUTE_PATH>-quantize-model<ABSOLUTE_PATH>-VL-4B-Instruct`|
|Effective bits (from search)|`7.9998`|
|Total AutoQuant score|`1.142873e+00`|
|Constraint satisfied|`True`|

## Quantization


|Key|Value|
| :--- | :--- |
|Base format|`INT4_WEIGHT_INT8_ACT_CFG`|
|Dtypes|`W=int4` / `A=int8`|

## Layer Sensitivity Table


Sorted by sensitivity (descending). Layer names are AutoQuant recipe handles; a trailing `.quant_recipe` suffix (if present) is stripped for readability.

|Layer|Num Bits|Sensitivity|Size Cost|
| :--- | :--- | :--- | :--- |
|model.visual.blocks.0.mlp.linear_fc2|4.0|1.082e+03|1.049e+06|
|model.visual.blocks.1.mlp.linear_fc2|4.0|3.777e+02|1.049e+06|
|model.visual.patch_embed.proj|4.0|2.455e+02|3.932e+05|
|model.visual.blocks.10.mlp.linear_fc2|4.0|8.829e+01|1.049e+06|
|model.visual.blocks.11.mlp.linear_fc2|4.0|8.818e+01|1.049e+06|
|model.visual.blocks.0.mlp.linear_fc1|4.0|8.409e+01|1.049e+06|
|model.visual.blocks.2.mlp.linear_fc2|4.0|6.168e+01|1.049e+06|
|model.visual.blocks.1.attn.proj|4.0|2.577e+01|2.621e+05|
|model.visual.blocks.0.attn.proj|4.0|2.122e+01|2.621e+05|
|model.visual.blocks.3.mlp.linear_fc2|4.0|1.938e+01|1.049e+06|
|model.visual.blocks.1.mlp.linear_fc1|4.0|1.383e+01|1.049e+06|
|model.visual.blocks.4.mlp.linear_fc2|4.0|9.905e+00|1.049e+06|
|model.visual.blocks.5.mlp.linear_fc2|4.0|7.400e+00|1.049e+06|
|model.visual.blocks.2.mlp.linear_fc1|4.0|7.286e+00|1.049e+06|
|model.visual.blocks.4.mlp.linear_fc1|4.0|7.239e+00|1.049e+06|
|model.visual.blocks.6.mlp.linear_fc2|4.0|7.163e+00|1.049e+06|
|model.visual.blocks.3.mlp.linear_fc1|4.0|6.874e+00|1.049e+06|
|model.visual.blocks.5.mlp.linear_fc1|4.0|6.750e+00|1.049e+06|
|model.visual.blocks.7.mlp.linear_fc2|4.0|5.789e+00|1.049e+06|
|model.visual.blocks.2.attn.proj|4.0|5.782e+00|2.621e+05|
|model.language_model.layers.4.mlp.down_proj|4.0|5.462e+00|6.226e+06|
|model.visual.blocks.6.mlp.linear_fc1|4.0|5.210e+00|1.049e+06|
|model.visual.blocks.5.attn.qkv|4.0|3.839e+00|7.864e+05|
|model.visual.blocks.8.mlp.linear_fc1|4.0|3.538e+00|1.049e+06|
|model.visual.blocks.7.mlp.linear_fc1|4.0|3.447e+00|1.049e+06|
|model.visual.blocks.2.attn.qkv|4.0|3.121e+00|7.864e+05|
|model.visual.blocks.12.mlp.linear_fc2|4.0|3.106e+00|1.049e+06|
|model.visual.blocks.11.mlp.linear_fc1|4.0|3.046e+00|1.049e+06|
|model.visual.blocks.8.mlp.linear_fc2|4.0|3.023e+00|1.049e+06|
|model.visual.blocks.9.mlp.linear_fc2|4.0|2.817e+00|1.049e+06|
|model.visual.blocks.1.attn.qkv|4.0|2.691e+00|7.864e+05|
|model.visual.blocks.4.attn.qkv|4.0|2.571e+00|7.864e+05|
|model.visual.blocks.4.attn.proj|4.0|2.505e+00|2.621e+05|
|model.language_model.layers.1.mlp.down_proj|4.0|2.338e+00|6.226e+06|
|model.visual.blocks.6.attn.proj|4.0|2.291e+00|2.621e+05|
|model.visual.blocks.5.attn.proj|4.0|2.288e+00|2.621e+05|
|model.visual.blocks.6.attn.qkv|4.0|2.219e+00|7.864e+05|
|model.visual.blocks.9.mlp.linear_fc1|4.0|2.210e+00|1.049e+06|
|model.visual.blocks.3.attn.proj|4.0|2.107e+00|2.621e+05|
|model.visual.blocks.8.attn.qkv|4.0|2.089e+00|7.864e+05|
|model.visual.blocks.10.mlp.linear_fc1|4.0|2.054e+00|1.049e+06|
|model.visual.blocks.23.mlp.linear_fc2|4.0|2.032e+00|1.049e+06|
|model.visual.blocks.7.attn.proj|4.0|2.000e+00|2.621e+05|
|model.visual.blocks.7.attn.qkv|4.0|1.940e+00|7.864e+05|
|model.visual.blocks.8.attn.proj|4.0|1.794e+00|2.621e+05|
|model.visual.blocks.9.attn.proj|4.0|1.677e+00|2.621e+05|
|model.visual.merger.linear_fc1|4.0|1.675e+00|4.194e+06|
|model.language_model.layers.6.mlp.down_proj|4.0|1.579e+00|6.226e+06|
|model.visual.blocks.13.mlp.linear_fc2|4.0|1.572e+00|1.049e+06|
|model.visual.blocks.3.attn.qkv|4.0|1.451e+00|7.864e+05|
|model.visual.blocks.21.mlp.linear_fc1|4.0|1.435e+00|1.049e+06|
|model.visual.blocks.0.attn.qkv|4.0|1.215e+00|7.864e+05|
|model.language_model.layers.2.mlp.down_proj|4.0|1.109e+00|6.226e+06|
|model.visual.blocks.9.attn.qkv|4.0|1.089e+00|7.864e+05|
|model.language_model.layers.6.mlp.gate_proj|4.0|1.035e+00|1.245e+07|
|model.visual.blocks.16.mlp.linear_fc1|4.0|1.019e+00|1.049e+06|
|model.visual.blocks.19.mlp.linear_fc1|4.0|9.973e-01|1.049e+06|
|model.visual.blocks.10.attn.qkv|4.0|9.697e-01|7.864e+05|
|model.language_model.layers.3.mlp.down_proj|4.0|8.950e-01|6.226e+06|
|model.visual.blocks.10.attn.proj|4.0|8.579e-01|2.621e+05|
|model.visual.blocks.11.attn.proj|4.0|7.350e-01|2.621e+05|
|model.visual.blocks.13.mlp.linear_fc1|4.0|7.094e-01|1.049e+06|
|model.visual.blocks.18.mlp.linear_fc1|4.0|7.063e-01|1.049e+06|
|model.visual.blocks.12.attn.qkv|4.0|6.759e-01|7.864e+05|
|model.visual.blocks.15.mlp.linear_fc1|4.0|6.740e-01|1.049e+06|
|model.visual.blocks.12.mlp.linear_fc1|4.0|6.682e-01|1.049e+06|
|model.visual.blocks.11.attn.qkv|4.0|6.575e-01|7.864e+05|
|model.visual.blocks.17.mlp.linear_fc1|4.0|6.344e-01|1.049e+06|
|model.visual.merger.linear_fc2|4.0|5.797e-01|2.621e+06|
|model.visual.blocks.14.mlp.linear_fc1|4.0|5.719e-01|1.049e+06|
|model.visual.blocks.12.attn.proj|4.0|5.372e-01|2.621e+05|
|model.visual.blocks.22.mlp.linear_fc1|4.0|5.101e-01|1.049e+06|
|model.language_model.layers.16.mlp.down_proj|4.0|4.744e-01|6.226e+06|
|model.visual.blocks.14.mlp.linear_fc2|4.0|4.718e-01|1.049e+06|
|model.visual.blocks.20.mlp.linear_fc1|4.0|4.378e-01|1.049e+06|
|model.language_model.layers.14.mlp.down_proj|4.0|4.295e-01|6.226e+06|
|model.visual.blocks.21.attn.qkv|4.0|3.860e-01|7.864e+05|
|model.language_model.layers.10.mlp.down_proj|4.0|3.641e-01|6.226e+06|
|model.language_model.layers.9.mlp.down_proj|4.0|3.600e-01|6.226e+06|
|model.visual.blocks.13.attn.proj|4.0|3.346e-01|2.621e+05|
|model.visual.blocks.15.mlp.linear_fc2|4.0|3.089e-01|1.049e+06|
|model.visual.deepstack_merger_list.1.linear_fc1|4.0|2.976e-01|4.194e+06|
|model.language_model.layers.12.mlp.down_proj|4.0|2.939e-01|6.226e+06|
|model.visual.blocks.13.attn.qkv|4.0|2.928e-01|7.864e+05|
|model.visual.blocks.23.mlp.linear_fc1|4.0|2.484e-01|1.049e+06|
|model.visual.blocks.16.mlp.linear_fc2|4.0|2.446e-01|1.049e+06|
|model.language_model.layers.7.mlp.down_proj|4.0|2.364e-01|6.226e+06|
|model.language_model.layers.8.mlp.down_proj|4.0|2.319e-01|6.226e+06|
|model.visual.blocks.17.mlp.linear_fc2|4.0|2.313e-01|1.049e+06|
|model.visual.blocks.14.attn.proj|4.0|2.248e-01|2.621e+05|
|model.visual.blocks.22.attn.qkv|4.0|2.237e-01|7.864e+05|
|model.visual.blocks.21.mlp.linear_fc2|4.0|2.143e-01|1.049e+06|
|model.language_model.layers.11.mlp.down_proj|4.0|2.084e-01|6.226e+06|
|model.language_model.layers.5.mlp.down_proj|4.0|2.078e-01|6.226e+06|
|model.language_model.layers.15.mlp.down_proj|4.0|1.965e-01|6.226e+06|
|model.language_model.layers.2.mlp.gate_proj|4.0|1.960e-01|1.245e+07|
|model.visual.deepstack_merger_list.2.linear_fc1|4.0|1.815e-01|4.194e+06|
|model.language_model.layers.34.mlp.down_proj|4.0|1.714e-01|6.226e+06|
|model.language_model.layers.0.mlp.down_proj|4.0|1.667e-01|6.226e+06|
|model.visual.blocks.15.attn.proj|4.0|1.651e-01|2.621e+05|
|model.visual.blocks.19.mlp.linear_fc2|4.0|1.625e-01|1.049e+06|
|model.visual.blocks.17.attn.proj|4.0|1.566e-01|2.621e+05|
|model.language_model.layers.18.mlp.down_proj|4.0|1.515e-01|6.226e+06|
|model.visual.blocks.18.mlp.linear_fc2|4.0|1.390e-01|1.049e+06|
|model.visual.blocks.16.attn.proj|4.0|1.329e-01|2.621e+05|
|model.visual.blocks.14.attn.qkv|4.0|1.276e-01|7.864e+05|
|model.visual.blocks.22.mlp.linear_fc2|4.0|1.264e-01|1.049e+06|
|model.visual.blocks.16.attn.qkv|4.0|1.249e-01|7.864e+05|
|model.visual.blocks.23.attn.qkv|4.0|1.165e-01|7.864e+05|
|model.language_model.layers.12.self_attn.q_proj|4.0|1.118e-01|3.932e+06|
|model.visual.blocks.20.mlp.linear_fc2|4.0|1.118e-01|1.049e+06|
|model.language_model.layers.10.self_attn.q_proj|4.0|1.074e-01|3.932e+06|
|model.visual.blocks.18.attn.proj|4.0|1.046e-01|2.621e+05|
|model.visual.blocks.15.attn.qkv|4.0|1.016e-01|7.864e+05|
|model.visual.blocks.17.attn.qkv|4.0|9.692e-02|7.864e+05|
|model.visual.blocks.20.attn.qkv|4.0|9.341e-02|7.864e+05|
|model.visual.blocks.19.attn.qkv|4.0|8.875e-02|7.864e+05|
|model.language_model.layers.13.mlp.down_proj|4.0|8.760e-02|6.226e+06|
|model.visual.blocks.19.attn.proj|4.0|8.595e-02|2.621e+05|
|model.visual.deepstack_merger_list.0.linear_fc1|4.0|8.135e-02|4.194e+06|
|model.visual.blocks.20.attn.proj|4.0|8.105e-02|2.621e+05|
|model.language_model.layers.7.mlp.gate_proj|4.0|8.055e-02|1.245e+07|
|model.visual.deepstack_merger_list.2.linear_fc2|4.0|8.047e-02|2.621e+06|
|model.visual.deepstack_merger_list.1.linear_fc2|4.0|7.879e-02|2.621e+06|
|model.language_model.layers.14.self_attn.q_proj|4.0|7.222e-02|3.932e+06|
|model.visual.blocks.21.attn.proj|4.0|7.149e-02|2.621e+05|
|model.language_model.layers.1.mlp.gate_proj|4.0|7.060e-02|1.245e+07|
|model.language_model.layers.15.self_attn.o_proj|4.0|7.027e-02|2.621e+06|
|model.language_model.layers.4.mlp.gate_proj|4.0|7.009e-02|1.245e+07|
|model.visual.blocks.18.attn.qkv|4.0|6.884e-02|7.864e+05|
|model.language_model.layers.9.self_attn.q_proj|4.0|6.864e-02|3.932e+06|
|model.language_model.layers.15.self_attn.q_proj|4.0|6.823e-02|3.932e+06|
|model.language_model.layers.6.self_attn.q_proj|4.0|6.502e-02|3.932e+06|
|model.language_model.layers.16.self_attn.q_proj|4.0|5.747e-02|3.932e+06|
|model.language_model.layers.3.mlp.gate_proj|4.0|5.593e-02|1.245e+07|
|model.language_model.layers.18.self_attn.q_proj|4.0|5.556e-02|3.932e+06|
|model.language_model.layers.20.self_attn.q_proj|4.0|5.516e-02|3.932e+06|
|model.language_model.layers.5.mlp.gate_proj|4.0|5.177e-02|1.245e+07|
|model.language_model.layers.8.self_attn.q_proj|4.0|5.123e-02|3.932e+06|
|model.language_model.layers.17.self_attn.q_proj|4.0|4.653e-02|3.932e+06|
|model.visual.blocks.22.attn.proj|4.0|4.541e-02|2.621e+05|
|model.language_model.layers.16.mlp.gate_proj|4.0|4.486e-02|1.245e+07|
|model.language_model.layers.11.self_attn.q_proj|4.0|4.418e-02|3.932e+06|
|model.language_model.layers.19.self_attn.q_proj|4.0|4.353e-02|3.932e+06|
|model.language_model.layers.17.self_attn.o_proj|4.0|3.999e-02|2.621e+06|
|model.language_model.layers.16.self_attn.o_proj|4.0|3.882e-02|2.621e+06|
|model.language_model.layers.14.mlp.gate_proj|4.0|3.876e-02|1.245e+07|
|model.language_model.layers.13.self_attn.q_proj|4.0|3.819e-02|3.932e+06|
|model.language_model.layers.34.self_attn.q_proj|4.0|3.751e-02|3.932e+06|
|model.language_model.layers.15.mlp.gate_proj|4.0|3.721e-02|1.245e+07|
|model.language_model.layers.9.mlp.gate_proj|4.0|3.539e-02|1.245e+07|
|model.language_model.layers.2.self_attn.q_proj|4.0|3.449e-02|3.932e+06|
|model.language_model.layers.13.mlp.gate_proj|4.0|3.320e-02|1.245e+07|
|model.language_model.layers.0.mlp.gate_proj|4.0|3.208e-02|1.245e+07|
|model.language_model.layers.7.self_attn.q_proj|4.0|3.197e-02|3.932e+06|
|model.visual.deepstack_merger_list.0.linear_fc2|4.0|3.162e-02|2.621e+06|
|model.language_model.layers.12.mlp.gate_proj|4.0|2.988e-02|1.245e+07|
|model.language_model.layers.35.mlp.gate_proj|4.0|2.795e-02|1.245e+07|
|model.language_model.layers.10.mlp.gate_proj|4.0|2.663e-02|1.245e+07|
|model.visual.blocks.23.attn.proj|4.0|2.642e-02|2.621e+05|
|model.language_model.layers.8.mlp.gate_proj|4.0|2.616e-02|1.245e+07|
|model.language_model.layers.11.mlp.gate_proj|4.0|2.558e-02|1.245e+07|
|model.language_model.layers.21.self_attn.q_proj|4.0|2.422e-02|3.932e+06|
|model.language_model.layers.14.self_attn.o_proj|4.0|2.314e-02|2.621e+06|
|model.language_model.layers.13.self_attn.o_proj|4.0|2.262e-02|2.621e+06|
|model.language_model.layers.31.mlp.down_proj|4.0|2.170e-02|6.226e+06|
|model.language_model.layers.28.mlp.down_proj|4.0|2.108e-02|6.226e+06|
|model.language_model.layers.35.self_attn.q_proj|4.0|2.101e-02|3.932e+06|
|model.language_model.layers.32.self_attn.q_proj|4.0|2.097e-02|3.932e+06|
|model.language_model.layers.30.mlp.down_proj|4.0|2.044e-02|6.226e+06|
|model.language_model.layers.19.mlp.down_proj|4.0|2.018e-02|6.226e+06|
|model.language_model.layers.17.mlp.down_proj|4.0|2.015e-02|6.226e+06|
|model.language_model.layers.22.self_attn.q_proj|4.0|1.933e-02|3.932e+06|
|model.language_model.layers.29.mlp.down_proj|4.0|1.825e-02|6.226e+06|
|model.language_model.layers.33.self_attn.q_proj|4.0|1.763e-02|3.932e+06|
|model.language_model.layers.5.self_attn.q_proj|4.0|1.690e-02|3.932e+06|
|model.language_model.layers.27.mlp.down_proj|4.0|1.621e-02|6.226e+06|
|model.language_model.layers.35.mlp.down_proj|4.0|1.557e-02|6.226e+06|
|model.language_model.layers.24.mlp.down_proj|4.0|1.545e-02|6.226e+06|
|model.language_model.layers.23.mlp.down_proj|4.0|1.544e-02|6.226e+06|
|model.language_model.layers.17.mlp.gate_proj|4.0|1.494e-02|1.245e+07|
|model.language_model.layers.33.mlp.down_proj|4.0|1.493e-02|6.226e+06|
|model.language_model.layers.18.mlp.gate_proj|4.0|1.408e-02|1.245e+07|
|model.language_model.layers.0.self_attn.q_proj|4.0|1.407e-02|3.932e+06|
|model.language_model.layers.32.mlp.down_proj|4.0|1.339e-02|6.226e+06|
|model.language_model.layers.10.self_attn.o_proj|4.0|1.302e-02|2.621e+06|
|model.language_model.layers.25.mlp.down_proj|4.0|1.287e-02|6.226e+06|
|model.language_model.layers.3.self_attn.q_proj|4.0|1.271e-02|3.932e+06|
|model.language_model.layers.26.mlp.down_proj|4.0|1.254e-02|6.226e+06|
|model.language_model.layers.1.self_attn.q_proj|4.0|1.118e-02|3.932e+06|
|model.language_model.layers.19.mlp.gate_proj|4.0|1.076e-02|1.245e+07|
|model.language_model.layers.23.self_attn.q_proj|4.0|1.065e-02|3.932e+06|
|model.language_model.layers.34.mlp.gate_proj|4.0|1.057e-02|1.245e+07|
|model.language_model.layers.9.self_attn.o_proj|4.0|1.038e-02|2.621e+06|
|model.language_model.layers.30.self_attn.q_proj|4.0|9.162e-03|3.932e+06|
|model.language_model.layers.8.self_attn.o_proj|4.0|9.067e-03|2.621e+06|
|model.language_model.layers.22.mlp.down_proj|4.0|8.989e-03|6.226e+06|
|model.language_model.layers.4.self_attn.q_proj|4.0|8.957e-03|3.932e+06|
|model.language_model.layers.12.self_attn.o_proj|4.0|8.292e-03|2.621e+06|
|model.language_model.layers.24.self_attn.q_proj|4.0|7.865e-03|3.932e+06|
|model.language_model.layers.21.mlp.down_proj|4.0|7.463e-03|6.226e+06|
|model.language_model.layers.6.self_attn.o_proj|4.0|7.292e-03|2.621e+06|
|model.language_model.layers.20.mlp.down_proj|4.0|7.178e-03|6.226e+06|
|model.language_model.layers.18.self_attn.o_proj|4.0|6.701e-03|2.621e+06|
|model.language_model.layers.29.self_attn.q_proj|4.0|6.561e-03|3.932e+06|
|model.language_model.layers.31.self_attn.q_proj|4.0|6.511e-03|3.932e+06|
|model.language_model.layers.28.self_attn.q_proj|4.0|6.462e-03|3.932e+06|
|model.language_model.layers.27.self_attn.q_proj|4.0|6.454e-03|3.932e+06|
|model.language_model.layers.11.self_attn.o_proj|4.0|6.446e-03|2.621e+06|
|model.language_model.layers.23.mlp.gate_proj|4.0|5.757e-03|1.245e+07|
|model.language_model.layers.22.mlp.gate_proj|4.0|5.709e-03|1.245e+07|
|model.language_model.layers.20.mlp.gate_proj|4.0|5.616e-03|1.245e+07|
|model.language_model.layers.24.mlp.gate_proj|4.0|5.381e-03|1.245e+07|
|model.language_model.layers.25.mlp.gate_proj|4.0|5.341e-03|1.245e+07|
|model.language_model.layers.19.self_attn.o_proj|4.0|5.228e-03|2.621e+06|
|model.language_model.layers.26.mlp.gate_proj|4.0|4.801e-03|1.245e+07|
|model.language_model.layers.2.self_attn.o_proj|4.0|4.741e-03|2.621e+06|
|model.language_model.layers.25.self_attn.q_proj|4.0|4.705e-03|3.932e+06|
|model.language_model.layers.21.mlp.gate_proj|4.0|4.408e-03|1.245e+07|
|model.language_model.layers.7.self_attn.o_proj|4.0|4.364e-03|2.621e+06|
|model.language_model.layers.27.mlp.gate_proj|4.0|4.136e-03|1.245e+07|
|model.language_model.layers.0.self_attn.o_proj|4.0|3.897e-03|2.621e+06|
|model.language_model.layers.28.mlp.gate_proj|4.0|3.591e-03|1.245e+07|
|model.language_model.layers.33.mlp.gate_proj|4.0|3.403e-03|1.245e+07|
|model.language_model.layers.26.self_attn.q_proj|4.0|3.396e-03|3.932e+06|
|model.language_model.layers.1.self_attn.o_proj|4.0|3.232e-03|2.621e+06|
|model.language_model.layers.4.self_attn.o_proj|4.0|3.184e-03|2.621e+06|
|model.language_model.layers.29.mlp.gate_proj|4.0|2.988e-03|1.245e+07|
|model.language_model.layers.5.self_attn.o_proj|4.0|2.817e-03|2.621e+06|
|model.language_model.layers.30.mlp.gate_proj|4.0|2.560e-03|1.245e+07|
|model.language_model.layers.32.mlp.gate_proj|4.0|2.545e-03|1.245e+07|
|model.language_model.layers.31.mlp.gate_proj|4.0|2.436e-03|1.245e+07|
|lm_head|4.0|2.347e-03|9.724e+07|
|model.language_model.layers.3.self_attn.o_proj|4.0|2.178e-03|2.621e+06|
|model.language_model.layers.20.self_attn.o_proj|4.0|2.176e-03|2.621e+06|
|model.language_model.layers.22.self_attn.o_proj|4.0|1.992e-03|2.621e+06|
|model.language_model.layers.23.self_attn.o_proj|4.0|1.789e-03|2.621e+06|
|model.language_model.layers.21.self_attn.o_proj|4.0|9.915e-04|2.621e+06|
|model.language_model.layers.24.self_attn.o_proj|4.0|9.447e-04|2.621e+06|
|model.language_model.layers.34.self_attn.o_proj|4.0|9.331e-04|2.621e+06|
|model.language_model.layers.33.self_attn.o_proj|4.0|6.347e-04|2.621e+06|
|model.language_model.layers.32.self_attn.o_proj|4.0|5.828e-04|2.621e+06|
|model.language_model.layers.31.self_attn.o_proj|4.0|5.587e-04|2.621e+06|
|model.language_model.layers.29.self_attn.o_proj|4.0|4.603e-04|2.621e+06|
|model.language_model.layers.30.self_attn.o_proj|4.0|3.901e-04|2.621e+06|
|model.language_model.layers.25.self_attn.o_proj|4.0|3.839e-04|2.621e+06|
|model.language_model.layers.35.self_attn.o_proj|4.0|3.714e-04|2.621e+06|
|model.language_model.layers.28.self_attn.o_proj|4.0|3.613e-04|2.621e+06|
|model.language_model.layers.26.self_attn.o_proj|4.0|3.018e-04|2.621e+06|
|model.language_model.layers.27.self_attn.o_proj|4.0|2.740e-04|2.621e+06|

## Composed Config (`composed-config.yaml`)


```yaml
script: run_qwen3_vl_4b_autoquant_all_layers.py
scheme:
  name: wint4_aint8_autoquant_all_layers
  auto_quantize_bits: 8.0
  auto_quantize_method: gradient
  auto_quantize_score_size: 128
  coverage_mode: full
  coverage_fraction: 1.0
  quant_formats:
  - INT4_WEIGHT_INT8_ACT_CFG
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
  quant_pair: wint4_aint8
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
  base_format_name: INT4_WEIGHT_INT8_ACT_CFG
  format_names:
  - INT4_WEIGHT_INT8_ACT_CFG
  quant_format: int8
  quant_pair:
    name: wint4_aint8
    weight: int4
    activation: int8
    format_name: INT4_WEIGHT_INT8_ACT_CFG
```