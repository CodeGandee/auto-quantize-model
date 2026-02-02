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
|Model|`<ABSOLUTE_PATH>-quantize-model<ABSOLUTE_PATH>-VL-8B-Instruct`|
|Effective bits (from search)|`7.9979`|
|Total AutoQuant score|`2.475781e+00`|
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
|model.visual.patch_embed.proj|4.0|3.258e+04|4.424e+05|
|model.visual.blocks.0.mlp.linear_fc2|4.0|8.161e+03|1.240e+06|
|model.visual.blocks.0.mlp.linear_fc1|4.0|7.549e+03|1.240e+06|
|model.visual.blocks.9.mlp.linear_fc2|4.0|1.347e+03|1.240e+06|
|model.visual.blocks.0.attn.proj|4.0|1.054e+03|3.318e+05|
|model.visual.blocks.1.mlp.linear_fc2|4.0|7.197e+02|1.240e+06|
|model.visual.blocks.1.attn.proj|4.0|2.827e+02|3.318e+05|
|model.visual.blocks.2.mlp.linear_fc2|4.0|2.638e+02|1.240e+06|
|model.visual.blocks.1.mlp.linear_fc1|4.0|2.250e+02|1.240e+06|
|model.visual.blocks.3.mlp.linear_fc2|4.0|1.353e+02|1.240e+06|
|model.visual.blocks.4.mlp.linear_fc2|4.0|9.096e+01|1.240e+06|
|model.visual.blocks.2.attn.proj|4.0|6.652e+01|3.318e+05|
|model.visual.blocks.2.mlp.linear_fc1|4.0|6.144e+01|1.240e+06|
|model.visual.blocks.5.mlp.linear_fc2|4.0|4.774e+01|1.240e+06|
|model.language_model.layers.6.mlp.gate_proj|4.0|4.417e+01|2.517e+07|
|model.visual.blocks.3.mlp.linear_fc1|4.0|3.705e+01|1.240e+06|
|model.visual.blocks.1.attn.qkv|4.0|3.509e+01|9.953e+05|
|model.visual.blocks.2.attn.qkv|4.0|3.454e+01|9.953e+05|
|model.visual.blocks.10.mlp.linear_fc2|4.0|3.359e+01|1.240e+06|
|model.visual.blocks.4.mlp.linear_fc1|4.0|2.910e+01|1.240e+06|
|model.visual.blocks.3.attn.qkv|4.0|2.745e+01|9.953e+05|
|model.visual.blocks.6.mlp.linear_fc2|4.0|2.669e+01|1.240e+06|
|model.visual.blocks.5.mlp.linear_fc1|4.0|2.563e+01|1.240e+06|
|model.visual.blocks.22.mlp.linear_fc2|4.0|2.506e+01|1.240e+06|
|model.visual.blocks.3.attn.proj|4.0|2.167e+01|3.318e+05|
|model.visual.blocks.7.mlp.linear_fc2|4.0|2.149e+01|1.240e+06|
|model.visual.blocks.4.attn.proj|4.0|2.037e+01|3.318e+05|
|model.visual.blocks.6.mlp.linear_fc1|4.0|1.851e+01|1.240e+06|
|model.visual.blocks.21.mlp.linear_fc2|4.0|1.824e+01|1.240e+06|
|model.visual.blocks.23.mlp.linear_fc2|4.0|1.579e+01|1.240e+06|
|model.visual.blocks.8.mlp.linear_fc1|4.0|1.505e+01|1.240e+06|
|model.visual.blocks.4.attn.qkv|4.0|1.477e+01|9.953e+05|
|model.visual.blocks.5.attn.proj|4.0|1.412e+01|3.318e+05|
|model.visual.blocks.7.mlp.linear_fc1|4.0|1.409e+01|1.240e+06|
|model.visual.blocks.5.attn.qkv|4.0|1.391e+01|9.953e+05|
|model.visual.blocks.8.mlp.linear_fc2|4.0|1.365e+01|1.240e+06|
|model.visual.blocks.6.attn.proj|4.0|1.344e+01|3.318e+05|
|model.visual.blocks.0.attn.qkv|4.0|1.195e+01|9.953e+05|
|model.language_model.layers.6.mlp.down_proj|4.0|1.174e+01|1.258e+07|
|model.visual.blocks.8.attn.proj|4.0|1.086e+01|3.318e+05|
|model.visual.blocks.7.attn.proj|4.0|9.596e+00|3.318e+05|
|model.visual.blocks.22.mlp.linear_fc1|4.0|9.488e+00|1.240e+06|
|model.visual.blocks.9.mlp.linear_fc1|4.0|9.124e+00|1.240e+06|
|model.visual.blocks.8.attn.qkv|4.0|8.670e+00|9.953e+05|
|model.visual.blocks.6.attn.qkv|4.0|8.079e+00|9.953e+05|
|model.language_model.layers.5.mlp.down_proj|4.0|7.858e+00|1.258e+07|
|model.visual.blocks.10.mlp.linear_fc1|4.0|7.793e+00|1.240e+06|
|model.visual.blocks.10.attn.proj|4.0|7.293e+00|3.318e+05|
|model.language_model.layers.4.mlp.down_proj|4.0|7.162e+00|1.258e+07|
|model.visual.blocks.11.mlp.linear_fc2|4.0|6.874e+00|1.240e+06|
|model.visual.blocks.7.attn.qkv|4.0|6.244e+00|9.953e+05|
|model.visual.blocks.10.attn.qkv|4.0|5.594e+00|9.953e+05|
|model.visual.blocks.11.mlp.linear_fc1|4.0|5.574e+00|1.240e+06|
|model.visual.blocks.9.attn.proj|4.0|5.284e+00|3.318e+05|
|model.visual.blocks.20.mlp.linear_fc2|4.0|4.693e+00|1.240e+06|
|model.visual.blocks.23.mlp.linear_fc1|4.0|4.545e+00|1.240e+06|
|model.visual.blocks.21.mlp.linear_fc1|4.0|4.471e+00|1.240e+06|
|model.visual.blocks.11.attn.qkv|4.0|4.179e+00|9.953e+05|
|model.visual.blocks.12.attn.qkv|4.0|4.092e+00|9.953e+05|
|model.visual.blocks.14.mlp.linear_fc2|4.0|3.832e+00|1.240e+06|
|model.language_model.layers.8.mlp.down_proj|4.0|3.598e+00|1.258e+07|
|model.visual.blocks.20.mlp.linear_fc1|4.0|3.576e+00|1.240e+06|
|model.visual.blocks.13.mlp.linear_fc2|4.0|3.564e+00|1.240e+06|
|model.visual.blocks.9.attn.qkv|4.0|3.501e+00|9.953e+05|
|model.visual.blocks.12.mlp.linear_fc1|4.0|3.453e+00|1.240e+06|
|model.visual.blocks.15.mlp.linear_fc2|4.0|3.387e+00|1.240e+06|
|model.visual.merger.linear_fc2|4.0|3.359e+00|4.719e+06|
|model.visual.blocks.12.mlp.linear_fc2|4.0|3.291e+00|1.240e+06|
|model.visual.blocks.24.mlp.linear_fc2|4.0|3.179e+00|1.240e+06|
|model.visual.blocks.18.mlp.linear_fc1|4.0|3.096e+00|1.240e+06|
|model.visual.blocks.13.mlp.linear_fc1|4.0|3.085e+00|1.240e+06|
|model.language_model.layers.2.mlp.down_proj|4.0|3.081e+00|1.258e+07|
|model.visual.blocks.11.attn.proj|4.0|2.907e+00|3.318e+05|
|model.language_model.layers.7.mlp.gate_proj|4.0|2.755e+00|2.517e+07|
|model.language_model.layers.1.mlp.down_proj|4.0|2.722e+00|1.258e+07|
|model.visual.blocks.17.mlp.linear_fc1|4.0|2.704e+00|1.240e+06|
|model.language_model.layers.0.mlp.down_proj|4.0|2.680e+00|1.258e+07|
|model.visual.blocks.12.attn.proj|4.0|2.601e+00|3.318e+05|
|model.language_model.layers.7.mlp.down_proj|4.0|2.578e+00|1.258e+07|
|model.language_model.layers.3.mlp.down_proj|4.0|2.552e+00|1.258e+07|
|model.visual.blocks.26.mlp.linear_fc2|4.0|2.529e+00|1.240e+06|
|model.language_model.layers.10.mlp.down_proj|4.0|2.473e+00|1.258e+07|
|model.visual.blocks.24.mlp.linear_fc1|4.0|2.378e+00|1.240e+06|
|model.visual.blocks.19.mlp.linear_fc1|4.0|2.259e+00|1.240e+06|
|model.visual.merger.linear_fc1|4.0|1.988e+00|5.308e+06|
|model.visual.blocks.15.mlp.linear_fc1|4.0|1.905e+00|1.240e+06|
|model.visual.blocks.25.attn.qkv|4.0|1.815e+00|9.953e+05|
|model.language_model.layers.9.mlp.down_proj|4.0|1.800e+00|1.258e+07|
|model.visual.blocks.14.mlp.linear_fc1|4.0|1.788e+00|1.240e+06|
|model.language_model.layers.5.mlp.gate_proj|4.0|1.660e+00|2.517e+07|
|model.visual.deepstack_merger_list.1.linear_fc1|4.0|1.624e+00|5.308e+06|
|model.visual.blocks.13.attn.proj|4.0|1.557e+00|3.318e+05|
|model.visual.blocks.13.attn.qkv|4.0|1.555e+00|9.953e+05|
|model.visual.blocks.23.attn.qkv|4.0|1.512e+00|9.953e+05|
|model.visual.deepstack_merger_list.2.linear_fc1|4.0|1.476e+00|5.308e+06|
|model.visual.blocks.16.mlp.linear_fc1|4.0|1.385e+00|1.240e+06|
|model.language_model.layers.12.mlp.down_proj|4.0|1.251e+00|1.258e+07|
|model.visual.blocks.16.mlp.linear_fc2|4.0|1.229e+00|1.240e+06|
|model.language_model.layers.3.mlp.gate_proj|4.0|1.171e+00|2.517e+07|
|model.visual.blocks.14.attn.proj|4.0|1.118e+00|3.318e+05|
|model.language_model.layers.4.mlp.gate_proj|4.0|1.027e+00|2.517e+07|
|model.visual.blocks.14.attn.qkv|4.0|9.952e-01|9.953e+05|
|model.visual.blocks.25.mlp.linear_fc1|4.0|9.302e-01|1.240e+06|
|model.language_model.layers.9.self_attn.q_proj|4.0|9.068e-01|6.291e+06|
|model.visual.blocks.17.mlp.linear_fc2|4.0|8.750e-01|1.240e+06|
|model.language_model.layers.6.self_attn.q_proj|4.0|8.556e-01|6.291e+06|
|model.visual.blocks.15.attn.proj|4.0|8.216e-01|3.318e+05|
|model.language_model.layers.2.mlp.gate_proj|4.0|7.807e-01|2.517e+07|
|model.visual.blocks.24.attn.qkv|4.0|7.794e-01|9.953e+05|
|model.visual.blocks.16.attn.proj|4.0|7.498e-01|3.318e+05|
|model.visual.blocks.15.attn.qkv|4.0|7.433e-01|9.953e+05|
|model.visual.blocks.18.mlp.linear_fc2|4.0|7.432e-01|1.240e+06|
|model.language_model.layers.11.mlp.down_proj|4.0|7.139e-01|1.258e+07|
|model.language_model.layers.8.self_attn.q_proj|4.0|6.463e-01|6.291e+06|
|model.language_model.layers.14.mlp.down_proj|4.0|6.179e-01|1.258e+07|
|model.visual.blocks.22.attn.proj|4.0|6.135e-01|3.318e+05|
|model.language_model.layers.7.self_attn.q_proj|4.0|6.101e-01|6.291e+06|
|model.visual.blocks.22.attn.qkv|4.0|5.656e-01|9.953e+05|
|model.visual.blocks.23.attn.proj|4.0|5.211e-01|3.318e+05|
|model.visual.blocks.26.attn.qkv|4.0|4.897e-01|9.953e+05|
|model.visual.blocks.16.attn.qkv|4.0|4.628e-01|9.953e+05|
|model.visual.blocks.19.mlp.linear_fc2|4.0|4.513e-01|1.240e+06|
|model.visual.blocks.17.attn.qkv|4.0|4.506e-01|9.953e+05|
|model.visual.blocks.21.attn.proj|4.0|4.316e-01|3.318e+05|
|model.visual.blocks.17.attn.proj|4.0|4.312e-01|3.318e+05|
|model.visual.deepstack_merger_list.0.linear_fc1|4.0|4.144e-01|5.308e+06|
|model.visual.blocks.20.attn.qkv|4.0|3.949e-01|9.953e+05|
|model.visual.blocks.21.attn.qkv|4.0|3.858e-01|9.953e+05|
|model.language_model.layers.13.mlp.down_proj|4.0|3.788e-01|1.258e+07|
|model.visual.blocks.18.attn.qkv|4.0|3.759e-01|9.953e+05|
|model.language_model.layers.16.mlp.down_proj|4.0|3.655e-01|1.258e+07|
|model.language_model.layers.1.mlp.gate_proj|4.0|3.423e-01|2.517e+07|
|model.visual.blocks.18.attn.proj|4.0|3.415e-01|3.318e+05|
|model.language_model.layers.10.self_attn.q_proj|4.0|3.335e-01|6.291e+06|
|model.visual.blocks.19.attn.proj|4.0|3.052e-01|3.318e+05|
|model.language_model.layers.8.mlp.gate_proj|4.0|2.983e-01|2.517e+07|
|model.visual.deepstack_merger_list.1.linear_fc2|4.0|2.937e-01|4.719e+06|
|model.visual.deepstack_merger_list.0.linear_fc2|4.0|2.819e-01|4.719e+06|
|model.visual.blocks.20.attn.proj|4.0|2.797e-01|3.318e+05|
|model.visual.blocks.26.mlp.linear_fc1|4.0|2.685e-01|1.240e+06|
|model.visual.blocks.25.mlp.linear_fc2|4.0|2.622e-01|1.240e+06|
|model.visual.deepstack_merger_list.2.linear_fc2|4.0|2.409e-01|4.719e+06|
|model.language_model.layers.13.self_attn.q_proj|4.0|2.271e-01|6.291e+06|
|model.visual.blocks.24.attn.proj|4.0|2.178e-01|3.318e+05|
|model.language_model.layers.11.self_attn.q_proj|4.0|2.166e-01|6.291e+06|
|model.visual.blocks.19.attn.qkv|4.0|1.986e-01|9.953e+05|
|model.language_model.layers.15.mlp.down_proj|4.0|1.909e-01|1.258e+07|
|model.language_model.layers.9.mlp.gate_proj|4.0|1.724e-01|2.517e+07|
|model.language_model.layers.34.self_attn.q_proj|4.0|1.379e-01|6.291e+06|
|model.language_model.layers.12.self_attn.q_proj|4.0|1.370e-01|6.291e+06|
|model.language_model.layers.5.self_attn.q_proj|4.0|1.340e-01|6.291e+06|
|model.language_model.layers.10.mlp.gate_proj|4.0|1.332e-01|2.517e+07|
|model.language_model.layers.34.mlp.down_proj|4.0|1.189e-01|1.258e+07|
|model.language_model.layers.3.self_attn.q_proj|4.0|1.157e-01|6.291e+06|
|model.language_model.layers.14.self_attn.q_proj|4.0|1.130e-01|6.291e+06|
|model.language_model.layers.8.self_attn.o_proj|4.0|1.116e-01|4.194e+06|
|model.language_model.layers.15.self_attn.q_proj|4.0|1.070e-01|6.291e+06|
|model.language_model.layers.6.self_attn.o_proj|4.0|1.061e-01|4.194e+06|
|model.language_model.layers.15.self_attn.o_proj|4.0|8.685e-02|4.194e+06|
|model.language_model.layers.11.mlp.gate_proj|4.0|7.858e-02|2.517e+07|
|model.language_model.layers.18.mlp.down_proj|4.0|7.835e-02|1.258e+07|
|model.language_model.layers.12.mlp.gate_proj|4.0|7.828e-02|2.517e+07|
|model.language_model.layers.0.self_attn.q_proj|4.0|7.365e-02|6.291e+06|
|model.language_model.layers.32.self_attn.q_proj|4.0|7.356e-02|6.291e+06|
|model.language_model.layers.4.self_attn.q_proj|4.0|7.126e-02|6.291e+06|
|model.visual.blocks.25.attn.proj|4.0|6.981e-02|3.318e+05|
|model.language_model.layers.16.mlp.gate_proj|4.0|6.597e-02|2.517e+07|
|model.language_model.layers.0.mlp.gate_proj|4.0|6.434e-02|2.517e+07|
|model.language_model.layers.1.self_attn.q_proj|4.0|6.146e-02|6.291e+06|
|model.language_model.layers.7.self_attn.o_proj|4.0|6.087e-02|4.194e+06|
|model.language_model.layers.9.self_attn.o_proj|4.0|6.075e-02|4.194e+06|
|model.visual.blocks.26.attn.proj|4.0|5.191e-02|3.318e+05|
|model.language_model.layers.16.self_attn.q_proj|4.0|5.042e-02|6.291e+06|
|model.language_model.layers.17.self_attn.q_proj|4.0|4.805e-02|6.291e+06|
|model.language_model.layers.14.mlp.gate_proj|4.0|4.684e-02|2.517e+07|
|model.language_model.layers.13.mlp.gate_proj|4.0|4.580e-02|2.517e+07|
|model.language_model.layers.4.self_attn.o_proj|4.0|4.578e-02|4.194e+06|
|model.language_model.layers.5.self_attn.o_proj|4.0|4.479e-02|4.194e+06|
|model.language_model.layers.22.self_attn.q_proj|4.0|4.450e-02|6.291e+06|
|model.language_model.layers.35.mlp.gate_proj|4.0|4.377e-02|2.517e+07|
|model.language_model.layers.3.self_attn.o_proj|4.0|4.254e-02|4.194e+06|
|model.language_model.layers.14.self_attn.o_proj|4.0|4.252e-02|4.194e+06|
|model.language_model.layers.31.mlp.down_proj|4.0|4.223e-02|1.258e+07|
|model.language_model.layers.15.mlp.gate_proj|4.0|3.786e-02|2.517e+07|
|model.language_model.layers.33.self_attn.q_proj|4.0|3.647e-02|6.291e+06|
|model.language_model.layers.28.mlp.down_proj|4.0|3.607e-02|1.258e+07|
|model.language_model.layers.10.self_attn.o_proj|4.0|3.558e-02|4.194e+06|
|model.language_model.layers.2.self_attn.q_proj|4.0|3.504e-02|6.291e+06|
|model.language_model.layers.29.mlp.down_proj|4.0|3.486e-02|1.258e+07|
|model.language_model.layers.30.mlp.down_proj|4.0|3.379e-02|1.258e+07|
|model.language_model.layers.24.mlp.down_proj|4.0|3.055e-02|1.258e+07|
|model.language_model.layers.33.mlp.down_proj|4.0|3.029e-02|1.258e+07|
|model.language_model.layers.26.mlp.down_proj|4.0|2.967e-02|1.258e+07|
|model.language_model.layers.35.mlp.down_proj|4.0|2.911e-02|1.258e+07|
|model.language_model.layers.0.self_attn.o_proj|4.0|2.756e-02|4.194e+06|
|model.language_model.layers.32.mlp.down_proj|4.0|2.719e-02|1.258e+07|
|model.language_model.layers.13.self_attn.o_proj|4.0|2.714e-02|4.194e+06|
|model.language_model.layers.11.self_attn.o_proj|4.0|2.678e-02|4.194e+06|
|model.language_model.layers.25.mlp.down_proj|4.0|2.673e-02|1.258e+07|
|model.language_model.layers.1.self_attn.o_proj|4.0|2.624e-02|4.194e+06|
|model.language_model.layers.21.self_attn.q_proj|4.0|2.538e-02|6.291e+06|
|model.language_model.layers.27.mlp.down_proj|4.0|2.517e-02|1.258e+07|
|model.language_model.layers.23.mlp.down_proj|4.0|2.411e-02|1.258e+07|
|model.language_model.layers.30.self_attn.q_proj|4.0|2.406e-02|6.291e+06|
|model.language_model.layers.12.self_attn.o_proj|4.0|2.387e-02|4.194e+06|
|model.language_model.layers.35.self_attn.q_proj|4.0|2.291e-02|6.291e+06|
|model.language_model.layers.18.self_attn.q_proj|4.0|2.229e-02|6.291e+06|
|model.language_model.layers.19.self_attn.q_proj|4.0|2.196e-02|6.291e+06|
|model.language_model.layers.16.self_attn.o_proj|4.0|2.174e-02|4.194e+06|
|model.language_model.layers.23.self_attn.q_proj|4.0|2.027e-02|6.291e+06|
|model.language_model.layers.24.self_attn.q_proj|4.0|1.974e-02|6.291e+06|
|model.language_model.layers.31.self_attn.q_proj|4.0|1.963e-02|6.291e+06|
|model.language_model.layers.27.self_attn.q_proj|4.0|1.927e-02|6.291e+06|
|model.language_model.layers.17.self_attn.o_proj|4.0|1.711e-02|4.194e+06|
|model.language_model.layers.2.self_attn.o_proj|4.0|1.479e-02|4.194e+06|
|model.language_model.layers.20.self_attn.q_proj|4.0|1.436e-02|6.291e+06|
|model.language_model.layers.25.self_attn.q_proj|4.0|1.392e-02|6.291e+06|
|model.language_model.layers.28.self_attn.q_proj|4.0|1.347e-02|6.291e+06|
|model.language_model.layers.17.mlp.down_proj|4.0|1.257e-02|1.258e+07|
|model.language_model.layers.21.mlp.down_proj|4.0|1.217e-02|1.258e+07|
|model.language_model.layers.22.mlp.down_proj|4.0|1.178e-02|1.258e+07|
|model.language_model.layers.34.mlp.gate_proj|4.0|1.086e-02|2.517e+07|
|model.language_model.layers.29.self_attn.q_proj|4.0|1.081e-02|6.291e+06|
|model.language_model.layers.19.mlp.down_proj|4.0|1.069e-02|1.258e+07|
|model.language_model.layers.17.mlp.gate_proj|4.0|9.857e-03|2.517e+07|
|model.language_model.layers.20.mlp.down_proj|4.0|9.272e-03|1.258e+07|
|model.language_model.layers.26.self_attn.q_proj|4.0|9.044e-03|6.291e+06|
|model.language_model.layers.23.mlp.gate_proj|4.0|7.153e-03|2.517e+07|
|model.language_model.layers.22.mlp.gate_proj|4.0|6.583e-03|2.517e+07|
|model.language_model.layers.24.mlp.gate_proj|4.0|6.472e-03|2.517e+07|
|model.language_model.layers.25.mlp.gate_proj|4.0|6.281e-03|2.517e+07|
|model.language_model.layers.18.mlp.gate_proj|4.0|6.245e-03|2.517e+07|
|model.language_model.layers.21.mlp.gate_proj|4.0|6.211e-03|2.517e+07|
|model.language_model.layers.20.mlp.gate_proj|4.0|5.804e-03|2.517e+07|
|model.language_model.layers.19.mlp.gate_proj|4.0|5.804e-03|2.517e+07|
|model.language_model.layers.26.mlp.gate_proj|4.0|5.485e-03|2.517e+07|
|model.language_model.layers.27.mlp.gate_proj|4.0|4.994e-03|2.517e+07|
|model.language_model.layers.28.mlp.gate_proj|4.0|4.778e-03|2.517e+07|
|model.language_model.layers.33.mlp.gate_proj|4.0|4.411e-03|2.517e+07|
|model.language_model.layers.29.mlp.gate_proj|4.0|3.642e-03|2.517e+07|
|model.language_model.layers.31.mlp.gate_proj|4.0|3.189e-03|2.517e+07|
|model.language_model.layers.30.mlp.gate_proj|4.0|3.176e-03|2.517e+07|
|model.language_model.layers.22.self_attn.o_proj|4.0|3.094e-03|4.194e+06|
|model.language_model.layers.23.self_attn.o_proj|4.0|3.082e-03|4.194e+06|
|model.language_model.layers.32.mlp.gate_proj|4.0|3.060e-03|2.517e+07|
|lm_head|4.0|2.905e-03|1.556e+08|
|model.language_model.layers.24.self_attn.o_proj|4.0|2.502e-03|4.194e+06|
|model.language_model.layers.18.self_attn.o_proj|4.0|2.160e-03|4.194e+06|
|model.language_model.layers.19.self_attn.o_proj|4.0|1.513e-03|4.194e+06|
|model.language_model.layers.34.self_attn.o_proj|4.0|1.436e-03|4.194e+06|
|model.language_model.layers.20.self_attn.o_proj|4.0|1.396e-03|4.194e+06|
|model.language_model.layers.21.self_attn.o_proj|4.0|1.254e-03|4.194e+06|
|model.language_model.layers.25.self_attn.o_proj|4.0|1.047e-03|4.194e+06|
|model.language_model.layers.33.self_attn.o_proj|4.0|8.263e-04|4.194e+06|
|model.language_model.layers.27.self_attn.o_proj|4.0|7.953e-04|4.194e+06|
|model.language_model.layers.28.self_attn.o_proj|4.0|7.281e-04|4.194e+06|
|model.language_model.layers.26.self_attn.o_proj|4.0|7.104e-04|4.194e+06|
|model.language_model.layers.32.self_attn.o_proj|4.0|7.033e-04|4.194e+06|
|model.language_model.layers.29.self_attn.o_proj|4.0|6.695e-04|4.194e+06|
|model.language_model.layers.31.self_attn.o_proj|4.0|6.022e-04|4.194e+06|
|model.language_model.layers.30.self_attn.o_proj|4.0|4.349e-04|4.194e+06|
|model.language_model.layers.35.self_attn.o_proj|4.0|3.947e-04|4.194e+06|

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
  model_dir: <ABSOLUTE_PATH>-quantize-model<ABSOLUTE_PATH>-VL-8B-Instruct
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