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
|Model|`<ABSOLUTE_PATH>-quantize-model<ABSOLUTE_PATH>-VL-8B-Instruct`|
|Effective bits (from search)|`7.9979`|
|Total AutoQuant score|`7.913743e-01`|
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
|model.visual.patch_embed.proj|4.0|3.289e+04|4.424e+05|
|model.visual.blocks.0.mlp.linear_fc2|4.0|2.231e+03|1.240e+06|
|model.visual.blocks.0.mlp.linear_fc1|4.0|9.344e+02|1.240e+06|
|model.visual.blocks.0.attn.proj|4.0|6.643e+02|3.318e+05|
|model.visual.blocks.1.attn.proj|4.0|2.230e+02|3.318e+05|
|model.visual.blocks.1.mlp.linear_fc2|4.0|2.158e+02|1.240e+06|
|model.visual.blocks.2.mlp.linear_fc2|4.0|1.381e+02|1.240e+06|
|model.visual.blocks.1.mlp.linear_fc1|4.0|1.082e+02|1.240e+06|
|model.visual.blocks.3.mlp.linear_fc2|4.0|6.739e+01|1.240e+06|
|model.visual.blocks.2.attn.proj|4.0|5.594e+01|3.318e+05|
|model.visual.blocks.2.mlp.linear_fc1|4.0|5.140e+01|1.240e+06|
|model.visual.blocks.4.mlp.linear_fc2|4.0|4.488e+01|1.240e+06|
|model.visual.blocks.5.mlp.linear_fc2|4.0|3.029e+01|1.240e+06|
|model.visual.blocks.3.mlp.linear_fc1|4.0|2.864e+01|1.240e+06|
|model.visual.blocks.2.attn.qkv|4.0|2.606e+01|9.953e+05|
|model.visual.blocks.22.mlp.linear_fc2|4.0|2.476e+01|1.240e+06|
|model.visual.blocks.3.attn.qkv|4.0|2.385e+01|9.953e+05|
|model.visual.blocks.1.attn.qkv|4.0|2.362e+01|9.953e+05|
|model.visual.blocks.4.mlp.linear_fc1|4.0|2.243e+01|1.240e+06|
|model.visual.blocks.6.mlp.linear_fc2|4.0|1.963e+01|1.240e+06|
|model.visual.blocks.4.attn.proj|4.0|1.829e+01|3.318e+05|
|model.visual.blocks.21.mlp.linear_fc2|4.0|1.758e+01|1.240e+06|
|model.visual.blocks.5.mlp.linear_fc1|4.0|1.723e+01|1.240e+06|
|model.visual.blocks.3.attn.proj|4.0|1.702e+01|3.318e+05|
|model.visual.blocks.23.mlp.linear_fc2|4.0|1.552e+01|1.240e+06|
|model.visual.blocks.7.mlp.linear_fc2|4.0|1.321e+01|1.240e+06|
|model.visual.blocks.5.attn.proj|4.0|1.258e+01|3.318e+05|
|model.visual.blocks.6.attn.proj|4.0|1.217e+01|3.318e+05|
|model.visual.blocks.4.attn.qkv|4.0|1.208e+01|9.953e+05|
|model.visual.blocks.5.attn.qkv|4.0|1.171e+01|9.953e+05|
|model.visual.blocks.6.mlp.linear_fc1|4.0|1.170e+01|1.240e+06|
|model.visual.blocks.8.attn.proj|4.0|1.024e+01|3.318e+05|
|model.visual.blocks.8.mlp.linear_fc2|4.0|1.007e+01|1.240e+06|
|model.visual.blocks.8.mlp.linear_fc1|4.0|9.724e+00|1.240e+06|
|model.visual.blocks.22.mlp.linear_fc1|4.0|9.273e+00|1.240e+06|
|model.visual.blocks.9.mlp.linear_fc2|4.0|8.957e+00|1.240e+06|
|model.visual.blocks.7.attn.proj|4.0|8.812e+00|3.318e+05|
|model.visual.blocks.7.mlp.linear_fc1|4.0|8.760e+00|1.240e+06|
|model.visual.blocks.0.attn.qkv|4.0|8.197e+00|9.953e+05|
|model.visual.blocks.8.attn.qkv|4.0|7.798e+00|9.953e+05|
|model.visual.blocks.6.attn.qkv|4.0|7.168e+00|9.953e+05|
|model.visual.blocks.9.mlp.linear_fc1|4.0|6.747e+00|1.240e+06|
|model.visual.blocks.10.attn.proj|4.0|6.264e+00|3.318e+05|
|model.visual.blocks.10.mlp.linear_fc1|4.0|5.928e+00|1.240e+06|
|model.visual.blocks.7.attn.qkv|4.0|5.788e+00|9.953e+05|
|model.visual.blocks.10.mlp.linear_fc2|4.0|5.034e+00|1.240e+06|
|model.visual.blocks.9.attn.proj|4.0|4.858e+00|3.318e+05|
|model.visual.blocks.10.attn.qkv|4.0|4.381e+00|9.953e+05|
|model.visual.blocks.23.mlp.linear_fc1|4.0|4.347e+00|1.240e+06|
|model.visual.blocks.11.mlp.linear_fc1|4.0|4.303e+00|1.240e+06|
|model.visual.blocks.21.mlp.linear_fc1|4.0|4.288e+00|1.240e+06|
|model.visual.blocks.20.mlp.linear_fc2|4.0|4.268e+00|1.240e+06|
|model.visual.blocks.20.mlp.linear_fc1|4.0|3.528e+00|1.240e+06|
|model.visual.blocks.11.mlp.linear_fc2|4.0|3.373e+00|1.240e+06|
|model.visual.blocks.12.attn.qkv|4.0|3.353e+00|9.953e+05|
|model.visual.blocks.11.attn.qkv|4.0|3.273e+00|9.953e+05|
|model.visual.blocks.9.attn.qkv|4.0|3.064e+00|9.953e+05|
|model.visual.blocks.24.mlp.linear_fc2|4.0|3.042e+00|1.240e+06|
|model.visual.blocks.12.mlp.linear_fc1|4.0|2.901e+00|1.240e+06|
|model.visual.blocks.18.mlp.linear_fc1|4.0|2.601e+00|1.240e+06|
|model.visual.blocks.11.attn.proj|4.0|2.575e+00|3.318e+05|
|model.visual.blocks.13.mlp.linear_fc1|4.0|2.392e+00|1.240e+06|
|model.visual.blocks.24.mlp.linear_fc1|4.0|2.356e+00|1.240e+06|
|model.visual.blocks.26.mlp.linear_fc2|4.0|2.333e+00|1.240e+06|
|model.visual.blocks.17.mlp.linear_fc1|4.0|2.190e+00|1.240e+06|
|model.visual.blocks.12.attn.proj|4.0|2.068e+00|3.318e+05|
|model.visual.blocks.19.mlp.linear_fc1|4.0|2.016e+00|1.240e+06|
|model.visual.blocks.12.mlp.linear_fc2|4.0|1.890e+00|1.240e+06|
|model.visual.blocks.13.mlp.linear_fc2|4.0|1.607e+00|1.240e+06|
|model.visual.blocks.15.mlp.linear_fc1|4.0|1.521e+00|1.240e+06|
|model.visual.blocks.14.mlp.linear_fc1|4.0|1.489e+00|1.240e+06|
|model.visual.blocks.25.attn.qkv|4.0|1.424e+00|9.953e+05|
|model.visual.blocks.13.attn.proj|4.0|1.407e+00|3.318e+05|
|model.visual.blocks.16.mlp.linear_fc1|4.0|1.324e+00|1.240e+06|
|model.visual.blocks.23.attn.qkv|4.0|1.280e+00|9.953e+05|
|model.visual.blocks.13.attn.qkv|4.0|1.257e+00|9.953e+05|
|model.visual.blocks.14.mlp.linear_fc2|4.0|9.517e-01|1.240e+06|
|model.visual.blocks.15.mlp.linear_fc2|4.0|9.506e-01|1.240e+06|
|model.visual.blocks.14.attn.proj|4.0|8.708e-01|3.318e+05|
|model.visual.blocks.25.mlp.linear_fc1|4.0|8.699e-01|1.240e+06|
|model.visual.blocks.14.attn.qkv|4.0|8.462e-01|9.953e+05|
|model.visual.blocks.15.attn.proj|4.0|6.265e-01|3.318e+05|
|model.visual.blocks.24.attn.qkv|4.0|6.241e-01|9.953e+05|
|model.visual.deepstack_merger_list.2.linear_fc1|4.0|6.154e-01|5.308e+06|
|model.visual.blocks.15.attn.qkv|4.0|6.073e-01|9.953e+05|
|model.visual.blocks.17.mlp.linear_fc2|4.0|5.984e-01|1.240e+06|
|model.language_model.layers.6.mlp.down_proj|4.0|5.948e-01|1.258e+07|
|model.visual.deepstack_merger_list.1.linear_fc1|4.0|5.918e-01|5.308e+06|
|model.visual.blocks.22.attn.proj|4.0|5.759e-01|3.318e+05|
|model.language_model.layers.7.mlp.gate_proj|4.0|5.646e-01|2.517e+07|
|model.visual.blocks.16.mlp.linear_fc2|4.0|5.646e-01|1.240e+06|
|model.language_model.layers.9.self_attn.q_proj|4.0|5.442e-01|6.291e+06|
|model.language_model.layers.6.mlp.gate_proj|4.0|5.312e-01|2.517e+07|
|model.language_model.layers.5.mlp.gate_proj|4.0|5.160e-01|2.517e+07|
|model.visual.blocks.16.attn.proj|4.0|5.131e-01|3.318e+05|
|model.visual.blocks.23.attn.proj|4.0|4.815e-01|3.318e+05|
|model.language_model.layers.3.mlp.gate_proj|4.0|4.502e-01|2.517e+07|
|model.visual.blocks.22.attn.qkv|4.0|4.238e-01|9.953e+05|
|model.visual.blocks.18.mlp.linear_fc2|4.0|4.077e-01|1.240e+06|
|model.visual.blocks.16.attn.qkv|4.0|3.693e-01|9.953e+05|
|model.visual.blocks.17.attn.proj|4.0|3.590e-01|3.318e+05|
|model.language_model.layers.4.mlp.gate_proj|4.0|3.584e-01|2.517e+07|
|model.visual.blocks.17.attn.qkv|4.0|3.567e-01|9.953e+05|
|model.visual.blocks.21.attn.proj|4.0|3.540e-01|3.318e+05|
|model.visual.blocks.26.attn.qkv|4.0|3.370e-01|9.953e+05|
|model.visual.blocks.20.attn.qkv|4.0|3.348e-01|9.953e+05|
|model.visual.blocks.19.mlp.linear_fc2|4.0|3.186e-01|1.240e+06|
|model.language_model.layers.2.mlp.gate_proj|4.0|3.076e-01|2.517e+07|
|model.visual.blocks.18.attn.qkv|4.0|2.982e-01|9.953e+05|
|model.visual.deepstack_merger_list.0.linear_fc1|4.0|2.970e-01|5.308e+06|
|model.visual.blocks.21.attn.qkv|4.0|2.960e-01|9.953e+05|
|model.language_model.layers.8.self_attn.q_proj|4.0|2.873e-01|6.291e+06|
|model.visual.blocks.18.attn.proj|4.0|2.742e-01|3.318e+05|
|model.visual.blocks.25.mlp.linear_fc2|4.0|2.448e-01|1.240e+06|
|model.visual.blocks.20.attn.proj|4.0|2.308e-01|3.318e+05|
|model.visual.blocks.19.attn.proj|4.0|2.211e-01|3.318e+05|
|model.language_model.layers.6.self_attn.q_proj|4.0|2.185e-01|6.291e+06|
|model.language_model.layers.7.self_attn.q_proj|4.0|2.042e-01|6.291e+06|
|model.visual.deepstack_merger_list.1.linear_fc2|4.0|2.036e-01|4.719e+06|
|model.visual.blocks.24.attn.proj|4.0|2.036e-01|3.318e+05|
|model.visual.deepstack_merger_list.2.linear_fc2|4.0|1.899e-01|4.719e+06|
|model.language_model.layers.8.mlp.gate_proj|4.0|1.763e-01|2.517e+07|
|model.language_model.layers.5.mlp.down_proj|4.0|1.625e-01|1.258e+07|
|model.visual.deepstack_merger_list.0.linear_fc2|4.0|1.585e-01|4.719e+06|
|model.visual.blocks.26.mlp.linear_fc1|4.0|1.490e-01|1.240e+06|
|model.visual.blocks.19.attn.qkv|4.0|1.399e-01|9.953e+05|
|model.language_model.layers.1.mlp.gate_proj|4.0|1.295e-01|2.517e+07|
|model.language_model.layers.10.self_attn.q_proj|4.0|1.248e-01|6.291e+06|
|model.language_model.layers.9.mlp.gate_proj|4.0|1.242e-01|2.517e+07|
|model.language_model.layers.4.mlp.down_proj|4.0|1.160e-01|1.258e+07|
|model.language_model.layers.7.mlp.down_proj|4.0|1.141e-01|1.258e+07|
|model.language_model.layers.8.self_attn.o_proj|4.0|1.004e-01|4.194e+06|
|model.language_model.layers.11.self_attn.q_proj|4.0|9.875e-02|6.291e+06|
|model.language_model.layers.3.mlp.down_proj|4.0|9.834e-02|1.258e+07|
|model.visual.merger.linear_fc2|4.0|9.422e-02|4.719e+06|
|model.language_model.layers.6.self_attn.o_proj|4.0|9.411e-02|4.194e+06|
|model.language_model.layers.8.mlp.down_proj|4.0|9.168e-02|1.258e+07|
|model.language_model.layers.10.mlp.gate_proj|4.0|8.135e-02|2.517e+07|
|model.language_model.layers.13.self_attn.q_proj|4.0|8.054e-02|6.291e+06|
|model.visual.merger.linear_fc1|4.0|7.781e-02|5.308e+06|
|model.language_model.layers.1.mlp.down_proj|4.0|7.521e-02|1.258e+07|
|model.language_model.layers.34.self_attn.q_proj|4.0|7.141e-02|6.291e+06|
|model.language_model.layers.12.self_attn.q_proj|4.0|7.018e-02|6.291e+06|
|model.language_model.layers.0.mlp.down_proj|4.0|6.769e-02|1.258e+07|
|model.language_model.layers.9.mlp.down_proj|4.0|6.527e-02|1.258e+07|
|model.visual.blocks.25.attn.proj|4.0|6.420e-02|3.318e+05|
|model.language_model.layers.14.self_attn.q_proj|4.0|5.892e-02|6.291e+06|
|model.language_model.layers.3.self_attn.q_proj|4.0|5.693e-02|6.291e+06|
|model.language_model.layers.2.mlp.down_proj|4.0|5.686e-02|1.258e+07|
|model.language_model.layers.5.self_attn.q_proj|4.0|5.427e-02|6.291e+06|
|model.language_model.layers.4.self_attn.q_proj|4.0|5.089e-02|6.291e+06|
|model.language_model.layers.11.mlp.gate_proj|4.0|5.076e-02|2.517e+07|
|model.language_model.layers.0.mlp.gate_proj|4.0|4.808e-02|2.517e+07|
|model.visual.blocks.26.attn.proj|4.0|4.761e-02|3.318e+05|
|model.language_model.layers.10.mlp.down_proj|4.0|4.750e-02|1.258e+07|
|model.language_model.layers.32.self_attn.q_proj|4.0|4.749e-02|6.291e+06|
|model.language_model.layers.9.self_attn.o_proj|4.0|4.736e-02|4.194e+06|
|model.language_model.layers.7.self_attn.o_proj|4.0|4.705e-02|4.194e+06|
|model.language_model.layers.12.mlp.gate_proj|4.0|4.423e-02|2.517e+07|
|model.language_model.layers.5.self_attn.o_proj|4.0|3.833e-02|4.194e+06|
|model.language_model.layers.4.self_attn.o_proj|4.0|3.830e-02|4.194e+06|
|model.language_model.layers.3.self_attn.o_proj|4.0|3.767e-02|4.194e+06|
|model.language_model.layers.0.self_attn.q_proj|4.0|3.761e-02|6.291e+06|
|model.language_model.layers.15.self_attn.q_proj|4.0|3.346e-02|6.291e+06|
|model.language_model.layers.1.self_attn.q_proj|4.0|3.128e-02|6.291e+06|
|model.language_model.layers.22.self_attn.q_proj|4.0|2.775e-02|6.291e+06|
|model.language_model.layers.11.mlp.down_proj|4.0|2.673e-02|1.258e+07|
|model.language_model.layers.10.self_attn.o_proj|4.0|2.613e-02|4.194e+06|
|model.language_model.layers.0.self_attn.o_proj|4.0|2.470e-02|4.194e+06|
|model.language_model.layers.33.self_attn.q_proj|4.0|2.397e-02|6.291e+06|
|model.language_model.layers.13.mlp.gate_proj|4.0|2.346e-02|2.517e+07|
|model.language_model.layers.12.mlp.down_proj|4.0|2.274e-02|1.258e+07|
|model.language_model.layers.1.self_attn.o_proj|4.0|2.256e-02|4.194e+06|
|model.language_model.layers.11.self_attn.o_proj|4.0|2.221e-02|4.194e+06|
|model.language_model.layers.2.self_attn.q_proj|4.0|2.156e-02|6.291e+06|
|model.language_model.layers.16.self_attn.q_proj|4.0|2.121e-02|6.291e+06|
|model.language_model.layers.14.mlp.gate_proj|4.0|2.112e-02|2.517e+07|
|model.language_model.layers.12.self_attn.o_proj|4.0|2.065e-02|4.194e+06|
|model.language_model.layers.17.self_attn.q_proj|4.0|2.047e-02|6.291e+06|
|model.language_model.layers.21.self_attn.q_proj|4.0|1.612e-02|6.291e+06|
|model.language_model.layers.27.self_attn.q_proj|4.0|1.450e-02|6.291e+06|
|model.language_model.layers.35.self_attn.q_proj|4.0|1.389e-02|6.291e+06|
|model.language_model.layers.31.self_attn.q_proj|4.0|1.382e-02|6.291e+06|
|model.language_model.layers.30.self_attn.q_proj|4.0|1.368e-02|6.291e+06|
|model.language_model.layers.15.mlp.gate_proj|4.0|1.340e-02|2.517e+07|
|model.language_model.layers.24.self_attn.q_proj|4.0|1.332e-02|6.291e+06|
|model.language_model.layers.13.mlp.down_proj|4.0|1.293e-02|1.258e+07|
|model.language_model.layers.23.self_attn.q_proj|4.0|1.232e-02|6.291e+06|
|model.language_model.layers.2.self_attn.o_proj|4.0|1.097e-02|4.194e+06|
|model.language_model.layers.19.self_attn.q_proj|4.0|1.008e-02|6.291e+06|
|model.language_model.layers.14.self_attn.o_proj|4.0|1.006e-02|4.194e+06|
|model.language_model.layers.25.self_attn.q_proj|4.0|9.740e-03|6.291e+06|
|model.language_model.layers.16.mlp.gate_proj|4.0|9.727e-03|2.517e+07|
|model.language_model.layers.14.mlp.down_proj|4.0|9.558e-03|1.258e+07|
|model.language_model.layers.18.self_attn.q_proj|4.0|8.873e-03|6.291e+06|
|model.language_model.layers.13.self_attn.o_proj|4.0|8.738e-03|4.194e+06|
|model.language_model.layers.15.self_attn.o_proj|4.0|8.670e-03|4.194e+06|
|model.language_model.layers.28.self_attn.q_proj|4.0|8.636e-03|6.291e+06|
|model.language_model.layers.20.self_attn.q_proj|4.0|8.261e-03|6.291e+06|
|model.language_model.layers.15.mlp.down_proj|4.0|7.539e-03|1.258e+07|
|model.language_model.layers.29.self_attn.q_proj|4.0|6.833e-03|6.291e+06|
|model.language_model.layers.23.mlp.gate_proj|4.0|6.049e-03|2.517e+07|
|model.language_model.layers.26.self_attn.q_proj|4.0|5.902e-03|6.291e+06|
|model.language_model.layers.17.mlp.gate_proj|4.0|5.900e-03|2.517e+07|
|model.language_model.layers.16.self_attn.o_proj|4.0|5.534e-03|4.194e+06|
|model.language_model.layers.22.mlp.gate_proj|4.0|5.513e-03|2.517e+07|
|model.language_model.layers.16.mlp.down_proj|4.0|5.489e-03|1.258e+07|
|model.language_model.layers.24.mlp.gate_proj|4.0|5.412e-03|2.517e+07|
|model.language_model.layers.25.mlp.gate_proj|4.0|5.259e-03|2.517e+07|
|model.language_model.layers.21.mlp.gate_proj|4.0|5.003e-03|2.517e+07|
|model.language_model.layers.18.mlp.gate_proj|4.0|4.406e-03|2.517e+07|
|model.language_model.layers.19.mlp.gate_proj|4.0|4.343e-03|2.517e+07|
|model.language_model.layers.34.mlp.gate_proj|4.0|4.331e-03|2.517e+07|
|model.language_model.layers.20.mlp.gate_proj|4.0|4.298e-03|2.517e+07|
|model.language_model.layers.26.mlp.gate_proj|4.0|4.240e-03|2.517e+07|
|model.language_model.layers.22.mlp.down_proj|4.0|3.963e-03|1.258e+07|
|model.language_model.layers.23.mlp.down_proj|4.0|3.937e-03|1.258e+07|
|model.language_model.layers.27.mlp.gate_proj|4.0|3.863e-03|2.517e+07|
|model.language_model.layers.28.mlp.gate_proj|4.0|3.768e-03|2.517e+07|
|model.language_model.layers.35.mlp.gate_proj|4.0|3.522e-03|2.517e+07|
|model.language_model.layers.33.mlp.gate_proj|4.0|3.400e-03|2.517e+07|
|model.language_model.layers.24.mlp.down_proj|4.0|3.367e-03|1.258e+07|
|model.language_model.layers.25.mlp.down_proj|4.0|3.142e-03|1.258e+07|
|model.language_model.layers.21.mlp.down_proj|4.0|3.093e-03|1.258e+07|
|model.language_model.layers.17.mlp.down_proj|4.0|3.054e-03|1.258e+07|
|model.language_model.layers.17.self_attn.o_proj|4.0|2.874e-03|4.194e+06|
|model.language_model.layers.19.mlp.down_proj|4.0|2.846e-03|1.258e+07|
|model.language_model.layers.18.mlp.down_proj|4.0|2.831e-03|1.258e+07|
|model.language_model.layers.23.self_attn.o_proj|4.0|2.796e-03|4.194e+06|
|model.language_model.layers.22.self_attn.o_proj|4.0|2.766e-03|4.194e+06|
|model.language_model.layers.32.mlp.gate_proj|4.0|2.766e-03|2.517e+07|
|model.language_model.layers.31.mlp.gate_proj|4.0|2.696e-03|2.517e+07|
|model.language_model.layers.29.mlp.gate_proj|4.0|2.694e-03|2.517e+07|
|model.language_model.layers.35.mlp.down_proj|4.0|2.578e-03|1.258e+07|
|model.language_model.layers.20.mlp.down_proj|4.0|2.517e-03|1.258e+07|
|model.language_model.layers.30.mlp.gate_proj|4.0|2.510e-03|2.517e+07|
|model.language_model.layers.26.mlp.down_proj|4.0|2.395e-03|1.258e+07|
|model.language_model.layers.27.mlp.down_proj|4.0|2.315e-03|1.258e+07|
|model.language_model.layers.34.mlp.down_proj|4.0|2.188e-03|1.258e+07|
|model.language_model.layers.28.mlp.down_proj|4.0|2.184e-03|1.258e+07|
|model.language_model.layers.33.mlp.down_proj|4.0|2.127e-03|1.258e+07|
|model.language_model.layers.29.mlp.down_proj|4.0|1.784e-03|1.258e+07|
|model.language_model.layers.24.self_attn.o_proj|4.0|1.694e-03|4.194e+06|
|model.language_model.layers.18.self_attn.o_proj|4.0|1.643e-03|4.194e+06|
|model.language_model.layers.30.mlp.down_proj|4.0|1.609e-03|1.258e+07|
|model.language_model.layers.31.mlp.down_proj|4.0|1.558e-03|1.258e+07|
|model.language_model.layers.32.mlp.down_proj|4.0|1.554e-03|1.258e+07|
|model.language_model.layers.19.self_attn.o_proj|4.0|1.281e-03|4.194e+06|
|model.language_model.layers.20.self_attn.o_proj|4.0|1.212e-03|4.194e+06|
|model.language_model.layers.21.self_attn.o_proj|4.0|1.129e-03|4.194e+06|
|model.language_model.layers.34.self_attn.o_proj|4.0|9.900e-04|4.194e+06|
|model.language_model.layers.25.self_attn.o_proj|4.0|9.010e-04|4.194e+06|
|lm_head|4.0|7.883e-04|1.556e+08|
|model.language_model.layers.27.self_attn.o_proj|4.0|6.317e-04|4.194e+06|
|model.language_model.layers.28.self_attn.o_proj|4.0|5.574e-04|4.194e+06|
|model.language_model.layers.33.self_attn.o_proj|4.0|4.960e-04|4.194e+06|
|model.language_model.layers.26.self_attn.o_proj|4.0|4.896e-04|4.194e+06|
|model.language_model.layers.31.self_attn.o_proj|4.0|4.593e-04|4.194e+06|
|model.language_model.layers.32.self_attn.o_proj|4.0|4.055e-04|4.194e+06|
|model.language_model.layers.29.self_attn.o_proj|4.0|3.883e-04|4.194e+06|
|model.language_model.layers.35.self_attn.o_proj|4.0|2.812e-04|4.194e+06|
|model.language_model.layers.30.self_attn.o_proj|4.0|2.679e-04|4.194e+06|

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