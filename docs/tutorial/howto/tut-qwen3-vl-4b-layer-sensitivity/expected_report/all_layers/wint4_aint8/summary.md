# Tutorial Pack Scenario Summary

| Key | Value |
|---|---|
| scenario_id | `all_layers/wint4_aint8` |
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
