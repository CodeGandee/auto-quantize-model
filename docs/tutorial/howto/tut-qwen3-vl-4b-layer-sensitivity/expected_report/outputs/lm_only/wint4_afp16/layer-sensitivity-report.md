# Tutorial Pack Scenario Summary

| Key | Value |
|---|---|
| scenario_id | `lm_only<ABSOLUTE_PATH> |
| mode | `lm_only` |
| quant_pair | `wint4_afp16` |
| dataset_size | `medium` |
| dataset_calib_seq_len | `512` |
| dataset_batch_size | `8` |
| dataset_num_calib_batches | `16` |
| dataset_num_calib_samples | `128` |
| dataset_max_calib_samples | `128` |
| auto_quantize_score_size | `128` |
| scheme_name | `wint4_afp16_autoquant_lm` |
| quant_formats | `["INT4_BLOCKWISE_WEIGHT_ONLY_CFG"]` |
| has_layer_sensitivity | `True` |
| has_autoquant_state | `True` |
| has_nonzero_sensitivity | `True` |
| manifest_keys | `["autoquant_state", "dataset", "layer_sensitivity", "layers", "model", "num_quantized_layers", "quantization", "run_config", "scheme", "sensitivity_ranking"]` |


---


AutoQuant Layer Sensitivity (wint4_afp16_autoquant_lm)
======================================================

## Summary


|Key|Value|
| :--- | :--- |
|Scheme|`wint4_afp16_autoquant_lm`|
|Model|`<ABSOLUTE_PATH>-quantize-model<ABSOLUTE_PATH>-VL-4B-Instruct`|
|Effective bits (from search)|`8.0000`|
|Total AutoQuant score|`6.688368e+00`|
|Constraint satisfied|`True`|

## Dataset


|Key|Value|
| :--- | :--- |
|Name|`vlm_coco2017_captions`|
|Size|`medium`|
|Root|`<ABSOLUTE_PATH>-quantize-model<ABSOLUTE_PATH>-quantize-calib`|
|Captions path|`<ABSOLUTE_PATH>-quantize-model<ABSOLUTE_PATH>-quantize-calib<ABSOLUTE_PATH>
|Calibration seq len|`512`|
|Batch size|`8`|
|Calibration batches|`16`|
|Calibration samples (used / max)|`128` / `128`|

## Quantization


|Key|Value|
| :--- | :--- |
|Base format|`INT4_BLOCKWISE_WEIGHT_ONLY_CFG`|
|Granularity|`default`|
|Quant cfg overrides|`see below`|

## Layer Sensitivity Table


Sorted by sensitivity (descending). Layer names are AutoQuant recipe handles; a trailing `.quant_recipe` suffix (if present) is stripped for readability.

|Layer|Num Bits|Sensitivity|Size Cost|
| :--- | :--- | :--- | :--- |
|layers.0.mlp.down_proj|4.0|9.321e+00|6.226e+06|
|layers.4.mlp.gate_proj|4.0|9.208e+00|1.245e+07|
|layers.2.mlp.gate_proj|4.0|8.337e+00|1.245e+07|
|layers.3.mlp.gate_proj|4.0|7.917e+00|1.245e+07|
|layers.6.mlp.gate_proj|4.0|4.487e+00|1.245e+07|
|layers.5.mlp.gate_proj|4.0|3.814e+00|1.245e+07|
|layers.1.mlp.gate_proj|4.0|3.581e+00|1.245e+07|
|layers.7.mlp.gate_proj|4.0|3.259e+00|1.245e+07|
|layers.10.mlp.gate_proj|4.0|2.649e+00|1.245e+07|
|layers.4.mlp.down_proj|4.0|2.645e+00|6.226e+06|
|layers.14.mlp.gate_proj|4.0|2.059e+00|1.245e+07|
|layers.5.mlp.down_proj|4.0|2.038e+00|6.226e+06|
|layers.0.mlp.gate_proj|4.0|2.038e+00|1.245e+07|
|layers.12.mlp.gate_proj|4.0|1.886e+00|1.245e+07|
|layers.13.mlp.gate_proj|4.0|1.876e+00|1.245e+07|
|layers.8.mlp.gate_proj|4.0|1.786e+00|1.245e+07|
|layers.9.mlp.gate_proj|4.0|1.684e+00|1.245e+07|
|layers.15.mlp.gate_proj|4.0|1.647e+00|1.245e+07|
|layers.11.mlp.gate_proj|4.0|1.558e+00|1.245e+07|
|layers.13.mlp.down_proj|4.0|1.536e+00|6.226e+06|
|layers.3.mlp.down_proj|4.0|1.506e+00|6.226e+06|
|layers.6.mlp.down_proj|4.0|1.385e+00|6.226e+06|
|layers.9.mlp.down_proj|4.0|1.232e+00|6.226e+06|
|layers.14.mlp.down_proj|4.0|1.214e+00|6.226e+06|
|layers.10.mlp.down_proj|4.0|1.180e+00|6.226e+06|
|layers.8.mlp.down_proj|4.0|1.109e+00|6.226e+06|
|layers.12.mlp.down_proj|4.0|1.092e+00|6.226e+06|
|layers.15.mlp.down_proj|4.0|9.888e-01|6.226e+06|
|layers.16.mlp.gate_proj|4.0|9.683e-01|1.245e+07|
|layers.7.mlp.down_proj|4.0|9.365e-01|6.226e+06|
|layers.1.mlp.down_proj|4.0|8.970e-01|6.226e+06|
|layers.11.mlp.down_proj|4.0|8.613e-01|6.226e+06|
|layers.2.mlp.down_proj|4.0|8.279e-01|6.226e+06|
|layers.17.mlp.gate_proj|4.0|7.492e-01|1.245e+07|
|layers.18.mlp.gate_proj|4.0|6.001e-01|1.245e+07|
|layers.16.mlp.down_proj|4.0|5.657e-01|6.226e+06|
|layers.35.mlp.gate_proj|4.0|4.424e-01|1.245e+07|
|layers.19.mlp.gate_proj|4.0|3.830e-01|1.245e+07|
|layers.17.mlp.down_proj|4.0|3.808e-01|6.226e+06|
|layers.18.mlp.down_proj|4.0|3.237e-01|6.226e+06|
|layers.20.mlp.gate_proj|4.0|2.559e-01|1.245e+07|
|layers.19.mlp.down_proj|4.0|2.421e-01|6.226e+06|
|layers.21.mlp.gate_proj|4.0|2.168e-01|1.245e+07|
|layers.22.mlp.gate_proj|4.0|2.056e-01|1.245e+07|
|layers.23.mlp.gate_proj|4.0|1.721e-01|1.245e+07|
|layers.24.mlp.gate_proj|4.0|1.437e-01|1.245e+07|
|layers.21.mlp.down_proj|4.0|1.380e-01|6.226e+06|
|layers.20.mlp.down_proj|4.0|1.344e-01|6.226e+06|
|layers.22.mlp.down_proj|4.0|1.279e-01|6.226e+06|
|layers.25.mlp.gate_proj|4.0|1.003e-01|1.245e+07|
|layers.23.mlp.down_proj|4.0|8.424e-02|6.226e+06|
|layers.26.mlp.gate_proj|4.0|7.666e-02|1.245e+07|
|layers.24.mlp.down_proj|4.0|6.494e-02|6.226e+06|
|layers.25.mlp.down_proj|4.0|4.649e-02|6.226e+06|
|layers.27.mlp.gate_proj|4.0|4.572e-02|1.245e+07|
|layers.26.mlp.down_proj|4.0|3.318e-02|6.226e+06|
|layers.34.mlp.gate_proj|4.0|2.748e-02|1.245e+07|
|layers.28.mlp.gate_proj|4.0|2.638e-02|1.245e+07|
|layers.27.mlp.down_proj|4.0|2.115e-02|6.226e+06|
|layers.29.mlp.gate_proj|4.0|1.799e-02|1.245e+07|
|layers.30.mlp.gate_proj|4.0|1.237e-02|1.245e+07|
|layers.28.mlp.down_proj|4.0|1.228e-02|6.226e+06|
|layers.31.mlp.gate_proj|4.0|9.300e-03|1.245e+07|
|layers.35.mlp.down_proj|4.0|9.194e-03|6.226e+06|
|layers.29.mlp.down_proj|4.0|8.653e-03|6.226e+06|
|layers.32.mlp.gate_proj|4.0|7.324e-03|1.245e+07|
|layers.30.mlp.down_proj|4.0|5.564e-03|6.226e+06|
|layers.33.mlp.gate_proj|4.0|5.242e-03|1.245e+07|
|layers.31.mlp.down_proj|4.0|4.474e-03|6.226e+06|
|layers.34.mlp.down_proj|4.0|4.233e-03|6.226e+06|
|layers.32.mlp.down_proj|4.0|3.100e-03|6.226e+06|
|layers.33.mlp.down_proj|4.0|1.881e-03|6.226e+06|
|layers.0.self_attn.q_proj|4.0|6.366e-04|3.932e+06|
|layers.35.self_attn.q_proj|4.0|5.569e-04|3.932e+06|
|layers.7.self_attn.q_proj|4.0|4.647e-04|3.932e+06|
|layers.6.self_attn.q_proj|4.0|4.501e-04|3.932e+06|
|layers.22.self_attn.q_proj|4.0|3.944e-04|3.932e+06|
|layers.23.self_attn.q_proj|4.0|3.846e-04|3.932e+06|
|layers.21.self_attn.q_proj|4.0|3.369e-04|3.932e+06|
|layers.34.self_attn.q_proj|4.0|3.322e-04|3.932e+06|
|layers.9.self_attn.q_proj|4.0|3.241e-04|3.932e+06|
|layers.24.self_attn.q_proj|4.0|3.130e-04|3.932e+06|
|layers.8.self_attn.q_proj|4.0|3.089e-04|3.932e+06|
|layers.10.self_attn.q_proj|4.0|2.831e-04|3.932e+06|
|layers.28.self_attn.q_proj|4.0|2.482e-04|3.932e+06|
|layers.32.self_attn.q_proj|4.0|2.452e-04|3.932e+06|
|layers.30.self_attn.q_proj|4.0|2.423e-04|3.932e+06|
|layers.14.self_attn.q_proj|4.0|2.255e-04|3.932e+06|
|layers.26.self_attn.q_proj|4.0|2.106e-04|3.932e+06|
|layers.31.self_attn.q_proj|4.0|2.086e-04|3.932e+06|
|layers.0.self_attn.o_proj|4.0|1.964e-04|2.621e+06|
|layers.5.self_attn.q_proj|4.0|1.949e-04|3.932e+06|
|layers.27.self_attn.q_proj|4.0|1.868e-04|3.932e+06|
|layers.15.self_attn.q_proj|4.0|1.854e-04|3.932e+06|
|layers.25.self_attn.q_proj|4.0|1.814e-04|3.932e+06|
|layers.18.self_attn.q_proj|4.0|1.750e-04|3.932e+06|
|layers.16.self_attn.q_proj|4.0|1.695e-04|3.932e+06|
|layers.17.self_attn.q_proj|4.0|1.657e-04|3.932e+06|
|layers.6.self_attn.o_proj|4.0|1.648e-04|2.621e+06|
|layers.19.self_attn.q_proj|4.0|1.642e-04|3.932e+06|
|layers.4.self_attn.q_proj|4.0|1.614e-04|3.932e+06|
|layers.11.self_attn.q_proj|4.0|1.606e-04|3.932e+06|
|layers.12.self_attn.q_proj|4.0|1.604e-04|3.932e+06|
|layers.20.self_attn.q_proj|4.0|1.559e-04|3.932e+06|
|layers.3.self_attn.q_proj|4.0|1.516e-04|3.932e+06|
|layers.29.self_attn.q_proj|4.0|1.474e-04|3.932e+06|
|layers.33.self_attn.q_proj|4.0|1.453e-04|3.932e+06|
|layers.13.self_attn.q_proj|4.0|1.219e-04|3.932e+06|
|layers.8.self_attn.o_proj|4.0|9.544e-05|2.621e+06|
|layers.23.self_attn.o_proj|4.0|9.398e-05|2.621e+06|
|layers.2.self_attn.q_proj|4.0|9.387e-05|3.932e+06|
|layers.1.self_attn.q_proj|4.0|9.353e-05|3.932e+06|
|layers.35.self_attn.o_proj|4.0|8.895e-05|2.621e+06|
|layers.15.self_attn.o_proj|4.0|8.725e-05|2.621e+06|
|layers.22.self_attn.o_proj|4.0|7.871e-05|2.621e+06|
|layers.14.self_attn.o_proj|4.0|7.144e-05|2.621e+06|
|layers.10.self_attn.o_proj|4.0|6.765e-05|2.621e+06|
|layers.12.self_attn.o_proj|4.0|6.708e-05|2.621e+06|
|layers.7.self_attn.o_proj|4.0|6.292e-05|2.621e+06|
|layers.24.self_attn.o_proj|4.0|6.197e-05|2.621e+06|
|layers.9.self_attn.o_proj|4.0|6.127e-05|2.621e+06|
|layers.5.self_attn.o_proj|4.0|6.044e-05|2.621e+06|
|layers.1.self_attn.o_proj|4.0|6.009e-05|2.621e+06|
|layers.34.self_attn.o_proj|4.0|5.886e-05|2.621e+06|
|layers.16.self_attn.o_proj|4.0|5.875e-05|2.621e+06|
|layers.11.self_attn.o_proj|4.0|5.133e-05|2.621e+06|
|layers.19.self_attn.o_proj|4.0|4.979e-05|2.621e+06|
|layers.21.self_attn.o_proj|4.0|4.927e-05|2.621e+06|
|layers.18.self_attn.o_proj|4.0|4.845e-05|2.621e+06|
|layers.4.self_attn.o_proj|4.0|4.743e-05|2.621e+06|
|layers.20.self_attn.o_proj|4.0|4.559e-05|2.621e+06|
|layers.17.self_attn.o_proj|4.0|4.375e-05|2.621e+06|
|layers.13.self_attn.o_proj|4.0|4.250e-05|2.621e+06|
|layers.28.self_attn.o_proj|4.0|4.187e-05|2.621e+06|
|layers.30.self_attn.o_proj|4.0|3.724e-05|2.621e+06|
|layers.26.self_attn.o_proj|4.0|3.670e-05|2.621e+06|
|layers.27.self_attn.o_proj|4.0|3.486e-05|2.621e+06|
|layers.25.self_attn.o_proj|4.0|3.374e-05|2.621e+06|
|layers.3.self_attn.o_proj|4.0|3.168e-05|2.621e+06|
|layers.31.self_attn.o_proj|4.0|2.708e-05|2.621e+06|
|layers.2.self_attn.o_proj|4.0|2.593e-05|2.621e+06|
|layers.32.self_attn.o_proj|4.0|2.564e-05|2.621e+06|
|layers.29.self_attn.o_proj|4.0|2.518e-05|2.621e+06|
|layers.33.self_attn.o_proj|4.0|1.790e-05|2.621e+06|

## Composed Config (`composed-config.yaml`)


```yaml
model:
  name: qwen3_vl_4b_instruct
  family: qwen
  variant: 3-vl-4b-instruct
  format: pytorch
  path: <ABSOLUTE_PATH>-quantize-model<ABSOLUTE_PATH>-VL-4B-Instruct
  dtype: bf16
dataset:
  name: vlm_coco2017_captions
  root: <ABSOLUTE_PATH>-quantize-model<ABSOLUTE_PATH>-quantize-calib
  size: medium
  captions_path: <ABSOLUTE_PATH>-quantize-model<ABSOLUTE_PATH>-quantize-calib<ABSOLUTE_PATH>
  size_to_max_samples:
    small: 16
    medium: 128
    large: 512
  max_calib_samples: null
  calib_seq_len: 512
quant_pair:
  name: wint4_afp16
  weight: int4
  activation: fp16
  format_name: INT4_BLOCKWISE_WEIGHT_ONLY_CFG
  experimental: true
quant_granularity:
  name: default
  quant_cfg_overrides: {}
autoquant:
  method: gradient
  device: cuda:0
  batch_size: 8
  effective_bits: 8.0
  score_size: 128
  verbose: true
output_layout:
  name: tmp
  mode: tmp
  root_dir: <ABSOLUTE_PATH>-quantize-model<ABSOLUTE_PATH>
hardware:
  device_index: 0
experiment: qwen3_lm_sensitivity
runner:
  report_only: false
  output_dir: <ABSOLUTE_PATH>-quantize-model<ABSOLUTE_PATH>
```