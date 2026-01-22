# Tutorial Pack Scenario Summary

| Key | Value |
|---|---|
| scenario_id | `lm_only<ABSOLUTE_PATH> |
| mode | `lm_only` |
| quant_pair | `wint4_aint8` |
| dataset_size | `medium` |
| dataset_calib_seq_len | `512` |
| dataset_batch_size | `8` |
| dataset_num_calib_batches | `16` |
| dataset_num_calib_samples | `128` |
| dataset_max_calib_samples | `128` |
| auto_quantize_score_size | `128` |
| scheme_name | `wint4_aint8_autoquant_lm` |
| quant_formats | `["INT4_WEIGHT_INT8_ACT_CFG"]` |
| has_layer_sensitivity | `True` |
| has_autoquant_state | `True` |
| has_nonzero_sensitivity | `True` |
| manifest_keys | `["autoquant_state", "dataset", "layer_sensitivity", "layers", "model", "num_quantized_layers", "quantization", "run_config", "scheme", "sensitivity_ranking"]` |


---


AutoQuant Layer Sensitivity (wint4_aint8_autoquant_lm)
======================================================

## Summary


|Key|Value|
| :--- | :--- |
|Scheme|`wint4_aint8_autoquant_lm`|
|Model|`<ABSOLUTE_PATH>-quantize-model<ABSOLUTE_PATH>-VL-8B-Instruct`|
|Effective bits (from search)|`8.0000`|
|Total AutoQuant score|`9.632868e+00`|
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
|Base format|`INT4_WEIGHT_INT8_ACT_CFG`|
|Granularity|`default`|
|Quant cfg overrides|`see below`|

## Layer Sensitivity Table


Sorted by sensitivity (descending). Layer names are AutoQuant recipe handles; a trailing `.quant_recipe` suffix (if present) is stripped for readability.

|Layer|Num Bits|Sensitivity|Size Cost|
| :--- | :--- | :--- | :--- |
|layers.1.mlp.down_proj|4.0|2.140e+01|1.258e+07|
|layers.16.mlp.down_proj|4.0|1.935e+01|1.258e+07|
|layers.14.mlp.down_proj|4.0|1.698e+01|1.258e+07|
|layers.6.mlp.gate_proj|4.0|1.416e+01|2.517e+07|
|layers.18.mlp.down_proj|4.0|1.328e+01|1.258e+07|
|layers.12.mlp.down_proj|4.0|1.075e+01|1.258e+07|
|layers.6.mlp.down_proj|4.0|8.082e+00|1.258e+07|
|layers.0.mlp.down_proj|4.0|4.453e+00|1.258e+07|
|layers.15.mlp.down_proj|4.0|3.935e+00|1.258e+07|
|layers.23.mlp.down_proj|4.0|3.533e+00|1.258e+07|
|layers.24.mlp.down_proj|4.0|3.531e+00|1.258e+07|
|layers.8.mlp.down_proj|4.0|3.406e+00|1.258e+07|
|layers.16.mlp.gate_proj|4.0|2.997e+00|2.517e+07|
|layers.10.mlp.down_proj|4.0|2.996e+00|1.258e+07|
|layers.13.mlp.down_proj|4.0|2.662e+00|1.258e+07|
|layers.22.mlp.down_proj|4.0|2.448e+00|1.258e+07|
|layers.25.mlp.down_proj|4.0|2.018e+00|1.258e+07|
|layers.21.mlp.down_proj|4.0|2.001e+00|1.258e+07|
|layers.9.mlp.down_proj|4.0|1.982e+00|1.258e+07|
|layers.11.mlp.down_proj|4.0|1.757e+00|1.258e+07|
|layers.5.mlp.down_proj|4.0|1.682e+00|1.258e+07|
|layers.35.mlp.gate_proj|4.0|1.663e+00|2.517e+07|
|layers.7.mlp.gate_proj|4.0|1.458e+00|2.517e+07|
|layers.7.mlp.down_proj|4.0|1.220e+00|1.258e+07|
|layers.19.mlp.down_proj|4.0|1.212e+00|1.258e+07|
|layers.26.mlp.down_proj|4.0|1.171e+00|1.258e+07|
|layers.15.mlp.gate_proj|4.0|1.139e+00|2.517e+07|
|layers.17.mlp.down_proj|4.0|9.599e-01|1.258e+07|
|layers.14.mlp.gate_proj|4.0|8.931e-01|2.517e+07|
|layers.4.mlp.down_proj|4.0|8.435e-01|1.258e+07|
|layers.17.mlp.gate_proj|4.0|8.400e-01|2.517e+07|
|layers.18.mlp.gate_proj|4.0|8.093e-01|2.517e+07|
|layers.19.mlp.gate_proj|4.0|8.090e-01|2.517e+07|
|layers.20.mlp.down_proj|4.0|7.908e-01|1.258e+07|
|layers.20.mlp.gate_proj|4.0|7.462e-01|2.517e+07|
|layers.13.mlp.gate_proj|4.0|6.961e-01|2.517e+07|
|layers.5.mlp.gate_proj|4.0|6.868e-01|2.517e+07|
|layers.21.mlp.gate_proj|4.0|6.342e-01|2.517e+07|
|layers.11.mlp.gate_proj|4.0|5.519e-01|2.517e+07|
|layers.1.mlp.gate_proj|4.0|5.412e-01|2.517e+07|
|layers.22.mlp.gate_proj|4.0|5.324e-01|2.517e+07|
|layers.12.mlp.gate_proj|4.0|4.992e-01|2.517e+07|
|layers.34.mlp.gate_proj|4.0|4.916e-01|2.517e+07|
|layers.23.mlp.gate_proj|4.0|4.904e-01|2.517e+07|
|layers.4.mlp.gate_proj|4.0|4.477e-01|2.517e+07|
|layers.10.mlp.gate_proj|4.0|4.384e-01|2.517e+07|
|layers.9.mlp.gate_proj|4.0|4.040e-01|2.517e+07|
|layers.27.mlp.down_proj|4.0|3.952e-01|1.258e+07|
|layers.24.mlp.gate_proj|4.0|3.751e-01|2.517e+07|
|layers.3.mlp.down_proj|4.0|3.151e-01|1.258e+07|
|layers.8.mlp.gate_proj|4.0|3.032e-01|2.517e+07|
|layers.28.mlp.down_proj|4.0|2.971e-01|1.258e+07|
|layers.25.mlp.gate_proj|4.0|2.594e-01|2.517e+07|
|layers.26.mlp.gate_proj|4.0|2.093e-01|2.517e+07|
|layers.2.mlp.down_proj|4.0|2.036e-01|1.258e+07|
|layers.2.mlp.gate_proj|4.0|1.858e-01|2.517e+07|
|layers.29.mlp.down_proj|4.0|1.834e-01|1.258e+07|
|layers.3.mlp.gate_proj|4.0|1.532e-01|2.517e+07|
|layers.27.mlp.gate_proj|4.0|1.159e-01|2.517e+07|
|layers.30.mlp.down_proj|4.0|1.137e-01|1.258e+07|
|layers.0.mlp.gate_proj|4.0|8.181e-02|2.517e+07|
|layers.28.mlp.gate_proj|4.0|7.799e-02|2.517e+07|
|layers.35.mlp.down_proj|4.0|6.850e-02|1.258e+07|
|layers.29.mlp.gate_proj|4.0|5.469e-02|2.517e+07|
|layers.31.mlp.down_proj|4.0|5.228e-02|1.258e+07|
|layers.30.mlp.gate_proj|4.0|3.713e-02|2.517e+07|
|layers.31.mlp.gate_proj|4.0|2.416e-02|2.517e+07|
|layers.32.mlp.down_proj|4.0|2.186e-02|1.258e+07|
|layers.32.mlp.gate_proj|4.0|2.022e-02|2.517e+07|
|layers.33.mlp.gate_proj|4.0|1.945e-02|2.517e+07|
|layers.34.mlp.down_proj|4.0|1.798e-02|1.258e+07|
|layers.33.mlp.down_proj|4.0|1.264e-02|1.258e+07|
|layers.35.self_attn.q_proj|4.0|1.267e-03|6.291e+06|
|layers.6.self_attn.q_proj|4.0|1.091e-03|6.291e+06|
|layers.34.self_attn.q_proj|4.0|1.057e-03|6.291e+06|
|layers.0.self_attn.q_proj|4.0|8.030e-04|6.291e+06|
|layers.3.self_attn.q_proj|4.0|5.110e-04|6.291e+06|
|layers.22.self_attn.q_proj|4.0|4.750e-04|6.291e+06|
|layers.21.self_attn.q_proj|4.0|4.341e-04|6.291e+06|
|layers.5.self_attn.q_proj|4.0|4.152e-04|6.291e+06|
|layers.32.self_attn.q_proj|4.0|3.783e-04|6.291e+06|
|layers.15.self_attn.o_proj|4.0|3.686e-04|4.194e+06|
|layers.23.self_attn.q_proj|4.0|3.616e-04|6.291e+06|
|layers.24.self_attn.q_proj|4.0|3.529e-04|6.291e+06|
|layers.10.self_attn.q_proj|4.0|3.372e-04|6.291e+06|
|layers.8.self_attn.q_proj|4.0|3.231e-04|6.291e+06|
|layers.7.self_attn.q_proj|4.0|3.199e-04|6.291e+06|
|layers.9.self_attn.q_proj|4.0|2.716e-04|6.291e+06|
|layers.30.self_attn.q_proj|4.0|2.693e-04|6.291e+06|
|layers.28.self_attn.q_proj|4.0|2.688e-04|6.291e+06|
|layers.4.self_attn.q_proj|4.0|2.484e-04|6.291e+06|
|layers.33.self_attn.q_proj|4.0|2.464e-04|6.291e+06|
|layers.19.self_attn.q_proj|4.0|2.329e-04|6.291e+06|
|layers.27.self_attn.q_proj|4.0|2.195e-04|6.291e+06|
|layers.26.self_attn.q_proj|4.0|2.173e-04|6.291e+06|
|layers.17.self_attn.q_proj|4.0|2.126e-04|6.291e+06|
|layers.14.self_attn.o_proj|4.0|2.067e-04|4.194e+06|
|layers.16.self_attn.q_proj|4.0|2.006e-04|6.291e+06|
|layers.31.self_attn.q_proj|4.0|1.992e-04|6.291e+06|
|layers.25.self_attn.q_proj|4.0|1.955e-04|6.291e+06|
|layers.11.self_attn.q_proj|4.0|1.941e-04|6.291e+06|
|layers.18.self_attn.q_proj|4.0|1.914e-04|6.291e+06|
|layers.15.self_attn.q_proj|4.0|1.887e-04|6.291e+06|
|layers.14.self_attn.q_proj|4.0|1.854e-04|6.291e+06|
|layers.20.self_attn.q_proj|4.0|1.755e-04|6.291e+06|
|layers.1.self_attn.q_proj|4.0|1.743e-04|6.291e+06|
|layers.2.self_attn.q_proj|4.0|1.684e-04|6.291e+06|
|layers.16.self_attn.o_proj|4.0|1.680e-04|4.194e+06|
|layers.12.self_attn.q_proj|4.0|1.637e-04|6.291e+06|
|layers.29.self_attn.q_proj|4.0|1.522e-04|6.291e+06|
|layers.34.self_attn.o_proj|4.0|1.450e-04|4.194e+06|
|layers.13.self_attn.q_proj|4.0|1.367e-04|6.291e+06|
|layers.0.self_attn.o_proj|4.0|1.320e-04|4.194e+06|
|layers.6.self_attn.o_proj|4.0|1.277e-04|4.194e+06|
|layers.22.self_attn.o_proj|4.0|1.049e-04|4.194e+06|
|layers.23.self_attn.o_proj|4.0|8.785e-05|4.194e+06|
|layers.13.self_attn.o_proj|4.0|7.358e-05|4.194e+06|
|layers.35.self_attn.o_proj|4.0|7.331e-05|4.194e+06|
|layers.24.self_attn.o_proj|4.0|7.282e-05|4.194e+06|
|layers.8.self_attn.o_proj|4.0|7.281e-05|4.194e+06|
|layers.10.self_attn.o_proj|4.0|6.524e-05|4.194e+06|
|layers.1.self_attn.o_proj|4.0|6.435e-05|4.194e+06|
|layers.21.self_attn.o_proj|4.0|5.565e-05|4.194e+06|
|layers.5.self_attn.o_proj|4.0|5.261e-05|4.194e+06|
|layers.7.self_attn.o_proj|4.0|5.171e-05|4.194e+06|
|layers.18.self_attn.o_proj|4.0|5.108e-05|4.194e+06|
|layers.12.self_attn.o_proj|4.0|4.886e-05|4.194e+06|
|layers.32.self_attn.o_proj|4.0|4.828e-05|4.194e+06|
|layers.33.self_attn.o_proj|4.0|4.669e-05|4.194e+06|
|layers.9.self_attn.o_proj|4.0|4.630e-05|4.194e+06|
|layers.19.self_attn.o_proj|4.0|4.570e-05|4.194e+06|
|layers.26.self_attn.o_proj|4.0|4.518e-05|4.194e+06|
|layers.30.self_attn.o_proj|4.0|4.487e-05|4.194e+06|
|layers.27.self_attn.o_proj|4.0|4.480e-05|4.194e+06|
|layers.20.self_attn.o_proj|4.0|4.269e-05|4.194e+06|
|layers.17.self_attn.o_proj|4.0|4.242e-05|4.194e+06|
|layers.4.self_attn.o_proj|4.0|4.188e-05|4.194e+06|
|layers.28.self_attn.o_proj|4.0|4.185e-05|4.194e+06|
|layers.11.self_attn.o_proj|4.0|4.024e-05|4.194e+06|
|layers.31.self_attn.o_proj|4.0|4.015e-05|4.194e+06|
|layers.25.self_attn.o_proj|4.0|3.962e-05|4.194e+06|
|layers.29.self_attn.o_proj|4.0|3.734e-05|4.194e+06|
|layers.3.self_attn.o_proj|4.0|3.235e-05|4.194e+06|
|layers.2.self_attn.o_proj|4.0|2.531e-05|4.194e+06|

## Composed Config (`composed-config.yaml`)


```yaml
model:
  name: qwen3_vl_8b_instruct
  family: qwen
  variant: 3-vl-8b-instruct
  format: pytorch
  path: <ABSOLUTE_PATH>-quantize-model<ABSOLUTE_PATH>-VL-8B-Instruct
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
  name: wint4_aint8
  weight: int4
  activation: int8
  format_name: INT4_WEIGHT_INT8_ACT_CFG
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