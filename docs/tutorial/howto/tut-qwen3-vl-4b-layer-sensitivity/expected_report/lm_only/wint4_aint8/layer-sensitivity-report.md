
AutoQuant Layer Sensitivity (wint4_aint8_autoquant_lm)
======================================================

## Summary


|Key|Value|
| :--- | :--- |
|Scheme|`wint4_aint8_autoquant_lm`|
|Model|`<ABSOLUTE_PATH>-quantize-model<ABSOLUTE_PATH>-VL-4B-Instruct`|
|Effective bits (from search)|`8.0000`|
|Total AutoQuant score|`1.537148e+01`|
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
|layers.1.mlp.down_proj|4.0|1.130e+02|6.226e+06|
|layers.6.mlp.down_proj|4.0|1.064e+02|6.226e+06|
|layers.6.mlp.gate_proj|4.0|9.966e+01|1.245e+07|
|layers.12.mlp.down_proj|4.0|7.610e+01|6.226e+06|
|layers.14.mlp.down_proj|4.0|5.381e+01|6.226e+06|
|layers.16.mlp.down_proj|4.0|3.360e+01|6.226e+06|
|layers.9.mlp.down_proj|4.0|2.689e+01|6.226e+06|
|layers.0.mlp.down_proj|4.0|2.398e+01|6.226e+06|
|layers.4.mlp.down_proj|4.0|2.381e+01|6.226e+06|
|layers.10.mlp.down_proj|4.0|2.342e+01|6.226e+06|
|layers.3.mlp.down_proj|4.0|2.337e+01|6.226e+06|
|layers.15.mlp.down_proj|4.0|2.118e+01|6.226e+06|
|layers.13.mlp.down_proj|4.0|1.963e+01|6.226e+06|
|layers.2.mlp.down_proj|4.0|1.804e+01|6.226e+06|
|layers.4.mlp.gate_proj|4.0|1.500e+01|1.245e+07|
|layers.3.mlp.gate_proj|4.0|1.361e+01|1.245e+07|
|layers.18.mlp.down_proj|4.0|1.287e+01|6.226e+06|
|layers.11.mlp.down_proj|4.0|1.267e+01|6.226e+06|
|layers.2.mlp.gate_proj|4.0|1.240e+01|1.245e+07|
|layers.5.mlp.down_proj|4.0|1.162e+01|6.226e+06|
|layers.8.mlp.down_proj|4.0|9.260e+00|6.226e+06|
|layers.7.mlp.down_proj|4.0|8.162e+00|6.226e+06|
|layers.7.mlp.gate_proj|4.0|6.914e+00|1.245e+07|
|layers.5.mlp.gate_proj|4.0|5.912e+00|1.245e+07|
|layers.1.mlp.gate_proj|4.0|4.976e+00|1.245e+07|
|layers.14.mlp.gate_proj|4.0|3.345e+00|1.245e+07|
|layers.10.mlp.gate_proj|4.0|3.101e+00|1.245e+07|
|layers.13.mlp.gate_proj|4.0|2.979e+00|1.245e+07|
|layers.16.mlp.gate_proj|4.0|2.967e+00|1.245e+07|
|layers.15.mlp.gate_proj|4.0|2.965e+00|1.245e+07|
|layers.12.mlp.gate_proj|4.0|2.882e+00|1.245e+07|
|layers.11.mlp.gate_proj|4.0|2.367e+00|1.245e+07|
|layers.8.mlp.gate_proj|4.0|2.336e+00|1.245e+07|
|layers.9.mlp.gate_proj|4.0|2.278e+00|1.245e+07|
|layers.0.mlp.gate_proj|4.0|2.085e+00|1.245e+07|
|layers.17.mlp.gate_proj|4.0|1.068e+00|1.245e+07|
|layers.24.mlp.down_proj|4.0|1.059e+00|6.226e+06|
|layers.17.mlp.down_proj|4.0|9.303e-01|6.226e+06|
|layers.18.mlp.gate_proj|4.0|7.970e-01|1.245e+07|
|layers.23.mlp.down_proj|4.0|7.404e-01|6.226e+06|
|layers.19.mlp.down_proj|4.0|6.922e-01|6.226e+06|
|layers.25.mlp.down_proj|4.0|6.511e-01|6.226e+06|
|layers.22.mlp.down_proj|4.0|5.594e-01|6.226e+06|
|layers.35.mlp.gate_proj|4.0|5.488e-01|1.245e+07|
|layers.19.mlp.gate_proj|4.0|4.910e-01|1.245e+07|
|layers.21.mlp.down_proj|4.0|3.912e-01|6.226e+06|
|layers.26.mlp.down_proj|4.0|3.604e-01|6.226e+06|
|layers.20.mlp.down_proj|4.0|3.571e-01|6.226e+06|
|layers.20.mlp.gate_proj|4.0|3.487e-01|1.245e+07|
|layers.21.mlp.gate_proj|4.0|2.819e-01|1.245e+07|
|layers.22.mlp.gate_proj|4.0|2.433e-01|1.245e+07|
|layers.23.mlp.gate_proj|4.0|2.284e-01|1.245e+07|
|layers.24.mlp.gate_proj|4.0|1.887e-01|1.245e+07|
|layers.28.mlp.down_proj|4.0|1.878e-01|6.226e+06|
|layers.27.mlp.down_proj|4.0|1.792e-01|6.226e+06|
|layers.25.mlp.gate_proj|4.0|1.394e-01|1.245e+07|
|layers.26.mlp.gate_proj|4.0|1.050e-01|1.245e+07|
|layers.27.mlp.gate_proj|4.0|5.988e-02|1.245e+07|
|layers.34.mlp.down_proj|4.0|5.725e-02|6.226e+06|
|layers.35.mlp.down_proj|4.0|5.614e-02|6.226e+06|
|layers.29.mlp.down_proj|4.0|5.508e-02|6.226e+06|
|layers.30.mlp.down_proj|4.0|5.039e-02|6.226e+06|
|layers.34.mlp.gate_proj|4.0|3.870e-02|1.245e+07|
|layers.28.mlp.gate_proj|4.0|3.275e-02|1.245e+07|
|layers.29.mlp.gate_proj|4.0|2.147e-02|1.245e+07|
|layers.31.mlp.down_proj|4.0|1.671e-02|6.226e+06|
|layers.30.mlp.gate_proj|4.0|1.370e-02|1.245e+07|
|layers.32.mlp.down_proj|4.0|1.173e-02|6.226e+06|
|layers.31.mlp.gate_proj|4.0|1.023e-02|1.245e+07|
|layers.32.mlp.gate_proj|4.0|8.148e-03|1.245e+07|
|layers.33.mlp.down_proj|4.0|7.638e-03|6.226e+06|
|layers.33.mlp.gate_proj|4.0|6.250e-03|1.245e+07|
|layers.35.self_attn.q_proj|4.0|9.492e-04|3.932e+06|
|layers.0.self_attn.q_proj|4.0|7.967e-04|3.932e+06|
|layers.6.self_attn.q_proj|4.0|7.935e-04|3.932e+06|
|layers.22.self_attn.q_proj|4.0|5.716e-04|3.932e+06|
|layers.23.self_attn.q_proj|4.0|5.426e-04|3.932e+06|
|layers.7.self_attn.q_proj|4.0|5.352e-04|3.932e+06|
|layers.21.self_attn.q_proj|4.0|5.148e-04|3.932e+06|
|layers.15.self_attn.o_proj|4.0|4.552e-04|2.621e+06|
|layers.34.self_attn.q_proj|4.0|4.453e-04|3.932e+06|
|layers.24.self_attn.q_proj|4.0|4.065e-04|3.932e+06|
|layers.9.self_attn.q_proj|4.0|3.770e-04|3.932e+06|
|layers.8.self_attn.q_proj|4.0|3.702e-04|3.932e+06|
|layers.10.self_attn.q_proj|4.0|3.524e-04|3.932e+06|
|layers.32.self_attn.q_proj|4.0|3.279e-04|3.932e+06|
|layers.28.self_attn.q_proj|4.0|3.242e-04|3.932e+06|
|layers.30.self_attn.q_proj|4.0|3.150e-04|3.932e+06|
|layers.3.self_attn.q_proj|4.0|2.931e-04|3.932e+06|
|layers.14.self_attn.q_proj|4.0|2.920e-04|3.932e+06|
|layers.5.self_attn.q_proj|4.0|2.918e-04|3.932e+06|
|layers.26.self_attn.q_proj|4.0|2.719e-04|3.932e+06|
|layers.31.self_attn.q_proj|4.0|2.714e-04|3.932e+06|
|layers.16.self_attn.o_proj|4.0|2.671e-04|2.621e+06|
|layers.4.self_attn.q_proj|4.0|2.638e-04|3.932e+06|
|layers.18.self_attn.q_proj|4.0|2.594e-04|3.932e+06|
|layers.16.self_attn.q_proj|4.0|2.544e-04|3.932e+06|
|layers.15.self_attn.q_proj|4.0|2.515e-04|3.932e+06|
|layers.25.self_attn.q_proj|4.0|2.458e-04|3.932e+06|
|layers.19.self_attn.q_proj|4.0|2.416e-04|3.932e+06|
|layers.17.self_attn.q_proj|4.0|2.386e-04|3.932e+06|
|layers.27.self_attn.q_proj|4.0|2.342e-04|3.932e+06|
|layers.0.self_attn.o_proj|4.0|2.237e-04|2.621e+06|
|layers.20.self_attn.q_proj|4.0|2.228e-04|3.932e+06|
|layers.14.self_attn.o_proj|4.0|2.091e-04|2.621e+06|
|layers.11.self_attn.q_proj|4.0|2.030e-04|3.932e+06|
|layers.12.self_attn.q_proj|4.0|1.979e-04|3.932e+06|
|layers.29.self_attn.q_proj|4.0|1.943e-04|3.932e+06|
|layers.13.self_attn.o_proj|4.0|1.941e-04|2.621e+06|
|layers.33.self_attn.q_proj|4.0|1.917e-04|3.932e+06|
|layers.6.self_attn.o_proj|4.0|1.752e-04|2.621e+06|
|layers.22.self_attn.o_proj|4.0|1.583e-04|2.621e+06|
|layers.13.self_attn.q_proj|4.0|1.581e-04|3.932e+06|
|layers.2.self_attn.q_proj|4.0|1.516e-04|3.932e+06|
|layers.34.self_attn.o_proj|4.0|1.479e-04|2.621e+06|
|layers.1.self_attn.q_proj|4.0|1.226e-04|3.932e+06|
|layers.23.self_attn.o_proj|4.0|1.163e-04|2.621e+06|
|layers.8.self_attn.o_proj|4.0|1.093e-04|2.621e+06|
|layers.35.self_attn.o_proj|4.0|1.077e-04|2.621e+06|
|layers.10.self_attn.o_proj|4.0|9.528e-05|2.621e+06|
|layers.24.self_attn.o_proj|4.0|7.507e-05|2.621e+06|
|layers.12.self_attn.o_proj|4.0|7.456e-05|2.621e+06|
|layers.7.self_attn.o_proj|4.0|7.291e-05|2.621e+06|
|layers.9.self_attn.o_proj|4.0|7.036e-05|2.621e+06|
|layers.5.self_attn.o_proj|4.0|6.836e-05|2.621e+06|
|layers.1.self_attn.o_proj|4.0|6.405e-05|2.621e+06|
|layers.19.self_attn.o_proj|4.0|6.167e-05|2.621e+06|
|layers.18.self_attn.o_proj|4.0|6.064e-05|2.621e+06|
|layers.28.self_attn.o_proj|4.0|5.740e-05|2.621e+06|
|layers.11.self_attn.o_proj|4.0|5.688e-05|2.621e+06|
|layers.17.self_attn.o_proj|4.0|5.669e-05|2.621e+06|
|layers.21.self_attn.o_proj|4.0|5.591e-05|2.621e+06|
|layers.26.self_attn.o_proj|4.0|5.551e-05|2.621e+06|
|layers.4.self_attn.o_proj|4.0|5.522e-05|2.621e+06|
|layers.27.self_attn.o_proj|4.0|5.464e-05|2.621e+06|
|layers.20.self_attn.o_proj|4.0|5.310e-05|2.621e+06|
|layers.30.self_attn.o_proj|4.0|5.203e-05|2.621e+06|
|layers.33.self_attn.o_proj|4.0|4.905e-05|2.621e+06|
|layers.32.self_attn.o_proj|4.0|4.761e-05|2.621e+06|
|layers.29.self_attn.o_proj|4.0|4.596e-05|2.621e+06|
|layers.25.self_attn.o_proj|4.0|4.543e-05|2.621e+06|
|layers.31.self_attn.o_proj|4.0|4.533e-05|2.621e+06|
|layers.3.self_attn.o_proj|4.0|3.577e-05|2.621e+06|
|layers.2.self_attn.o_proj|4.0|2.884e-05|2.621e+06|

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