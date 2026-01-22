
AutoQuant Layer Sensitivity (wint4_afp16_autoquant_lm)
======================================================

## Summary


|Key|Value|
| :--- | :--- |
|Scheme|`wint4_afp16_autoquant_lm`|
|Model|`<ABSOLUTE_PATH>-quantize-model<ABSOLUTE_PATH>-VL-8B-Instruct`|
|Effective bits (from search)|`8.0000`|
|Total AutoQuant score|`4.948071e+00`|
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
|layers.35.mlp.gate_proj|4.0|1.581e+00|2.517e+07|
|layers.0.mlp.down_proj|4.0|7.088e-01|1.258e+07|
|layers.15.mlp.gate_proj|4.0|6.601e-01|2.517e+07|
|layers.16.mlp.gate_proj|4.0|6.596e-01|2.517e+07|
|layers.19.mlp.gate_proj|4.0|6.238e-01|2.517e+07|
|layers.17.mlp.gate_proj|4.0|6.145e-01|2.517e+07|
|layers.18.mlp.gate_proj|4.0|6.046e-01|2.517e+07|
|layers.20.mlp.gate_proj|4.0|5.753e-01|2.517e+07|
|layers.14.mlp.gate_proj|4.0|5.521e-01|2.517e+07|
|layers.21.mlp.gate_proj|4.0|4.938e-01|2.517e+07|
|layers.1.mlp.down_proj|4.0|4.570e-01|1.258e+07|
|layers.13.mlp.gate_proj|4.0|4.441e-01|2.517e+07|
|layers.22.mlp.gate_proj|4.0|4.351e-01|2.517e+07|
|layers.11.mlp.gate_proj|4.0|4.303e-01|2.517e+07|
|layers.23.mlp.gate_proj|4.0|4.300e-01|2.517e+07|
|layers.7.mlp.gate_proj|4.0|4.080e-01|2.517e+07|
|layers.34.mlp.gate_proj|4.0|3.989e-01|2.517e+07|
|layers.19.mlp.down_proj|4.0|3.909e-01|1.258e+07|
|layers.12.mlp.gate_proj|4.0|3.736e-01|2.517e+07|
|layers.17.mlp.down_proj|4.0|3.717e-01|1.258e+07|
|layers.6.mlp.gate_proj|4.0|3.710e-01|2.517e+07|
|layers.6.mlp.down_proj|4.0|3.682e-01|1.258e+07|
|layers.14.mlp.down_proj|4.0|3.534e-01|1.258e+07|
|layers.15.mlp.down_proj|4.0|3.495e-01|1.258e+07|
|layers.16.mlp.down_proj|4.0|3.432e-01|1.258e+07|
|layers.10.mlp.down_proj|4.0|3.224e-01|1.258e+07|
|layers.20.mlp.down_proj|4.0|3.210e-01|1.258e+07|
|layers.10.mlp.gate_proj|4.0|3.194e-01|2.517e+07|
|layers.9.mlp.gate_proj|4.0|3.100e-01|2.517e+07|
|layers.5.mlp.gate_proj|4.0|3.087e-01|2.517e+07|
|layers.21.mlp.down_proj|4.0|3.015e-01|1.258e+07|
|layers.24.mlp.gate_proj|4.0|3.004e-01|2.517e+07|
|layers.18.mlp.down_proj|4.0|2.911e-01|1.258e+07|
|layers.23.mlp.down_proj|4.0|2.732e-01|1.258e+07|
|layers.22.mlp.down_proj|4.0|2.694e-01|1.258e+07|
|layers.13.mlp.down_proj|4.0|2.639e-01|1.258e+07|
|layers.11.mlp.down_proj|4.0|2.422e-01|1.258e+07|
|layers.25.mlp.gate_proj|4.0|2.318e-01|2.517e+07|
|layers.8.mlp.gate_proj|4.0|2.300e-01|2.517e+07|
|layers.12.mlp.down_proj|4.0|2.300e-01|1.258e+07|
|layers.26.mlp.gate_proj|4.0|1.846e-01|2.517e+07|
|layers.4.mlp.gate_proj|4.0|1.796e-01|2.517e+07|
|layers.1.mlp.gate_proj|4.0|1.782e-01|2.517e+07|
|layers.24.mlp.down_proj|4.0|1.666e-01|1.258e+07|
|layers.8.mlp.down_proj|4.0|1.621e-01|1.258e+07|
|layers.9.mlp.down_proj|4.0|1.591e-01|1.258e+07|
|layers.25.mlp.down_proj|4.0|1.361e-01|1.258e+07|
|layers.27.mlp.gate_proj|4.0|1.051e-01|2.517e+07|
|layers.5.mlp.down_proj|4.0|1.047e-01|1.258e+07|
|layers.7.mlp.down_proj|4.0|1.028e-01|1.258e+07|
|layers.26.mlp.down_proj|4.0|8.588e-02|1.258e+07|
|layers.4.mlp.down_proj|4.0|7.980e-02|1.258e+07|
|layers.0.mlp.gate_proj|4.0|7.563e-02|2.517e+07|
|layers.2.mlp.gate_proj|4.0|6.902e-02|2.517e+07|
|layers.28.mlp.gate_proj|4.0|6.840e-02|2.517e+07|
|layers.3.mlp.gate_proj|4.0|5.397e-02|2.517e+07|
|layers.27.mlp.down_proj|4.0|5.307e-02|1.258e+07|
|layers.29.mlp.gate_proj|4.0|5.062e-02|2.517e+07|
|layers.30.mlp.gate_proj|4.0|3.400e-02|2.517e+07|
|layers.28.mlp.down_proj|4.0|3.107e-02|1.258e+07|
|layers.29.mlp.down_proj|4.0|2.571e-02|1.258e+07|
|layers.31.mlp.gate_proj|4.0|2.281e-02|2.517e+07|
|layers.2.mlp.down_proj|4.0|2.142e-02|1.258e+07|
|layers.32.mlp.gate_proj|4.0|1.987e-02|2.517e+07|
|layers.33.mlp.gate_proj|4.0|1.870e-02|2.517e+07|
|layers.3.mlp.down_proj|4.0|1.769e-02|1.258e+07|
|layers.30.mlp.down_proj|4.0|1.589e-02|1.258e+07|
|layers.35.mlp.down_proj|4.0|1.466e-02|1.258e+07|
|layers.31.mlp.down_proj|4.0|1.066e-02|1.258e+07|
|layers.32.mlp.down_proj|4.0|6.726e-03|1.258e+07|
|layers.33.mlp.down_proj|4.0|5.197e-03|1.258e+07|
|layers.34.mlp.down_proj|4.0|5.190e-03|1.258e+07|
|layers.35.self_attn.q_proj|4.0|5.374e-04|6.291e+06|
|layers.34.self_attn.q_proj|4.0|5.241e-04|6.291e+06|
|layers.0.self_attn.q_proj|4.0|4.236e-04|6.291e+06|
|layers.6.self_attn.q_proj|4.0|3.308e-04|6.291e+06|
|layers.21.self_attn.q_proj|4.0|2.726e-04|6.291e+06|
|layers.22.self_attn.q_proj|4.0|2.679e-04|6.291e+06|
|layers.24.self_attn.q_proj|4.0|2.320e-04|6.291e+06|
|layers.32.self_attn.q_proj|4.0|2.301e-04|6.291e+06|
|layers.23.self_attn.q_proj|4.0|2.122e-04|6.291e+06|
|layers.7.self_attn.q_proj|4.0|1.942e-04|6.291e+06|
|layers.10.self_attn.q_proj|4.0|1.932e-04|6.291e+06|
|layers.8.self_attn.q_proj|4.0|1.871e-04|6.291e+06|
|layers.5.self_attn.q_proj|4.0|1.674e-04|6.291e+06|
|layers.9.self_attn.q_proj|4.0|1.647e-04|6.291e+06|
|layers.33.self_attn.q_proj|4.0|1.626e-04|6.291e+06|
|layers.30.self_attn.q_proj|4.0|1.613e-04|6.291e+06|
|layers.28.self_attn.q_proj|4.0|1.587e-04|6.291e+06|
|layers.27.self_attn.q_proj|4.0|1.375e-04|6.291e+06|
|layers.17.self_attn.q_proj|4.0|1.353e-04|6.291e+06|
|layers.25.self_attn.q_proj|4.0|1.320e-04|6.291e+06|
|layers.31.self_attn.q_proj|4.0|1.285e-04|6.291e+06|
|layers.19.self_attn.q_proj|4.0|1.283e-04|6.291e+06|
|layers.0.self_attn.o_proj|4.0|1.271e-04|4.194e+06|
|layers.16.self_attn.q_proj|4.0|1.267e-04|6.291e+06|
|layers.3.self_attn.q_proj|4.0|1.234e-04|6.291e+06|
|layers.14.self_attn.q_proj|4.0|1.223e-04|6.291e+06|
|layers.15.self_attn.q_proj|4.0|1.193e-04|6.291e+06|
|layers.6.self_attn.o_proj|4.0|1.167e-04|4.194e+06|
|layers.26.self_attn.q_proj|4.0|1.150e-04|6.291e+06|
|layers.11.self_attn.q_proj|4.0|1.142e-04|6.291e+06|
|layers.18.self_attn.q_proj|4.0|1.128e-04|6.291e+06|
|layers.20.self_attn.q_proj|4.0|1.088e-04|6.291e+06|
|layers.4.self_attn.q_proj|4.0|1.064e-04|6.291e+06|
|layers.12.self_attn.q_proj|4.0|1.045e-04|6.291e+06|
|layers.29.self_attn.q_proj|4.0|9.177e-05|6.291e+06|
|layers.1.self_attn.q_proj|4.0|9.057e-05|6.291e+06|
|layers.13.self_attn.q_proj|4.0|8.245e-05|6.291e+06|
|layers.2.self_attn.q_proj|4.0|7.642e-05|6.291e+06|
|layers.23.self_attn.o_proj|4.0|6.573e-05|4.194e+06|
|layers.8.self_attn.o_proj|4.0|6.420e-05|4.194e+06|
|layers.35.self_attn.o_proj|4.0|6.283e-05|4.194e+06|
|layers.24.self_attn.o_proj|4.0|5.960e-05|4.194e+06|
|layers.1.self_attn.o_proj|4.0|5.905e-05|4.194e+06|
|layers.34.self_attn.o_proj|4.0|5.854e-05|4.194e+06|
|layers.22.self_attn.o_proj|4.0|5.814e-05|4.194e+06|
|layers.14.self_attn.o_proj|4.0|5.058e-05|4.194e+06|
|layers.21.self_attn.o_proj|4.0|4.990e-05|4.194e+06|
|layers.15.self_attn.o_proj|4.0|4.905e-05|4.194e+06|
|layers.5.self_attn.o_proj|4.0|4.693e-05|4.194e+06|
|layers.16.self_attn.o_proj|4.0|4.573e-05|4.194e+06|
|layers.10.self_attn.o_proj|4.0|4.489e-05|4.194e+06|
|layers.12.self_attn.o_proj|4.0|4.291e-05|4.194e+06|
|layers.7.self_attn.o_proj|4.0|4.011e-05|4.194e+06|
|layers.9.self_attn.o_proj|4.0|3.850e-05|4.194e+06|
|layers.19.self_attn.o_proj|4.0|3.820e-05|4.194e+06|
|layers.18.self_attn.o_proj|4.0|3.785e-05|4.194e+06|
|layers.20.self_attn.o_proj|4.0|3.707e-05|4.194e+06|
|layers.17.self_attn.o_proj|4.0|3.659e-05|4.194e+06|
|layers.11.self_attn.o_proj|4.0|3.604e-05|4.194e+06|
|layers.4.self_attn.o_proj|4.0|3.391e-05|4.194e+06|
|layers.28.self_attn.o_proj|4.0|3.242e-05|4.194e+06|
|layers.27.self_attn.o_proj|4.0|2.991e-05|4.194e+06|
|layers.25.self_attn.o_proj|4.0|2.740e-05|4.194e+06|
|layers.3.self_attn.o_proj|4.0|2.706e-05|4.194e+06|
|layers.26.self_attn.o_proj|4.0|2.701e-05|4.194e+06|
|layers.13.self_attn.o_proj|4.0|2.698e-05|4.194e+06|
|layers.30.self_attn.o_proj|4.0|2.649e-05|4.194e+06|
|layers.2.self_attn.o_proj|4.0|2.307e-05|4.194e+06|
|layers.32.self_attn.o_proj|4.0|2.067e-05|4.194e+06|
|layers.31.self_attn.o_proj|4.0|2.027e-05|4.194e+06|
|layers.33.self_attn.o_proj|4.0|1.663e-05|4.194e+06|
|layers.29.self_attn.o_proj|4.0|1.660e-05|4.194e+06|

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