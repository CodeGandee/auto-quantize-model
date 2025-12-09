# Task: INC Op Sensitivity for Qwen2.5-VL-3B W8A8

## Current Status

- **INC monkey-patch for MSE-V2**: Implemented in `src/auto_quantize_model/inc_pytorch_mse_patching.py`. It wraps `get_mse_order_per_fp32` and `get_mse_order_per_int8` to record per-op MSE during INC PTQ runs.
- **Qwen2.5-VL-3B sensitivity driver**: Implemented in `scripts/qwen/inc_qwen2_5_vl_3b_sensitivity.py`. Uses COCO2017 captions to run static PTQ (CPU, PyTorch FX backend) with `mse_v2` strategy and captures:
  - `fp32_fallback_sensitivity`: MSE when treating INT8 ops as candidates for FP32 fallback.
  - `int8_requant_sensitivity`: MSE when re-quantizing FP32 ops.
- **Output artifacts**:
  - JSON: `tmp/qwen2_5_vl_3b_inc/op_sensitivity_mse_v2_cpu.json`
  - Markdown: `tmp/qwen2_5_vl_3b_inc/op_sensitivity_mse_v2_cpu.md`
  - Optional quantized model checkpoint: `tmp/qwen2_5_vl_3b_inc/q_model.pt`
- **Environment**: Designed to run under the Pixi RTX 5090 env, but currently using CPU for INC (PyTorch FX backend expectations).

## Gaps / TODOs

- **W8A8 / spectrum mapping**:
  - No dedicated visualization or ranking utility yet to map the captured MSE spectrum into concrete W8A8 “keep high-precision here” decisions.
  - No automated policy to select top-N most sensitive ops for FP32/BF16 fallback based on the JSON.
- **GPU-aware analysis**:
  - Current sensitivity run is CPU-only; no variant that measures sensitivity under a GPU-executed path (e.g., to account for kernel/layout differences).
- **Integration hooks**:
  - No direct wiring from the sensitivity JSON into downstream quantization/export pipelines (e.g., feeding a “do-not-quantize” list into ONNX or ModelOpt configs).

## Next Steps

- Add a small analysis script (e.g. `scripts/qwen/analyze_inc_op_sensitivity.py`) that:
  - Loads `op_sensitivity_mse_v2_cpu.json`.
  - Produces sorted tables/plots of op sensitivity.
  - Emits candidate op-keep lists (e.g., top-K ops by MSE) suitable for W8A8 schemes.
- Wire the sensitivity output into the chosen W8A8 config format:
  - Generate a machine-readable “fallback ops” file for Qwen2.5-VL-3B.
  - Ensure it can be consumed by existing quantization/export tooling in this repo.
- (Optional) Explore a GPU-backed variant of the INC run if/when INC backend support makes sense for RTX 5090.

