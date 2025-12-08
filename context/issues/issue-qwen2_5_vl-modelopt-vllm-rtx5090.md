# Issue: Qwen2.5-VL ModelOpt quantization + vLLM on RTX 5090

## Summary

We can quantize `Qwen2.5-VL-3B-Instruct` with NVIDIA ModelOpt and export a W8A8 (INT8 SmoothQuant + FP8 KV cache) Hugging Face checkpoint, but:

- vLLM refuses to run that checkpoint because its ModelOpt integration only supports `FP8` and `NVFP4` quantizations.
- A direct FP8 quantization run currently fails on RTX 5090 with a CUDA device-side assert in the FP8 path.

For now, the W8A8 + FP8 KV model is usable via HF/ModelOpt, but **not** via vLLM; a pure FP8 model is not stable on this GPU/toolchain combination yet.

## Environment

- GPU: RTX 5090 (Blackwell, `sm_120`)
- CUDA toolkit (pixi, `rtx5090` env): `cuda-toolkit 12.8.1.*` from `conda-forge`
- PyTorch (`rtx5090` env, pinned for vLLM):
  - `torch == 2.9.1+cu128`
  - `torchvision == 0.24.1+cu128`
  - `torchaudio == 2.9.1+cu128`
- vLLM:
  - Custom wheel built from `extern/vllm`, branch `autoq-vllm-v0.10.1`
  - Built in `rtx5090` env with the above torch/CUDA stack
- ModelOpt:
  - `extern/TensorRT-Model-Optimizer` workspace, using `examples/llm_ptq/hf_ptq.py`

## W8A8 + FP8 KV quantization (works, but vLLM rejects it)

Script:

- `scripts/qwen/quantize_qwen2_5_vl_3b_w8a8_coco2017.sh`
  - `--qformat "int8_sq"`
  - `--kv_cache_qformat "fp8"`
  - `--dataset "coco2017_captions_local"`
  - `--calib_size "4096"`, `--calib_seq 512`
  - `--export_path models/qwen2_5_vl_3b_instruct/quantized/w8a8_int8_sq_coco2017`

Local dataset hook added to ModelOpt:

- File: `extern/TensorRT-Model-Optimizer/modelopt/torch/utils/dataset_utils.py`
- We added a special case in `_get_dataset_samples(dataset_name, num_samples)`:
  - `dataset_name == "coco2017_captions_local"`:
    - Reads captions from:
      - `COCO2017_CAPTIONS_LOCAL_PATH` (if set), or
      - `datasets/vlm-quantize-calib/coco2017_captions.txt` (newline-delimited).
    - Returns up to `num_samples` captions.

Export path change for VLMs:

- File: `extern/TensorRT-Model-Optimizer/examples/llm_ptq/hf_ptq.py`
- Original behavior: for `int8_sq` and some other conditions, ModelOpt attempted a TensorRT-LLM checkpoint export, which is **not** supported for Qwen2.5-VL and raised:
  - `NotImplementedError: Cannot export tensorrt_llm checkpoint for model qwen: Qwen2_5_VLForConditionalGeneration.`
- We changed the export condition so that:
  - TensorRT-LLM export is used for `int8_sq` **only when the model is not a VLM**.
  - For VLMs (`is_vlm == True`), even with `qformat="int8_sq"`, we go through the unified HF export path instead.

Result:

- Command:
  - `pixi run -e rtx5090 bash scripts/qwen/quantize_qwen2_5_vl_3b_w8a8_coco2017.sh`
- Outcome:
  - Quantization completes.
  - HF checkpoint (plus processor/tokenizer config) exported to:
    - `models/qwen2_5_vl_3b_instruct/quantized/w8a8_int8_sq_coco2017`

vLLM behavior on this checkpoint:

- Run:

  ```python
  from vllm import LLM, SamplingParams
  from pathlib import Path

  path = Path("models/qwen2_5_vl_3b_instruct/quantized/w8a8_int8_sq_coco2017").resolve()
  llm = LLM(str(path), tensor_parallel_size=1)
  ```

- Error (from vLLM’s ModelOpt config validation):

  - `Value error, ModelOpt currently only supports: ['FP8', 'NVFP4'] quantizations in vLLM.`
  - It inspects `hf_quant_config.json` and rejects the `INT8_SMOOTHQUANT` configuration.

Conclusion:

- The **INT8 SmoothQuant + FP8 KV** (W8A8) model is currently **not usable from vLLM**; ModelOpt’s vLLM integration only supports FP8/NVFP4 configs.

## FP8 + FP8 attempt (fails with CUDA device-side assert)

Script added for FP8:

- `scripts/qwen/quantize_qwen2_5_vl_3b_fp8_coco2017.sh`
  - Uses the same checkpoint and calibration captions as the W8A8 script.
  - Invokes `hf_ptq.py` with:
    - `--qformat "fp8"`
    - `--kv_cache_qformat "fp8"`
    - `--dataset "coco2017_captions_local"`
    - `--export_path models/qwen2_5_vl_3b_instruct/quantized/fp8_fp8_coco2017`

Run (RTX 5090 env):

- Command:
  - `pixi run -e rtx5090 bash scripts/qwen/quantize_qwen2_5_vl_3b_fp8_coco2017.sh`

Observed behavior:

- Model and quantization proceed through the calibration loop, but:
  - ModelOpt logs:
    - `Unknown CUDA arch (10.1) or GPU not supported`
    - `CUDA extension for FP8 quantization could not be built and loaded, FP8 simulated quantization will not be available.`
  - After quantization, when `hf_ptq.py` runs a preview `generate` call, we get:
    - `_assert_async_cuda_kernel: ... Assertion 'probability tensor contains either 'inf', 'nan' or element < 0' failed.`
    - Followed by `torch.AcceleratorError: CUDA error: device-side assert triggered`.
  - No `Quantized model exported to` message is printed; the script exits with non-zero status.

Interpretation:

- On RTX 5090 / CUDA 12.8.1 / torch 2.9.1, the ModelOpt FP8 path for Qwen2.5-VL:
  - Cannot build/load the `cuda_ext_fp8` extension for this arch.
  - Falls back to a simulated FP8 behavior, which appears numerically unstable for this model: probabilities contain invalid values (NaN/Inf/negative), triggering an internal CUDA assertion during generation.
- Therefore we currently **cannot obtain a stable FP8+FP8 Qwen2.5-VL model** from this toolchain on this GPU.

## Current status / recommendations

- **Working path today:**
  - W8A8 (INT8 SmoothQuant) + FP8 KV cache via ModelOpt HF PTQ.
  - Exported checkpoint: `models/qwen2_5_vl_3b_instruct/quantized/w8a8_int8_sq_coco2017`.
  - Usable via HF/ModelOpt runtime (not vLLM).

- **Not working / blocked:**
  - vLLM on the W8A8 model:
    - Blocked by vLLM’s ModelOpt integration only supporting `FP8` and `NVFP4` with its current `VllmConfig` validation.
  - FP8+FP8 quantization:
    - FP8 CUDA extension not built for this arch, simulated FP8 path hits device-side assert in `generate`.

## Possible future directions

1. **Upstream vLLM support for INT8_SMOOTHQUANT**  
   - Extend vLLM’s ModelOpt integration to accept `INT8_SMOOTHQUANT` (and other INT8 formats) in `hf_quant_config.json`, plus any kernel/runtime support needed.

2. **Revisit FP8 on RTX 5090 once ModelOpt adds support**  
   - Track ModelOpt releases for explicit RTX 5090/Blackwell FP8 support.
   - Once `cuda_ext_fp8` builds cleanly for `sm_120`, retry the FP8+FP8 quantization path.

3. **Interim: keep W8A8 for HF, non-ModelOpt quantization for vLLM**  
   - If vLLM integration is required sooner, consider:
     - Using vLLM’s own quantization options (e.g., Marlin or other backends) on the original FP16/BF16 checkpoint instead of feeding it a ModelOpt-quantized one.

