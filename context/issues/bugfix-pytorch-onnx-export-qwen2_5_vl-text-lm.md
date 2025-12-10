# Status: Open (blocked by upstream)

Bugfix: PyTorch ONNX / export failures for Qwen2.5‑VL text LM
=============================================================

Summary
-------

- Exporting the **text-only** branch of **Qwen2.5‑VL‑3B‑Instruct** to ONNX fails in both legacy and new PyTorch exporters.
- This appears to be a **PyTorch `torch.onnx` / `torch.export` bug**, not a model‑specific or repo‑specific issue.
- The failure blocks our plan to:
  - Run **INC + ONNXRuntime GPU** sensitivity for the text tower.
  - Use a clean ONNX text LM as part of mixed-precision profiling for Qwen2.5‑VL.


Environment
-----------

- Repo: `auto-quantize-model`
- Model: `Qwen2.5-VL-3B-Instruct`
  - Local HF checkpoint:  
    `models/qwen2_5_vl_3b_instruct/checkpoints/Qwen2.5-VL-3B-Instruct`
- Export script (text-only LM wrapper):  
  `models/qwen2_5_vl_3b_instruct/helpers/convert_to_onnx_text_fp16.py`
- Environments:
  - `pixi run -e rtx5090 python ...` (INC / quantization-focused env)
  - `pixi run -e rtx5090-vllm python ...` (vLLM / newer PyTorch + ONNX exporter)


Reproduction
------------

1. Ensure the Qwen2.5‑VL checkpoint is bootstrapped:

   - `models/qwen2_5_vl_3b_instruct/bootstrap.sh`

2. Run the text LM ONNX export in the **rtx5090** env:

   - `pixi run -e rtx5090 python models/qwen2_5_vl_3b_instruct/helpers/convert_to_onnx_text_fp16.py`

3. Run the same export in the **rtx5090-vllm** env:

   - `pixi run -e rtx5090-vllm python models/qwen2_5_vl_3b_instruct/helpers/convert_to_onnx_text_fp16.py`

4. Logs:
   - Legacy exporter (rtx5090):  
     `models/qwen2_5_vl_3b_instruct/onnx/qwen2_5_vl_3b_text_fp16_export_error.log`
   - New exporter (rtx5090-vllm):  
     `tmp/rtx5090-vllm-exp-001/qwen2_5_vl_3b_text_fp16_export_error_vllm_env.log`


Observed behavior
-----------------

**rtx5090 env (legacy `torch.onnx.export` / TorchScript path):**

- `convert_to_onnx_text_fp16.py`:
  - Loads `Qwen2_5_VLForConditionalGeneration` with `attn_implementation="eager"`.
  - Wraps it in `Qwen25VLTextWrapper` exposing `(input_ids, attention_mask) -> logits`.
  - Calls `torch.onnx.export(...)` with `opset_version=17`, dynamic axes, and `do_constant_folding=True`.
- Export fails during TorchScript tracing with:
  - `RuntimeError: _Map_base::at`
  - Stack goes through:
    - `torch.onnx.utils._model_to_graph(...)`
    - `torch.jit._get_trace_graph(...)`
    - `torch._functorch.autograd_function.custom_function_call_vmap_generate_rule(...)`
- No `qwen2_5_vl_3b_text_fp16.onnx` is produced.

**rtx5090-vllm env (new `torch.export`-based ONNX path):**

- Same script, but `torch.onnx.export` now routes through the new exporter:
  - `torch.onnx._internal.exporter._core.export(...)`
  - `torch.export.export(...)` + AOTAutograd + FX/ProxyTensor + FakeTensor.
- Export fails earlier with:
  - `torch.onnx._internal.exporter._errors.TorchExportError: Failed to export the model with torch.export. This is step 1/3 of exporting the model to ONNX.`
  - Root cause:
    - `RuntimeError: 8*s72 (...) is not tracked with proxy for <torch.fx.experimental.proxy_tensor._ModuleStackTracer ...>`
  - Stack passes through:
    - `torch.export._trace._export_to_aten_ir_make_fx`
    - `torch.fx.experimental.proxy_tensor.dispatch_trace`
    - `transformers.masking_utils.create_causal_mask -> sdpa_mask_recent_torch -> _vmap_for_bhqkv(...)`
    - `torch._functorch.vmap` and custom higher-order autograd functions
    - `aten.index` handling in FakeTensor / ProxyTensor.
- Again, no `qwen2_5_vl_3b_text_fp16.onnx` is produced.

**What does work in this repo:**

- Vision-only ONNX export for Qwen2.5‑VL at fixed 672×672:
  - `models/qwen2_5_vl_3b_instruct/onnx/qwen2_5_vl_3b_vision_672_fp32.onnx`
  - Uses custom `build_patches_and_grid` and avoids the problematic masking/vmap stack.
- CPU-only INC sensitivity on the PyTorch FX graph (no ONNX) for the full model.


Analysis / suspected root cause
-------------------------------

- The failures are **not** classic “unsupported ONNX op” errors and **not** tied to flash-attention CUDA kernels:
  - We explicitly load with `attn_implementation="eager"`.
  - The crash occurs on both CPU and GPU.
- Both exporters fail inside **PyTorch’s tracing / export engine**, not in our model code:
  - Legacy path: TorchScript + functorch → `_Map_base::at`.
  - New path: `torch.export` + FX + ProxyTensor/FakeTensor → “not tracked with proxy” error.
- The call stack points at **new masking utilities with vmap + higher-order autograd**:
  - `create_causal_mask` → `sdpa_mask_recent_torch` → `_vmap_for_bhqkv(...)` → functorch custom functions.
  - These use autograd/functorch machinery even in nominal “inference mode” because `torch.export` simulates execution with symbolic shapes.
- This matches **known upstream PyTorch export bugs**:
  - `/pytorch/pytorch#163713`: `[export] 8*s72 is not tracked with proxy for torch.fx.experimental.proxy_tensor._ModuleStackTracer object`
    - Reported when exporting a Transformers model (`transformers==4.53.0`).
  - `/pytorch/pytorch#163146`: `[export] Data dependent error on slices in autograd::applySlicing`
    - Data-dependent slicing / masking causing `torch.export` failures.
  - Older `_Map_base::at` exporter bugs:
    - `/pytorch/pytorch#106972`, `NVIDIA/TransformerEngine#269`, `NVlabs/FB-BEV#37`.
  - Our stack traces and symptoms are consistent with these issues.
- Conclusion:
  - This is effectively an **upstream PyTorch ONNX/exporter bug** triggered by Qwen2.5‑VL’s attention/masking stack, not a bug in this repo’s wrappers.


Impact
------

- Blocks **text-only ONNX export** for Qwen2.5‑VL‑3B in our current environments.
- Prevents:
  - Running **INC+ONNXRuntime GPU** layer sensitivity on the text tower.
  - Building a clean ONNX text LM to pair with the vision ONNX for VLM‑wide sensitivity profiles.
- Forces our INC workflow to:
  - Use **CPU-side FX models** for full-graph sensitivity.
  - Use **ONNX only for vision subgraphs** (where export is stable).


Proposed handling in this repo
------------------------------

- Treat this as **“blocked by upstream PyTorch exporter”**, not something we patch locally:
  - Do **not** fork HF Qwen2.5‑VL or hand‑rewrite the masking/vmap stack just to satisfy `torch.export`.
  - Keep using the **vision-only ONNX** + **CPU FX** approach for sensitivity until PyTorch is fixed.
- Track upstream progress:
  - Monitor `/pytorch/pytorch#163713` and `/pytorch/pytorch#163146`.
  - Re‑test the export when:
    - PyTorch or Transformers version is bumped in this repo, or
    - Exporter docs / release notes mention fixes for vmap / masking / ProxyTensor issues.
- For cross‑stack comparisons (INC vs ModelOpt vs TensorRT‑LLM):
  - Use **Qwen2‑VL (non‑2.5)** ONNX community models as a proxy where we need full ONNX VLM graphs.
  - Use Qwen2.5‑VL PyTorch FX sensitivity + vision ONNX as complementary signals.


Next steps / TODO
-----------------

- [ ] Add a short note in:
  - `context/issues/known/issue-inc-gpu-limitations-for-vlm-layer-sensitivity.md`
  - `context/hints/howto-export-qwen2_5_vl-to-onnx-for-layer-sensitivity.md`
  to explicitly link to this bugfix entry and the upstream PyTorch issues.
- [ ] When upgrading PyTorch / Transformers:
  - Re‑run both export commands (rtx5090 and rtx5090-vllm).
  - If successful, document:
    - Required versions.
    - Any additional flags (e.g., `dynamic_shapes` instead of `dynamic_axes`).
    - Any model code changes needed.
- [ ] If upstream provides a minimal repro and workaround for the masking/vmap issue:
  - Evaluate whether a **small, local shim** (e.g., replacing `create_causal_mask` usage in our wrapper) is acceptable.

