# Subtask 4.1: Confirm AutoQuant hooks and constraints for Qwen2.5-VL-3B LM-only

## Scope

Understand how ModelOpt AutoQuant is currently wired in this repo (via `hf_ptq.py`, `model_quant.py`, and related utilities), what options control sensitivity-based mixed-precision search, and how to constrain AutoQuant to operate on the Qwen2.5-VL-3B language model only while keeping the vision tower in BF16/FP16. This subtask is purely investigative and design-oriented; it does not implement new scripts yet.

## Planned outputs

- A concise written summary of how `mtq.auto_quantize` is invoked in `extern/TensorRT-Model-Optimizer/examples/llm_ptq/hf_ptq.py`, including key arguments (`auto_quantize_bits`, `auto_quantize_method`, `num_score_steps`, scoring modes, etc.).
- A clear description of how ModelOpt represents per-layer quantization decisions/configs, and how those can be exported or post-processed to derive LM-only mixed-precision schemes.
- A documented approach for excluding the vision tower and restricting AutoQuant to the language model stack (e.g., by module name patterns, traversal filters, or config constraints).
- A short note describing any constraints or caveats for using AutoQuant with Qwen2.5-VL-3B (e.g., memory usage, calibration loop expectations, unsupported layers).

## TODOs

- [ ] Job-004-101-001 Read `extern/TensorRT-Model-Optimizer/examples/llm_ptq/hf_ptq.py` to identify where and how `mtq.auto_quantize` is called, including the meaning of `auto_quantize_bits`, `auto_quantize_method`, `score_mode`, and `num_score_steps`.
- [ ] Job-004-101-002 Inspect `extern/TensorRT-Model-Optimizer/modelopt/torch/quantization/model_quant.py` and `extern/TensorRT-Model-Optimizer/modelopt/torch/quantization/utils.py` to understand how AutoQuant constructs the quantized model and quantization config objects (e.g., how per-layer decisions are stored).
- [ ] Job-004-101-003 Determine how to restrict AutoQuant to LM-only for Qwen2.5-VL-3B (e.g., by passing an LM-only module, using include/exclude patterns, or post-filtering quantization configs), and write down the preferred approach.
- [ ] Job-004-101-004 Identify any AutoQuant configuration knobs that could affect vLLM compatibility (e.g., weight packing formats, extra wrapper modules, or tensor name changes), and note them for later subtasks.
- [ ] Job-004-101-005 Summarize findings in a short note under `context/plans/plan-modelopt-autoquant-fp8-qwen2_5-vl-mixed-schemes.md` or a linked context file, so later subtasks can reference the agreed constraints.

## Notes

- Prefer reading ModelOpt’s vendored sources in `extern/TensorRT-Model-Optimizer/` as reference only; do not modify them in this repo.
- Keep an eye on how calibration forward loops are expected to be structured, since this will drive the design of the Qwen2.5-VL-3B AutoQuant driver in a later subtask.

