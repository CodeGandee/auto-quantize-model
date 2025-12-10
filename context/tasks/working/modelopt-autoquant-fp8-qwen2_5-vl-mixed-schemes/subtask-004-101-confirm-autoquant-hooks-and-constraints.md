# Subtask 4.1: Confirm AutoQuant hooks and constraints for Qwen2.5-VL-3B LM-only

## Scope

Understand how ModelOpt AutoQuant is currently wired in this repo (via `hf_ptq.py`, `model_quant.py`, and related utilities), what options control sensitivity-based mixed-precision search, and how to constrain AutoQuant to operate on the Qwen2.5-VL-3B language model only while keeping the vision tower in BF16/FP16. This subtask is purely investigative and design-oriented; it does not implement new scripts yet, and it is **not accuracy-driven**—the goal is to map out the search hooks and configuration levers we will use to compare different quantization schemes and sensitivity-analysis strategies.

## Planned outputs

- A concise written summary of how `mtq.auto_quantize` is invoked in `extern/TensorRT-Model-Optimizer/examples/llm_ptq/hf_ptq.py`, including key arguments (`auto_quantize_bits`, `auto_quantize_method`, `num_score_steps`, scoring modes, etc.).
- A clear description of how ModelOpt represents per-layer quantization decisions/configs, and how those can be exported or post-processed to derive LM-only mixed-precision schemes.
- A documented approach for excluding the vision tower and restricting AutoQuant to the language model stack (e.g., by module name patterns, traversal filters, or config constraints).
- A short note describing any constraints or caveats for using AutoQuant with Qwen2.5-VL-3B (e.g., memory usage, calibration loop expectations, unsupported layers).
- An explicit description of which parts of the quantization config are **name-pattern based** (e.g., `_default_disabled_quantizer_cfg`) and how we can override them with **custom configs** (including all-layers FP8) for research experiments.

## TODOs

- [x] Job-004-101-001 Read `extern/TensorRT-Model-Optimizer/examples/llm_ptq/hf_ptq.py` to identify where and how `mtq.auto_quantize` is called, including the meaning of `auto_quantize_bits`, `auto_quantize_method`, `score_mode`, and `num_score_steps`.
- [x] Job-004-101-002 Inspect `extern/TensorRT-Model-Optimizer/modelopt/torch/quantization/model_quant.py` and `extern/TensorRT-Model-Optimizer/modelopt/torch/quantization/utils.py` to understand how AutoQuant constructs the quantized model and quantization config objects (e.g., how per-layer decisions are stored).
- [x] Job-004-101-003 Determine how to restrict AutoQuant to LM-only for Qwen2.5-VL-3B (e.g., by passing an LM-only module, using include/exclude patterns, or post-filtering quantization configs), and write down the preferred approach.
- [x] Job-004-101-004 Identify any AutoQuant configuration knobs that could affect vLLM compatibility (e.g., weight packing formats, extra wrapper modules, or tensor name changes), and note them for later subtasks.
- [x] Job-004-101-005 Summarize findings in a short note under `context/plans/plan-modelopt-autoquant-fp8-qwen2_5-vl-mixed-schemes.md` or a linked context file, so later subtasks can reference the agreed constraints.

All findings for this subtask are captured in:

- `context/plans/plan-modelopt-autoquant-fp8-qwen2_5-vl-mixed-schemes.md`  
  - Section **5. AutoQuant hooks and LM-only constraints (Subtask 4.1)**, which documents:
    - How `hf_ptq.py` calls `mtq.auto_quantize` and how `auto_quantize_bits`, `auto_quantize_method`, and `num_score_steps` interact.
    - How per-layer decisions are represented via `QuantRecipe`, `QuantRecipeHparam`, and AutoQuant’s `state_dict`, and how to derive LM-only mixed-precision manifests.
    - The LM-only restriction pattern for Qwen2.5-VL based on `get_language_model_from_vl` and disabled quant configs for the vision tower.
    - AutoQuant knobs that are relevant for vLLM compatibility (FP8 formats, KV cache quantization, grouping rules, and effective bits vs memory/accuracy trade-offs).

## Summary

This subtask finalized the design for using ModelOpt AutoQuant on Qwen2.5-VL-3B LM-only:

- Clarified the `hf_ptq.py` AutoQuant wrapper around `mtq.auto_quantize`, including how `auto_quantize_bits`, FP8 `quantization_formats`, calibration vs scoring steps, methods (`gradient` vs `kl_div`), and `disabled_layers` are wired.
- Described how AutoQuant stores per-layer decisions using `QuantRecipe` / `QuantRecipeHparam` and the searcher `state_dict`, and how these can be post-processed into an LM-only mixed-precision manifest independent of the exported checkpoint.
- Selected an LM-only pattern for Qwen2.5-VL-3B based on `get_language_model_from_vl`: quantize the extracted language model module while leaving the vision tower BF16/FP16, then export the full VLM with a quantized LM.
- Identified key knobs for vLLM compatibility (FP8-only formats, KV cache options, grouping rules, effective bits vs memory) that will constrain scheme design and driver implementation in later subtasks.

## Notes

- Prefer reading ModelOpt’s vendored sources in `extern/TensorRT-Model-Optimizer/` as reference only; do not modify them in this repo.
- Keep an eye on how calibration forward loops are expected to be structured, since this will drive the design of the Qwen2.5-VL-3B AutoQuant driver in a later subtask.
