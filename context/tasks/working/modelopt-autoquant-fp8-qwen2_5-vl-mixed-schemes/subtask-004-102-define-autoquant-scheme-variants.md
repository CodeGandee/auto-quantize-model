# Subtask 4.2: Define AutoQuant FP8 scheme variants and coverage targets

## Scope

Design a small, well-motivated set of AutoQuant FP8 mixed-precision scheme variants for the Qwen2.5-VL-3B language model, varying the fraction of layers quantized and/or effective bits budgets while keeping the vision tower unquantized. This subtask focuses on specifying scheme names, high-level configuration knobs, and how to map AutoQuant outputs into discrete schemes, but does not yet implement the driver script. The emphasis is on **defining schemes for comparison** (including future all-layers and custom configs), not on picking a single “best” accuracy configuration.

## Planned outputs

- A list of 2–3 named schemes (e.g., `fp8_autoquant_top25`, `fp8_autoquant_top50`, `fp8_autoquant_full`), each with:
  - Target effective bits or other AutoQuant settings (e.g., `auto_quantize_bits`, scoring mode, number of scoring steps).
  - A clear definition of how many transformer blocks or which sensitivity percentile should be quantized for that scheme.
- A document or small config file (e.g., JSON or YAML) mapping scheme names to AutoQuant configuration parameters and any post-filtering rules.
- A brief rationale for each scheme, explaining expected quality vs. aggressiveness and how it will be used in experiments.

## TODOs

- [x] Job-004-102-001 Based on Subtask 4.1 findings and the main plan, choose 2–3 AutoQuant operating points (e.g., different `effective_bits` or scoring budgets) that are likely to produce distinct mixed-precision profiles for the LM.
- [x] Job-004-102-002 Decide how to translate AutoQuant sensitivity scores into “top-K layers quantized” schemes (e.g., quantize top 25% and 50% of blocks by sensitivity, plus an all-eligible baseline), and write down the mapping rules.
- [x] Job-004-102-003 Assign stable scheme names (e.g., `fp8_autoquant_top25`, `fp8_autoquant_top50`, `fp8_autoquant_full`) and define their expected coverage and high-level goals.
- [x] Job-004-102-004 Draft a small machine-readable config (JSON/YAML or Python dict) that maps scheme names to AutoQuant configuration knobs and any post-processing parameters; decide where this config should live (e.g., alongside the driver script).
- [x] Job-004-102-005 Record the scheme catalog and rationale in either the main plan or a short sub-doc, so later subtasks can treat it as authoritative.

## Notes

- Keep the number of schemes small enough to be practical for experimentation but rich enough to reveal meaningful quality vs. compression trade-offs.
- When deciding coverage rules, consider future reuse for larger Qwen2.5-VL variants or related models.

## Summary

This subtask defines the initial FP8 AutoQuant scheme catalog for Qwen2.5-VL-3B LM-only:

- We selected three schemes—`fp8_autoquant_top25`, `fp8_autoquant_top50`, and `fp8_autoquant_full`—with progressively more aggressive effective bits targets and FP8 coverage.
- We specified how to translate AutoQuant’s per-layer search results into “top-K LM blocks quantized” schemes by ranking blocks and pruning FP8 recipes according to a coverage fraction.
- We outlined a small machine-readable config (an `AUTOQUANT_FP8_SCHEMES` dict) that maps scheme names to `auto_quantize_bits`, method, score size, coverage mode, and allowed quantization formats.
- All details, including rationale and mapping rules, are recorded in `context/plans/plan-modelopt-autoquant-fp8-qwen2_5-vl-mixed-schemes.md` under **6. AutoQuant FP8 scheme variants (Subtask 4.2)**.
