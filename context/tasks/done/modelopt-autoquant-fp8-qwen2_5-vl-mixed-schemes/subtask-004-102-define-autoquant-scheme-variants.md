# Subtask 4.2: Define AutoQuant FP8 scheme variants and coverage targets

## Scope

Design a small, well-motivated set of AutoQuant FP8 mixed-precision scheme variants for the Qwen2.5-VL-3B language model, varying the fraction of layers quantized and/or effective bits budgets while keeping the vision tower unquantized. This subtask focuses on specifying scheme names, high-level configuration knobs, and how to map AutoQuant outputs into discrete schemes, but does not yet implement the driver script. The emphasis is on **defining schemes for comparison** (including future all-layers and custom configs), not on picking a single “best” accuracy configuration.

## Planned outputs

- A **family** of named schemes that sweep LM-only FP8 coverage in 10% increments, e.g., `fp8_autoquant_top10`, `fp8_autoquant_top20`, …, `fp8_autoquant_top100`, each with:
  - Target effective bits or other AutoQuant settings (e.g., `auto_quantize_bits`, scoring mode, number of scoring steps).
  - A clear definition of what “top XX% of layers” means in terms of transformer blocks and AutoQuant sensitivity rankings.
- A document or small config file (e.g., JSON or YAML) mapping scheme names to AutoQuant configuration parameters and any post-filtering rules.
- A brief rationale for the coverage grid (why 10% steps) and how these schemes will be used in experiments.

## TODOs

- [x] Job-004-102-001 Based on Subtask 4.1 findings and the main plan, choose representative AutoQuant operating points (e.g., a baseline `effective_bits` and scoring budget) that are suitable for sweeping coverage across multiple LM-only schemes.
- [x] Job-004-102-002 Decide how to translate AutoQuant sensitivity scores into “top-K layers quantized” schemes, using coverage fractions at 10% steps (e.g., top 10%, 20%, 30%, …, 100% of LM blocks by sensitivity), and write down the mapping rules.
- [x] Job-004-102-003 Assign stable scheme names (`fp8_autoquant_top10`, `fp8_autoquant_top20`, …, `fp8_autoquant_top100`) and define their expected coverage and high-level goals.
- [x] Job-004-102-004 Draft a small machine-readable config (JSON/YAML or Python dict) that maps scheme names to AutoQuant configuration knobs and any post-processing parameters; decide where this config should live (e.g., alongside the driver script).
- [x] Job-004-102-005 Record the scheme catalog and rationale in either the main plan or a short sub-doc, so later subtasks can treat it as authoritative.

## Notes

- Keep the number of schemes small enough to be practical for experimentation but rich enough to reveal meaningful quality vs. compression trade-offs.
- When deciding coverage rules, consider future reuse for larger Qwen2.5-VL variants or related models.

## Summary

This subtask defines the initial FP8 AutoQuant scheme catalog for Qwen2.5-VL-3B LM-only:

- We defined a **coverage grid** of schemes—`fp8_autoquant_top10`, `fp8_autoquant_top20`, …, `fp8_autoquant_top100`—that sweep how many LM blocks are quantized to FP8 in 10% increments, while keeping the vision tower in BF16/FP16.
- We specified a **two-stage** procedure: first run a full LM-only sensitivity analysis with all eligible blocks participating in AutoQuant, then rank blocks by sensitivity and, for each scheme, quantize only the top-X% most sensitive LM blocks while leaving the rest in BF16/FP16.
- We outlined a small machine-readable config (an `AUTOQUANT_FP8_SCHEMES` dict) that maps scheme names to `auto_quantize_bits`, method, score size, coverage mode, coverage fraction, and allowed quantization formats.
- All details, including rationale and mapping rules, are recorded in `context/plans/done/plan-modelopt-autoquant-fp8-qwen2_5-vl-mixed-schemes.md` under **6. AutoQuant FP8 scheme variants (Subtask 4.2)**.
