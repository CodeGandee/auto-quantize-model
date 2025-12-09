# Subtask 4.2: Define AutoQuant FP8 scheme variants and coverage targets

## Scope

Design a small, well-motivated set of AutoQuant FP8 mixed-precision scheme variants for the Qwen2.5-VL-3B language model, varying the fraction of layers quantized and/or effective bits budgets while keeping the vision tower unquantized. This subtask focuses on specifying scheme names, high-level configuration knobs, and how to map AutoQuant outputs into discrete schemes, but does not yet implement the driver script.

## Planned outputs

- A list of 2–3 named schemes (e.g., `fp8_autoquant_top25`, `fp8_autoquant_top50`, `fp8_autoquant_full`), each with:
  - Target effective bits or other AutoQuant settings (e.g., `auto_quantize_bits`, scoring mode, number of scoring steps).
  - A clear definition of how many transformer blocks or which sensitivity percentile should be quantized for that scheme.
- A document or small config file (e.g., JSON or YAML) mapping scheme names to AutoQuant configuration parameters and any post-filtering rules.
- A brief rationale for each scheme, explaining expected quality vs. aggressiveness and how it will be used in experiments.

## TODOs

- [ ] Job-004-102-001 Based on Subtask 4.1 findings and the main plan, choose 2–3 AutoQuant operating points (e.g., different `effective_bits` or scoring budgets) that are likely to produce distinct mixed-precision profiles for the LM.
- [ ] Job-004-102-002 Decide how to translate AutoQuant sensitivity scores into “top-K layers quantized” schemes (e.g., quantize top 25% and 50% of blocks by sensitivity, plus an all-eligible baseline), and write down the mapping rules.
- [ ] Job-004-102-003 Assign stable scheme names (e.g., `fp8_autoquant_top25`, `fp8_autoquant_top50`, `fp8_autoquant_full`) and define their expected coverage and high-level goals.
- [ ] Job-004-102-004 Draft a small machine-readable config (JSON/YAML or Python dict) that maps scheme names to AutoQuant configuration knobs and any post-processing parameters; decide where this config should live (e.g., alongside the driver script).
- [ ] Job-004-102-005 Record the scheme catalog and rationale in either the main plan or a short sub-doc, so later subtasks can treat it as authoritative.

## Notes

- Keep the number of schemes small enough to be practical for experimentation but rich enough to reveal meaningful quality vs. compression trade-offs.
- When deciding coverage rules, consider future reuse for larger Qwen2.5-VL variants or related models.

