# Subtask 4.6: Write FP8 AutoQuant schemes report

**Status**: CANCEL

## Scope

Create a report summarizing the AutoQuant FP8 scheme experiments for Qwen2.5-VL-3B, including settings used, per-scheme quantization coverage, layer sensitivity observations, and vLLM compatibility results. This subtask focuses on documentation rather than new code, and the report should emphasize **comparative insights** across schemes and configs rather than picking a single “best” accuracy setting.

## Planned outputs

- A new report file `models/qwen2_5_vl_3b_instruct/reports/report-fp8-autoquant-schemes.md` containing:
  - A brief overview of the motivation and setup.
  - A description of each scheme’s configuration (effective bits, coverage, key AutoQuant flags).
  - Summaries of quantization coverage (e.g., number of FP8 vs BF16/FP16 layers) and any noteworthy patterns.
  - Highlights from **layer-sensitivity analysis** (e.g., which blocks consistently appear as high-sensitivity across configs).
  - A section on vLLM compatibility and observations.
- Optional tables or bullet lists that cross-reference scheme names, checkpoint paths, and compatibility status.

## TODOs

- [ ] **CANCEL** Job-004-106-001 Create the initial `models/qwen2_5_vl_3b_instruct/reports/report-fp8-autoquant-schemes.md` file with a header, context section, and placeholders for schemes and results.
- [ ] **CANCEL** Job-004-106-002 Populate a section describing the experimental setup, including calibration data, hardware, and key AutoQuant configuration defaults.
- [ ] **CANCEL** Job-004-106-003 For each scheme, document the AutoQuant settings, quantization coverage (e.g., approximate fraction of LM blocks quantized), **layer-sensitivity trends**, and any qualitative quality notes (if available).
- [ ] **CANCEL** Job-004-106-004 Summarize vLLM compatibility findings, referencing artifacts from Subtask 4.5 and clearly calling out which schemes are recommended for vLLM use.
- [ ] **CANCEL** Job-004-106-005 Add cross-links to relevant context files and scripts (plan, driver, export helper) so future readers can reproduce or extend the experiments.

## Notes

- Keep the report focused on the language model; any future work on vision-aware quantization can be referenced as out of scope.
