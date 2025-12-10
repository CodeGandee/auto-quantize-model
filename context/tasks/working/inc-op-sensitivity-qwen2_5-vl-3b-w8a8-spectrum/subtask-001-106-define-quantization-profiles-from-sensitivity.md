# Subtask 1.6: Define quantization profiles from sensitivity rankings

## Scope

Use the sensitivity rankings (from MSE_V2 and optionally HAWQ_V2) to design several quantization profiles that span a spectrum from minimal to aggressive W8A8 INT8 quantization. Each profile specifies which layers/ops remain in higher precision and which are quantized to INT8. These profiles are **analysis artifacts first**: they should be usable both for INC PTQ runs and for cross-framework comparison (e.g., against ModelOpt W8A8 / FP8 schemes), even when the underlying INC PTQ runs that produced the rankings did not yield high-accuracy quantized models.

## Planned outputs

- A small set of profiles (e.g., A–D) such as:
  - Minimal quantization (only least sensitive layers quantized).
  - Moderate quantization (bottom ~50% of layers quantized).
  - Aggressive quantization (bottom ~75–80% quantized).
  - Max W8A8 (all but a small set of sensitive ops quantized).
- A machine-readable profile file (JSON/YAML/Markdown) mapping:
  - Profile name → list of ops to quantize vs keep higher precision.

## TODOs

- [ ] Job-001-106-001 Analyze the sensitivity ranking and determine reasonable cut points for different quantization levels.
- [ ] Job-001-106-002 Define profile schemas (e.g., by op name patterns or layer indices) that can be fed into INC configs.
- [ ] Job-001-106-003 Serialize the profiles to a configuration file (e.g., `qwen2_5_vl_3b_quant_profiles.json` or `.md`) under `context/summaries/inc-kb/`.
- [ ] Job-001-106-004 Validate that the profiles align with practical constraints (e.g., always keep embeddings/LayerNorm/Softmax in higher precision).

## Notes

- Profiles should be designed to be easy to reuse across scripts and potentially other models with similar architectures.
- Treat the INC-derived sensitivity rankings as one input among several: when defining profiles, consider how they compare to ModelOpt / manual schemes, and do not assume that every INC “most sensitive” op must stay in higher precision—these profiles are primarily for experimentation and trade-off exploration, not hard production rules.
