# Subtask 1.8: Extend sanity-check script for multiple Qwen2.5-VL variants

## Scope

Extend or wrap `scripts/qwen/run_qwen2_5_vl_3b_sanity.py` so it can evaluate both the baseline and multiple INC-quantized variants using a consistent set of prompts, and label outputs by variant for easier comparison. The same machinery should be usable to compare ModelOpt and INC variants side by side, since the updated main plan treats INC as one of several frameworks under test.

## Planned outputs

- A simple way (CLI flags or wrapper script) to:
  - Run sanity checks on the baseline model.
  - Run sanity checks on each quantized variant (per profile).
- Saved outputs in variant-specific files/directories (e.g., `tmp/qwen2_5_vl_3b_sanity/<profile-name>/`).

## TODOs

- [ ] Job-001-108-001 Update `run_qwen2_5_vl_3b_sanity.py` or add a wrapper to accept a `--model-dir` and `--label` (profile name) for output naming.
- [ ] Job-001-108-002 Define a fixed set of prompts (text-only and image+text) to reuse across all variants.
- [ ] Job-001-108-003 Run sanity checks for baseline and each quantized profile and save outputs under clearly labeled paths.
- [ ] Job-001-108-004 Document how to invoke these sanity runs in the main plan or a small README.

## Notes

- Keep the prompt set small but diverse enough to highlight any obvious regressions (e.g., hallucinations, failure to describe images).
- Because some INC-derived W8A8 profiles are expected to be “bad” on purpose (for sensitivity research), the sanity script should focus on making differences easy to see (qualitatively and with simple scalar metrics) rather than enforcing any hard pass/fail accuracy threshold. This keeps it aligned with the exploratory nature of the main INC plan.
