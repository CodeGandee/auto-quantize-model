# Requirements: Revise Qwen3-VL-8B Tutorial Pack (Introduce + Layer Sensitivity)

## Status

- **Status:** Draft / Open
- **Owner:** huangzhe + assistants
- **Date:** 2026-01-21
- **Scope:** `docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/`

## Goal

Revise the Qwen3-VL-8B tutorial pack so readers can run it in the **default Pixi env** end-to-end,
understand what happens under the hood, and not be confused by the current LM-only output behavior.

## Background / Problem Statement

The tutorial pack currently snapshots real run outputs, but the **LM-only INT8** report shows
`0.000e+00` sensitivities due to AutoQuant degenerating to `NONE(effective-bits: 16.0)` candidates.
This differs from the older committed Hydra artifacts (e.g. 2025-12-17) where LM-only sensitivity
values are non-zero.

Tracking issue and analysis:

- `context/issues/known/issue-qwen3-vl-lm-only-tutorial-zero-sensitivity.md`

## Requirements (captured from conversation)

### R1. Default Pixi env only

- All tutorial commands must run via `pixi run ...` in the **default** environment.
- Never use `pip` directly; if a dependency is missing, add it to `pyproject.toml` (PyPI or
  conda-forge) and update the lock.

### R2. Tutorial-pack properties

- Keep the tutorial as a **self-contained tutorial pack**:
  - `inputs/` contains minimal tracked inputs.
  - `expected_report/` contains **sanitized** but **real** run artifacts (not placeholders).
  - `run_demo.sh` runs end-to-end in a gitignored workspace under `tmp/`.
  - `run_demo.sh --snapshot-report` refreshes `expected_report/`.
- The pack must be runnable and verifiable (diff against `expected_report/`).

### R3. Explain what happens “under the hood”

- The tutorial MUST explain what `run_demo.sh` does (step-by-step), not only “run this script”.
- The tutorial MUST explain which Python driver(s) are invoked and what each stage produces.

### R4. Explain the “4B helper script” naming

- The tutorial must explicitly explain why it calls helpers under
  `models/qwen3_vl_4b_instruct/helpers/...` even for 8B (the drivers are parameterized by
  `--model-dir`, but the filenames/log strings still include “4b”).

### R5. LM-only result must not confuse readers

The tutorial must address the LM-only “all zero sensitivities” behavior so that readers do not
assume the tutorial is broken.

At minimum, include:

- A clear explanation of why the LM-only tutorial output can collapse to `NONE`-only and therefore
  yield zero sensitivity values.
- An explicit “how to get meaningful LM-only sensitivity values” path (e.g., Hydra runner with a
  non-trivial dataset size), with commands and expected artifacts.

Preferred outcome (if feasible without breaking portability/CI expectations):

- Make the tutorial-pack LM-only run produce **non-zero** sensitivity values in its
  `expected_report/` by adjusting the LM-only configuration (quant format + score/calib budget) while
  keeping runtime reasonable.

### R6. Use specific quant-pair examples

For sensitivity analysis examples in the revised tutorial, use these quant pairs as the primary
worked examples:

- `wint4_afp16`
- `wint4_aint8`

### R7. Ensure non-zero sensitivities for both modes

Choose calibration and AutoQuant settings (dataset size / seq len / batch size / score size, and any
required quant-format enablement) so that the tutorial’s expected outputs contain **non-zero**
sensitivity values for:

- **all-layers** sensitivity runs, and
- **LM-only** sensitivity runs.

In particular:

- Set `scheme.auto_quantize_score_size = 128` for the runs documented by this tutorial.
- Ensure the run has enough calibration batches/samples for scoring (otherwise the effective
  `num_score_steps` will be capped by the available data).

### R8. Dataset sizing guidance

- The tutorial pack should use the **medium** dataset variants by default (so that the documented
  results are meaningful and stable).
- The tutorial must explicitly call out:
  - Use **small** for quick smoke tests / iteration.
  - Use **large** for real applications (most stable sensitivity rankings).

### R9. Use full dataset subset (no partial sampling)

- For the tutorial runs, set `dataset.max_calib_samples` so that **all samples** in the selected
  dataset subset are used (i.e., for `medium`, use all 128; for `small`, all 16; for `large`, all
  512).
- Ensure the generated manifests record:
  - `dataset.num_calib_samples` = number of samples actually used
  - `dataset.max_calib_samples` = maximum allowed samples
  and that these match the full subset size for the tutorial pack.

### R10. Calibration shape + batching (target settings)

Use these dataset calibration settings in the tutorial runs:

- `dataset.calib_seq_len = 512`
- `dataset.batch_size = 8`
- `dataset.num_calib_batches = 16`

Note: `num_calib_batches=16` with `batch_size=8` implies `dataset.num_calib_samples=128` (i.e., the
`medium` subset).

## Non-goals

- Do not commit model weights or environment-specific symlinks.
- Do not require a specific GPU SKU (e.g., RTX5090-only docs); the tutorial should prefer the
  default env and work anywhere CUDA is available.

## Acceptance Criteria

- `bash docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/run_demo.sh` succeeds
  in the default Pixi env and verifies against `expected_report/`.
- The tutorial README explains:
  - model introduction (checkpoint symlink),
  - workspace + input generation,
  - the invoked drivers and produced artifacts,
  - the “4B helper script” naming,
  - the LM-only sensitivity caveat + the recommended way to obtain meaningful values.
- `expected_report/` contains real, sanitized outputs for the documented runs.

## References

- Tutorial pack: `docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/`
- Known issue: `context/issues/known/issue-qwen3-vl-lm-only-tutorial-zero-sensitivity.md`
- Historical non-zero LM-only artifacts:
  - `models/qwen3_vl_4b_instruct/layer-analysis/lm-only/2025-12-17_05-10-42/wint8_aint8/per-layer-sensitivity.md`
