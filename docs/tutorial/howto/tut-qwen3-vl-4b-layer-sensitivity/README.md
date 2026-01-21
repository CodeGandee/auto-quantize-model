# Tutorial Pack: Qwen3-VL-4B Layer Sensitivity (End-to-End)

This tutorial pack runs Qwen3-VL-4B-Instruct per-layer sensitivity analysis end-to-end using the repo’s drivers, producing artifacts under a gitignored workspace in `tmp/` and (optionally) snapshotting a small, sanitized “expected report” into this directory for verification.

## What this demonstrates

- All-layers sensitivity (vision + text) using `models/qwen3_vl_4b_instruct/helpers/qwen3_vl_4b_autoquant_all_layers/run_qwen3_vl_4b_autoquant_all_layers.py`
- LM-only sensitivity (text tower) using `models/qwen3_vl_4b_instruct/helpers/qwen3_vl_4b_autoquant_int8_lm/run_qwen3_vl_4b_autoquant_int8_lm.py`
- A minimal, self-contained calibration setup (a tiny synthetic COCO-like root + a 1-row SQLite DB + a 1-line captions file)

The runs are intentionally configured as a “smoke test” (very small sample size and score size) so it’s feasible to execute on a single GPU.

## Prerequisites

- A CUDA-capable GPU (recommended for this tutorial; CPU execution is possible but can be extremely slow)
- Pixi installed
- A local Qwen3-VL-4B-Instruct snapshot linked into the repo:
  - `models/qwen3_vl_4b_instruct/checkpoints/Qwen3-VL-4B-Instruct` must exist and be a directory (usually a symlink)
  - If missing, run: `./models/qwen3_vl_4b_instruct/bootstrap.sh --yes`

## How to run

From the repo root:

```bash
bash docs/tutorial/howto/tut-qwen3-vl-4b-layer-sensitivity/run_demo.sh
```

This writes outputs under a workspace like:

- `tmp/tutorial_workspace_qwen3_vl_4b_layer_sensitivity_<epoch>/`

## Verify against the expected report

If `expected_report/` is present, `run_demo.sh` will compare the generated summaries against it and fail the run if they differ.

The `expected_report/` directory includes both:

- A stable, sanitized `summary.{json,md}` used for verification
- The detailed run artifacts (sanitized for portability): `layer-sensitivity-report.{md,json}`, and `quant_manifest.json` when present

To regenerate the expected report (maintenance mode):

```bash
bash docs/tutorial/howto/tut-qwen3-vl-4b-layer-sensitivity/run_demo.sh --snapshot-report
```

## Notes

- This tutorial uses the Pixi default environment (`pixi run ...`), not any GPU-specific feature env.
- The “expected report” contains only sanitized summaries (no large tensors, no absolute paths), so it is suitable to commit.
