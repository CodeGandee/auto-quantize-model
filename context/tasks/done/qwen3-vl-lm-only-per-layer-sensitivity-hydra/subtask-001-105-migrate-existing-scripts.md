# Subtask 1.5: Migrate existing Qwen3 sensitivity scripts to Hydra

## Scope

Update or wrap existing Qwen3 sensitivity drivers and orchestration scripts so the Hydra runner becomes the primary workflow, while preserving or cleanly deprecating the old entry points.

This subtask focuses on the “outer layers” (scripts, wrappers, READMEs), not the shared core implementation.

## Planned outputs

- A Hydra-based replacement for the 3-phase bash runner (small/medium/large), either by:
  - Updating `scripts/qwen/run_qwen3_vl_4b_int8_sensitivity_3phase.sh` to call Hydra, or
  - Replacing it with a new Hydra-friendly runner script (bash or Python) that delegates to `scripts/qwen/qwen3_lm_sensitivity.py`.
- Updated helper READMEs / pointers so users find the Hydra workflow first.
- Removal of `sys.path` hacks and duplicated logic in legacy scripts (as enabled by Subtask 1.3).

## TODOs

- [x] Job-001-105-001 Decide compatibility policy for existing scripts
  - Keep argparse scripts as supported wrappers (preferred if low cost), OR
  - Mark them deprecated with a pointer to the Hydra runner and keep only `report_only` compatibility.

- [x] Job-001-105-002 Update the 3-phase runner to delegate to Hydra
  - For Qwen3 INT8 LM-only sensitivity:
    - Replace the inner `python ...run_qwen3_vl_4b_autoquant_int8_lm.py` call with `python scripts/qwen/qwen3_lm_sensitivity.py ...`.
  - Ensure small/medium/large can be done via:
    - 3 explicit runs, or
    - Hydra multirun with `dataset.size=small,medium,large`.

- [x] Job-001-105-003 Ensure output layout matches the published analysis structure
  - Decide where the “official” committed artifacts should live:
    - `models/qwen3_vl_4b_instruct/layer-analysis/weight-<w>-act-<a>/...`
  - Ensure wrappers pass the right output dir to Hydra for publishing.

- [x] Job-001-105-004 Update Qwen3 model analysis READMEs to reference Hydra
  - Update:
    - `models/qwen3_vl_4b_instruct/layer-analysis/README.md`
    - `models/qwen3_vl_4b_instruct/layer-analysis/weight-int8-act-int8/README.md`
  - Add “How to regenerate with Hydra” command examples.

- [x] Job-001-105-005 Remove or quarantine legacy script assumptions
  - Eliminate reliance on `sys.path` insertion between helper directories.
  - Ensure any remaining legacy scripts import from `src/auto_quantize_model/` instead of other scripts.

## Notes

- Keep changes minimal for existing published artifacts: changing filenames or directory layout should be done intentionally, with a migration note in the READMEs.

## Status

Completed.

## What changed (high level)

- Kept legacy Qwen3 drivers as supported wrappers, but updated the “official” workflow/docs to point to Hydra first.
- Updated the Qwen3 INT8 3-phase runner to:
  - Publish into `models/qwen3_vl_4b_instruct/layer-analysis/weight-int8-act-int8/` by default.
  - Use Hydra (`scripts/qwen/qwen3_lm_sensitivity.py`) for the LM-only INT8 pass.
  - Support `OUTPUT_MODE=tmp` to keep the previous `tmp/` layout.
- Added a legacy quant-pair config (`quant_pair=wint8_aint8`) with naming overrides so Hydra can write into the existing published INT8 LM-only folders.
- Updated layer-analysis READMEs to include Hydra regeneration commands.
