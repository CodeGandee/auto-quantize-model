# Subtask 1.5: Update tmp/publish output naming for collision-free sweeps

## Scope

Ensure that multi-run grid-searches across `quant_pair` × `quant_granularity` × `dataset.size` do not overwrite artifacts, both in:
- Hydra tmp runs (`hydra.run.dir` and `hydra.sweep.subdir`), and
- “publish” mode (`models/qwen3_vl_4b_instruct/layer-analysis/...`).

In scope:
- Update Hydra directory templates to include `${quant_granularity.name}`.
- Update publish path naming helpers (likely `src/auto_quantize_model/experiment_layout.py`) to include granularity in output directories.
- Ensure any per-run filenames that depend on scheme name remain unique when granularity differs (or ensure directories differ).

Out of scope:
- Writing docs and examples (Subtask 1.6).

## Planned outputs

- Updated `conf/preset/qwen3_lm_sensitivity.yaml` run/sweep directories include quant granularity.
- Updated publish output directory naming includes quant granularity (new folder layer or suffix).
- Confirmed non-colliding path layout for `hydra -m` sweeps.

## TODOs

- [ ] Job-001-105-001 Update `conf/preset/qwen3_lm_sensitivity.yaml`:
  - `hydra.run.dir` includes `${quant_granularity.name}`.
  - `hydra.sweep.subdir` includes `${quant_granularity.name}`.
- [ ] Job-001-105-002 Update publish layout naming in `src/auto_quantize_model/experiment_layout.py` to incorporate granularity:
  - Decide between a new directory layer (preferred) vs suffixes.
  - Keep existing `weight-<w>-act-<a>` organization intact.
- [ ] Job-001-105-003 Ensure `scripts/qwen/qwen3_lm_sensitivity.py` uses the updated layout helper API in publish mode.
- [ ] Job-001-105-004 Validate the resulting paths are stable and readable for large sweeps (avoid deeply nested, overly verbose names).

## Notes

- Output naming is part of the user-facing API; treat it as stable once merged.
- Prefer path semantics that remain meaningful when browsing results manually (weight/act grouping first, then granularity, then dataset size).

