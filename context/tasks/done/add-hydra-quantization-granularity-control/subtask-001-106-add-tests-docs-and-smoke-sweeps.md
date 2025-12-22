# Subtask 1.6: Add unit tests, docs, and smoke multirun examples

## Scope

Finish the feature with the minimum validation and documentation needed for reliable grid-search usage:
- Unit tests for overlay and naming logic (no GPU dependency).
- Documentation updates with example commands for sweeping granularity.
- A tiny smoke sweep in the `rtx5090-vllm` Pixi environment to validate that ModelOpt accepts the overridden configs and outputs don’t collide.

In scope:
- Add unit tests under `tests/unit/` for:
  - axis/block_sizes exclusivity behavior
  - block_sizes key normalization
  - deterministic naming/path building from `(quant_pair, quant_granularity, dataset.size)`
- Update `models/qwen3_vl_4b_instruct/layer-analysis/README.md` with example multirun sweeps using `quant_granularity`.
- Run a tiny sweep (small dataset size, small score_size) using:
  - `pixi run -e rtx5090-vllm python scripts/qwen/qwen3_lm_sensitivity.py -m ...`

Out of scope:
- Long-running accuracy evaluations; this is a “wiring + collision check”.

## Planned outputs

- New unit tests in `tests/unit/` (fast, deterministic).
- Updated `models/qwen3_vl_4b_instruct/layer-analysis/README.md` with usage examples.
- A recorded smoke-run log path under `tmp/` (not committed) and/or notes in this subtask file.

## TODOs

- [x] Job-001-106-001 Add unit tests for the overlay helper (happy-path and invalid overlays).
- [x] Job-001-106-002 Add unit tests for output naming helpers (tmp + publish) to ensure granularity is included and stable.
- [x] Job-001-106-003 Update `models/qwen3_vl_4b_instruct/layer-analysis/README.md` with:
  - how to select `quant_granularity`
  - at least one example `-m` sweep across multiple granularities
  - a note about the required Pixi env (`rtx5090-vllm`)
- [x] Job-001-106-004 Run a tiny smoke sweep (e.g., 2 quant pairs × 2 granularities × `dataset.size=small`) and confirm:
  - runs complete
  - no artifact collisions
  - manifests record base + overlay metadata

## Notes

- Use the session’s Python env guidance: `context/instructions/prep-rtx5090-vllm.md` (always run Python via `pixi run -e rtx5090-vllm ...`).

## Validation notes

- Unit tests: `pixi run -e rtx5090-vllm pytest` (configured via `pyproject.toml` `testpaths=["tests"]` to avoid collecting vendored `extern/` tests).
- Smoke logs (not committed): `tmp/add-hydra-quant-granularity-smoke/`
  - Single-run: `single_run_gradient.log`
  - Multirun: `multirun.log`
- Smoke multirun outputs (not committed) landed under:
  - `tmp/model-experiments/qwen3_lm_sensitivity/qwen3_vl_4b_instruct/multirun/2025-12-17_14-27-33/`
