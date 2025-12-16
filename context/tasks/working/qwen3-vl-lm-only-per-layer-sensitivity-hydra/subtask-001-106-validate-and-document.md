# Subtask 1.6: Validate, publish artifacts, add tests and docs

## Scope

Close the loop on the Hydra migration by validating correctness and reproducibility:

- Regenerate the existing Qwen3 INT8 LM-only sensitivity artifacts via Hydra (parity check).
- Run at least one new (W,A) pair sensitivity analysis and publish artifacts under `models/qwen3_vl_4b_instruct/layer-analysis/`.
- Add unit tests for non-GPU logic introduced during refactors.
- Update documentation so future runs follow the Hydra workflow.

## Planned outputs

- Parity-confirmed Hydra run for INT8 LM-only sensitivity (small/medium/large) producing the same artifact set as existing folders.
- At least one new analysis folder committed, e.g.:
  - `models/qwen3_vl_4b_instruct/layer-analysis/weight-fp4-act-fp8/` (or the canonical naming chosen in Subtask 1.1)
- Updated docs:
  - `models/qwen3_vl_4b_instruct/layer-analysis/README.md`
  - Any per-analysis README(s) as needed
- Unit tests under `tests/unit/` covering pure helpers (no model weights / no GPU required).

## TODOs

- [x] Job-001-106-001 Run Hydra parity check for existing INT8 LM-only sensitivity
  - Run Hydra for `quant_pair=wint8_...` across `dataset.size=small,medium,large`.
  - Compare produced artifacts against the committed baseline in `models/qwen3_vl_4b_instruct/layer-analysis/weight-int8-act-int8/`:
    - Presence of manifest + Markdown + JSON
    - No obvious schema drift (key names)

- [x] Job-001-106-002 Run at least one new precision pair and publish artifacts
  - Pick one new pair (based on Subtask 1.1 feasibility), for example:
    - `W=FP4, A=FP8` (hybrid W4A8)
    - or `W=FP8, A=FP16` (weight-only)
  - Publish results under the agreed folder structure.

- [x] Job-001-106-003 Ensure `.pt` artifacts are not committed
  - Confirm `.pt` outputs remain ignored (existing `models/qwen3_vl_4b_instruct/.gitignore` and layer-analysis rules).
  - Ensure Markdown/JSON are tracked for reproducibility.

- [x] Job-001-106-004 Update the layer-analysis documentation
  - Update `models/qwen3_vl_4b_instruct/layer-analysis/README.md` to include:
    - The new (W,A) pair folders
    - The Hydra commands to regenerate (single run + sweep)
    - Any environment prerequisites (Pixi env)

- [x] Job-001-106-005 Add/expand unit tests
  - Add tests for:
    - Precision-pair mapping resolution (Subtask 1.1)
    - Output directory naming/layout logic
    - Report/manifest writer helpers that are pure or file-based without model execution

- [x] Job-001-106-006 Run the repo test/lint loop for changed code
  - Run (as applicable):
    - `pixi run pytest`
    - `pixi run ruff check .`
    - `pixi run mypy .`

## Notes

- If GPU runs are too expensive for CI, treat artifact regeneration as a documented manual step but keep unit tests focused on the pure-Python logic.

## Status

Completed.

## What was validated / published

- Hydra `report_only` regeneration works for the existing INT8 LM-only manifests under `models/qwen3_vl_4b_instruct/layer-analysis/weight-int8-act-int8/`.
- Hydra INT8 LM-only runs (`quant_pair=wint8_aint8`) were executed for `dataset.size=small,medium,large` (written to `tmp/model-experiments/...`) to validate end-to-end execution.
- A new published analysis folder was generated:
  - `models/qwen3_vl_4b_instruct/layer-analysis/weight-fp8-act-fp16/` (LM-only, `quant_pair=wfp8_afp16`, `dataset.size=small`).

## Notes / caveats

- Qwen3-VL requires a Transformers version that recognizes `model_type: qwen3_vl`; use the Pixi env that provides `transformers>=4.57`.
- The legacy committed INT8 “LM-only” manifests include vision-module entries under `.layers` because they were generated from a full-model quantization scope; the Hydra LM-only runner quantizes only the extracted language model by default.
