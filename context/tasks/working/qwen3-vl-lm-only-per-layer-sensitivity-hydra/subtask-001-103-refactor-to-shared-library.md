# Subtask 1.3: Refactor to shared library code

## Scope

Refactor the existing Qwen3/Qwen sensitivity drivers so core logic lives in `src/auto_quantize_model/` and can be reused by:

- The new Hydra runner (Subtask 1.4)
- Existing argparse-based scripts (kept as thin wrappers, if we keep them)

Core logic includes:

- Loading the Qwen3‑VL checkpoint and extracting the language model module for LM-only flows
- Building the captions calibration dataloader
- Running ModelOpt AutoQuant (forward/loss functions, score sizing)
- Building/writing manifests and per-layer sensitivity reports (Markdown + JSON)

This subtask does not require Hydra integration; it prepares the codebase so Hydra can be a thin orchestration layer.

## Planned outputs

- A new shared module (or small set of modules) under `src/auto_quantize_model/` for Qwen LM-only sensitivity:
  - Example: `src/auto_quantize_model/qwen/autoquant_sensitivity.py`
- Updated Qwen3 driver(s) to reuse shared code without behavior drift:
  - `models/qwen3_vl_4b_instruct/helpers/qwen3_vl_4b_autoquant_int8_lm/run_qwen3_vl_4b_autoquant_int8_lm.py`
  - (Optionally) `models/qwen3_vl_4b_instruct/helpers/qwen3_vl_4b_autoquant_all_layers/run_qwen3_vl_4b_autoquant_all_layers.py` for shared report/manifest helpers
- A clean API surface suitable for unit tests (pure helpers separated from GPU work).

## TODOs

- [x] Job-001-103-001 Create a Qwen helper package under `src/auto_quantize_model/`
  - Add `src/auto_quantize_model/qwen/__init__.py`.
  - Decide module layout (single file vs `datasets.py`, `model.py`, `reports.py`).

- [x] Job-001-103-002 Extract captions dataset + dataloader builders
  - Move the `CocoCaptionsDataset` and `build_lm_calib_dataloader` logic out of the Qwen3 script(s).
  - Ensure deterministic ordering and explicit typing.
  - Keep tokenization settings configurable (seq_len, max_samples, padding side).

- [x] Job-001-103-003 Extract LM loss/forward step utilities
  - Provide:
    - A `forward_step` builder compatible with ModelOpt AutoQuant methods (`gradient`, `kl_div`).
    - A loss function for LM-only that uses `lm_head(last_hidden_state)` (matching current Qwen3 INT8 LM script).

- [x] Job-001-103-004 Add a single “run LM AutoQuant sensitivity” entry function
  - Implement a function that:
    - Accepts model path, device, dtype, quantization format(s), AutoQuant settings, and dataset settings.
    - Loads the model, extracts LM, runs AutoQuant, builds the manifest, and returns the manifest + (optionally) state.
  - Keep file I/O separate where possible so we can unit-test core pieces.

- [x] Job-001-103-005 Consolidate manifest + report writing helpers
  - Decide whether to:
    - Move `build_quant_manifest`, `write_layer_sensitivity_md/json` into `src/auto_quantize_model/`, or
    - Keep them in the existing Qwen3 all-layers helper but import from there
  - Goal: eliminate cross-script `sys.path` hacks (e.g., current Qwen3 LM driver importing from the all-layers driver via path injection).

- [x] Job-001-103-006 Update existing scripts to use the shared library
  - Refactor `run_qwen3_vl_4b_autoquant_int8_lm.py` to:
    - Call the shared functions.
    - Preserve current CLI behavior and output filenames for backward compatibility.
  - Ensure the LM-only script no longer depends on `sys.path` insertion to reach shared helpers.

- [x] Job-001-103-007 Add unit tests for pure helpers
  - Add tests under `tests/unit/` for:
    - Mapping/selection logic (from Subtask 1.1 outputs)
    - Output path/layout builder (if introduced)
    - Manifest/report serialization helpers that do not require model weights

## Notes

- Prefer to keep the shared library free of Hydra-specific imports; pass in plain Python values so the runner can adapt configurations cleanly.
