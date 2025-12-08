# Subtask 1.4: Implement MSE_V2 sensitivity script for Qwen2.5-VL-3B

## Scope

Create a driver script (e.g., `scripts/qwen/inc_qwen2_5_vl_3b_sensitivity.py`) that uses Intel Neural Compressor’s `mse_v2` tuning strategy to compute per-op/layer sensitivity scores for Qwen2.5-VL-3B. The script should run a sensitivity analysis pass (without necessarily producing a quantized model) and persist a ranked list of ops by MSE impact.

## Planned outputs

- A working sensitivity script that:
  - Loads the Qwen2.5-VL model via the chosen INC path.
  - Builds a `PostTrainingQuantConfig` with `strategy="mse_v2"` and reasonable `confidence_batches`.
  - Applies `op_type_dict` / `op_name_dict` to keep fragile ops in higher precision.
  - Calls `calculate_op_sensitivity` (or uses INC’s fallback flow) to compute per-op MSE scores.
- A saved artifact (JSON/Markdown) containing:
  - Ordered list of ops and their MSE scores.
  - Any relevant metadata (e.g., which ops were excluded from quantization).

## TODOs

- [ ] Job-001-104-001 Design the `PostTrainingQuantConfig` for Qwen2.5-VL with `strategy="mse_v2"` and appropriate `confidence_batches`.
- [ ] Job-001-104-002 Implement the driver script to load the model, run sensitivity analysis, and log/store the ranked op list, using a **runtime monkeypatch** to tap into INC’s internal MSE computations.
- [ ] Job-001-104-003 Ensure the script can run within reasonable time on the RTX 5090 environment (adjust sample counts or confidence batches if needed).
- [ ] Job-001-104-004 Save the sensitivity results to a stable path (e.g., `context/summaries/inc-kb/qwen2_5_vl_3b_mse_sensitivity.json`) for later use.

## Notes

- Consider printing both human-readable and machine-readable outputs so that developers can inspect sensitivity quickly while scripts can consume the data programmatically.
- For this task we explicitly **target the current INC 2.6 PyTorch FX backend** and are willing to use internal APIs via monkeypatching (without editing the vendored `extern/neural-compressor` sources) to extract per-op MSE values:
  - The per-op sensitivity metrics for PyTorch FX are computed inside:
    - `neural_compressor.adaptor.torch_utils.util.get_mse_order_per_fp32(...)`
    - `neural_compressor.adaptor.torch_utils.util.get_mse_order_per_int8(...)`
  - These functions build an internal `fallback_order: Dict[(op_name, op_type) -> mse_val]` and then return only a sorted list of keys to the strategy.
  - The `MSE_V2TuneStrategy` (`neural_compressor.strategy.mse_v2.MSE_V2TuneStrategy`) calls `adaptor.calculate_op_sensitivity(...)` repeatedly and only consumes the ordered op list; it never exposes the raw MSE values.
- Implementation plan for monkeypatching (high-level):
  - In the sensitivity driver script (e.g., `scripts/qwen/inc_qwen2_5_vl_3b_sensitivity.py` or a variant), **before** calling `quantization.fit(...)`:
    - Import the real module from the installed INC package:
      - `import neural_compressor.adaptor.torch_utils.util as nc_util`
    - Save the original functions:
      - `_orig_get_mse_order_per_fp32 = nc_util.get_mse_order_per_fp32`
      - `_orig_get_mse_order_per_int8 = nc_util.get_mse_order_per_int8`
    - Define patched wrappers that:
      - Call the original implementations (copied or delegated) to preserve behavior.
      - Capture the computed `fallback_order` dict into a global or module-level structure (e.g., `nc_util._last_mse_fp32`, `nc_util._last_mse_int8`) or append snapshots per call.
      - Return the same `ordered_ops` list that INC expects so the tuning flow remains unchanged.
    - Assign the monkeypatch:
      - `nc_util.get_mse_order_per_fp32 = patched_get_mse_order_per_fp32`
      - `nc_util.get_mse_order_per_int8 = patched_get_mse_order_per_int8`
  - Run `quantization.fit(...)` with `strategy="mse_v2"` on CPU+PyTorch FX as in Subtask 1.3/1.2; during tuning INC will call the patched functions and populate the captured per-op MSE maps.
  - After tuning completes (or even after the first sensitivity phase), read the captured `fallback_order` and write a Markdown report under `tmp/qwen2_5_vl_3b_inc/`, for example:
    - `tmp/qwen2_5_vl_3b_inc/op_sensitivity_mse_v2_cpu.md` with a table:
      - Columns: `rank`, `op_name`, `op_type`, `mse_value`.
      - Ordered from **least** sensitive (lowest MSE) to **most** sensitive (highest MSE), mirroring INC’s `get_mse_order_*` sorting.
- This approach keeps the INC installation itself untouched and fully reproducible: sensitivity extraction is done entirely through runtime monkeypatching in our driver script, scoped to this repository and INC version.
- To keep monkeypatching logic reusable and testable, centralize it in a small helper module under `src/auto_quantize_model/`, for example:
  - `src/auto_quantize_model/inc_pytorch_mse_patching.py` exposing:
    - A context manager such as `capture_mse_v2_sensitivity()` that installs and later restores the patched INC functions and returns the captured per-op MSE maps.
  - The sensitivity driver script should import and use this helper rather than inlining monkeypatch code, so other experiments (e.g., alternate configs or ONNX/PT backends) can reuse the same capture tooling.
