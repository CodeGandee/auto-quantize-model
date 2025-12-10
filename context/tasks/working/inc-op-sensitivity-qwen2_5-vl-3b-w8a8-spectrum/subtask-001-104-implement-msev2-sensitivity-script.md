# Subtask 1.4: Implement MSE_V2 sensitivity script for Qwen2.5-VL-3B

## Scope

Create a driver script (e.g., `scripts/qwen/inc_qwen2_5_vl_3b_sensitivity.py`) that uses Intel Neural Compressor’s `mse_v2` tuning strategy to compute per-op/layer sensitivity scores for Qwen2.5-VL-3B. The script should run a sensitivity analysis pass (without necessarily producing a quantized model) and persist a ranked list of ops by MSE impact. It must **force INC’s PyTorch FX adaptor to execute its MSE helpers** at least once (via `calculate_op_sensitivity(...)` and the underlying `get_mse_order_per_fp32` / `get_mse_order_per_int8` functions), even when no quantized configuration meets any accuracy goal.

## Planned outputs

- A working sensitivity script that:
  - Loads the Qwen2.5-VL model via the chosen INC path.
  - Builds a `PostTrainingQuantConfig` with `strategy="mse_v2"`, small but configurable `confidence_batches`, and a loose or trivial `AccuracyCriterion` so the sensitivity computation can run independently of PTQ “success”.
  - Applies `op_type_dict` / `op_name_dict` to keep fragile ops in higher precision.
  - Uses `calculate_op_sensitivity(...)` (via the PyTorch FX adaptor) together with a **runtime monkeypatch** of INC’s internal MSE helpers to compute and capture per-op MSE scores, treating INC as a layer-wise scoring engine rather than as the owner of the final quantized model.
- A saved artifact (JSON/Markdown) containing:
  - Ordered list of ops and their MSE scores.
  - Any relevant metadata (e.g., which ops were excluded from quantization).

## TODOs

- [x] Job-001-104-001 Design the `PostTrainingQuantConfig` for Qwen2.5-VL with `strategy="mse_v2"` and appropriate `confidence_batches`.
- [x] Job-001-104-002 Implement the driver script to load the model, run sensitivity analysis, and log/store the ranked op list, using a **runtime monkeypatch** to tap into INC’s internal MSE computations.
- [x] Job-001-104-003 Ensure the script can run within reasonable time on the RTX 5090 environment (adjust sample counts or confidence batches if needed).
- [x] Job-001-104-004 Save the sensitivity results to a stable path (e.g., `context/summaries/inc-kb/qwen2_5_vl_3b_mse_sensitivity.json`) for later use.

## Notes

- Consider printing both human-readable and machine-readable outputs so that developers can inspect sensitivity quickly while scripts can consume the data programmatically.
- For this task we explicitly **target the current INC 2.6 PyTorch FX backend** and are willing to use internal APIs via monkeypatching (without editing the vendored `extern/neural-compressor` sources) to extract per-op MSE values:
  - The per-op sensitivity metrics for PyTorch FX are computed inside:
    - `neural_compressor.adaptor.torch_utils.util.get_mse_order_per_fp32(...)`
    - `neural_compressor.adaptor.torch_utils.util.get_mse_order_per_int8(...)`
  - These functions build an internal `fallback_order: Dict[(op_name, op_type) -> mse_val]` and then return only a sorted list of keys to the strategy.
  - The `MSE_V2TuneStrategy` (`neural_compressor.strategy.mse_v2.MSE_V2TuneStrategy`) calls `adaptor.calculate_op_sensitivity(...)` repeatedly and only consumes the ordered op list; it never exposes the raw MSE values.
- Implementation plan for monkeypatching (high-level):
  - In the sensitivity driver script (e.g., `scripts/qwen/inc_qwen2_5_vl_3b_sensitivity.py` or a variant), **before** calling any INC APIs that may trigger sensitivity computation:
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
  - Prefer calling the adaptor’s `calculate_op_sensitivity(...)` **directly** (on a constructed `tune_cfg` and the shared calibration dataloader) to force a single, bounded sensitivity pass that invokes the patched helpers, even if `quantization.fit` itself never finds an acceptable quantized configuration. If you still use `quantization.fit(...)` with `strategy="mse_v2"`, call `calculate_op_sensitivity(...)` afterward to guarantee that the patched functions run at least once.
  - After the sensitivity pass completes, read the captured MSE maps and write a report under `tmp/qwen2_5_vl_3b_inc/`, for example:
    - `tmp/qwen2_5_vl_3b_inc/op_sensitivity_mse_v2_cpu.md` with a table:
      - Columns: `rank`, `op_name`, `op_type`, `mse_value`.
      - Ordered from **least** sensitive (lowest MSE) to **most** sensitive (highest MSE), mirroring INC’s `get_mse_order_*` sorting.
- This approach keeps the INC installation itself untouched and fully reproducible: sensitivity extraction is done entirely through runtime monkeypatching in our driver script, scoped to this repository and INC version.
- To keep monkeypatching logic reusable and testable, centralize it in a small helper module under `src/auto_quantize_model/`, for example:
  - `src/auto_quantize_model/inc_pytorch_mse_patching.py` exposing:
    - A context manager such as `capture_mse_v2_sensitivity()` that installs and later restores the patched INC functions and returns the captured per-op MSE maps, and a helper such as `run_single_mse_v2_sensitivity_pass(...)` that wraps a one-shot `calculate_op_sensitivity(...)` pass via `MSE_V2TuneStrategy`.
  - The sensitivity driver script should import and use this helper rather than inlining monkeypatch code, so other experiments (e.g., alternate configs or ONNX/PT backends) can reuse the same capture tooling.
- Current implementation status:
  - `src/auto_quantize_model/inc_pytorch_mse_patching.py` implements both the monkeypatch context manager and `run_single_mse_v2_sensitivity_pass(...)`.
  - `scripts/qwen/inc_qwen2_5_vl_3b_sensitivity.py` uses `run_single_mse_v2_sensitivity_pass(...)` on a small calibration set to produce per-op MSE rankings and writes them to:
    - a tmp run directory (e.g., `tmp/qwen2_5_vl_3b_inc/.../op_sensitivity_mse_v2_cpu.{json,md}`), and
    - stable copies under `context/summaries/inc-kb/qwen2_5_vl_3b_mse_sensitivity.{json,md}` for reuse by later subtasks.
