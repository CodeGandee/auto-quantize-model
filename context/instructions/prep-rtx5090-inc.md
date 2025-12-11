# Prep Instructions: RTX 5090 INC Session

This file describes how to work in the current dialog session when using the **Intel Neural Compressor–focused** Pixi environment.

- If you want to run Python, **do not** use the system `python`.
- Always run Python via Pixi with the RTX 5090 INC environment:

  - `pixi run -e rtx5090-inc python ...`

All Python code changes that rely on INC or ONNX PTQ flows should be checked with:

- `pixi run -e rtx5090-inc mypy .`
- `pixi run -e rtx5090-inc ruff check .`

For debugging, any temporary scripts, notebooks, logs, or outputs should be saved under a task-specific subdirectory of `tmp/`, for example: `tmp/inc-quantization-debug/` or `tmp/rtx5090-inc-exp-001/`.

Follow the existing repository guidelines in `AGENTS.md` and prefer Pixi-managed tools and environments over any system-level installations.

If any rules in `AGENTS.md` conflict with this prep document, prefer this prep document for this session.

## Know your tools in the RTX 5090 INC env

The `rtx5090-inc` Pixi environment is built on top of the base `rtx5090` stack and adds:

- **Intel Neural Compressor** (`neural-compressor`, `neural-compressor-pt`):
  - Used by scripts like `scripts/qwen/inc_qwen2_5_vl_3b_sensitivity.py` and the helpers in `src/auto_quantize_model/inc_pytorch_mse_patching.py`.
- **ONNX tooling for INC**:
  - `onnxruntime-tools`, `onnxscript`, and `optimum-onnx[onnxruntime]` for ONNX export and INC-driven sensitivity analysis.

Use this env whenever you are:

- Running INC-based PTQ, op-level MSE sensitivity, or ONNX Runtime INC flows.
- Working with the `extern/neural-compressor` reference code or the `context/summaries/inc-kb` guides.

