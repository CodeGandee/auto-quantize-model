# Prep Instructions: RTX 5090 Session

This file describes how to work in the current dialog session for this project.

- If you want to run Python, **do not** use the system `python`.
- Always run Python via Pixi with the RTX 5090 environment:

  - `pixi run -e rtx5090 python ...`

All Python code changes should be checked with:

- `pixi run -e rtx5090 mypy .`
- `pixi run -e rtx5090 ruff check .`

For debugging, any temporary scripts, notebooks, logs, or outputs should be saved under a task-specific subdirectory of `tmp/`, for example: `tmp/quantization-debug/` or `tmp/rtx5090-exp-001/`.

Follow the existing repository guidelines in `AGENTS.md` and prefer Pixi-managed tools and environments over any system-level installations.

If any rules in `AGENTS.md` conflict with this prep document, prefer this prep document for this session.

## ModelOpt per-layer sensitivity runs

Per-layer quantization sensitivity analysis for Qwen2.5‑VL / Qwen3‑VL (FP8 or INT8)
is done with NVIDIA ModelOpt `auto_quantize` drivers under `models/qwen*/helpers/`.
Run them in this env and write outputs under `tmp/` (for example
`tmp/qwen3_vl_4b_autoquant_all_layers_int8_large/`), then compare
`layer-sensitivity-report.{md,json}` across schemes.

## Know your tools in the RTX 5090 env

The `rtx5090` Pixi environment already includes several key packages for ONNX and quantization work:

- **optimum-onnx**:
  - Installed as `optimum-onnx[onnxruntime]>=0.0.3,<0.0.4` via `pyproject.toml`.
  - Use `optimum.exporters.onnx` and `optimum.onnxruntime` for model export / optimization instead of hand-rolling `torch.onnx.export` where possible.
- **onnx / onnxruntime-gpu**:
  - ONNX core + GPU runtime are available for running exported graphs and PTQ flows.
- **neural-compressor**:
  - Present via the `rtx5090` dependency group for INC-based PTQ and sensitivity analysis.

Source code mirrors for some tools live under `extern/` (e.g. `extern/optimum-onnx`, `extern/TensorRT-Model-Optimizer`, `extern/neural-compressor`, `extern/vllm`), but the **runtime libraries** used in this env come from the Pixi-managed Python packages above. Prefer importing from the installed packages and treat `extern/` as read-only reference code unless explicitly modifying vendored sources.
