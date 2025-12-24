# Prep Instructions: CUDA 12.8 (`cu128`) Session

This file describes how to work in the current dialog session when using the generic CUDA 12.8 Pixi environment (`cu128`).

- If you want to run Python, **do not** use the system `python`.
- Always run Python via Pixi with the `cu128` environment:

  - `pixi run -e cu128 python ...`

All Python code changes should be checked with:

- `pixi run -e cu128 mypy .`
- `pixi run -e cu128 ruff check .`

For debugging, any temporary scripts, notebooks, logs, or outputs should be saved under a task-specific subdirectory of `tmp/`, for example: `tmp/quantization-debug/` or `tmp/cu128-exp-001/`.

Follow the existing repository guidelines in `AGENTS.md` and prefer Pixi-managed tools and environments over any system-level installations.

If any rules in `AGENTS.md` conflict with this prep document, prefer this prep document for this session.

## GPU / driver assumptions

This `cu128` environment is designed for systems like the RTX 3090 nodes shown by `nvidia-smi`:

- NVIDIA driver provides CUDA runtime 13.0 (or later), which is compatible with a CUDA 12.8 toolkit in the env.
- GPUs are Ampere (e.g., RTX 3090, compute capability 8.6), which are fully supported by CUDA 12.x and the PyTorch/cu128 wheels used here.

If you move this project to different hardware, ensure:

- `nvidia-smi` reports a recent enough driver (>= the CUDA runtime version the environment expects).
- The GPU architecture is supported by CUDA 12.x.

## Know your tools in the `cu128` env

The `cu128` Pixi environment is a generic CUDA 12.8 / PyTorch stack that mirrors the non‑vLLM `rtx5090` software versions but avoids any custom-built wheels:

- **cuda-toolkit 12.8.1** (conda-forge):
  - Provides `nvcc` and CUDA headers/libs suitable for building GPU extensions for Ampere GPUs (e.g., RTX 3090).
- **PyTorch stack**:
  - `torch == 2.9.0`
  - `torchvision == 0.24.0`
  - `torchaudio == 2.9.0`
- **onnx / onnxruntime-gpu**:
  - Uses the standard upstream `onnxruntime-gpu` wheel (no local `.whl` path), suitable for general ONNX + CUDA workflows on RTX 3090.
- **nvidia-modelopt == 0.40.0**:
  - NVIDIA Model Optimizer for PTQ / QAT and export into downstream inference frameworks.

Source code mirrors for some tools live under `extern/` (e.g. `extern/optimum-onnx`, `extern/TensorRT-Model-Optimizer`, `extern/neural-compressor`, `extern/vllm`), but the **runtime libraries** used in this env come from the Pixi-managed Python packages above. Prefer importing from the installed packages and treat `extern/` as read-only reference code unless explicitly modifying vendored sources.

