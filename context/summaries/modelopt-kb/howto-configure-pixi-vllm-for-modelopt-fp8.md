Howto: Configure Pixi + vLLM for ModelOpt FP8 HF checkpoints

## HEADER
- **Purpose**: Show how to configure a Pixi environment so that vLLM can run NVIDIA ModelOpt–quantized FP8 Hugging Face checkpoints (with `quantization_config.quant_method = "modelopt"`). Based on Qwen2.5‑VL‑3B FP8, but applicable to similar models.
- **Status**: Draft
- **Date**: 2025-12-09
- **Owner**: AI assistant (Codex CLI)
- **Source**:
  - This repo’s `pyproject.toml` (RTX 5090 vLLM env).
  - ModelOpt LLM PTQ example: `extern/TensorRT-Model-Optimizer/examples/llm_ptq/hf_ptq.py` and README.
  - vLLM docs for `modelopt` quantization and `qwen2_5_vl` model executor.
  - xFormers releases for Torch 2.9.1 / CUDA 12.8.

## 1. Scenario and goals

We start from:

- A Hugging Face–style checkpoint produced by NVIDIA ModelOpt PTQ:
  - Exported via `export_hf_checkpoint` from the PTQ example.
  - Stored under `models/<name>/quantized/fp8_*`.
  - Contains:
    - `config.json` with a `quantization_config` block.
    - `hf_quant_config.json` with ModelOpt metadata.
    - `model.safetensors` with low‑precision weights (FP8, etc.).
- A Pixi-managed GPU environment dedicated to vLLM (here: `rtx5090-vllm`).

Goal: configure the Pixi env so that:

- vLLM imports successfully (no missing CUDA/torch deps).
- The ModelOpt FP8 HF checkpoint loads with `quantization="modelopt"`.
- Architectures like Qwen2.5‑VL (which use a ViT vision tower) have their attention backend dependencies satisfied (xFormers).

This document focuses on the environment wiring. For a higher‑level overview of how to *use* ModelOpt checkpoints with vLLM/TRT‑LLM, see:

- `context/summaries/modelopt-kb/howto-infer-modelopt-quantized-hf-checkpoints.md`

## 2. Verify the ModelOpt FP8 HF checkpoint

Before touching the environment, confirm you indeed have a ModelOpt FP8 HF checkpoint.

Example (Qwen2.5‑VL‑3B‑Instruct):

- Directory: `models/qwen2_5_vl_3b_instruct/quantized/fp8_fp8_coco2017`
- Key files:
  - `config.json`:
    - `model_type = "qwen2_5_vl"`
    - `quantization_config.quant_method = "modelopt"`
    - `quantization_config.quant_algo = "FP8"`
    - `quantization_config.kv_cache_scheme` is FP8.
  - `hf_quant_config.json`:

    ```json
    {
      "producer": {"name": "modelopt", "version": "0.33.1"},
      "quantization": {
        "quant_algo": "FP8",
        "kv_cache_quant_algo": "FP8",
        "exclude_modules": ["lm_head", "model.visual*"]
      }
    }
    ```

If those fields are present, vLLM’s `modelopt` backend can interpret this checkpoint.

## 3. Shape your Pixi env for vLLM + ModelOpt

We use a dedicated Pixi feature/env (here called `rtx5090-vllm`) separate from the TensorRT‑LLM stack, so torch / CUDA / quantization deps can evolve independently.

### 3.1. GPU toolchain and Python

In `pyproject.toml`, under `[tool.pixi.feature.rtx5090-vllm.dependencies]`:

```toml
[tool.pixi.feature.rtx5090-vllm.dependencies]
# RTX 5090 vLLM CUDA toolchain pin.
cuda-toolkit = { version = "12.8.1.*", channel = "conda-forge" }
cuda-nvtx-dev = ">=12.4.99,<13"
```

At the workspace level:

```toml
[tool.pixi.dependencies]
python = "3.12.*"
```

This gives a CUDA 12.8 toolchain and Python 3.12 that matches the vLLM / torch wheels we use.

### 3.2. Torch + vLLM + ModelOpt + xFormers

The heart of the vLLM environment is its PyTorch and quantization stack. For our Qwen2.5‑VL‑3B FP8 run, we used Torch 2.9.1 and CUDA 12.8:

```toml
[tool.pixi.feature.rtx5090-vllm.pypi-dependencies]
# PyTorch stack (nightly cu128).
torch = "==2.9.1"
torchvision = "==0.24.1"
torchaudio = "==2.9.1"

onnxruntime-gpu = "*"
cupy-cuda12x = "*"

# ModelOpt runtime; version is allowed to float here so that
# env can resolve a compatible release for vLLM and TRT-LLM.
nvidia-modelopt = "*"

# Required by vLLM's Qwen2.5-VL vision transformer backend.
# Qwen2.5-VL's vLLM implementation falls back to xFormers when
# vllm-flash-attn has a known vision bug.
xformers = ">=0.0.33"

# vLLM itself is installed via a Pixi task, e.g.:
# [tool.pixi.tasks]
# postinstall-vllm-rtx5090 = "python -m pip install --no-deps custom-build/vllm-....whl"
```

And we point the PyPI index to the PyTorch cu128 nightly wheels:

```toml
[tool.pixi.feature.rtx5090-vllm.pypi-options]
index-url = "https://download.pytorch.org/whl/nightly/cu128"
extra-index-urls = ["https://pypi.org/simple"]
index-strategy = "unsafe-best-match"
```

Finally, wire the environment into Pixi:

```toml
[tool.pixi.environments]
rtx5090-vllm = { features = ["rtx5090-vllm"], solve-group = "rtx5090-vllm" }
```

### 3.3. Activation env (CUDA layout)

The vLLM env needs to expose CUDA headers/libs in the layout used by `cuda-toolkit`:

```toml
[tool.pixi.feature.rtx5090-vllm.activation.env]
CUDAToolkit_ROOT = "${CONDA_PREFIX}/targets/x86_64-linux"
CUDA_TOOLKIT_ROOT_DIR = "${CONDA_PREFIX}/targets/x86_64-linux"
CUDA_HOME = "${CONDA_PREFIX}/targets/x86_64-linux"
CUDA_PATH = "${CONDA_PREFIX}/targets/x86_64-linux"
```

This is enough for PyTorch and xFormers wheels that are already compiled for CUDA 12.8; we do **not** build them from source inside Pixi.

### 3.4. Installing and sanity‑checking the env

After editing `pyproject.toml`:

```bash
pixi install

# Sanity checks:
pixi run -e rtx5090-vllm python -c \
  "import torch, vllm, xformers; \
   print('torch', torch.__version__); \
   print('xformers', xformers.__version__)"
```

If this succeeds, you have a usable vLLM + ModelOpt + xFormers stack in the Pixi env.

## 4. Running vLLM on a ModelOpt FP8 checkpoint inside Pixi

With the env configured, you can use the vLLM Python API from within Pixi. For example:

```python
from pathlib import Path

from vllm import LLM, SamplingParams

model_dir = Path(
    "models/qwen2_5_vl_3b_instruct/quantized/fp8_fp8_coco2017"
).resolve()

llm = LLM(
    model=str(model_dir),
    quantization="modelopt",
    trust_remote_code=True,
)

prompts = [
    "Write a short haiku about GPUs and quantization.",
    "Explain FP8 quantization to a senior ML engineer in three sentences.",
]

outputs = llm.generate(prompts, SamplingParams(max_tokens=64, temperature=0.7))

for out in outputs:
    print("Prompt:", repr(out.prompt))
    print("Output:", out.outputs[0].text)
    print("---")
```

Run this via Pixi:

```bash
pixi run -e rtx5090-vllm python your_script.py
```

In this repo, we also added a convenience script that follows the same pattern:

- `scripts/qwen/run_qwen2_5_vl_3b_vllm_fp8.py`
  - Loads the FP8 ModelOpt checkpoint.
  - Runs several text-only prompts.
  - Saves prompt+response pairs under `tmp/qwen2_5_vl_3b_vllm_fp8/`.

## 5. Common pitfalls and how the Pixi config avoids them

### 5.1. `ModuleNotFoundError: No module named 'vllm'`

**Symptom**:

- `pixi run -e rtx5090-vllm python -c "import vllm"` fails.

**Fix**:

- Ensure that vLLM is installed *inside* the Pixi env. In this repo, we use a custom wheel and a Pixi task:

  ```toml
  [tool.pixi.tasks]
  postinstall-vllm-rtx5090 = "python -m pip install --no-deps custom-build/vllm-0.10.2.dev2+g926b2b1d9.d20251208-*.whl"
  ```

- Run that task from Pixi (or equivalent) after `pixi install`.

### 5.2. `ModuleNotFoundError: No module named 'xformers'` in `qwen2_5_vl.py`

**Symptom**:

- vLLM starts to load `Qwen2_5_VLForConditionalGeneration` and then fails with:

  ```text
  ModuleNotFoundError: No module named 'xformers'
  ```

- Logs show a warning like:

  ```text
  Current `vllm-flash-attn` has a bug inside vision module,
  so we use xformers backend instead.
  ```

**Cause**:

- Qwen2.5‑VL’s vLLM backend uses a ViT attention implementation that falls back to xFormers for vision attention when the `vllm-flash-attn` path is disabled.
- If `xformers` is not installed in the environment, you get a runtime import error.

**Fix (what we did in Pixi)**:

- Added `xformers = ">=0.0.33"` to `[tool.pixi.feature.rtx5090-vllm.pypi-dependencies]`.
- Re‑ran `pixi install`.
- Verified with:

  ```bash
  pixi run -e rtx5090-vllm python -c \
    "import xformers; print(xformers.__version__)"
  ```

After that, vLLM can load the Qwen2.5‑VL FP8 checkpoint and run inference.

### 5.3. Flash-Attn build issues inside Pixi

In earlier iterations, we tried to let vLLM use `flash-attn` inside Pixi by adding:

```toml
flash-attn = "*"
```

to the Pixi env. This caused build failures because:

- `flash-attn`’s `setup.py` expects `torch` to be available at build time.
- The Pixi/uv build isolation environment didn’t have torch preinstalled.

**Resolution**:

- Removed `flash-attn` from the Pixi env and relied on xFormers as the vision attention backend, as suggested by vLLM’s own warning.
- This is sufficient for Qwen2.5‑VL in our setup.

If you explicitly need `flash-attn` inside Pixi, follow the hints in the error message (e.g., adding torch to `tool.uv.extra-build-dependencies` or installing `flash-attn` in a non‑isolated context) and ensure the versions are compatible with your torch/CUDA combo.

### 5.4. Torch / vLLM version mismatches

vLLM wheels are tied to specific Torch and CUDA versions. If you see:

- Segfaults in low‑level CUDA kernels.
- `RuntimeError: expected device cuda:0 but got cuda:1` or layout mismatches.

Check that:

- The vLLM wheel you install was built against the same Torch version as your Pixi env (here: 2.9.1+cu128).
- The CUDA version in `cuda-toolkit` matches the wheel’s CUDA ABI (here: 12.8).

In this repo, both are aligned to CUDA 12.8 / Torch 2.9.1 via:

- `cuda-toolkit = "12.8.1.*"`
- `torch == 2.9.1` from the `cu128` nightly index.

## 6. Summary checklist

To make a ModelOpt FP8 HF checkpoint run under vLLM inside a Pixi env:

1. **Confirm the checkpoint is ModelOpt FP8**:
   - `quantization_config.quant_method = "modelopt"`.
   - `quantization_config.quant_algo = "FP8"`.
   - `hf_quant_config.json` present.
2. **Create a dedicated Pixi feature/env for vLLM**:
   - Pin a compatible CUDA toolchain, Python, and torch stack.
3. **Add runtime deps to the Pixi env**:
   - `nvidia-modelopt` (un-pinned).
   - `xformers` (for Qwen2.5‑VL and similar models).
   - vLLM wheel built against the same Torch/CUDA versions.
4. **Configure PyPI indexes**:
   - Use the appropriate PyTorch wheel index (e.g., cu128 nightly).
5. **Sanity check the env**:
   - `import torch, vllm, xformers` succeeds.
6. **Run vLLM with `quantization="modelopt"`**:
   - Point `model=` at the HF directory containing the ModelOpt checkpoint.

Following this pattern, you can reuse the same Pixi/vLLM env to serve multiple ModelOpt‑quantized FP8 models, as long as they are compatible with the chosen Torch/CUDA stack and their vLLM backends’ extra dependencies (like xFormers) are present.

## 7. Troubleshooting log (what actually broke here)

This section records the concrete issues we hit in this repo and how they were resolved, so future changes can be debugged faster.

### 7.1. Pixi solve failure when adding `flash-attn`

- **Error** (during `pixi run -e rtx5090-vllm ...` after adding `flash-attn = "*"`)

  ```text
  × Failed to update PyPI packages for environment 'rtx5090-vllm'
    ...
    Failed to build `flash-attn==2.8.3`
    ...
    ModuleNotFoundError: No module named 'torch'
  ```

- **Cause**:
  - `flash-attn`’s build backend expects `torch` to be importable during build, but Pixi’s isolated build env didn’t contain `torch` yet.
- **Fix**:
  - Removed `flash-attn` entirely from `[tool.pixi.feature.rtx5090-vllm.pypi-dependencies]`.
  - Relied on xFormers as the attention backend instead (see 7.2).

### 7.2. vLLM crash: `ModuleNotFoundError: No module named 'xformers'`

- **Error** (while loading the Qwen2.5‑VL model in vLLM):

  ```text
  WARNING ... Current `vllm-flash-attn` has a bug inside vision module,
  so we use xformers backend instead.
  ...
  ModuleNotFoundError: No module named 'xformers'
  ```

- **Cause**:
  - vLLM’s `qwen2_5_vl` backend uses an xFormers-based vision attention path as a fallback when `vllm-flash-attn` is disabled.
  - `xformers` was not installed in the Pixi env.
- **Fix**:
  - Added `xformers = ">=0.0.33"` under `[tool.pixi.feature.rtx5090-vllm.pypi-dependencies]`.
  - Re-ran `pixi install`.
  - Verified with:

    ```bash
    pixi run -e rtx5090-vllm python -c \
      "import xformers, torch; print('xformers', xformers.__version__, 'torch', torch.__version__)"
    ```

  - After that, `LLM(..., quantization="modelopt")` successfully loaded the FP8 Qwen2.5‑VL checkpoint and generated text.

### 7.3. Engine core dying after successful generation

- **Symptom**:
  - After a short interactive script finishes generating responses, vLLM logs:

    ```text
    ERROR ... Engine core proc EngineCore_0 died unexpectedly, shutting down client.
    ```

  - This happens *after* responses are printed and the Python process exits.
- **Cause**:
  - The main Python process exits, and the engine worker process is torn down. vLLM logs the worker exit as an error even though work completed.
- **Fix / Guidance**:
  - No configuration change was required; this did not affect correctness.
  - For long-running services, prefer the `vllm serve` / server entrypoint instead of short-lived Python scripts, so engine lifecycle is clearly managed.

