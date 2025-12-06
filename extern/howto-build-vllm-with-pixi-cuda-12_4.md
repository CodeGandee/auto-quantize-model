Howto: Build vLLM inside the Pixi CUDA 12.4 environment

## HEADER
- Purpose: Document what is needed to build a CUDA-enabled vLLM wheel inside this repo’s Pixi environment (CUDA 12.4, torch 2.6.0), and how to avoid the common build and OOM pitfalls.
- Status: Draft (build is very close to working; last observed failures are due to heavy CUDA compilation and resource limits, not missing toolchain pieces).
- Date: 2025-12-06
- Owner: AI assistant (Codex CLI)
- References:
  - vLLM repo: https://github.com/vllm-project/vllm
  - vLLM installation docs (GPU): https://docs.vllm.ai/en/stable/getting_started/installation/gpu/
  - NVIDIA TensorRT Model Optimizer “Unified HuggingFace Checkpoint”: https://nvidia.github.io/TensorRT-Model-Optimizer/deployment/3_unified_hf.html
  - CUDA toolkit meta-packages on conda-forge: https://anaconda.org/conda-forge/cuda-toolkit
  - NVTX dev packages: https://anaconda.org/conda-forge/cuda-nvtx-dev

## 1. Goal and constraints

We want to build vLLM from source under `extern/vllm` using:

- Python: 3.12 (managed by Pixi)
- PyTorch: `torch==2.6.0+cu124` (from the official CUDA 12.4 wheel index)
- CUDA: 12.4 toolchain managed fully inside Pixi (no reliance on `/usr/local/cuda` for `nvcc`)
- Target vLLM: 0.8.5 (highest version that works reliably with `torch==2.6.0` and supports NVIDIA ModelOpt FP8 unified checkpoints)

This is needed to serve NVIDIA ModelOpt-quantized FP8 models with `LLM(..., quantization="modelopt")` from inside this repo’s Pixi env.

## 2. Required Pixi dependencies and env variables

In `pyproject.toml` we need, under `[tool.pixi.dependencies]`:

```toml
[tool.pixi.dependencies]
python = "3.12.*"
cuda-toolkit = { version = "12.4.0.*", channel = "conda-forge" }
cuda-nvtx-dev = ">=12.4.99,<13"
```

Key points:

- `cuda-toolkit=12.4.0.*` from `conda-forge` provides:
  - `nvcc` and core CUDA headers under `${CONDA_PREFIX}`.
  - CUDA driver/runtime libs under `${CONDA_PREFIX}/lib` and `${CONDA_PREFIX}/targets/x86_64-linux/lib`.
- `cuda-nvtx-dev` provides NVTX development headers and stubs vLLM/Torch expect:
  - NVTX headers:
    - `${CONDA_PREFIX}/targets/x86_64-linux/include/nvToolsExt.h`
    - `${CONDA_PREFIX}/targets/x86_64-linux/include/nvtx3/nvToolsExt.h`
  - NVTX libs:
    - `${CONDA_PREFIX}/lib/libnvToolsExt.so[.1[.0.0]]`

Under `[tool.pixi.activation.env]` we additionally need:

```toml
[tool.pixi.activation.env]
CUDA_HOME = "${CONDA_PREFIX}"
CUDA_PATH = "${CONDA_PREFIX}"
LD_LIBRARY_PATH = "${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"
PATH = "${CONDA_PREFIX}/bin:/opt/tensorrt/bin:${PATH}"
# Optional: tells Torch’s CMake where the Python-level NVTX headers live
NVTOOLSEXT_PATH = "${CONDA_PREFIX}/lib/python3.12/site-packages/nvidia/nvtx"
```

After editing `pyproject.toml`:

```bash
pixi install
pixi run nvcc --version
```

You should see CUDA 12.4 (e.g., `Cuda compilation tools, release 12.4, V12.4.99`).

## 3. vLLM version and build script settings

We’ve observed that:

- vLLM ≥ 0.9.0 expects PyTorch 2.7.x and uses APIs (e.g. `at::Float8_e8m0fnu`) not available in `torch==2.6.0`.
- vLLM 0.8.x is commonly paired with `torch==2.6.0` in real deployments and satisfies NVIDIA ModelOpt’s requirement that vLLM must be ≥ 0.6.5 for FP8 unified HF checkpoints.

Therefore, in `extern/build-vllm.sh` we set:

```bash
VLLM_TAG="${VLLM_TAG:-v0.8.5}"
```

This ensures that `pixi run bash extern/build-vllm.sh` checks out `v0.8.5` before building.

## 4. Making vLLM’s CMake cooperate with setup.py

vLLM’s `CMakeLists.txt` requires `VLLM_PYTHON_EXECUTABLE` to identify the Python interpreter and enforce supported Python versions, otherwise it fails with:

> Please set VLLM_PYTHON_EXECUTABLE to the path of the desired python version before running cmake configure.

`setup.py` already passes that as a CMake argument, but CMake can re-run configure and forget the cache. To make this more robust, we patched `extern/vllm/CMakeLists.txt` to accept `VLLM_PYTHON_EXECUTABLE` from the environment:

```cmake
# Supported python versions (in CMakeLists.txt)
set(PYTHON_SUPPORTED_VERSIONS "3.9" "3.10" "3.11" "3.12")

# Try to find python package with an executable that exactly matches
# `VLLM_PYTHON_EXECUTABLE` and is one of the supported versions.
if (NOT VLLM_PYTHON_EXECUTABLE AND DEFINED ENV{VLLM_PYTHON_EXECUTABLE})
  set(VLLM_PYTHON_EXECUTABLE "$ENV{VLLM_PYTHON_EXECUTABLE}")
endif()

if (VLLM_PYTHON_EXECUTABLE)
  find_python_from_executable(${VLLM_PYTHON_EXECUTABLE} "${PYTHON_SUPPORTED_VERSIONS}")
else()
  message(FATAL_ERROR
    "Please set VLLM_PYTHON_EXECUTABLE to the path of the desired python version"
    " before running cmake configure.")
endif()
```

When invoking the build, we then export the variable:

```bash
pixi run bash -lc '
  export VLLM_PYTHON_EXECUTABLE="$CONDA_PREFIX/bin/python"
  bash extern/build-vllm.sh
'
```

This avoids the `VLLM_PYTHON_EXECUTABLE` fatal error during CMake configure.

## 5. NVTX and Torch’s CMake CUDA integration

PyTorch’s CMake integration (`Caffe2/public/cuda.cmake` and its copy of `FindCUDAToolkit.cmake`) expects to be able to create `CUDA::nvToolsExt` and `torch::nvtoolsext` imported targets:

- It searches for:
  - `CUDAToolkit_nvToolsExt_INCLUDE_DIR` (which should contain `nvToolsExt.h`).
  - `CUDA_nvToolsExt_LIBRARY` (which should be `libnvToolsExt.so`).
- With `cuda-toolkit=12.4.0` + `cuda-nvtx-dev>=12.4.99,<13`, and `CUDA_HOME=$CONDA_PREFIX`, these are found under:
  - `${CONDA_PREFIX}/targets/x86_64-linux/include/nvToolsExt.h`
  - `${CONDA_PREFIX}/lib/libnvToolsExt.so*`

If this still fails in a new environment, check:

```bash
pixi run bash -lc '
  echo "CONDA_PREFIX=$CONDA_PREFIX"
  find "$CONDA_PREFIX/targets" -maxdepth 5 -iname "nvToolsExt.h"
  ls "$CONDA_PREFIX"/lib | grep -i nvToolsExt || echo "no nvToolsExt in lib"
'
```

If necessary, you can also point `NVTOOLSEXT_PATH` at the Python NVTX package (as we do in `[tool.pixi.activation.env]`) so `FindCUDAToolkit` has another hint path.

## 6. End-to-end summary

To build vLLM 0.8.5 inside this repo’s Pixi CUDA 12.4 env:

1. Ensure `pyproject.toml` has:
   - `cuda-toolkit = { version = "12.4.0.*", channel = "conda-forge" }`
   - `cuda-nvtx-dev = ">=12.4.99,<13"`
   - `[tool.pixi.activation.env]` sets `CUDA_HOME`, `CUDA_PATH`, `PATH`, `LD_LIBRARY_PATH`, and optionally `NVTOOLSEXT_PATH`.
2. Run:
   ```bash
   pixi install
   ```
3. Confirm CUDA:
   ```bash
   pixi run nvcc --version
   ```
4. Make sure `extern/build-vllm.sh` uses:
   ```bash
   VLLM_TAG="${VLLM_TAG:-v0.8.5}"
   ```
5. Build (optionally with conservative parallelism):
   ```bash
   pixi run bash -lc '
     export VLLM_PYTHON_EXECUTABLE="$CONDA_PREFIX/bin/python"
     bash extern/build-vllm.sh -j 4 --nvcc-threads 1
   '
   ```

If the build still fails, inspect the latest log in `build-vllm/vllm-build-*.log` for:

- NVTX-related CMake issues (missing `CUDA::nvToolsExt`).
- OOM or `ninja` errors (reduce parallelism or increase RAM).
