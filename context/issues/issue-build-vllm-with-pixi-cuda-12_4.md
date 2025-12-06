Issue: Build vLLM inside Pixi env with CUDA 12.4 and torch 2.6.0
================================================================

Context
-------

- Goal: Build a GPU-enabled vLLM wheel inside this repo’s Pixi environment so we can run NVIDIA ModelOpt-quantized FP8 models via `quantization="modelopt"`.
- Environment:
  - Python: 3.12 (Pixi, `[tool.pixi.dependencies]`)
  - PyTorch: `torch==2.6.0+cu124` (from the CUDA 12.4 wheel index)
  - GPU: Blackwell-class, `compute_cap=12.0`


What we tried
-------------

1. Initial attempt: `pixi run bash extern/build-vllm.sh`
   - Default tag: `v0.10.1`
   - Failure 1: `nvcc` not found in Pixi env (`.pixi/envs/default/bin/nvcc`).
   - Fix attempt A: added `cudatoolkit-dev >=11.7,<12` via Pixi. This provided some CUDA headers/libs but **no `nvcc`** inside the env, so vLLM’s `setup.py` still failed when probing `nvcc`.

2. Switch to a proper CUDA 12.4 toolchain in Pixi
   - Replaced `cudatoolkit-dev` with a compiler-only package:
     - First tried `cuda-nvcc=12.4.131` (conda-forge). This exposed `nvcc` but not the full CUDA library set expected by PyTorch’s CMake (`FindCUDAToolkit` / `FindCUDA`), so we hit missing `CUDA::cuda_driver` / `CUDA_nvrtc_LIBRARY` / `CUDA::nvToolsExt` targets.
   - Final fix on the CUDA side:
     - Removed `cuda-nvcc`.
     - Added full CUDA toolkit 12.4 from conda-forge:
       - `pyproject.toml` → `[tool.pixi.dependencies]`:
         - `cuda-toolkit = { version = "12.4.0.*", channel = "conda-forge" }`
       - `[tool.pixi.activation.env]` already sets:
         - `CUDA_HOME = "${CONDA_PREFIX}"`
         - `CUDA_PATH = "${CONDA_PREFIX}"`
         - `PATH` includes `${CONDA_PREFIX}/bin`
         - `LD_LIBRARY_PATH` includes `${CONDA_PREFIX}/lib` and `${CONDA_PREFIX}/lib64`
     - Result:
       - `pixi run which nvcc` → `.pixi/envs/default/bin/nvcc`
       - `pixi run nvcc --version` → CUDA 12.4, V12.4.99
     - This gives a *Pixi-managed* CUDA 12.4 toolchain; we no longer depend on `/usr/local/cuda` for `nvcc`.

3. Align vLLM version with `torch==2.6.0`
   - Observations:
     - vLLM `v0.10.1`’s C++ code uses `at::Float8_e8m0fnu` in `gptq_marlin.cu`, which is only available in newer PyTorch builds (2.7.x+).
     - Upstream docs / issues indicate:
       - vLLM ≥ 0.9.0 expects **PyTorch 2.7.x**.
       - vLLM 0.8.x is commonly used with **PyTorch 2.6.0** (e.g. `vllm==0.8.5` + `torch==2.6.0` in several real deployments).
   - Change:
     - `extern/build-vllm.sh`:
       - Default tag: `VLLM_TAG="${VLLM_TAG:-v0.8.5}"` (was `v0.10.1`).
   - Rationale:
     - vLLM 0.8.5 is the highest vLLM version that:
       - Works with `torch==2.6.0`.
       - Satisfies NVIDIA ModelOpt’s requirement of vLLM ≥ 0.6.5 for FP8 unified HF checkpoints.

4. Making vLLM’s CMake tolerate `setup.py`-driven builds
   - vLLM’s `CMakeLists.txt` requires a `VLLM_PYTHON_EXECUTABLE` CMake variable:
     - If unset, it errors with:
       - “Please set VLLM_PYTHON_EXECUTABLE to the path of the desired python version before running cmake configure.”
     - `setup.py` already passes `-DVLLM_PYTHON_EXECUTABLE=<path>` in the command line, but CMake re-runs configure in some cases and can forget the cache.
   - Local patch:
     - In `extern/vllm/CMakeLists.txt`, we added:
       ```cmake
       if (NOT VLLM_PYTHON_EXECUTABLE AND DEFINED ENV{VLLM_PYTHON_EXECUTABLE})
         set(VLLM_PYTHON_EXECUTABLE "$ENV{VLLM_PYTHON_EXECUTABLE}")
       endif()
       ```
       right before the `if (VLLM_PYTHON_EXECUTABLE)` block.
     - Then we call the build with:
       - `VLLM_PYTHON_EXECUTABLE=$CONDA_PREFIX/bin/python pixi run bash extern/build-vllm.sh`
   - Outcome:
     - The “please set VLLM_PYTHON_EXECUTABLE” CMake fatal error is gone.


Remaining build issues
----------------------

1. `CUDA::nvToolsExt` target resolution (earlier failures)
   - With `cuda-nvcc` only or when pointing CMake at `/usr/local/cuda`, PyTorch’s `Caffe2/public/cuda.cmake` and `FindCUDAToolkit.cmake` sometimes failed to create the `CUDA::nvToolsExt` imported target:
     - Symptom:
       - “The link interface of target `torch::nvtoolsext` contains: `CUDA::nvToolsExt` but the target was not found.”
     - Root cause:
       - `FindCUDAToolkit.cmake` only creates `CUDA::nvToolsExt` if it can find both:
         - `nvToolsExt.h` (via `CUDAToolkit_nvToolsExt_INCLUDE_DIR`).
         - `libnvToolsExt.so` (via `CUDA_nvToolsExt_LIBRARY`).
       - In our Pixi env:
         - Libraries exist at:
           - `${CONDA_PREFIX}/lib/libnvToolsExt.so.1.0.0`
           - `${CONDA_PREFIX}/targets/x86_64-linux/lib/libnvToolsExt.so.1.0.0`
         - Headers are **not** under `${CONDA_PREFIX}/include`, but instead at:
           - `${CONDA_PREFIX}/lib/python3.12/site-packages/nvidia/nvtx/include/nvToolsExt.h`
     - Planned fix:
       - Use `NVTOOLSEXT_PATH` to point CMake at the proper include/lib root:
         - `NVTOOLSEXT_PATH = "${CONDA_PREFIX}/lib/python3.12/site-packages/nvidia/nvtx"`
       - `FindCUDAToolkit.cmake` honours `ENV NVTOOLSEXT_PATH` for both `find_path(CUDAToolkit_nvToolsExt_INCLUDE_DIR ...)` and `_CUDAToolkit_find_and_add_import_lib(nvToolsExt ...)`, which should allow it to create `CUDA::nvToolsExt` correctly.
       - This environment override is appropriate to set in `[tool.pixi.activation.env]`.

2. Current status with `cuda-toolkit=12.4.0` and vLLM 0.8.5
   - `nvcc` is available from Pixi (`V12.4.99`) and used via `CMAKE_CUDA_COMPILER=/workspace/.../.pixi/envs/default/bin/nvcc`.
   - vLLM 0.8.5 configure and code generation phases complete (Marlin, FlashAttn, Machete, etc.).
   - The last failing step (as of the latest run) is still a CMake/Torch CUDA integration error rooted in `FindCUDAToolkit` not fully wiring some CUDA components (notably `nvToolsExt`), which we plan to address by exporting `NVTOOLSEXT_PATH` as described above.


Next steps
----------

- Add `NVTOOLSEXT_PATH` to `pyproject.toml` / `[tool.pixi.activation.env]` so that Torch’s CMake CUDA helpers can discover `nvToolsExt` inside the Pixi env.
- Re-run:
  - `pixi install`
  - `pixi run bash extern/build-vllm.sh`
- If further errors arise, inspect:
  - `extern/vllm/build/temp.linux-x86_64-cpython-312/CMakeCache.txt`
  - `build-vllm/vllm-build-*.log`
  - and adjust vLLM’s CMake/Torch configuration accordingly (preferably without heavy patching of upstream CMake files).

