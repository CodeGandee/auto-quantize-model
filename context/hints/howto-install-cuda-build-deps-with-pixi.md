Howto: Install CUDA build dependencies in a Pixi environment

## HEADER
- **Purpose**: Show how to use Pixi (with conda-forge / NVIDIA channels and PyPI) to install CUDA build dependencies such as CUDA toolkit, cuDNN, cuSPARSELt, and cuDSS so that complex C++/CUDA libraries can be built inside a Pixi-managed environment.
- **Status**: Draft
- **Date**: 2025-12-06
- **Owner**: AI assistant (Codex CLI)
- **References**:
  - Pixi `pyproject.toml` docs: https://prefix-dev.github.io/pixi/dev/python/pyproject_toml/
  - conda-forge `cudatoolkit-dev` feedstock: https://github.com/conda-forge/cudatoolkit-dev-feedstock
  - conda-forge `cudnn` package: https://anaconda.org/conda-forge/cudnn
  - conda-forge `cusparselt` feedstock: https://github.com/conda-forge/cusparselt-feedstock
  - conda-forge `libcudss` / `libcudss-dev` packages: https://anaconda.org/conda-forge/libcudss, https://anaconda.org/conda-forge/libcudss-dev
  - NVIDIA conda packages index (cudnn / cusparselt / cudss): https://anaconda.org/nvidia/repo

## 1. When you want CUDA build deps inside Pixi

You want Pixi-managed CUDA build dependencies when you need to compile C++/CUDA projects fully inside the Pixi environment, without relying on whatever CUDA is installed on the host OS. Typical cases:

- Building complex C++/CUDA libraries that call `nvcc` and link against cuDNN / cuSPARSELt / cuDSS.
- Running CMake-based builds that expect `cudnn.h`, `cusparseLt.h`, and `cudss.h` to be discoverable under the active environment’s prefix (`$CONDA_PREFIX` inside Pixi).
- Sharing a reproducible build recipe that works on other machines where there may be no system-wide CUDA toolkit installed.

Important caveat for PyTorch-based builds: PyTorch wheels (e.g., `torch==2.7.1+cu12x` from the official CUDA wheel index) are compiled with a fixed set of CUDA features (USE_CUDNN, USE_CUSPARSELT, USE_CUDSS, etc.). Adding extra CUDA libs into the Pixi env does not retroactively flip those feature flags to 1; you only gain those flags by building PyTorch itself from source against those libraries. The instructions below are still very useful for building *other* C++/CUDA projects, and for PyTorch-from-source builds, but they do not change how a prebuilt PyTorch wheel was configured.

## 2. Ensure Pixi workspace uses conda-forge (and optionally NVIDIA)

Pixi uses the `[tool.pixi.workspace]` section in `pyproject.toml` to control which conda channels are available for dependencies. For CUDA dev packages you typically want at least `conda-forge`, and sometimes `nvidia` if you prefer NVIDIA’s own channel:

```toml
[tool.pixi.workspace]
channels = ["conda-forge", "nvidia"]
platforms = ["linux-64"]
```

This matches the setup in this repo and enables you to add `cudatoolkit-dev`, `cudnn`, `cusparselt`, `libcudss-dev` and friends directly with `pixi add`.

## 3. Add a CUDA toolkit with nvcc to the Pixi env

For many C++/CUDA builds you need more than just runtime libraries; you also need `nvcc` and the CUDA headers. On conda-forge there are two main options:

- `cudatoolkit` – provides the runtime, but traditionally does not bundle `nvcc`.
- `cudatoolkit-dev` – a “developer toolkit” meta-package that downloads and installs a full CUDA toolkit including compiler and headers.

For C++/CUDA builds inside Pixi, `cudatoolkit-dev` is usually the simplest choice:

```bash
pixi add cudatoolkit-dev
```

This will update `pyproject.toml` under `[tool.pixi.dependencies]` and ensure that `nvcc`, CUDA headers, and the core libs live under the Pixi env (e.g. `$CONDA_PREFIX/bin/nvcc`, `$CONDA_PREFIX/include`, `$CONDA_PREFIX/lib`), which CMake can then discover via the usual `find_package(CUDA)` / `find_package(CUDAToolkit)` mechanisms or via environment variables like `CUDA_HOME`.

If you are already relying on a system CUDA install (e.g. `/usr/local/cuda`) and only want cuDNN/cuSPARSELt/cuDSS from conda, you can skip `cudatoolkit-dev` and only add the individual libraries in the next section, but the most reproducible path is to keep everything in the Pixi env when possible.

## 4. Add cuDNN / cuSPARSELt / cuDSS in Pixi

Once CUDA itself is available, you can layer the higher-level libraries on top. On conda-forge and NVIDIA channels you have several relevant packages:

- cuDNN (deep-learning primitives)
  - `conda-forge::cudnn` – runtime cuDNN library (headers + `.so` in the env prefix).
  - NVIDIA’s `nvidia::cudnn` – alternative source for cuDNN, typically aligned with specific CUDA versions.
- cuSPARSELt (structured sparsity)
  - `conda-forge::cusparselt` – cuSPARSELt runtime library, suitable for linking sparse GEMM kernels.
- cuDSS (direct sparse solver)
  - `conda-forge::libcudss` – runtime cuDSS library.
  - `conda-forge::libcudss-dev` – development package providing headers and static libs for building against cuDSS.

In a Pixi workspace with `conda-forge` and `nvidia` channels enabled, you can add these into the environment with:

```bash
# Pure conda-forge sources (recommended when possible)
pixi add cudnn cusparselt libcudss-dev

# Or mix in NVIDIA’s channel explicitly for some components
pixi add conda-forge::cudnn conda-forge::cusparselt conda-forge::libcudss-dev
pixi add nvidia::cudnn nvidia::cusparselt nvidia::cudss  # if you need NVIDIA channel builds
```

After `pixi add`, Pixi writes these packages into `[tool.pixi.dependencies]` and pins exact builds into `pixi.lock`, so all team members get the same CUDA, cuDNN, cuSPARSELt, and cuDSS builds when they run `pixi install`.

## 5. How CMake / complex C++ libraries see these CUDA deps

When you build a C++/CUDA library inside the Pixi env, its CMake configuration sees the environment prefix as the “system root” for dependencies. That means:

- Headers live under `$CONDA_PREFIX/include` (e.g. `cudnn.h`, `cusparseLt.h`, `cudss.h`).
- Libraries live under `$CONDA_PREFIX/lib` or `$CONDA_PREFIX/lib64` (e.g. `libcudnn.so`, `libcusparseLt.so`, `libcudss.so`).

Typical patterns inside a project’s `CMakeLists.txt` look like this:

```cmake
find_package(CUDAToolkit REQUIRED)

# Optionally, if the project ships FindCUDNN.cmake / FindcuDSS.cmake modules:
find_package(CUDNN REQUIRED)
find_package(cuDSS REQUIRED)

include_directories(${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(my_cuda_lib PRIVATE ${CUDAToolkit_LIBRARIES} cudnn cusparseLt cudss)
```

Because Pixi sets `$CONDA_PREFIX` to the environment root and installs all of these libraries into that prefix, standard CMake `find_package` and `find_library` calls typically succeed without extra flags. If a library is not picked up automatically, you can hint CMake by exporting environment variables in `pyproject.toml` under `[tool.pixi.activation.env]`:

```toml
[tool.pixi.activation.env]
CUDA_HOME = "${CONDA_PREFIX}"
CUDA_PATH = "${CONDA_PREFIX}"
LD_LIBRARY_PATH = "${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib64:${LD_LIBRARY_PATH}"
```

Then rebuild after reactivating the Pixi environment.

## 6. PyTorch-specific caveats (USE_CUDNN, USE_CUSPARSELT, USE_CUDSS)

For PyTorch itself and libraries that depend on PyTorch’s CMake scripts, it is important to understand what Pixi can and cannot change:

- If you install `torch` via a prebuilt wheel (`pip install` from the official CUDA index), the wheel has already been compiled with a fixed configuration of `USE_CUDNN`, `USE_CUSPARSELT`, `USE_CUDSS`, `USE_CUFILE`, etc. Adding conda packages like `cudnn` or `cusparselt` into the Pixi env later does not retroactively enable these flags.
- Those flags (and the “Compiling without cuDNN support” warnings) are determined when PyTorch itself is compiled, not when a third-party C++/CUDA extension is built.
- Pixi-managed `cudnn` / `cusparselt` / `libcudss-dev` are still useful for building your own CUDA code or other libraries, and for building **PyTorch from source** inside Pixi, but they won’t change a prebuilt PyTorch wheel’s feature set.

If you need PyTorch with full cuDNN/cuSPARSELt/cuDSS support under Pixi, the recommended pattern is:

1. Use Pixi to install `cudatoolkit-dev`, `cudnn`, `cusparselt`, and `libcudss-dev` into a dedicated “build” environment.
2. Build PyTorch from source inside that Pixi env, following the official build docs and making sure Torch’s CMake picks up the Pixi-managed CUDA libs.
3. Install the resulting wheel into the same Pixi env (or a separate one) and then build your C++/CUDA projects against that source-built PyTorch.

## 7. Quick checklist

When setting up a Pixi environment to build a complex C++/CUDA library:

- [ ] Decide whether you rely on system CUDA or a Pixi-managed CUDA toolkit; for full reproducibility prefer `cudatoolkit-dev` from conda-forge.
- [ ] Add `cudnn`, `cusparselt`, and `libcudss-dev` (and any other needed CUDA libs) via `pixi add` so they appear under `[tool.pixi.dependencies]`.
- [ ] Ensure CMake sees the environment prefix via `CUDA_HOME`, `CUDA_PATH`, and `LD_LIBRARY_PATH` if the project’s CMake files don’t already use `CUDAToolkit_ROOT` or the default prefix search paths.
- [ ] For PyTorch-specific builds, remember that prebuilt wheels keep their original CUDA feature flags; use a source build if you need different `USE_*` settings.

