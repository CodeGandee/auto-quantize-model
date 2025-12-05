Howto: Build vLLM from source with CUDA 12.4

## HEADER
- **Purpose**: Provide a focused recipe for building a recent vLLM (with ModelOpt quantization support) from source on a Linux machine with CUDA 12.4, suitable for running NVIDIA ModelOpt-quantized HF checkpoints.
- **Status**: Draft
- **Date**: 2025-12-05
- **Owner**: AI assistant (Codex CLI)
- **References**:
  - vLLM GPU install guide (CUDA 12.4, source build): https://docs.vllm.ai/en/stable/getting_started/installation/gpu.html
  - vLLM ModelOpt quantization API: https://docs.vllm.ai/en/stable/api/vllm/model_executor/layers/quantization/modelopt.html
  - vLLM source repository: https://github.com/vllm-project/vllm

## 1. When you need to build vLLM from source

You should consider a source build if:

- You need a **newer vLLM** than the one available via your package manager (e.g., you want `quantization="modelopt"` support for ModelOpt HF checkpoints).
- You want to align vLLM with an **existing PyTorch + CUDA 12.4** installation that doesn’t match prebuilt wheels.
- You plan to **modify vLLM C++/CUDA kernels** or stay very close to `main` for latest features/bug fixes.

For CUDA 12.4 on standard NVIDIA GPUs (compute capability ≥ 7.0), vLLM’s official docs recommend:

- Python 3.9–3.12
- A fresh virtualenv/conda env
- PyTorch installed from the official wheel index

## 2. Prepare a clean Python environment (CUDA 12.4)

The vLLM docs strongly recommend a fresh environment to avoid ABI mismatches between vLLM’s compiled kernels and your PyTorch/CUDA stack.

Example using `conda` (just for env management):

```bash
conda create -n vllm-src python=3.12 -y
conda activate vllm-src
```

Install PyTorch with CUDA 12.4 wheels (matching your system’s driver):

```bash
pip install --upgrade pip
pip install "torch==2.6.0+cu124" "torchvision==0.21.0+cu124" --index-url https://download.pytorch.org/whl/cu124
```

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

This should report `2.6.0+cu124` / `12.4` or similar.

## 3. Clone vLLM and choose a version with ModelOpt support

vLLM’s ModelOpt integration (the `modelopt` and `modelopt_fp4` quantization backends) is documented in the vLLM 0.10.1 API:

- `vllm.model_executor.layers.quantization.modelopt` defines `ModelOptFp8Config` and `ModelOptNvFp4Config`.
- `vllm.model_executor.layers.quantization.__init__` maps:

```python
method_to_config = {
    ...
    "modelopt": ModelOptFp8Config,
    "modelopt_fp4": ModelOptNvFp4Config,
    ...
}
```

To build a recent version that is known to support ModelOpt HF checkpoints and is compatible with CUDA 12.4:

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm

# Recommended: use a stable tag with documented ModelOpt support
git fetch --tags
git checkout v0.10.1

# Optional: if you need bleeding-edge features, use main
# (but be prepared for API changes):
# git checkout main
```

If you decide to use a newer tag in the future, verify two things before building:

- `vllm/model_executor/layers/quantization/__init__.py` still contains entries for `"modelopt"` and `"modelopt_fp4"`.
- The docs at https://docs.vllm.ai/en/stable/api/vllm/model_executor/layers/quantization/modelopt/ match your chosen version (or equivalent versioned docs, e.g., `/en/v0.10.1/`).

## 4. Install vLLM from source (editable mode)

From the `vllm/` repo root:

```bash
pip install -e .
```

This will:

- Compile vLLM’s C++/CUDA kernels against your current PyTorch and CUDA (12.4) install.
- Install an editable `vllm` Python package that you can import from anywhere in this environment.

On first build, expect several minutes of compilation. If you rebuild frequently, consider:

- Keeping the same env and repo to reuse ccache/compilation artifacts.
- Using `CCACHE` where applicable (see vLLM docs for build tips).

Verify the install:

```bash
python -c "import vllm; print('vLLM version:', vllm.__version__)"
```

## 5. Basic vLLM usage (no quantization) to sanity-check

Before testing ModelOpt, confirm that vLLM works with a standard model:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3-8B-Instruct",  # example HF model
    dtype="bfloat16",
)

params = SamplingParams(max_tokens=32, temperature=0.7, top_p=0.9)
outputs = llm.generate("Hello from vLLM!", params)
print(outputs[0].outputs[0].text)
```

If this runs without errors, your vLLM + CUDA 12.4 setup is sane.

## 6. Using vLLM with ModelOpt-quantized HF checkpoints

For a ModelOpt HF checkpoint exported with `quant_method = "modelopt"` (e.g., Qwen or Llama FP8/INT8 from NVIDIA’s collection):

- The model directory should contain:
  - `config.json` with `quantization_config` and `quant_method: "modelopt"` or `"modelopt_fp4"`.
  - `hf_quant_config.json` with `quantization.quant_algo` and related settings.
  - `model.safetensors`, tokenizer files, etc.

Example vLLM usage:

```python
from vllm import LLM, SamplingParams

model_path = "/abs/path/to/modelopt_hf_checkpoint"

llm = LLM(
    model=model_path,
    quantization="modelopt",   # or "modelopt_fp4" for FP4 NVFP4 checkpoints
    dtype="bfloat16",
    trust_remote_code=True,    # needed for many NVIDIA / custom HF models
)

params = SamplingParams(max_tokens=64, temperature=0.7, top_p=0.9)
outputs = llm.generate("Write a short haiku about quantization.", params)
print(outputs[0].outputs[0].text)
```

If vLLM complains about mismatched quantization methods:

- Check `config.json["quantization_config"]["quant_method"]` and `hf_quant_config.json["quantization"]["quant_algo"]`.
- Ensure that `quantization="modelopt"` or `"modelopt_fp4"` matches what’s in the HF config.

## 7. Common pitfalls and troubleshooting

- **EngineArgs signature mismatch**:
  - If you see `TypeError: EngineArgs.__init__() got an unexpected keyword argument 'quantization'`, you are using an older vLLM that predates the new quantization API.
  - Fix: make sure you’ve `pip install -e .` from a recent vLLM tag (≥ 0.8.0) and that your environment is not picking up a system vLLM.

- **CUDA / PyTorch ABI issues**:
  - If you change CUDA or PyTorch version, you must rebuild vLLM from source inside a fresh env.
  - Avoid mixing conda-installed PyTorch with pip-built vLLM; use official PyTorch wheels from `https://download.pytorch.org/whl/cu124` instead.

- **Missing Triton**:
  - Some vLLM quantization paths (compressed tensors, certain kernels) require Triton.
  - If you see `ModuleNotFoundError: No module named 'triton'` during import, install Triton via pip or use a prebuilt vLLM wheel that bundles the necessary kernels; for most ModelOpt FP8/NVFP4 cases, the standard GPU build is sufficient.

- **ModelOpt HF checkpoint mismatch**:
  - If vLLM warns that the model’s `quantization_config` doesn’t match the requested quantization backend, double-check the HF config and `hf_quant_config.json` values and adjust your `quantization=` argument accordingly.

## 8. References and further reading

- vLLM GPU installation (CUDA 12.x, build from source): https://docs.vllm.ai/en/stable/getting_started/installation/gpu.html
- vLLM ModelOpt quantization (`modelopt`, `modelopt_fp4`): https://docs.vllm.ai/en/stable/api/vllm/model_executor/layers/quantization/modelopt.html
- NVIDIA ModelOpt HF checkpoints (examples): https://huggingface.co/collections/nvidia/inference-optimized-checkpoints-with-model-optimizer
- vLLM GitHub repo: https://github.com/vllm-project/vllm

## 9. Speeding up vLLM source builds (CUDA 12.4)

Building vLLM from source can easily take several minutes, especially on the first build. The options below come from the vLLM docs and build system in `extern/vllm` (`setup.py`, `vllm/envs.py`, `docs/getting_started/installation/gpu/cuda.inc.md`, `docs/contributing/incremental_build.md`).

### 9.1 Use precompiled kernels when you only change Python

If you are not touching C++/CUDA kernels, you can avoid compiling them entirely and still get an editable install by using precompiled wheels:

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
VLLM_USE_PRECOMPILED=1 uv pip install --editable .
```

This does:

- Locates the base commit for your current branch.
- Downloads the matching pre-built wheel (or a wheel from `VLLM_PRECOMPILED_WHEEL_LOCATION` if set).
- Reuses the wheel’s compiled `.so` libraries inside your editable install.

Good use cases:

- Early experimentation where you only edit Python code.
- Preparing an editable environment before switching to the incremental CMake workflow (Section 9.5).

If you later modify kernels, switch back to a full source build (Section 4) or the incremental workflow; otherwise you will eventually hit import/ABI errors.

### 9.2 Enable ccache / sccache for repeated builds

`extern/vllm/setup.py` automatically wires in `sccache` or `ccache` if they are on `PATH`:

- If `sccache` is found, it sets `CMAKE_*_COMPILER_LAUNCHER=sccache`.
- Else if `ccache` is found, it sets `CMAKE_*_COMPILER_LAUNCHER=ccache`.

Practical recipe:

```bash
# One-time setup
sudo apt-get install -y ccache    # or: conda install ccache

# Verify
which ccache
```

For pip-based editable builds, follow the pattern recommended in the vLLM docs so `ccache` can actually hit:

```bash
CCACHE_NOHASHDIR=true pip install --no-build-isolation -e .
```

Notes:

- `CCACHE_NOHASHDIR=true` prevents the random temporary build dirs created by pip from defeating the cache.
- `sccache` works similarly but can use remote storage; configure via `SCCACHE_BUCKET`, `SCCACHE_REGION`, `SCCACHE_S3_NO_CREDENTIALS`, etc., if you want shared caches.

On large CUDA projects, once the cache is warm, rebuilds typically drop from minutes to seconds.

### 9.3 Tune parallelism with MAX_JOBS and NVCC_THREADS

The vLLM build exposes two installation-time env vars in `vllm/envs.py` and uses them in `setup.py`:

- `MAX_JOBS`: maximum number of compilation jobs (if unset, defaults to detected CPU core count).
- `NVCC_THREADS`: number of threads to use per `nvcc` compilation; when set, the build logic reduces `MAX_JOBS` to avoid oversubscribing the machine.

Example on a 32-core workstation with enough RAM:

```bash
export MAX_JOBS=32        # cap jobs at core count (or slightly below)
export NVCC_THREADS=4     # let nvcc use 4 threads per compilation
pip install -e .
```

Guidelines:

- On strong machines where CPU cores are underutilized by default, setting `MAX_JOBS` explicitly can speed up builds.
- On memory-constrained machines, you may want a *lower* `MAX_JOBS` (for example `MAX_JOBS=6`–`8`) to avoid swapping; faster builds are about keeping all cores busy *without* thrashing memory.
- For CUDA ≥ 11.2, `NVCC_THREADS` controls internal nvcc threading and is also passed via `-DNVCC_THREADS` into CMake, which appends `--threads=<NVCC_THREADS>` to nvcc’s flags.

### 9.4 Restrict CUDA architectures with TORCH_CUDA_ARCH_LIST

By default, PyTorch/vLLM may compile kernels for multiple SM architectures (e.g., 7.5, 8.0, 8.9, 9.0, etc.). Limiting this list to just the architectures you actually run on can significantly reduce compile time and binary size.

1. Find your GPU’s compute capability:

```bash
python -c "import torch; print(torch.cuda.get_device_capability())"
# e.g. (8, 0) for A100, (8, 9) for L40S
```

2. Set `TORCH_CUDA_ARCH_LIST` accordingly before building:

```bash
# A100 (sm_80)
export TORCH_CUDA_ARCH_LIST="8.0"

# L40S (sm_89), example for a single dev GPU
# export TORCH_CUDA_ARCH_LIST="8.9"

pip install -e .
```

Make sure **all** GPUs you plan to serve on are covered; otherwise the resulting wheel may not contain kernels for those devices.

The same principle applies if you build via `uv` or through this repo’s tooling; just ensure `TORCH_CUDA_ARCH_LIST` is exported in the environment before the build command runs.

### 9.5 Use the incremental CMake workflow for kernel development

For frequent C++/CUDA kernel changes (everything under `extern/vllm/csrc/`), using the incremental CMake workflow from `docs/contributing/incremental_build.md` is much faster than repeatedly running `pip install -e .`.

High-level flow:

1. One-time editable install (ideally with precompiled kernels to keep the first step quick):

    ```bash
    uv venv --python 3.12 --seed
    source .venv/bin/activate

    # Python-only editable install that reuses precompiled kernels
    VLLM_USE_PRECOMPILED=1 uv pip install -U -e . --torch-backend=auto

    # Build-time tools: cmake, ninja, etc.
    uv pip install -r requirements/build.txt --torch-backend=auto
    ```

2. Generate CMake presets using vLLM’s helper:

    ```bash
    python tools/generate_cmake_presets.py
    ```

    This script:

    - Detects `nvcc` and your Python executable.
    - Chooses reasonable `NVCC_THREADS` and CMake `jobs` based on CPU cores.
    - Enables `sccache`/`ccache` automatically if available.
    - Emits `CMakeUserPresets.json` (for example with `generator: "Ninja"`).

3. Configure and build once:

    ```bash
    cmake --preset release
    cmake --build --preset release --target install
    ```

After that, kernel edits only require re-running:

```bash
cmake --build --preset release --target install
```

CMake/Ninja will recompile only the affected translation units and reinstall the updated `.so` files into your editable `vllm` tree, making inner-loop kernel development much faster than full re-installs.
