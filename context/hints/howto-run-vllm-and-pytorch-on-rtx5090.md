howto: run vLLM and PyTorch on RTX 5090 (Blackwell, sm_120)
===========================================================

This note captures a **working combination of PyTorch and vLLM for NVIDIA RTX 5090 (Blackwell, compute capability 12.0 / sm_120)** based on early community reports, plus practical setup tips. It is meant for experimental environments only; expect breakage and changes over time.

Summary of recommended versions (early 2025)
-------------------------------------------

- **GPU**: NVIDIA GeForce RTX 5090 (GB202, Blackwell, `sm_120`)
- **Driver**: NVIDIA 575.xx or newer with CUDA 12.8 runtime support
- **CUDA toolkit**: 12.8 (or matching what your driver supports for sm_120)
- **Python**: 3.12
- **PyTorch**: **2.9.0 nightly with CUDA 12.8 (cu128) wheels** for Linux
  - Example: `torch==2.9.0.dev*+cu128` from the official PyTorch nightly index
  - Rationale: stable PyTorch releases up to at least 2.8.x only support up to `sm_90` and will print the familiar warning:
    - “NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible with the current PyTorch installation.”
    - Followed by `RuntimeError: CUDA error: no kernel image is available for execution on the device`
- **vLLM**: **nightly / source build around 0.10.1** (or newer) compiled against the above nightly PyTorch
  - Example from community report:
    - `vllm --version` → `0.10.1rc2.dev413+g5438967fb.d20250901`
  - Built from GitHub `main` with CUDA 12.8 and the nightly PyTorch 2.9 stack

Key references
--------------

- vLLM community thread with a working RTX 5090 setup:
  - https://discuss.vllm.ai/t/vllm-on-rtx5090-working-gpu-setup-with-torch-2-9-0-cu128/1492
- vLLM thread discussing general install guidance for 5090:
  - https://discuss.vllm.ai/t/vllm-install-for-5090/1268
- vLLM Blackwell (RTX 6000) discussion reinforcing the “build from source” recommendation:
  - https://discuss.vllm.ai/t/support-for-rtx-6000-blackwell-96gb-card/1707
- PyTorch discussion about RTX 5090 support:
  - https://discuss.pytorch.org/t/nvidia-geforce-rtx-5090-with-cuda-capability-sm-120-is-not-compatible-with-the-current-pytorch-installation/216518
- PyTorch issue tracking official sm_120 support:
  - https://github.com/pytorch/pytorch/issues/159207

Do I need to build from source?
-------------------------------

Short version from the current ecosystem:

- **PyTorch**: on Linux, you generally do **not** need to build from source if you are willing to use **nightly cu128 binaries** (e.g. `torch==2.9.0.dev*+cu128`) that include `sm_120` support.
- **vLLM**: you should plan to **build from source** for RTX 5090 right now, unless you explicitly trust community-provided wheels that were built against a Blackwell-capable PyTorch.

Details:

- The vLLM RTX5090 guide above describes a successful setup using:
  - PyTorch `2.9.0.dev20250831+cu128` installed from the nightly CUDA 12.8 index (binary wheel).
  - vLLM built from GitHub `main` (around `0.10.1rc2`) with `TORCH_CUDA_ARCH_LIST="12.0"` and other Blackwell-friendly flags, installed via `pip install -e . --no-deps --no-build-isolation`.
  - Motivation: pre-built vLLM wheels at that time depended on stable PyTorch builds that only supported up to `sm_90`, so they could not run on 5090.
- A separate vLLM thread for RTX 6000 Blackwell explicitly recommends:
  - Building vLLM from source and setting `torch_cuda_arch_list="12.0 12.1"` during build to enable SM120 support.
  - Avoiding official Docker images and wheels until they are rebuilt with sm_120-aware toolchains.
- There are reports (e.g. DeepSeek-OCR issues) of users sharing **unofficial vLLM wheels** compiled for RTX 5090, but those are community artifacts, not official vLLM releases; they can work but are less reproducible and harder to keep aligned with your PyTorch version.

In practice, the most robust option today for RTX 5090 is:

- Install a **nightly PyTorch wheel** that advertises CUDA 12.8+ and sm_120 support.
- **Build vLLM from source against that PyTorch**, with `TORCH_CUDA_ARCH_LIST` including `12.0`, and rebuild any other CUDA extension libraries you depend on.

What does **not** work
----------------------

- **Stable PyTorch 2.6–2.8 with cu12x** (e.g. `torch==2.6.0+cu124`, `torch==2.8.0+cu12x`):
  - These builds are compiled only up to `sm_90`.
  - On RTX 5090 they emit warnings and crash on the first CUDA kernel launch:
    - `UserWarning: NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible with the current PyTorch installation.`
    - `RuntimeError: CUDA error: no kernel image is available for execution on the device`
  - vLLM built against these PyTorch wheels will inherit the same limitation.
- **Trying to run vLLM with pre-built wheels compiled against older CUDA/PyTorch**:
  - vLLM wheels built for `sm_90` or lower cannot execute kernels on `sm_120` GPUs.

High-level setup recipe
-----------------------

The concrete commands here are intentionally schematic; adapt versions to what the PyTorch and vLLM projects publish at the time you read this.

1. Install a compatible NVIDIA driver and CUDA
   - Use an NVIDIA driver that supports Blackwell (`sm_120`) and CUDA 12.8+.
   - On Ubuntu, this typically means a 575+ driver series with bundled CUDA runtime.

2. Create a fresh Python 3.12 environment
   - Use your preferred environment manager (conda, pixi, venv, uv, etc.).
   - Example (conda-style):
     ```bash
     conda create -n rtx5090-vllm python=3.12
     conda activate rtx5090-vllm
     ```

3. Install **PyTorch 2.9 nightly cu128** (Linux, sm_120-enabled)
   - Use the official nightly index, **not** the stable one:
     ```bash
     # Example only; check https://pytorch.org/get-started/locally/ for updated URLs
     pip install --pre torch torchvision torchaudio \
       --index-url https://download.pytorch.org/whl/nightly/cu128
     ```
   - Verify that CUDA is visible and that a simple kernel runs:
     ```python
     import torch
     print(torch.__version__)
     print(torch.cuda.get_device_name(0))
     x = torch.ones(1, device="cuda") * 2
     print(x)
     ```

4. Build vLLM from source against this PyTorch
   - Clone vLLM:
     ```bash
     git clone https://github.com/vllm-project/vllm.git
     cd vllm
     git checkout main  # or a recent tag that supports PyTorch 2.9
     ```
   - Set Blackwell-friendly build flags (example from community reports):
     ```bash
     export TORCH_CUDA_ARCH_LIST="12.0"
     export MAX_JOBS=8
     # Optional: ensure sccache/ccache are configured if you use them
     ```
   - Install vLLM (editable is convenient while iterating):
     ```bash
     pip install -e . --no-deps --no-build-isolation
     ```
   - Smoke-test:
     ```bash
     python -c "import vllm; print(vllm.__version__)"
     vllm --version
     ```

5. Run a minimal vLLM example
   - Use a small model first to validate the GPU stack:
     ```python
     from vllm import LLM, SamplingParams

     llm = LLM(model="gpt2", dtype="bfloat16")  # or another small HF model
     sampling_params = SamplingParams(temperature=0.7, max_tokens=32)
     outputs = llm.generate(["Hello from RTX 5090!"], sampling_params)
     print(outputs[0].outputs[0].text)
     ```

6. Integrate with your project
   - Once the RTX 5090 setup is validated in a standalone environment, mirror the versions into your project’s env:
     - Pin `torch` to the same 2.9 nightly cu128 build.
     - Build vLLM from the same git revision.
     - Ensure `TORCH_CUDA_ARCH_LIST` includes `12.0` when building any custom CUDA extensions.

Installing TensorRT‑LLM in the `rtx5090` pixi environment
---------------------------------------------------------

For this repository, we use a pixi environment named `rtx5090` that already carries:

- CUDA toolkit 12.8.1 (via `cuda-toolkit` from `conda-forge`).
- PyTorch pinned at `2.9.1+cu128` (nightly cu128 wheels).
- vLLM built from source against that PyTorch.

The recommended way to install TensorRT‑LLM here is:

1. Use the pixi environment

   Either open a shell:

   ```bash
   pixi run -e rtx5090 bash
   ```

   or prefix commands with:

   ```bash
   pixi run -e rtx5090 <command>
   ```

2. Ensure CUDA paths are visible to pip-installed wheels

   Inside the `rtx5090` env, pixi already defines `CUDAToolkit_ROOT`. Make `CUDA_HOME` consistent:

   ```bash
   export CUDA_HOME="${CUDAToolkit_ROOT}"
   ```

3. Install TensorRT‑LLM from NVIDIA’s PyPI index

   With torch 2.9.x cu128 already installed in `rtx5090`, install TensorRT‑LLM via:

   ```bash
   python -m pip install --pre \
     tensorrt_llm \
     --extra-index-url https://pypi.nvidia.com \
     --extra-index-url https://download.pytorch.org/whl/nightly/cu128
   ```

   Notes:

   - `--pre` is recommended because TensorRT‑LLM wheels are often published as pre‑releases.
   - `--extra-index-url https://download.pytorch.org/whl/nightly/cu128` ensures any PyTorch‑related deps are compatible with the cu128 stack we already use.

4. Sanity‑check the install

   Still in `rtx5090`:

   ```bash
   python -c "import tensorrt_llm; print('TensorRT‑LLM version:', tensorrt_llm.__version__)"
   ```

   If this import succeeds, the Python wheel is correctly installed into the pixi environment and can be used by your scripts or by ModelOpt / vLLM integration that expects TensorRT‑LLM to be importable.

Notes on stability and future updates
-------------------------------------

- The combination above is **not a long-term guarantee**; it reflects what early adopters reported working:
  - Stable PyTorch: no official sm_120 support yet at the time of the cited threads.
  - Nightly PyTorch 2.9 cu128 + vLLM main: works for at least one documented RTX 5090 setup.
- As soon as a **stable PyTorch release with sm_120 support** appears (tracked in the PyTorch issue above), prefer:
  - That stable version (e.g. “PyTorch 2.x with CUDA 13, sm_120 support”), plus
  - A matching vLLM release that officially supports it (likely vLLM ≥ 0.10+ with updated docs).

If you see the `no kernel image is available for execution on the device` error on RTX 5090, the most common fix is:

- Upgrade to a PyTorch build compiled with sm_120 support (nightly or future stable), then rebuild any CUDA-dependent libraries (vLLM, custom ops) against that toolchain.
