# Plan: Build vLLM from source with CUDA 12.4 and ModelOpt support

## HEADER
- **Purpose**: Build and validate a recent vLLM version from source (with ModelOpt quantization support) on this CUDA 12.x multi-GPU host, placing the vLLM source tree under `extern/` and preparing it for running ModelOpt-quantized Qwen checkpoints.
- **Status**: Draft
- **Date**: 2025-12-05
- **Dependencies**:
  - `extern/` directory for third-party source trees
  - CUDA 12.x driver and runtime (`nvidia-smi` shows CUDA 12.6)
  - Python 3.12 environment (Pixi default) with PyTorch CUDA 12.4 wheels
  - vLLM GPU install docs: https://docs.vllm.ai/en/stable/getting_started/installation/gpu.html
  - vLLM ModelOpt quantization docs: https://docs.vllm.ai/en/stable/api/vllm/model_executor/layers/quantization/modelopt.html
  - ModelOpt HF checkpoint docs and our local Qwen plan: `context/plans/cancel/plan-quantize-qwen2_5-vl-3b-w8a8-modelopt.md`
- **Target**: Auto-quantize-model developers and AI assistants who need a vLLM runtime that understands ModelOpt-quantized HF checkpoints.

---

## 1. Purpose and Outcome

We want a reproducible, repo-local procedure to build vLLM from source in `extern/vllm` for CUDA 12.4, enabling `quantization="modelopt"` and modern HF model support (e.g., Qwen2.5-VL). The outcome should be:

- A vLLM source checkout under `extern/vllm` at a stable tag (e.g., `v0.10.1`) that is documented to support ModelOpt (`modelopt`, `modelopt_fp4`).
- A working build of vLLM in the Pixi Python environment (or a clearly documented companion env) that can:
  - Run standard FP16/BF16 models for sanity checks.
  - Load and run NVIDIA ModelOpt HF checkpoints that use `quantization_config.quant_method = "modelopt"`.
- Minimal, documented glue so future developers can rebuild or upgrade vLLM without re-discovering CUDA/ABI issues.

Success criteria:

- `python -c "import vllm; print(vllm.__version__)"` reports the chosen tag version from within the target env.
- A small FP16 model (e.g., Llama-3-8B-Instruct) can generate text via vLLM on at least one GPU.
- vLLM recognizes ModelOpt quantization methods (`modelopt`, `modelopt_fp4`) in `vllm.model_executor.layers.quantization` and can be pointed at a ModelOpt HF checkpoint without immediate configuration errors.

---

## 2. Implementation Approach

### 2.1 High-level flow

1. **Decide version and env strategy**
   - Choose a vLLM tag with documented ModelOpt support (e.g., `v0.10.1`) and confirm compatibility with Python 3.12 and CUDA 12.x.
   - Decide whether to build vLLM directly inside the Pixi env or use a dedicated `conda`/`venv` env; document the tradeoffs.
2. **Prepare CUDA toolkit visibility**
   - Verify presence of CUDA toolkit binaries (e.g., `nvcc`) and include the appropriate `/usr/local/cuda-12.x/bin` in `PATH` for builds that require it.
3. **Install matching PyTorch CUDA 12.4 wheels**
   - In the chosen env, install PyTorch and torchvision built for CUDA 12.4 (e.g., via `pip` with the PyTorch cu124 index), matching what we already use in Pixi where possible.
4. **Clone vLLM into `extern/vllm` at the chosen tag**
   - Create `extern/vllm` via `git clone`, fetch tags, and check out `v0.10.1` (or newer) under that directory.
   - Optionally record the exact commit hash in a small text file for reproducibility.
5. **Build and install vLLM from source**
   - From `extern/vllm`, run `pip install -e .` within the target env.
   - Address any build failures by adjusting environment variables (e.g., `CUDA_HOME`, `PATH`) or adding minimal build dependencies.
6. **Sanity-check vLLM with a standard model**
   - Use a small open HF model (e.g., `meta-llama/Llama-3-8B-Instruct` or similar) to test plain FP16/BF16 inference.
   - Confirm multi-GPU visibility if needed (e.g., `tensor_parallel_size > 1`).
7. **Verify ModelOpt quantization integration**
   - Confirm that `vllm.model_executor.layers.quantization.__init__` includes `"modelopt"` and `"modelopt_fp4"` in `QUANTIZATION_METHODS` and `method_to_config`.
   - Dry-run vLLM pointing at a ModelOpt HF checkpoint (e.g., one of NVIDIA’s public `*FP8` models) with `quantization="modelopt"` to confirm basic compatibility.
8. **Integrate usage into this repo**
   - Document how to activate the vLLM env and run inference against our Qwen2.5-VL ModelOpt-quantized checkpoint.
   - Optionally add a small helper script under `scripts/qwen/` that calls vLLM with `quantization="modelopt"` for the local checkpoint.

### 2.2 Sequence diagram (steady-state usage)

```mermaid
sequenceDiagram
    participant Dev as Developer
    participant Env as Python Env (Pixi/conda)
    participant Git as GitHub (vLLM)
    participant CUDA as CUDA Toolkit + Driver
    participant vLLM as vLLM (source build)
    participant Qwen as Qwen2.5-VL ModelOpt HF Checkpoint

    Dev->>Env: activate env (Pixi shell / conda activate)
    Dev->>Git: git clone https://github.com/vllm-project/vllm.git extern/vllm
    Dev->>Git: git checkout v0.10.1
    Dev->>Env: pip install torch+cu124, torchvision+cu124
    Dev->>CUDA: ensure nvcc / CUDA libs visible on PATH/LD_LIBRARY_PATH
    Dev->>vLLM: pip install -e extern/vllm
    vLLM-->>Env: compiled CUDA kernels and Python package installed

    Dev->>vLLM: python -c "import vllm; print(vllm.__version__)"
    vLLM-->>Dev: prints v0.10.1

    Dev->>vLLM: LLM(model=Qwen, quantization=\"modelopt\", ...)
    vLLM->>Qwen: load config.json, hf_quant_config.json, model.safetensors
    vLLM->>CUDA: run int8/FP8 kernels via ModelOpt quantization
    vLLM-->>Dev: generated text/image-text outputs for Qwen2.5-VL
```

---

## 3. Files to Modify or Add

- **extern/**:
  - Add `extern/vllm/` as a Git subdirectory containing the vLLM source tree cloned from GitHub at a specific tag (e.g., `v0.10.1`).
- **context/hints/howto-build-vllm-from-source-cuda124.md**:
  - Ensure this hint stays in sync with the actual version and steps used for building vLLM from source on CUDA 12.4.
- **context/summaries/modelopt-kb/howto-infer-modelopt-quantized-hf-checkpoints.md**:
  - Optionally add a short note that vLLM must be at a recent version (e.g., ≥ 0.8.x, tested with v0.10.1) and built/installed as described in the vLLM build hint.
- **scripts/qwen/** (optional, for convenience):
  - Add a helper script such as `scripts/qwen/run_qwen2_5_vl_3b_vllm_modelopt.py` that uses vLLM to run inference on the local ModelOpt-quantized Qwen2.5-VL HF checkpoint.

---

## 4. TODOs (Implementation Steps)

- [ ] **Confirm CUDA + PyTorch baseline** Verify CUDA 12.x driver, locate toolkit (optional), and decide whether to build vLLM inside Pixi or a dedicated env, installing PyTorch 2.6.0+cu124 / torchvision accordingly.
- [ ] **Clone vLLM under extern/** Create `extern/vllm` via `git clone`, fetch tags, and check out `v0.10.1` (or a documented ModelOpt-capable tag).
- [ ] **Build vLLM from source** Run `pip install -e extern/vllm` in the chosen env, adjusting `PATH`, `CUDA_HOME`, or build dependencies if necessary.
- [ ] **Sanity-check vLLM inference** Use a small FP16/BF16 HF model (e.g., Llama) with vLLM to confirm that inference works on at least one GPU.
- [ ] **Validate ModelOpt integration** Confirm that `modelopt`/`modelopt_fp4` quantization methods are available in vLLM and run a smoke test against a public ModelOpt HF checkpoint.
- [ ] **Wire into Qwen workflow** (Optional) Add or update a helper script to run the ModelOpt-quantized Qwen2.5-VL checkpoint via vLLM and document the command in the Qwen quantization plan.
