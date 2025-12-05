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
