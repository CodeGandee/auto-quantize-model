Howto: Inference over ModelOpt-quantized HF checkpoints

## HEADER
- **Purpose**: Explain how to run inference on NVIDIA ModelOpt-quantized Hugging Face checkpoints (with `quantization_config.quant_method = "modelopt"`), and which runtimes understand this format.
- **Status**: Draft
- **Date**: 2025-12-05
- **Owner**: AI assistant (Codex CLI)
- **Source**:
  - NVIDIA ModelOpt LLM PTQ docs (`extern/TensorRT-Model-Optimizer/examples/llm_ptq/README.md`)
  - vLLM `modelopt` quantization docs (https://docs.vllm.ai/en/stable/api/vllm/model_executor/layers/quantization/modelopt.html)
  - NVIDIA ModelOpt collection models on Hugging Face (e.g. https://huggingface.co/nvidia/Qwen3-8B-FP8, https://huggingface.co/nvidia/Llama-3.1-8B-Instruct-NVFP4)

## 1. What a ModelOpt HF checkpoint looks like

When you run NVIDIA ModelOpt’s HF PTQ flow and call `export_hf_checkpoint`, you get a Hugging Face-style directory with:

- `config.json`:
  - Regular HF config fields: `model_type`, `architectures`, etc.
  - A `quantization_config` block, for example:

```json
"quantization_config": {
  "quant_method": "modelopt",
  "quant_algo": "W8A8_SQ_PER_CHANNEL",
  "kv_cache_scheme": {"type": "float", "num_bits": 8, "dynamic": false},
  "ignore": ["lm_head", "model.visual*"],
  "producer": {"name": "modelopt", "version": "0.39.0"}
}
```

- `hf_quant_config.json`:
  - Additional ModelOpt metadata, for example:

```json
{
  "producer": {"name": "modelopt", "version": "0.39.0"},
  "quantization": {
    "quant_algo": "W8A8_SQ_PER_CHANNEL",
    "kv_cache_quant_algo": "FP8",
    "exclude_modules": ["lm_head", "model.visual*"]
  }
}
```

- `model.safetensors`:
  - Contains mixed dtypes:
    - Many linear weights stored as `torch.int8` (or FP8 / FP4 etc. for other schemes).
    - Scale tensors in `float32`.
    - Some remaining weights in `float16` / `bfloat16`.

Plain `transformers` does not know how to interpret `quant_method = "modelopt"` or the packed low-precision weight layout; you must use a runtime that implements ModelOpt’s quantization kernels.

## 2. Why plain transformers.from_pretrained fails

Symptoms when you try to do:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "/path/to/modelopt-quantized-checkpoint",
    trust_remote_code=True,
    device_map="auto",
)
```

Typical behaviors:

- Transformers warns: `Unknown quantization type, got modelopt - supported types are: [...]`
- If the meta / low-memory loading path is triggered, it will try to construct `nn.Parameter` objects directly from `int8` weight tensors, and you may see:

```text
Error(s) in loading state_dict for Linear:
  While copying the parameter named "weight" ... an exception occurred:
  ('Only Tensors of floating point and complex dtype can require gradients',).
```

Root cause:

- ModelOpt rewrites many `Linear` weights into low-precision (INT8 / FP8 / FP4) plus separate scale tensors.
- Transformers’ generic HF loader does not implement the dequantization logic for `quant_method = "modelopt"`, and assumes all trainable parameters are floating point.
- As a result, you cannot reliably use these checkpoints via plain `AutoModelForCausalLM.from_pretrained` alone.

## 3. Supported inference backends for ModelOpt HF checkpoints

According to NVIDIA’s LLM PTQ README (see `extern/TensorRT-Model-Optimizer/examples/llm_ptq/README.md` and https://nvidia.github.io/TensorRT-Model-Optimizer), unified HF checkpoints with ModelOpt quantization metadata are intended to be deployed on:

- **TensorRT-LLM**:
  - Either via explicit TensorRT-LLM checkpoints exported with `export_tensorrt_llm_checkpoint`.
  - Or via the TensorRT-LLM PyTorch backend, which can understand ModelOpt HF checkpoints.
- **vLLM**:
  - vLLM has a `modelopt` quantization backend that can read `quantization_config` / `hf_quant_config.json` and run the quantized weights directly.
- **SGLang**:
  - Similar to vLLM, SGLang can use ModelOpt-quantized checkpoints with a `modelopt` quantization mode.

These runtimes implement the dequantization math and custom kernels needed to run low-precision weights efficiently.

## 4. Inference with vLLM (recommended pattern)

vLLM supports ModelOpt quantization via the `modelopt` backend. Key points from the vLLM docs:

- The vLLM config loader checks:
  - `config["quantization_config"]` (if present).
  - Or `hf_quant_config.json` inside the model directory.
- If it sees `quant_method = "modelopt"`, it initializes layers using vLLM’s ModelOpt kernels.

### 4.1 vLLM server example

```bash
vllm serve \
  --model /path/to/modelopt-hf-checkpoint \
  --quantization modelopt \
  --trust-remote-code \
  --max-model-len 8192
```

Notes:

- `--model` should point at the HF directory that contains `config.json`, `model.safetensors`, `tokenizer.json`, and `hf_quant_config.json`.
- `--quantization modelopt` tells vLLM to use its ModelOpt backend; if `quantization_config` is present in `config.json`, vLLM will cross-check and may error if methods disagree.

### 4.2 vLLM Python API example

```python
from vllm import LLM, SamplingParams

model_path = "/path/to/modelopt-hf-checkpoint"

llm = LLM(
    model=model_path,
    quantization="modelopt",
    trust_remote_code=True,
)

outputs = llm.generate(
    "Write a short haiku about quantization.",
    SamplingParams(max_tokens=64),
)
print(outputs[0].outputs[0].text)
```

If the HF checkpoint was exported by ModelOpt’s `export_hf_checkpoint`, and the GPU/driver stack supports the quantization type (e.g., FP8, NVFP4, W8A8 SmoothQuant), this is typically the most straightforward way to run inference.

## 5. Inference with TensorRT-LLM

There are two main pathways described in NVIDIA’s docs:

### 5.1 Export TensorRT-LLM checkpoints and build engines

From a quantized PyTorch model (produced by ModelOpt PTQ/QAT), you can call:

```python
from modelopt.torch.export import export_tensorrt_llm_checkpoint

with torch.inference_mode():
    export_tensorrt_llm_checkpoint(
        model,                  # Quantized PyTorch model
        model_type,             # e.g., "gpt", "llama", "qwen"
        export_dir="/path/to/trtllm_ckpt",
        inference_tensor_parallel=tp,
        inference_pipeline_parallel=pp,
    )
```

Then, use TensorRT-LLM’s `trtllm-build` to build engines from that checkpoint (see TensorRT-LLM docs).

This path is independent of the HF directory; you export a dedicated TRT-LLM checkpoint and build optimized TensorRT engines for deployment.

### 5.2 TensorRT-LLM PyTorch backend with unified HF checkpoints

ModelOpt also supports a “unified HF checkpoint” path for TRT-LLM’s PyTorch backend:

- You keep the ModelOpt HF checkpoint (like the one produced by `export_hf_checkpoint`).
- You run inference via TRT-LLM’s PyTorch runtime, passing `quantization="modelopt"` or equivalent configuration so TRT-LLM understands the quantization layout.

Exact API details are evolving; consult:

- ModelOpt LLM PTQ README: https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/main/examples/llm_ptq/README.md
- TensorRT-LLM docs: https://github.com/NVIDIA/TensorRT-LLM

## 6. Diffusers and NVIDIAModelOptConfig (for non-LLM models)

For diffusion/image models, Hugging Face Diffusers has first-class integration with ModelOpt via `NVIDIAModelOptConfig`:

```python
import torch
from diffusers import AutoModel, NVIDIAModelOptConfig

model_id = "Efficient-Large-Model/Sana_600M_1024px_diffusers"
dtype = torch.bfloat16

quant_cfg = NVIDIAModelOptConfig(
    quant_type="FP8",
    quant_method="modelopt",
)

transformer = AutoModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=quant_cfg,
    torch_dtype=dtype,
)
```

This is a separate but related integration: the runtime is still HF Diffusers, but quantization is handled by ModelOpt’s HF plugins rather than vLLM/TRT-LLM.

## 7. Practical checklist when debugging inference on ModelOpt checkpoints

If inference over a ModelOpt HF checkpoint fails or behaves strangely:

1. **Confirm it is a ModelOpt checkpoint**:
   - `config.json` has a `quantization_config` with `"quant_method": "modelopt"`.
   - `hf_quant_config.json` is present with a `producer.name = "modelopt"`.
2. **Do not use plain transformers for quantized inference**:
   - Avoid `AutoModelForCausalLM.from_pretrained(..., device_map="auto")` directly on the quantized directory; use vLLM, TRT-LLM, SGLang, or HF Diffusers (for vision) instead.
3. **Use a compatible backend**:
   - vLLM: `quantization="modelopt"`, plus `trust_remote_code=True` if needed.
   - TensorRT-LLM: export TRT-LLM checkpoints or use the PyTorch backend with ModelOpt support.
4. **Watch for mismatched quantization metadata**:
   - vLLM and SGLang cross-check `quantization_config` in `config.json` against the CLI/argument `--quantization modelopt_fpX`; if they disagree (e.g., config says `fp8` but runtime says `modelopt_fp4`), they will error.
5. **Verify GPU and driver capabilities**:
   - FP8 / NVFP4 / INT8 acceleration may require specific GPU architectures (e.g., Hopper, Blackwell) and CUDA versions; check ModelOpt and vLLM/TRT-LLM docs for support matrices.

## 8. Useful upstream references

- NVIDIA ModelOpt repository and docs:
  - https://github.com/NVIDIA/TensorRT-Model-Optimizer
  - LLM PTQ README: https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/main/examples/llm_ptq/README.md
  - General quantization guide: https://nvidia.github.io/TensorRT-Model-Optimizer/guides/1_quantization.html
- vLLM ModelOpt quantization:
  - https://docs.vllm.ai/en/stable/api/vllm/model_executor/layers/quantization/modelopt.html
- Example ModelOpt HF checkpoints on Hugging Face:
  - https://huggingface.co/collections/nvidia/inference-optimized-checkpoints-with-model-optimizer
  - https://huggingface.co/nvidia/Qwen3-8B-FP8
  - https://huggingface.co/nvidia/Llama-3.1-8B-Instruct-NVFP4

