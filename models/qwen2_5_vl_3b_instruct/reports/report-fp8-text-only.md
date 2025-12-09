# Qwen2.5‑VL‑3B FP8 vLLM Test Summary

## Models and Sizes

| Model                                   | Format                    | Weight size (approx.) |
| --------------------------------------- | ------------------------- | ---------------------- |
| Original FP16/BF16 checkpoint           | HF, FP16/BF16             | 7.1 GiB               |
| FP8 LM-only (ModelOpt, vLLM-compatible) | HF + ModelOpt FP8 (LM+KV) | 4.5 GiB               |

Paths:
- Original: `models/qwen2_5_vl_3b_instruct/checkpoints/Qwen2.5-VL-3B-Instruct` (symlink to `/workspace/llm-models/Qwen2.5-VL-3B-Instruct`)
- FP8 LM-only: `models/qwen2_5_vl_3b_instruct/quantized/fp8_fp8_coco2017`

## Calibration Dataset

- Local calibration text file:
  - `datasets/vlm-quantize-calib/coco2017_captions.txt`
- Origin:
  - Built from COCO 2017 captions via the project script.
  - Registered inside ModelOpt’s PTQ pipeline as dataset name
    `coco2017_captions_local`.
- Key PTQ settings (from `quantize_qwen2_5_vl_3b_fp8_coco2017.sh`):
  - `qformat = "fp8"`
  - `calib_size = 4096` text samples
  - `calib_seq = 512`
  - KV cache quantization: `kv_cache_qformat = "fp8"`

- Number of calibration samples used: **4096** text captions.

## vLLM FP8 Inference Test

- Script:
  - `scripts/qwen/run_qwen2_5_vl_3b_vllm_fp8.py`
  - Environment: `rtx5090-vllm` Pixi env
  - Model argument (default):
    - `--model-dir models/qwen2_5_vl_3b_instruct/quantized/fp8_fp8_coco2017`
  - Output directory:
    - `--out-dir tmp/qwen2_5_vl_3b_vllm_fp8`

- Test prompts (text‑only):
  | # | Prompt (text-only)                                                                 | FP8 output (vLLM, truncated)                                                                                           |
  | - | ---------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
  | 1 | Write a short haiku about GPUs and quantization.                                   | “Sure! Here's a short haiku … Quantized data flows, / GPU crunches numbers fast, / Precision in action.”              |
  | 2 | Explain FP8 quantization to a senior ML engineer in three sentences.              | “FP8 quantization is a form of quantization that represents floating-point numbers using 8-bit fixed-point arithmetic…” |
  | 3 | List three practical benefits of using TensorRT-LLM with FP8 quantization.        | “1. Reduced memory usage … 2. Improved inference speed …”                                                             |
  | 4 | Describe a cute animal in one sentence.                                           | “The cute animal is a fluffy, little squirrel with a big, round nose and bright, green eyes…”                         |
  | 5 | Generate a short conversation between a data scientist and a GPU about compression. | “Data Scientist: Hi, I'm looking for ways to compress my deep learning models. GPU: Hello! I can definitely help with that…” |

- Saved outputs (full text):
  - `tmp/qwen2_5_vl_3b_vllm_fp8/sample_01.txt` … `sample_05.txt`

- Notes on vision stack quantization and vLLM:
  - The FP8 checkpoint used here (`fp8_fp8_coco2017`) follows the
    **LM-only FP8** pattern used by official / community Qwen2.5‑VL FP8
    recipes (language model quantized to FP8; vision tower left in
    BF16/FP16).
  - vLLM’s ModelOpt integration is currently wired to that LM-only FP8
    layout; **vision-stack FP8 Qwen2.5‑VL checkpoints are not supported**
    in vLLM 0.10.x.
  - A VLM-calibrated FP8 checkpoint that also quantizes vision
    (`fp8_fp8_coco2017_vlm`) exists for experimentation with HF/ModelOpt
    and TRT‑LLM, but it fails to load in vLLM due to missing tensor names
    expected by the current loader. See:
      `models/qwen2_5_vl_3b_instruct/reports/fp8-vlm-vs-textonly-vllm-compat.md`

## Quantization Error / Accuracy Notes

- The FP8 PTQ step was run via:

  ```bash
  pixi run -e rtx5090 bash scripts/qwen/quantize_qwen2_5_vl_3b_fp8_coco2017.sh
  ```

- The `hf_ptq.py` example from TensorRT Model Optimizer:
  - Performs calibration and quantization.
  - Logs timing and a small “before vs after PTQ” text sample.
  - **Does not** compute or persist a numeric accuracy / error metric
    (e.g., perplexity or MSE) by default.

- **Quantization error status for this run:**
  - No numeric error figure was produced or stored by the PTQ script.
  - To obtain quantitative accuracy / error metrics, a separate evaluation
    pipeline (e.g., the `llm_eval` tools in
    `extern/TensorRT-Model-Optimizer/examples/llm_eval`) or another benchmark
    (MMLU, MTBench, etc.) must be run against:
    - The original FP16/bfloat16 checkpoint, and
    - The FP8‑quantized checkpoint.

## Mixed-precision scheme (how FP8 vs BF16/FP16 is chosen)

For the `fp8_fp8_coco2017` checkpoint, we use ModelOpt’s default FP8 config
via `hf_ptq.py`:

- Command (from `quantize_qwen2_5_vl_3b_fp8_coco2017.sh`):

  ```bash
  pixi run -e rtx5090 bash scripts/qwen/quantize_qwen2_5_vl_3b_fp8_coco2017.sh
  ```

- `hf_ptq.py` is invoked with:
  - `--qformat fp8`
  - `--kv_cache_qformat fp8`
  - `--auto_quantize_bits 0` (no AutoQuant search)

Inside `hf_ptq.py`, this selects the FP8 config from
`QUANT_CFG_CHOICES["fp8"]`, which maps to
`modelopt.torch.quantization.FP8_DEFAULT_CFG`, and then merges in FP8 KV
cache config. The resulting **mixed-precision** behavior is fully driven by
name/pattern rules in `quant_cfg`.

### 1. Structural scope: language model only, vision excluded

Before applying FP8, `hf_ptq.py` extracts the language model and disables
quantization on all other modules:

```python
# extern/TensorRT-Model-Optimizer/examples/llm_ptq/hf_ptq.py:348–368
# We only quantize the language model for VLMs other than the type supported above.
language_model_lineage = get_language_model_from_vl(full_model)
if language_model_lineage is not None:
    language_model = language_model_lineage.pop(-1)
    ancestors = language_model_lineage
    # Apply disabled quant to all modules that are not part of language_model
    disabled_quant_cfg = {
        "quant_cfg": {"default": {"enable": False}},
        "algorithm": "max",
    }
    ...
    model = language_model
    model_type = get_model_type(model)
```

This means:

- The **vision stack** is excluded structurally and remains BF16/FP16.
- Subsequent FP8 rules apply only inside the extracted language model.

### 2. Pattern-based quantization inside the LM

`FP8_DEFAULT_CFG["quant_cfg"]` (from `modelopt.torch.quantization`) looks like:

```python
{
    "*weight_quantizer": {"num_bits": (4, 3), "axis": None},
    "*input_quantizer": {"num_bits": (4, 3), "axis": None},
    "nn.BatchNorm1d": {"*": {"enable": False}},
    "nn.BatchNorm2d": {"*": {"enable": False}},
    "nn.BatchNorm3d": {"*": {"enable": False}},
    "nn.LeakyReLU": {"*": {"enable": False}},
    "*lm_head*": {"enable": False},
    "*proj_out.*": {"enable": False},
    "*block_sparse_moe.gate*": {"enable": False},
    "*router*": {"enable": False},
    "*mlp.gate.*": {"enable": False},
    "*mlp.shared_expert_gate.*": {"enable": False},
    "*output_layer*": {"enable": False},
    "output.*": {"enable": False},
    "default": {"enable": False},
}
```

and the FP8 KV cache config is:

```python
FP8_KV_CFG["quant_cfg"] = {
    "*[kv]_bmm_quantizer": {"num_bits": (4, 3), "axis": None, "enable": True},
    "default": {"enable": False},
}
```

These are combined in `hf_ptq.py` via
`update_quant_cfg_with_kv_cache_quant` (see
`extern/TensorRT-Model-Optimizer/modelopt/torch/quantization/utils.py:760–772`).

At calibration time, ModelOpt walks the LM, inserts quantizers, and for each
quantizer/module applies the first matching pattern. The effect is:

| Pattern / module match                 | Precision / behavior                                   |
| -------------------------------------- | ------------------------------------------------------ |
| `*weight_quantizer`, `*input_quantizer` | FP8 (E4M3) weights + activations enabled               |
| `*[kv]_bmm_quantizer`                  | FP8 KV cache scaling enabled                           |
| `*lm_head*`, `*proj_out.*`, `*router*`, `*mlp.gate.*`, etc. | Quantization explicitly disabled → stays BF16/FP16 |
| BatchNorm / LeakyReLU patterns         | Quantization disabled                                  |
| `default`                              | Quantization disabled if no earlier pattern matches    |

In other words:

- **Language model transformer blocks and KV cache**:
  - Quantized to FP8 wherever a `*weight_quantizer` / `*input_quantizer` /
    `*[kv]_bmm_quantizer` is attached and not excluded by name.
- **Parts of the LM such as `lm_head`, certain outputs/gates, norms,
  embeddings, etc.**:
  - Remain in BF16/FP16 because they match “disable” patterns (or only the
    `default` rule) and therefore have quantization turned off.
- **Vision tower**:
  - Remains entirely BF16/FP16 because it was excluded before this pattern
    pass via the language‑model‑only extraction step.

This is why `fp8_fp8_coco2017` is a **mixed‑precision LM‑only FP8 checkpoint**:
the FP8 vs BF16/FP16 choice is driven by these name‑pattern rules in the
ModelOpt config, not by per‑layer sensitivity search, and the vision stack is
never quantized for this variant.
