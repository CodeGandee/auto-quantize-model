Howto: force all-layer sensitivity analysis with ModelOpt AutoQuant

## HEADER
- **Purpose**: Show how to force ModelOpt AutoQuant to score sensitivity for all quantizable layers (including ones that are normally excluded) and what issues to expect.
- **Status**: Draft, based on Qwen2.5‑VL and YOLO-style experiments
- **Date**: 2025-12-10
- **Owner**: AI assistant (Codex CLI)
- **Scope**: PyTorch `modelopt.torch.quantization.auto_quantize` on large models (LLMs / VLMs / CNNs), focusing on effective-bits constrained FP8 and INT8 searches.

## 1. What “all-layer sensitivity” means in ModelOpt

ModelOpt AutoQuant normally works under two kinds of constraints:
- Structural constraints: some modules are never quantized (for example VLM vision towers in hf_ptq, or modules excluded by `disabled_quant_cfg`).
- Name/pattern constraints: the default quant configs (for example `FP8_DEFAULT_CFG`, `INT8_DEFAULT_CFG`) have patterns like `*lm_head*`, `*router*`, `*output_layer*`, `default: {enable: False}` which deliberately keep certain layers in higher precision.

If you just call:
- `mtq.auto_quantize(model, quantization_formats=[mtq.FP8_DEFAULT_CFG], constraints={"effective_bits": 11.0}, ...)`

then AutoQuant will only compute sensitivity for layers that are allowed to toggle between “quantized format(s)” and “NONE(...)” according to that config and any model-level restrictions.

**Forcing all-layer sensitivity** means:
- Every quantizable linear/conv module should have a candidate recipe (for example FP8 vs NONE) so AutoQuant produces a meaningful score.
- There are no default “enable: False” patterns silently excluding whole families of layers you care about.

## 2. Step 1 – Build an “all-layers” quantization config

The first step is to derive a custom quantization config from a default one by removing the name-based disable patterns and enabling quantization by default.

Example: FP8 all-layers config (from this repo’s `src/auto_quantize_model/modelopt_configs.py`):

```python
from copy import deepcopy
from typing import Any, Dict

import modelopt.torch.quantization as mtq

def build_fp8_all_layers_cfg() -> Dict[str, Any]:
    cfg: Dict[str, Any] = deepcopy(mtq.FP8_DEFAULT_CFG)
    quant_cfg = cfg.get("quant_cfg", {})

    # Keep only generic weight/input quantizer patterns.
    keep_keys = {"*weight_quantizer", "*input_quantizer"}
    for key in list(quant_cfg.keys()):
        if key not in keep_keys:
            quant_cfg.pop(key, None)

    # Enable quantization by default with FP8 attributes similar to input quantizers.
    input_cfg = quant_cfg.get("*input_quantizer", {})
    num_bits = input_cfg.get("num_bits", (4, 3))
    axis = input_cfg.get("axis", None)
    quant_cfg["default"] = {
        "num_bits": num_bits,
        "axis": axis,
        "enable": True,
    }
    return cfg

FP8_ALL_LAYERS_CFG = build_fp8_all_layers_cfg()
```

Key points:
- Removing the “disable” patterns (`*lm_head*`, `*output_layer*`, `default: {enable: False}`, etc.) allows AutoQuant to consider **every attached quantizer** as a candidate.
- Setting `default.enable = True` ensures any quantizer that does not match a more specific rule is still part of the search space.
- You can do the same starting from `INT8_DEFAULT_CFG` if you want INT8 all-layer sensitivity instead of FP8.

References:
- ModelOpt PyTorch quantization guide (`auto_quantize` and config structure): https://nvidia.github.io/TensorRT-Model-Optimizer/guides/_pytorch_quantization.html
- API reference for `auto_quantize`: https://nvidia.github.io/TensorRT-Model-Optimizer/reference/generated/modelopt.torch.quantization.model_quant.html

## 3. Step 2 – Ensure your calibration data exercises all subsystems

AutoQuant’s gradient-based scoring is driven by a loss function.
- If the loss only depends on some submodules, layers outside that path will show sensitivity scores very close to zero even if they have quantizers attached.

Examples:
- For a pure LLM, a causal LM loss over tokens exercises only the transformer stack and head, not any external encoders.
- For Qwen2.5‑VL:
  - If you calibrate on **text-only captions**, the vision tower (`visual.blocks.*`) never affects the LM loss, so its sensitivity scores stay at 0.0.
  - If you calibrate on **image + text pairs** using `AutoProcessor` and `qwen_vl_utils.process_vision_info`, then the vision tower participates in the forward + loss, and you see non-zero sensitivity scores for `visual.blocks.*` as well.

Checklist:
- For CNNs: use real images from your deployment domain and the detection/segmentation/classification loss you care about.
- For VLMs: build a calibration dataset that includes both visual and textual inputs so that both towers and cross-modal layers influence the loss.
- For encoder–decoder architectures: make sure the loss involves both encoder and decoder outputs if you want sensitivity for both sides.

In code (Qwen2.5‑VL style image+text calibration dataset):

```python
from pathlib import Path
from torch.utils.data import Dataset
from transformers import AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info

class CocoVlmDataset(Dataset):
    def __init__(self, calib_db: Path, coco_root: Path,
                 tokenizer: AutoTokenizer, processor: AutoProcessor,
                 max_samples: int, max_length: int) -> None:
        # Load (image_relpath, caption) rows from SQLite and resolve paths...
        self.m_samples = ...
        self.m_tokenizer = tokenizer
        self.m_processor = processor
        self.m_max_length = max_length

    def __getitem__(self, idx):
        image_path, caption = self.m_samples[idx]
        messages = [{"role": "user", "content": [
            {"type": "image", "image": str(image_path)},
            {"type": "text", "text": caption},
        ]}]
        text = self.m_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.m_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding="max_length",
            max_length=self.m_max_length,
            return_tensors="pt",
        )
        inputs["labels"] = inputs["input_ids"].clone()
        return inputs
```

## 4. Step 3 – Wire `auto_quantize` with the custom config

Once you have:
- A model in eval mode on the target device.
- A calibration iterable over batches that exercise all subsystems.
- A custom quant config like `FP8_ALL_LAYERS_CFG`.

You can call `auto_quantize` with that config as the only quantization format:

```python
import torch
import modelopt.torch.quantization as mtq

from myproject.modelopt_configs import FP8_ALL_LAYERS_CFG

device = torch.device("cuda")
model = my_large_model.to(device).eval()

def forward_step(model, batch):
    batch_on_device = {k: v.to(device) for k, v in batch.items()}
    return model(**batch_on_device)

def loss_func(output, batch):
    labels = batch["labels"].to(device)
    logits = output.logits if hasattr(output, "logits") else output
    # Simple causal LM loss; adjust to your task.
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    return loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1))

quantization_formats = [FP8_ALL_LAYERS_CFG]

calib_batches = list(calib_loader)
num_score_steps = min(8, len(calib_batches))  # tune based on budget

quantized_model, state = mtq.auto_quantize(
    model,
    constraints={"effective_bits": 11.0},
    quantization_formats=quantization_formats,
    data_loader=calib_batches,
    forward_step=forward_step,
    loss_func=loss_func,
    num_calib_steps=len(calib_batches),
    num_score_steps=num_score_steps,
    verbose=True,
)
```

With this setup:
- Every enabled quantizer has at least two candidate recipes internally: “quantized” (your FP8/INT8 config) and “NONE(...)” (no quantization).
- AutoQuant solves a budgeted assignment problem over these recipes to meet `effective_bits`, and returns:
  - A quantized model.
  - A `state` dict that contains per-layer candidate stats and the best solution.

## 5. Step 4 – Extract and inspect per-layer sensitivity

ModelOpt’s AutoQuant state object exposes a `candidate_stats` map with per-layer information:
- `candidate_stats[name]["formats"]`: list of recipe labels (for example `CUSTOM_0(effective-bits: 8.0)`, `NONE(effective-bits: 16.0)`).
- `candidate_stats[name]["scores"]`: sensitivity scores per recipe.
- `candidate_stats[name]["costs"]`: cost / size contribution per recipe.

The **sensitivity** signal you usually care about is the score associated with the quantized recipe versus the `NONE(...)` recipe:
- High score for FP8/INT8 and low score for NONE means the layer is sensitive (quantizing hurts loss more).
- Low score for FP8/INT8 implies the layer is robust to quantization.

In this repo we store a JSON manifest and a Markdown summary:
- Manifest field: `layer_sensitivity[name] = {"formats": [...], "scores": [...], "costs": [...]}`.
- Markdown table column “Sensitivity”: formatted from those `scores` values, with `NONE(...)` rows filtered out.

Minimal extraction sketch:

```python
candidate_stats = state["candidate_stats"]
for name, stats in candidate_stats.items():
    formats = [str(f) for f in stats["formats"]]
    scores = [float(s) for s in stats["scores"]]
    costs = [float(c) for c in stats["costs"]]
    # Find quantized vs NONE entries
    for fmt, score, cost in zip(formats, scores, costs):
        if fmt.startswith("NONE("):
            continue
        print(name, fmt, "sensitivity:", score, "cost:", cost)
```

## 6. Issues and gotchas you are likely to hit

Based on running all-layer FP8 AutoQuant on Qwen2.5‑VL‑3B and similar models, these are the most common pitfalls:

1. **Zero sensitivity for entire subsystems**
   - Symptom: all `visual.blocks.*` or another whole module family have scores `[0.0, 0.0]` for every recipe.
   - Cause: the loss does not depend on those modules for the chosen calibration data (for example text-only prompts for a VLM).
   - Fix: switch to calibration batches that exercise those paths (image+text, encoder+decoder, etc.).

2. **Structural exclusions still apply**
   - For VLMs, tools like `hf_ptq.py` may:
     - Use `get_language_model_from_vl` and quantize only the language model.
     - Apply a `disabled_quant_cfg` to everything outside the LM.
   - If you copy those patterns, your all-layer config might still be limited to LM-only sensitivity.
   - Fix: if you truly want full VLM sensitivity, use the full model as the AutoQuant target and avoid structural disabled-quant configs, or accept that you are doing “LM + some attached modules” sensitivity only.

3. **OOM and long runtimes**
   - All-layer FP8/INT8 on large VLMs is memory-heavy:
     - AutoQuant temporarily inserts many quantizers (we saw >1400 on Qwen2.5‑VL‑3B), and score estimation runs backward hooks through all of them.
   - Typical mitigations:
     - Reduce `max_calib_samples`, `batch_size`, and `num_score_steps`.
     - Use shorter sequence lengths (`calib-seq-len` in our driver).
     - Accept that sensitivity scores are noisy but still useful for ranking.

4. **FP8 CUDA extension warnings**
   - On some GPUs / arch combinations, ModelOpt prints:
     - “CUDA extension for FP8 quantization could not be built, FP8 simulated quantization will not be available.”
   - This means:
     - You still get AutoQuant scores, but FP8 behavior is simulated rather than using the custom kernel.
   - For sensitivity analysis this is usually acceptable, but be aware it may not perfectly match production kernels.

5. **Interpreting `scores` vs `costs`**
   - `scores` correspond to how much quantizing that recipe increases the AutoQuant target loss.
   - `costs` correspond to approximate compressed weight size for that recipe.
   - Together they define a trade-off frontier: you typically prefer recipes with low score and low cost, within your `effective_bits` budget.

6. **Name noise and `quant_recipe` suffixes**
   - ModelOpt’s AutoQuant often uses internal “quant recipe” handles:
     - For example `language_model.layers.8.self_attn.q_proj.quant_recipe`.
   - The `quant_recipe` suffix is not a real module in the original model; it is a ModelOpt-inserted hook representing the quantization hyperparameter group.
   - When generating human-readable reports, you may want to strip `.quant_recipe` so the path matches the underlying module, while keeping a note that the JSON still uses the full name.

## 7. Practical pattern for this repo

For LLM/VLM experiments in this repository:
- We define `FP8_ALL_LAYERS_CFG` as above and register it under a friendly name so high-level drivers can reference it as a “format”.
- We add an AutoQuant scheme:
  - `fp8_autoquant_all_layers_fp8` with `quant_formats=["FP8_ALL_LAYERS_CFG"]`, `coverage_mode="full"`, `coverage_fraction=1.0`.
- For Qwen2.5‑VL‑3B:
  - Text-only schemes use COCO captions and produce LM-only sensitivity (vision blocks show up as quantized modules but had zero scores until we switched to image+text calibration).
  - The all-layers scheme uses COCO2017 image+caption pairs via `CocoVlmDataset`, so both vision and language towers get non-zero sensitivity scores.
- We write:
  - A JSON manifest with `layer_sensitivity`.
  - A `layer-sensitivity-report.md` Markdown table (sorted by sensitivity, ignoring `NONE(...)` recipes) for human inspection.

You can reuse this pattern for other large models:
- Clone the “all-layers config” idea on top of the default config for your target datatype (FP8 / INT8 / NVFP4).
- Ensure calibration data and loss functions cover the parts of the model you want to analyze.
- Use the AutoQuant state’s `candidate_stats` to build your own reports and drive mixed-precision decisions.

## 8. Additional references

- ModelOpt PyTorch PTQ overview (includes AutoQuant examples): https://nvidia.github.io/TensorRT-Model-Optimizer/guides/_pytorch_quantization.html
- ModelOpt `auto_quantize` API reference: https://nvidia.github.io/TensorRT-Model-Optimizer/reference/generated/modelopt.torch.quantization.model_quant.html
- NVIDIA blog on PTQ and AutoQuant for LLMs: https://developer.nvidia.com/blog/optimizing-llms-for-performance-and-accuracy-with-post-training-quantization/
- vLLM + ModelOpt integration docs (for using exported ModelOpt checkpoints): https://docs.vllm.ai/en/latest/features/quantization/modelopt/
