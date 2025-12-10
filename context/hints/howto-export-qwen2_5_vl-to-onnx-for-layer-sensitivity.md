# How to Export Qwen2.5‑VL to ONNX for Layer Sensitivity Analysis

This hint summarizes what works (and what currently does not) when exporting **Qwen2.5‑VL‑3B‑Instruct** to ONNX for use with ONNX Runtime / Intel Neural Compressor as a *layer sensitivity engine*.

It is based on:
- The official Qwen2.5‑VL implementation in Transformers (vision patching and attention internals).
- Public examples for exporting the **vision encoder** to ONNX.
- Hands‑on experiments in this repo with `torch.onnx.export` and `optimum-onnx` on RTX 5090.

We focus on:
- Why full **VLM** (vision + text) ONNX export is fragile right now.
- Why **image size constraints** matter.
- A practical recipe using **separate ONNX exports** (vision encoder + text LM), with a **fixed image size 672×672** for the vision path.

## 1. Key constraints from Qwen2.5‑VL vision implementation

The Qwen2.5‑VL vision tower uses a patch+merge scheme (`patch_size`, `spatial_merge_size`, `temporal_patch_size`) and constructs a `grid_thw` tensor that drives the windowed attention.

From `transformers.models.qwen2_5_vl.modeling_qwen2_5_vl` (current HF version), the visual config for Qwen2.5‑VL‑3B looks like:

```python
from transformers import Qwen2_5_VLForConditionalGeneration

model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
vcfg = model.visual.config
print(vcfg.patch_size, vcfg.spatial_merge_size, vcfg.temporal_patch_size)
# patch_size: 14, spatial_merge_size: 2, temporal_patch_size: 2
```

A reference export script (see below) uses:

```python
def build_patches_and_grid(pixel_values, temporal_patch_size, patch_size, merge_size):
    assert pixel_values.dim() == 4, "pixel_values must be (N, C, H, W)"
    N, C, H, W = pixel_values.shape
    if H % patch_size != 0 or W % patch_size != 0:
        raise ValueError("H, W must be divisible by patch_size")
    if (H // patch_size) % merge_size != 0 or (W // patch_size) % merge_size != 0:
        raise ValueError("(H/patch_size, W/patch_size) must be divisible by merge_size")
    ...
    grid_t = pixel_values.shape[0] // temporal_patch_size
    grid_h = H // patch_size
    grid_w = W // patch_size
    ...
    grid_thw = torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.int32, device=flatten_patches.device)
    return flatten_patches, grid_thw
```

With `patch_size=14` and `merge_size=2`, the constraints are:

- `H % 14 == 0` and `W % 14 == 0`.
- `(H / 14) % 2 == 0` and `(W / 14) % 2 == 0`.

Equivalently, **H and W must be multiples of 28**.

Examples:
- 476×476 is valid (`476 = 14 × 34`, and 34 is divisible by 2).
- 640×640 is **not** valid (`640 / 14` is not an integer).
- 672×672 is valid (`672 = 14 × 48`, and 48 is divisible by 2).

This is why many public examples for Qwen2.5‑VL vision exports default to **476×476** or other multiples of 28.

Reference:
- HF vision export example (vision‑only):  
  `happyme531/Qwen2.5-VL-3B-Instruct-RKLLM/export_vision_onnx.py`  
  https://huggingface.co/happyme531/Qwen2.5-VL-3B-Instruct-RKLLM/blob/main/export_vision_onnx.py

## 2. Why full VLM ONNX export is fragile

In this repo we attempted to export **the entire Qwen2.5‑VL model (vision + text)** to ONNX (FP16) by:

- Building a single multimodal batch using `AutoProcessor` + `qwen_vl_utils.process_vision_info` (image+text).
- Wrapping `Qwen2_5_VLForConditionalGeneration` in a small module that exposes:
  - `input_ids`, `attention_mask`, `pixel_values`, `image_grid_thw` as inputs.
  - `logits` as the only output.
- Calling `torch.onnx.export(...)` with `attn_implementation="eager"`:
  ```python
  model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
      ckpt_dir,
      torch_dtype=torch.float16,
      device_map=None,
      attn_implementation="eager",
  )
  ```

On RTX 5090 (PyTorch 2.7.1) this consistently fails during export with:

- `RuntimeError: _Map_base::at` raised deep inside:
  - `torch._functorch.autograd_function.custom_function_call_vmap(...)`
  - called from Qwen2.5‑VL attention stacks (both visual and text) when tracing.

Key observations:

- The failure happens for both **CUDA** and **CPU** export.
- Forcing `attn_implementation="eager"` does not avoid the crash.
- `optimum-onnx` does not yet provide a dedicated `OnnxConfig` for `qwen2_5_vl`; internally it also calls `torch.onnx.export`, so it inherits the same issue.
- The Qwen2.5‑VL code itself contains comments aimed at making ONNX export possible (e.g. ensuring `cu_seqlens` dtype matches `grid_thw` when `torch.jit.is_tracing()`), but the combination of functorch/custom attention and multimodal logic is still fragile for a **single monolithic ONNX graph**.

Conclusion: as of this writing, exporting the **full VLM** Qwen2.5‑VL‑3B model to a single ONNX file (vision + text) is unreliable with the current `torch` / `transformers` stack used in this repo. The more robust path is to **export towers separately**.

## 3. Recommended pattern: separate ONNX exports

For layer sensitivity analysis (e.g., with ONNX Runtime + INC), the most practical pattern is:

- Export **vision encoder only** as ONNX (fixed image size, e.g. 672×672).
- Export **text LM only** as ONNX (standard text inputs).
- Run sensitivity analysis separately on the vision and text graphs (or primarily on the LM if that’s your focus).

This matches how many VLM deploy pipelines treat the model: a vision encoder + language model, often quantized or analyzed separately.

### 3.1 Vision encoder ONNX (672×672 image)

We adopt **672×672** as the default fixed resolution for ONNX vision sensitivity, because:

- It satisfies the patch/merge constraints (multiples of 28).
- It is high enough to be representative for common VLM tasks.
- It plays well with downstream backends that prefer square inputs.

Basic recipe (adapted from the HF example):

```python
import torch
from pathlib import Path
from transformers import Qwen2_5_VLForConditionalGeneration


def build_patches_and_grid(pixel_values, temporal_patch_size, patch_size, merge_size):
    assert pixel_values.dim() == 4, "pixel_values must be (N, C, H, W)"
    N, C, H, W = pixel_values.shape
    if H % patch_size != 0 or W % patch_size != 0:
        raise ValueError("H, W must be divisible by patch_size")
    if (H // patch_size) % merge_size != 0 or (W // patch_size) % merge_size != 0:
        raise ValueError("(H/patch_size, W/patch_size) must be divisible by merge_size")

    if N == 1:
        pixel_values = pixel_values.repeat(temporal_patch_size, 1, 1, 1)

    grid_t = pixel_values.shape[0] // temporal_patch_size
    grid_h = H // patch_size
    grid_w = W // patch_size

    patches = pixel_values.reshape(
        grid_t,
        temporal_patch_size,
        C,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flatten_patches = patches.reshape(
        grid_t * grid_h * grid_w, C * temporal_patch_size * patch_size * patch_size
    )
    grid_thw = torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.int32, device=flatten_patches.device)
    return flatten_patches, grid_thw


ckpt_dir = Path("models/qwen2_5_vl_3b_instruct/checkpoints/Qwen2.5-VL-3B-Instruct")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    str(ckpt_dir),
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
    attn_implementation="eager",
).eval()

vcfg = model.visual.config
merge_size = int(vcfg.spatial_merge_size)
patch_size = int(vcfg.patch_size)              # 14
temporal_patch_size = int(vcfg.temporal_patch_size)  # 2

N, C, H, W = 1, 3, 672, 672                    # our chosen resolution
pixel_values = torch.randn(N, C, H, W, dtype=torch.float32)

with torch.no_grad():
    fp, gthw = build_patches_and_grid(pixel_values, temporal_patch_size, patch_size, merge_size)
    vision_features = model.visual(fp, gthw)
    print("vision features shape:", vision_features.shape)


def vision_forward(pixel_values_in: torch.Tensor) -> torch.Tensor:
    fp, gthw = build_patches_and_grid(pixel_values_in, temporal_patch_size, patch_size, merge_size)
    return model.visual(fp, gthw)


model.forward = vision_forward

torch.onnx.export(
    model,
    (pixel_values,),
    "models/qwen2_5_vl_3b_instruct/onnx/qwen2_5_vl_3b_vision_672_fp16.onnx",
    input_names=["pixel_values"],
    output_names=["vision_features"],
    opset_version=17,
)
```

Notes:

- You can cast weights to FP16 before export or post‑process the ONNX graph to FP16 with `onnxconverter-common.float16`.
- For ONNX Runtime + INC sensitivity, the key is a stable, fully traceable **vision encoder graph** with fixed shape 672×672.

### 3.2 Text‑only LM ONNX

The text tower can be exported using a simpler wrapper similar to a standard decoder‑only LLM:

```python
import torch
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration


class Qwen25VLTextWrapper(torch.nn.Module):
    def __init__(self, model: Qwen2_5_VLForConditionalGeneration) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits


model_dir = "models/qwen2_5_vl_3b_instruct/checkpoints/Qwen2.5-VL-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_dir,
    torch_dtype=torch.float16,
    device_map=None,
    attn_implementation="eager",
).eval()

wrapped = Qwen25VLTextWrapper(base_model)

dummy = tokenizer("Hello, world", return_tensors="pt", max_length=256, padding="max_length", truncation=True)
input_ids = dummy["input_ids"]
attention_mask = dummy["attention_mask"]

torch.onnx.export(
    wrapped,
    (input_ids, attention_mask),
    "models/qwen2_5_vl_3b_instruct/onnx/qwen2_5_vl_3b_text_fp16.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "logits": {0: "batch", 1: "seq"},
    },
    opset_version=17,
)
```

This ONNX is straightforward and does not have the vision‑specific attention / patching complications.

## 4. Using ONNX models for layer sensitivity

Once you have:

- `qwen2_5_vl_3b_vision_672_fp16.onnx` (vision features from images).
- `qwen2_5_vl_3b_text_fp16.onnx` (logits from tokenized text).

You can:

- Build ONNXRuntime calibration dataloaders that:
  - Feed fixed‑resolution images (672×672) for the vision model.
  - Feed tokenized prompts for the text model.
- Use INC’s ONNXRuntime adaptor (`backend="onnxrt_qdq"`) and `strategy="mse_v2"` to compute per‑op sensitivity on each graph separately.
- Design mixed‑precision / quantization profiles based on:
  - Vision sensitivity (which layers in the visual transformer are most fragile).
  - Text sensitivity (which decoder blocks / projections are most sensitive).

This approach avoids the current fragility of full VLM ONNX export while still giving you actionable per‑layer sensitivity for each tower. In this repo we standardize on **672×672** as the fixed image size for vision ONNX exports because it matches the Qwen2.5‑VL patching constraints and is convenient for downstream tools. 

