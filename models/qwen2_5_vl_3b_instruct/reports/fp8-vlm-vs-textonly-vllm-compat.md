# FP8 Qwen2.5‑VL: why text‑only works with vLLM but VLM FP8 does not

## Context

We produced two FP8‑quantized checkpoints for Qwen2.5‑VL‑3B‑Instruct using NVIDIA ModelOpt:

- **Text‑only FP8**: `models/qwen2_5_vl_3b_instruct/quantized/fp8_fp8_coco2017`
  - Produced via ModelOpt’s `hf_ptq.py` script.
  - Calibration: COCO2017 captions (`coco2017_captions_local`, text only).
- **VLM FP8 (image+text)**: `models/qwen2_5_vl_3b_instruct/quantized/fp8_fp8_coco2017_vlm`
  - Produced via a custom driver script
    (`scripts/qwen/quantize_qwen2_5_vl_3b_fp8_coco2017_vlm.py`) that
    runs ModelOpt FP8 calibration on COCO2017 image+caption pairs.

When loading these checkpoints with vLLM (`quantization="modelopt"`) in the
`rtx5090-vllm` Pixi env:

- **Text‑only FP8** loads and generates successfully.
- **VLM FP8** fails in the weight‑loading stage with a missing‑tensor `KeyError`.

This document explains why.

## 1. Quantization configs differ (vision excluded vs included)

Comparing the ModelOpt metadata in `hf_quant_config.json`:

- **Text‑only FP8** (`fp8_fp8_coco2017`):

  ```json
  {
    "producer": {"name": "modelopt", "version": "0.33.1"},
    "quantization": {
      "quant_algo": "FP8",
      "kv_cache_quant_algo": "FP8",
      "exclude_modules": ["lm_head", "model.visual*"]
    }
  }
  ```

  - Only the language model is quantized to FP8 (plus FP8 KV cache).
  - The entire vision stack (`model.visual*`) is explicitly excluded and remains
    in bf16/fp16.

- **VLM FP8** (`fp8_fp8_coco2017_vlm`):

  ```json
  {
    "producer": {"name": "modelopt", "version": "0.33.1"},
    "quantization": {
      "quant_algo": "FP8",
      "kv_cache_quant_algo": "FP8",
      "exclude_modules": ["lm_head"]
    }
  }
  ```

  - Language model **and** vision modules are quantized to FP8.
  - No `model.visual*` exclusion, so more layers receive FP8 quantizers, scales,
    and auxiliary tensors.

Although the base HF architecture is the same, these two quantization configs
produce **different sets of parameters** in the exported `model.safetensors`
state dict (different modules are quantized, with different naming and layout
for their scale tensors).

## 2. vLLM expects the `hf_ptq.py` LM‑only FP8 layout

vLLM’s ModelOpt integration for Qwen2.5‑VL:

- Assumes the checkpoint comes from ModelOpt’s official LLM PTQ flow
  (`extern/TensorRT-Model-Optimizer/examples/llm_ptq/hf_ptq.py`) with:
  - `qformat="fp8"`,
  - FP8 KV cache enabled,
  - and, critically, **only the LM quantized** (vision excluded).
- Its `qwen2_5_vl` loader and `hf_to_vllm_mapper` are implemented to match the
  parameter names, dtypes, and structures emitted by that flow.

For the text‑only FP8 checkpoint, this contract holds:

- The quantization config matches what vLLM expects (FP8 LM, FP8 KV, vision off).
- The exported state dict contains the extra FP8 tensors with the exact names
  vLLM looks for (e.g. per‑proj scale tensors such as
  `blocks.0.attn.k_proj.k_scale`).
- Result: vLLM can traverse the checkpoint, map HF → vLLM parameters, and build
  its internal quantized layers successfully.

## 3. What goes wrong for the VLM‑calibrated FP8 checkpoint

When we run:

```bash
pixi run -e rtx5090-vllm python scripts/qwen/run_qwen2_5_vl_3b_vllm_fp8.py \
  --model-dir models/qwen2_5_vl_3b_instruct/quantized/fp8_fp8_coco2017_vlm
```

vLLM log excerpt:

- Detects the FP8 ModelOpt checkpoint and begins loading:

  ```text
  WARNING ... Detected ModelOpt fp8 checkpoint. Please note that
  the format is experimental and could change.
  ```

- Fails in the weight loader:

  ```text
  KeyError: 'blocks.0.attn.k_proj.k_scale'
  ```

Interpretation:

- vLLM’s `qwen2_5_vl` loader is iterating its expected list of parameters for
  a ModelOpt FP8 checkpoint and tries to fetch
  `blocks.0.attn.k_proj.k_scale` from the HF state dict.
- In the VLM FP8 export we produced:
  - That exact tensor name **does not exist** in `model.safetensors`.
  - Either the corresponding layer is quantized differently, or its scale
    tensors are stored under different names / structures due to the different
    quantization scope (vision included) and our custom quantization config + export.
- vLLM does not dynamically adapt to the new layout. It hard‑codes the
  expected names and thus hits a `KeyError` when the checkpoint doesn’t match.

In other words, vLLM’s current ModelOpt integration knows how to read the
**LM‑only FP8** export from `hf_ptq.py`, but it does **not** know how to read
our more aggressive VLM‑quantized FP8 checkpoint.

## 4. Why text‑only vs VLM export behave differently

The two flows differ in several ways:

1. **Which modules are quantized**
   - Text‑only: quantization confined to LM layers; vision modules remain
     unquantized.
   - VLM: both LM and vision modules are quantized (no `model.visual*`
     exclusion).

2. **Quantization config construction**
   - Text‑only: uses ModelOpt’s `hf_ptq.py` `build_quant_cfg` +
     `update_quant_cfg_with_kv_cache_quant(...)`, which encodes the exact
     behavior vLLM’s heuristics were tested against.
   - VLM: we manually build `quant_cfg` from `mtq.FP8_DEFAULT_CFG` + `FP8_KV_CFG`
     and run quantization via a custom forward loop over VLM data; this is a
     supported ModelOpt pattern, but not one vLLM’s loader has been wired to
     understand for Qwen2.5‑VL.

3. **Export helper and metadata**
   - Both flows call `export_hf_checkpoint`, but the starting model and
     quantization config differ, so the resulting arrangement of weight tensors
     and scales in `model.safetensors` diverges.
   - The text‑only path reproduces the “canonical” layout vLLM expects; the VLM
     path does not, even though the architecture is the same.

4. **hf_ptq.py calibration for Qwen2.5‑VL is text‑only**

   The stock ModelOpt PTQ script (`extern/TensorRT-Model-Optimizer/examples/llm_ptq/hf_ptq.py`)
   has a VLM calibration path **only for specific model types**:

   - In the calibration setup:

     ```python
     if model_type == "mllama":
         # VLM path for Mllama – uses images + text
         calib_dataloader = get_vlm_dataset_dataloader(
             dataset_name=args.dataset[0] if args.dataset else "scienceqa",
             processor=processor,
             batch_size=args.batch_size,
             num_samples=args.calib_size[0],
         )
         elif model_type == "whisper":
             # audio path
             calib_dataloader, first_text = get_speech_dataset_dataloader(...)
         else:
             # generic LLM path – text only
             calib_dataloader = get_dataset_dataloader(
                 dataset_name=args.dataset,
                 tokenizer=tokenizer,
                 batch_size=args.batch_size,
                 num_samples=args.calib_size,
                 device=device,
                 include_labels=include_labels,
             )
     ```

   - And in `example_utils.get_processor`:

     ```python
     if model_type == "whisper":
         ...
     elif model_type == "mllama":
         # Only mllama gets an MllamaImageProcessor for images
         return MllamaImageProcessor(processor, device)
     ```

   For Qwen2.5‑VL, `model_type` is `qwen2_5_vl`, so neither the `mllama` nor
   `whisper` branch triggers. The script falls into the generic `else:` branch
   and builds a text‑only `calib_dataloader` via `get_dataset_dataloader`, even
   though the underlying model is multimodal. In other words:

   - **hf_ptq.py does not use image data when quantizing Qwen2.5‑VL** in its
     current form; it treats it like a standard LLM for calibration purposes.
   - That is consistent with the LM‑only FP8 checkpoint (`fp8_fp8_coco2017`)
     that vLLM understands.

   The exact locations in this repo are:

   - Language‑model‑only extraction for VLMs:
     - `extern/TensorRT-Model-Optimizer/examples/llm_ptq/hf_ptq.py:340–369`
   - Calibration dataloader selection (`mllama` / `whisper` / generic text):
     - `extern/TensorRT-Model-Optimizer/examples/llm_ptq/hf_ptq.py:440–486`
   - Processor selection with image support only for `mllama`:
     - `extern/TensorRT-Model-Optimizer/examples/llm_ptq/example_utils.py:207–242`

   On the vLLM side, the relevant pieces backing the discussion above are:

   - Weight loading that assumes a fixed mapping and will raise `KeyError`
     when a requested parameter name is missing:
     - `extern/vllm/vllm/model_executor/models/qwen2_5_vl.py:762–780`
   - The Qwen2.5‑VL HF→vLLM weight prefix mapper:
     - `extern/vllm/vllm/model_executor/models/qwen2_5_vl.py:818–827`
   - ModelOpt FP8/NVFP4 config parsing and `exclude_modules` handling:
     - `extern/vllm/vllm/model_executor/layers/quantization/modelopt.py:661–671, 692–707`

## 5. Community FP8 recipes also avoid quantizing the vision tower

The broader ModelOpt / LLM‑Compressor ecosystem makes similar design choices
for Qwen2.5‑VL FP8:

- **NVIDIA’s official Qwen2.5‑VL‑7B FP8 model**  
  The `nvidia/Qwen2.5-VL-7B-Instruct-FP8` model card explicitly states:

  > “Only the weights and activations of the linear operators within transformer blocks of the **language model** are quantized.”

  (Model card: `https://huggingface.co/nvidia/Qwen2.5-VL-7B-Instruct-FP8`)

- **Neural Magic / Red Hat FP8‑dynamic recipe for Qwen2.5‑VL‑72B**  
  The `RedHatAI/Qwen2.5-VL-72B-Instruct-FP8-dynamic` model card shows the
  LLM‑Compressor recipe used to build a vLLM‑ready FP8 checkpoint:

  ```python
  QuantizationModifier(
      targets="Linear",
      scheme="FP8_DYNAMIC",
      sequential_targets=["MistralDecoderLayer"],
      ignore=["re:.*lm_head", "re:vision_tower.*", "re:multi_modal_projector.*"],
  )
  ```

  Note that `vision_tower.*` and `multi_modal_projector.*` are explicitly
  ignored, i.e. the vision stack is left in higher precision.

- **LLM‑Compressor documentation**  
  The LLM‑Compressor docs on memory requirements state (for Qwen2.5‑VL‑7B):

  > “At this time LLM Compressor does not quantise the vision tower as quantization is generally not worth the tradeoff between latency/throughput and accuracy loss.”

  (Docs: `https://docs.vllm.ai/projects/llm-compressor/en/latest/getting-started/compress/`)

Taken together, the “standard” FP8 flows for Qwen2.5‑VL are **LM‑only FP8 by
design**:

- Vision towers are intentionally kept in BF16/FP16 (or similar) due to
  accuracy/performance trade‑offs and memory behavior.
- vLLM’s ModelOpt integration and community FP8 checkpoints are aligned with
  this assumption.

In this context:

- The absence of an “official vision‑FP8 Qwen2.5‑VL” recipe is primarily a
  **ModelOpt / LLM‑Compressor design choice** today.
- The fact that our custom FP8‑VLM checkpoint (which does quantize vision) does
  **not** load in vLLM is a **vLLM integration limitation**: its loader is
  only wired for the LM‑only FP8 layouts those tools currently produce.

This is backed by:

> “This model was obtained by quantizing the weights and activations of Qwen2.5-VL-7B-Instruct to FP8 data type, ready for inference with TensorRT-LLM. **Only the weights and activations of the linear operators within transformer blocks of the language model are quantized.**”  
> — `nvidia/Qwen2.5-VL-7B-Instruct-FP8` model card

and:

> `QuantizationModifier( targets="Linear", scheme="FP8_DYNAMIC", sequential_targets=["MistralDecoderLayer"], ignore=["re:.*lm_head", "re:vision_tower.*", "re:multi_modal_projector.*"], )`  
> “This model was obtained by quantizing the weights … to FP8 data type, **ready for inference with vLLM >= 0.5.2.**”  
> — `RedHatAI/Qwen2.5-VL-72B-Instruct-FP8-dynamic` model card

and from the LLM‑Compressor docs:

> “At this time **LLM Compressor does not quantise the vision tower** as quantization is generally not worth the tradeoff between latency/throughput and accuracy loss.”  
> — LLM‑Compressor “Compress Your Model” guide

## 5. Can we make a VLM‑calibrated FP8 checkpoint that vLLM accepts?

With the current tool versions in this repo:

- **Reliable path today**:
  - Use the official `hf_ptq.py` FP8 flow (LM‑only FP8 + FP8 KV, vision
    excluded) to produce a vLLM‑compatible checkpoint.
  - That’s what `scripts/qwen/quantize_qwen2_5_vl_3b_fp8_coco2017.sh` does
    for `fp8_fp8_coco2017`.

- **What we cannot do reliably yet**:
  - Quantize the vision tower (full VLM FP8) *and* expect vLLM 0.10.x to load
    it, unless:
    - vLLM’s ModelOpt/Qwen2.5‑VL integration is extended upstream to recognize
      the new layout, or
    - We re‑implement vLLM’s internal mapping logic and manufacture a state
      dict that exactly matches its current expectations (fragile and tightly
      coupled to vLLM internals).

**Practical compromise**:

- Keep the vLLM‑facing checkpoint in the **LM‑only FP8** format that vLLM
  understands.
- If you want VLM‑aware calibration, adjust the *text* calibration dataset
  (e.g., captions / prompts derived from image tasks) while still excluding
  `model.visual*` from quantization, so the layout remains compatible with
  vLLM’s current expectations.

## 6. Summary

- The base HF model architecture is identical in both cases.
- The difference is in the **quantization scope and export format**:
  - Text‑only FP8 uses the canonical LM‑only FP8 path that vLLM expects and
    loads fine.
  - VLM FP8 quantizes more modules (vision included) using a custom config, so
    the exported checkpoint’s tensor names/layout deviate from what vLLM
    looks for, causing missing‑tensor `KeyError`s during weight loading.

Until vLLM’s ModelOpt support for Qwen2.5‑VL is extended to handle VLM‑wide
FP8, the safe choice for vLLM is to stick to the `fp8_fp8_coco2017` LM‑only
FP8 checkpoint and treat the VLM‑calibrated FP8 checkpoint as a HF/ModelOpt or
TRT‑LLM artifact.
