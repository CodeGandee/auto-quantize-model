Short answer: **yes, this record is technically very reasonable**, and it lines up with what I can see in upstream ModelOpt + vLLM code and with current bug reports around FP8 / ModelOpt in vision-language models. I’ll call out a couple of small nuances you might optionally tighten, but the core story is solid.

---

## 1. Your high-level causal story checks out

### hf_ptq.py really is LM-only for Qwen2.5-VL

From the current `examples/llm_ptq/hf_ptq.py` in NVIDIA Model-Optimizer:

* After loading the model, they compute `model_type = get_model_type(model)`.
* For multimodal models that aren’t explicitly handled (`mllama`, `whisper`), they do:

```python
# We only quantize the language model for VLMs other than the type supported above.
language_model_lineage = get_language_model_from_vl(full_model)
if language_model_lineage is not None:
    language_model = language_model_lineage.pop(-1)
    ...
    # disable quantization on all modules not in language_model
    disabled_quant_cfg = {"quant_cfg": {"default": {"enable": False}}, "algorithm": "max"}
    ...
    mtq.quantize(module, disabled_quant_cfg, forward_loop=None)
    ...
    model = language_model
    model_type = get_model_type(model)
```

Then they build a *text* calibration dataloader with `get_dataset_dataloader(...)` for all non-`mllama` / non-`whisper` types. ([GitHub][1])

So for **Qwen2.5-VL (`model_type == "qwen2_5_vl"`)**:

* Images *do not* enter the calibration path in `hf_ptq.py`.
* Only the extracted language model is actually quantized; the vision tower is explicitly disabled by the “disabled quant” pass.

That’s exactly what you say in §1 and §4: the “text-only FP8” checkpoint is LM-only FP8, vision left in bf16/fp16, even though the base model is multimodal.

Your reconstruction of the `mllama` / `whisper` / `else` branches for calibration is also accurate: they use `get_vlm_dataset_dataloader` only when `model_type == "mllama"`, `get_speech_dataset_dataloader` only for Whisper, and fall back to `get_dataset_dataloader` (text-only) for everything else. ([GitHub][1])

So the claim:

> **hf_ptq.py does not use image data when quantizing Qwen2.5-VL; it treats it like a standard LLM.

…is correct.

### The difference in quant scopes you show is plausible

You show two `hf_quant_config.json` snippets, one with:

```json
"exclude_modules": ["lm_head", "model.visual*"]
```

and another with only:

```json
"exclude_modules": ["lm_head"]
```

This is consistent with ModelOpt’s config structure and with vLLM’s `ModelOptQuantizationConfig`, which reads `hf_quant_config.json` and uses the `exclude_modules` list (after applying `hf_to_vllm_mapper`) to decide what to quantize. ([vLLM][2])

Given hf_ptq’s behaviour above, it’s totally reasonable that:

* The **LM-only** export you get from `hf_ptq.py` ends up with `model.visual*` excluded.
* Your **VLM FP8** custom script (built from `mtq.FP8_DEFAULT_CFG` + FP8 KV) ends up with no `model.visual*` exclusion, so both LM and vision get FP8 quantizers and extra scale tensors.

And yes, that **does** produce a different set of tensors in `model.safetensors` (more modules quantized, more `*_scale`/aux weights, and potentially slightly different naming).

That part of your explanation is solid.

---

## 2. vLLM really is tuned to the “canonical” ModelOpt LM-only layout

For Qwen2 / Qwen2-VL / Qwen2.5-VL, vLLM defines an `hf_to_vllm_mapper` like:

````python
hf_to_vllm_mapper = WeightsMapper(
    orig_to_new_prefix={
        "model.language_model.": "language_model.model.",
        "model.visual.": "visual.",
        "lm_head.": "language_model.lm_head.",
        "model.": "language_model.model.",
    }
)
``` :contentReference[oaicite:3]{index=3}  

And `ModelOptQuantizationConfig`:

- Detects `modelopt` from `hf_quant_config["quantization"]["quant_algo"]` (FP8 / NVFP4, etc.).
- Uses `hf_to_vllm_mapper` to rewrite `exclude_modules` and other patterns from HF names to vLLM’s internal module names. :contentReference[oaicite:4]{index=4}  

Combined with hf_ptq’s “only quantize the language model for VLMs” behavior, vLLM is implicitly assuming:

- **Scope**: FP8 applies to the *language_model* submodule only.
- **Layout**: the FP8 scales and related tensors follow the naming patterns produced by hf_ptq’s `build_quant_cfg` / `update_quant_cfg_with_kv_cache_quant`.

This isn’t just theoretical — there are multiple vLLM issues where ModelOpt FP8 checkpoints fail to load because expected scale tensors (e.g. `...k_scale`, `...v_scale`, or `...input_scale`) don’t match what vLLM expects for that model family. :contentReference[oaicite:5]{index=5}

There’s also a specific bug report that **Qwen2.5-VL cannot use fp8 quantization in vLLM**, even for official models, in some versions. :contentReference[oaicite:6]{index=6} And a very similar issue on vllm-ascend where a Qwen2.5-VL-based **FP8 vision-language model** (allenai/olmOCR-2-7B-1025-FP8) throws a KeyError on a *visual* weight name (`visual.blocks.0.attn.qkv.weight`) during loading. :contentReference[oaicite:7]{index=7}

So your statement:

> vLLM’s current ModelOpt integration knows how to read the **LM-only FP8** export from `hf_ptq.py`, but it does **not** know how to read our more aggressive VLM-quantized FP8 checkpoint.

…is exactly what we see in the wild.

Given that you’re seeing:

```text
KeyError: 'blocks.0.attn.k_proj.k_scale'
````

for your VLM-quantized FP8 checkpoint, it fits the same pattern as those upstream bugs: the loader is iterating through a hard-coded list of expected weights/scales for an FP8-ModelOpt checkpoint and fails when the checkpoint layout doesn’t match.

You don’t over-claim here; “hard-codes the expected names” is accurate for the current implementation.

---

## 3. Your explanation of *why* VLM FP8 breaks is reasonable

Your logic in §3 and §4 boils down to:

1. **LM-only FP8 path** = hf_ptq (LLM PTQ flow):

   * Uses text-only calibration for Qwen2.5-VL.
   * Calls `get_language_model_from_vl` and disables quantization on vision modules.
   * Produces a HF checkpoint whose FP8 aux tensors and naming match vLLM’s assumptions.

2. **VLM FP8 path** = custom MTQ config:

   * You start from `mtq.FP8_DEFAULT_CFG` + FP8 KV, no visual exclusion.
   * You calibrate on image+caption pairs.
   * You export with `export_hf_checkpoint` from a *different* quantized model and config.
   * Resulting `model.safetensors` has a different pattern of `*_scale` tensors and, in your case, is missing `blocks.0.attn.k_proj.k_scale`.

3. vLLM’s `qwen2_5_vl` implementation doesn’t dynamically infer the FP8 layout. It assumes **exactly** what hf_ptq produces for an LM-only FP8 ModelOpt checkpoint.

That matches what we know:

* ModelOpt quantization for VLMs is still moving fast and somewhat model-specific (see the NVIDIA Qwen2.5-VL FP8 / FP4 model cards and the “only official Qwen2.5-VL works” HF discussions). ([Hugging Face][3])
* vLLM’s ModelOpt integration is mainly documented and tested around Llama-family LLMs; the docs show examples for Llama-3.1 FP8, not VLM FP8. ([vLLM][4])

So it’s entirely believable that:

* LM-only FP8 → looks like the “canonical” pattern and works.
* VLM-wide FP8 → layout diverges enough to trip the loader.

---

## 4. Minor nits / possible refinements

These are small, optional tweaks if you want to be ultra-precise:

1. **Where you mention `build_quant_cfg` + `update_quant_cfg_with_kv_cache_quant`**

   * hf_ptq indeed uses something like `build_quant_cfg(...)` and then updates it with KV cache quantization when `kv_cache_qformat` is set. That’s accurate in spirit. ([GitHub][1])
   * You could explicitly mention that this is the path used when `--qformat fp8` and `--kv_cache_qformat fp8` in hf_ptq, just to tie it directly to the command line.

2. **“Different naming and layout for scale tensors”**

   * That’s very likely true, but you could soften it a bit to something like “may produce different naming and/or presence of scale tensors (e.g., `k_scale` vs `kv_scale`), which is enough to confuse vLLM’s loader.”
   * This lines up with an actual vLLM bug where `k_scale` / `v_scale` names had to be remapped to `kv_scale` for Qwen3-MOE FP8. ([GitHub][5])

3. **Scope it explicitly to your vLLM version**

   You say “vLLM 0.10.x” in the narrative; that’s consistent with the docs I’m seeing and with the fact that FP8/VLM support is still evolving. ([vLLM][6])
   You might add one sentence like:

   > “This behaviour is as of vLLM 0.10.x; future vLLM releases may add broader ModelOpt/VLM FP8 support.”

   just to time-box the observation.

But these are polish-level comments; nothing here contradicts the core conclusions.

---

## 5. Your recommended “practical compromise” is exactly what I’d suggest

* Using the **hf_ptq LM-only FP8 + FP8 KV** checkpoint as your **vLLM-facing artifact** is the path that aligns with both NVIDIA’s official Qwen2.5-VL FP8 checkpoints and vLLM’s current ModelOpt integration. ([Hugging Face][7])
* Keeping your **VLM-calibrated FP8** checkpoint as a **ModelOpt / TRT-LLM / HF-Transformers artifact** (for experiments or other runtimes) and *not* trying to feed it directly to vLLM until upstream adds support is the sane choice.

And your suggestion of “VLM-aware calibration but LM-only quantization” (i.e., using image-derived text prompts for calibration while still excluding `model.visual*`) is a nice, pragmatic compromise that keeps the vLLM-compatible layout intact.

---

### Verdict

If you paste this writeup into a repo as an internal design note / RCA, **I would consider it accurate and well-grounded given the current upstream code and issues.**

If you want, I can help you add a short “Future Work / Upstream TODOs” section (e.g. what changes would be needed in vLLM’s `qwen2_5_vl` loader + ModelOptQuantizationConfig to support full VLM FP8).

[1]: https://raw.githubusercontent.com/NVIDIA/Model-Optimizer/main/examples/llm_ptq/hf_ptq.py "raw.githubusercontent.com"
[2]: https://docs.vllm.ai/en/stable/api/vllm/model_executor/layers/quantization/modelopt/?utm_source=chatgpt.com "vllm.model_executor.layers.quantization.modelopt"
[3]: https://huggingface.co/nvidia/Qwen2.5-VL-7B-Instruct-FP4?utm_source=chatgpt.com "nvidia/Qwen2.5-VL-7B-Instruct-FP4"
[4]: https://docs.vllm.ai/en/latest/features/quantization/modelopt/?utm_source=chatgpt.com "NVIDIA Model Optimizer - vLLM"
[5]: https://github.com/vllm-project/vllm/issues/25047?utm_source=chatgpt.com "vLLM fails to load k_scale and v_scale for FP8 quantized ..."
[6]: https://docs.vllm.ai/en/stable/features/quantization/fp8/?utm_source=chatgpt.com "FP8 W8A8 - vLLM"
[7]: https://huggingface.co/nvidia/Qwen2.5-VL-7B-Instruct-FP8/discussions/2 "nvidia/Qwen2.5-VL-7B-Instruct-FP8 · How to quantize VL models to NVFP4"
