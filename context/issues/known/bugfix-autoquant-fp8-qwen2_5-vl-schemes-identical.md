# Bug: AutoQuant FP8 all-layers `topXX` schemes share identical HF weights

## Summary

The Qwen2.5-VL-3B AutoQuant FP8 **all-layers** checkpoints exported for different coverage levels (`top10`, `top20`, …, `top100`) under:

- `models/qwen2_5_vl_3b_instruct/quantized/fp8_autoquant_all_layers_top*_coco2017`

appear to share **identical HF weights**, despite being labeled as different coverage fractions in their names and metadata. As a result:

- Text-only and VLM evaluation scripts report **identical metrics** (perplexity, logit MSE/KL) for all `topXX` schemes.
- HF-based evaluation currently measures “FP16 vs one FP8 all-layers scheme”, not a spectrum of distinct coverage points.

## Symptoms

- For all `fp8_autoquant_all_layers_topXX_coco2017` schemes, the VLM eval summaries look identical, for example:

  - `tmp/modelopt-autoquant-fp8/eval-all-layers-top10-vlm-gpu/summary.md`
  - `tmp/modelopt-autoquant-fp8/eval-all-layers-top50-vlm-gpu/summary.md`
  - `tmp/modelopt-autoquant-fp8/eval-all-layers-top100-vlm-gpu/summary.md`

  All show:

  - `Num samples (loaded): 100`
  - `Num batches (used): 100`
  - `VLM logit MSE (last token): 8.423863e+01`
  - `VLM logit KL (last token, KL(fp16 || quant)): 1.120675e+01`

- The same pattern holds for text-only metrics produced by:

  - `models/qwen2_5_vl_3b_instruct/helpers/autoquant_eval/compare_qwen2_5_vl_3b_schemes_vs_fp16.py`
  - Output: `tmp/modelopt-autoquant-fp8/eval-all-layers-schemes-top10-100/summary.md`

## Reproduction steps

1. Ensure Qwen2.5-VL-3B checkpoint and AutoQuant FP8 schemes exist:

   - `models/qwen2_5_vl_3b_instruct/checkpoints/Qwen2.5-VL-3B-Instruct`
   - `models/qwen2_5_vl_3b_instruct/quantized/fp8_autoquant_all_layers_top*_coco2017`

2. Run the combined eval wrapper (RTX 5090 vLLM env):

   ```bash
   pixi run -e rtx5090-vllm bash \
     models/qwen2_5_vl_3b_instruct/helpers/autoquant_eval/run_all_autoquant_fp8_schemes_eval.sh
   ```

3. Inspect VLM summaries, for example:

   ```bash
   sed -n '10,40p' \
     tmp/modelopt-autoquant-fp8/eval-all-layers-top10-vlm-gpu/summary.md

   sed -n '10,40p' \
     tmp/modelopt-autoquant-fp8/eval-all-layers-top100-vlm-gpu/summary.md
   ```

   Metrics will be identical across all `topXX` runs.

4. Compare weights between different `topXX` checkpoints:

   ```bash
   python - << 'PY'
   from pathlib import Path
   import torch
   from transformers import Qwen2_5_VLForConditionalGeneration

   root = Path("models") / "qwen2_5_vl_3b_instruct" / "quantized"
   paths = {
       "top10": root / "fp8_autoquant_all_layers_top10_coco2017",
       "top50": root / "fp8_autoquant_all_layers_top50_coco2017",
       "top100": root / "fp8_autoquant_all_layers_top100_coco2017",
   }

   models = {name: Qwen2_5_VLForConditionalGeneration.from_pretrained(str(p))
             for name, p in paths.items()}

   layer_name = "model.language_model.layers.0.mlp.gate_proj.weight"
   w10 = dict(models["top10"].named_parameters())[layer_name]
   w50 = dict(models["top50"].named_parameters())[layer_name]
   w100 = dict(models["top100"].named_parameters())[layer_name]

   print("dtypes:", w10.dtype, w50.dtype, w100.dtype)
   with torch.no_grad():
       d10_50 = (w10.float() - w50.float()).abs().max().item()
       d10_100 = (w10.float() - w100.float()).abs().max().item()
       d50_100 = (w50.float() - w100.float()).abs().max().item()
   print("max|top10-top50|:", d10_50)
   print("max|top10-top100|:", d10_100)
   print("max|top50-top100|:", d50_100)
   PY
   ```

   Expected output (as observed):

   - All dtypes: `torch.float8_e4m3fn`
   - `max|top10 - top50| = 0.0`
   - `max|top10 - top100| = 0.0`
   - `max|top50 - top100| = 0.0`

   This shows the FP8 weights are **bit-identical** across these `topXX` schemes.

5. Compare base vs one quantized scheme (to confirm they *are* quantized w.r.t. FP16):

   ```bash
   python - << 'PY'
   from pathlib import Path
   import torch
   from transformers import Qwen2_5_VLForConditionalGeneration

   base_dir = Path("models") / "qwen2_5_vl_3b_instruct" / "checkpoints" / "Qwen2.5-VL-3B-Instruct"
   quant_dir = Path("models") / "qwen2_5_vl_3b_instruct" / "quantized" / "fp8_autoquant_all_layers_top10_coco2017"

   base = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(base_dir))
   quant = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(quant_dir))

   layer_name = "model.language_model.layers.0.mlp.gate_proj.weight"
   wb = dict(base.named_parameters())[layer_name]
   wq = dict(quant.named_parameters())[layer_name]

   print("dtypes:", wb.dtype, wq.dtype)
   with torch.no_grad():
       diff = (wb.float() - wq.float()).abs()
       print("max|base-top10|:", diff.max().item())
       print("mean|base-top10|:", diff.mean().item())
   PY
   ```

   Observed:

   - Base dtype: `float32`, quant dtype: `float8_e4m3fn`.
   - `max|base - top10| ≈ 4.48e2`, `mean|diff| ≈ 2.14e1`.  
   → Quantization is real vs base; the problem is **no difference between schemes**.

## Likely cause

- The ModelOpt AutoQuant pipeline (LM-only all-layers search) appears to be:
  - Producing a **single FP8 all-layers checkpoint**, and
  - Exporting / copying it under multiple scheme names:

    - `fp8_autoquant_all_layers_top10_coco2017`
    - `fp8_autoquant_all_layers_top20_coco2017`
    - …
    - `fp8_autoquant_all_layers_top100_coco2017`

- Per-scheme differences (coverage fractions, sensitivity scores, etc.) seem to be captured only in:

  - `hf_quant_config.json`
  - `layer-sensitivity/` manifests

  but not reflected in distinct HF weight tensors.

- Our HF evaluation scripts:

  - Only look at the actual `model-*.safetensors` parameters, not at `hf_quant_config.json` or `layer-sensitivity/` manifests.
  - Therefore, they see identical FP8 models and produce identical metrics.

## Impact

- The current “all-layers top-XX” evaluation (text-only and VLM) does **not** show a meaningful trade-off curve vs coverage:

  - All points correspond to the same FP8 all-layers checkpoint.
  - Metrics may be valid as a **single** all-layers FP8 baseline, but not as a sweep over top-10/20/…/100% coverage.

- Sprint tasks that expect:

  - Per-coverage perplexity and logit MSE/KL curves, and
  - Qualitative VLM differences between `top10` vs `top50` vs `top100`

  are currently blocked or misleading.

## Suggested next steps

1. **Confirm export behavior in the AutoQuant driver**  
   Check `scripts/qwen/qwen2_5_vl_3b_autoquant_fp8_schemes.py` and the slicer/export helper described in:

   - `context/tasks/working/modelopt-autoquant-fp8-qwen2_5-vl-mixed-schemes/subtask-004-104-export-hf-checkpoints-per-scheme.md`

   That subtask explicitly states that:

   - A **single all-layers baseline** `fp8_autoquant_all_layers_fp8` is run with full sensitivity analysis.
   - A slicer helper (e.g. `slice_qwen2_5_vl_3b_autoquant_all_layers_schemes.py`) derives `top10/20/…/100` schemes by:
     - Selecting least-sensitive X% layers from the baseline’s `sensitivity_ranking`.
     - Exporting scheme-specific HF checkpoint directories such as:
       - `fp8_autoquant_all_layers_top10_coco2017`
       - `fp8_autoquant_all_layers_top20_coco2017`
     - Making each directory “self-contained” (weights + `hf_quant_config.json` + `layer-sensitivity/` + per-scheme coverage manifest).

   Given that, areas to inspect closely are:

   - How many **distinct quantized models** ModelOpt actually returns:
     - Are we reusing the same `quantized_model` object and only updating metadata?
   - In the slicer helper:
     - Do we actually **apply** the per-scheme coverage manifest to the model before calling `export_hf_checkpoint`, or are we just:
       - Copying the baseline checkpoint directory, and
       - Dropping a different `hf_quant_config.json` / manifest into it?
     - Is there any code path that rewrites `model-*.safetensors` differently per scheme, or do all schemes copy the same safetensors from the baseline?

   - How per-scheme manifests are generated from the baseline `state_dict` / `layer_sensitivity`.
   - How (and whether) distinct FP8 weights are exported per scheme vs copying/cloning the same all-layers FP8 checkpoint.

2. **Decide on representation for coverage-specific schemes**:

   - Option A: Re-run AutoQuant and export separate HF checkpoints per coverage level, each with different FP8 weights.
   - Option B: Keep one FP8 “super” checkpoint plus scheme-specific manifests, and implement a loader/wrapper that:
     - Reads the manifest and selectively applies FP8 vs higher-precision weights per layer.
     - Exposes this as a scheme-specific HF model for evaluation.

3. **Update eval scripts once representation is fixed**:

   - Ensure `compare_qwen2_5_vl_3b_schemes_vs_fp16.py` and VLM eval helpers:
     - Either load distinct scheme checkpoints, or
     - Use a manifest-aware wrapper that configures the same base checkpoint differently per scheme.

4. **Add a regression check**:

   - A small test/util that, for any set of scheme dirs, verifies:

     ```python
     max_diff = max(
         (w_i.float() - w_j.float()).abs().max().item()
         for each pair of scheme weights
     )
     ```

   - And fails if `max_diff == 0.0` across all non-base schemes, catching this earlier.
