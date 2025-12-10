# Subtask 4.4: Export HF checkpoints per AutoQuant FP8 scheme

## Scope

Using the quantized models and manifests produced by the AutoQuant driver, export one Hugging Face-style checkpoint directory per scheme under `models/qwen2_5_vl_3b_instruct/quantized/`, following a consistent naming convention and ensuring that each checkpoint is self-contained and ready for downstream consumers (including vLLM and pure PyTorch-based experiments). This subtask is primarily about **artifact organization and comparability**, not about optimizing accuracy of any particular scheme.

## Preparation: Mixed-precision sensitivity inputs

Before exporting any HF checkpoints, we rely on the mixed-precision layer sensitivity information produced by the AutoQuant driver in Subtask 4.3.

**AutoQuant outputs and manifests**

- The AutoQuant driver `qwen2_5_vl_3b_autoquant_fp8_schemes.py` runs ModelOpt `mtq.auto_quantize` in a small number of **baseline modes**: an LM-only full-coverage run (all LM blocks eligible) and an all-layers full-coverage run using `FP8_ALL_LAYERS_CFG`. Each baseline produces a quantized model and a `state_dict` with full per-layer sensitivity information.
- The AutoQuant searcher records per-layer candidate recipes, sensitivity scores, and approximate cost/size metrics in its `state_dict` (e.g., `candidate_stats`, `best`, and constraint satisfaction fields).
- The driver uses this `state_dict` plus a walk over the quantized model (`named_modules()` + `is_quantized_linear`) to build a quantization manifest (JSON) with:
  - A `layer_sensitivity` section that maps each logical block (`...quant_recipe`) to candidate formats, scores, and costs.
  - A `sensitivity_ranking` list that sorts layers by importance (e.g., max sensitivity score) so that **top-X% quantized schemes can be derived later without rerunning AutoQuant**.
- A companion Markdown report (`per-layer-sensitivity.md`) summarizes this manifest as a per-layer sensitivity table, sorted by score and focused on FP8 vs BF16/FP16 decisions.

**Calibration data used for AutoQuant**

- LM-only full-sensitivity baseline (Stage A for `fp8_autoquant_topXX`):
  - Data: text-only COCO captions.
  - Source: `datasets/vlm-quantize-calib/coco2017_captions.txt`.
  - Shape: each sample is a single caption, tokenized to a maximum sequence length of 512 tokens.
  - Count: up to **4096** caption samples by default (configurable via the driver CLI).
  - All LM-only coverage schemes (`fp8_autoquant_top10`, `fp8_autoquant_top20`, …, `fp8_autoquant_top100`) are **derived from this single full-coverage AutoQuant run**; they do not each rerun AutoQuant with separate calibration.
- All-layers full-sensitivity baseline (`fp8_autoquant_all_layers_fp8`):
  - Data: multimodal COCO2017 samples (image + caption).
  - Sources: calibration DB `datasets/vlm-quantize-calib/coco2017_vlm_calib.db` and COCO2017 image root under `datasets/coco2017/source-data/`.
  - Shape: each sample is **one image + one caption** combined into a single multimodal prompt via the Qwen2.5-VL processor.
  - Count: up to **4096** calibration samples by default.

**Scheme naming and coverage intent**

- The LM-only family `fp8_autoquant_top10`, `fp8_autoquant_top20`, …, `fp8_autoquant_top100` encodes **coverage targets** based on AutoQuant sensitivity rankings over LM transformer blocks: for `fp8_autoquant_topXX`, we first run a full LM-only sensitivity analysis, then select the **top `XX%` most sensitive LM blocks** and keep their FP8 (quantized) recipes while reverting the remaining blocks to BF16/FP16.
- `fp8_autoquant_top100` corresponds to a “full LM coverage” regime (all LM blocks remain eligible for FP8 under the AutoQuant budget), while `fp8_autoquant_all_layers_fp8` represents an all-layers sensitivity analysis setting that uses a custom config to include non-LM components as well; from that analysis we can similarly derive all-layers schemes by quantizing the top-X% most sensitive layers across the full model, if desired.

These manifests and reports are the **inputs** to this export subtask: we treat them as the authoritative description of each scheme’s mixed precision pattern when deciding what to export and how to name the resulting checkpoints.

## Planned outputs

Stage 1: all-layers FP8 analysis (priority)

- One HF checkpoint directory for at least one **all-layers-derived scheme** under `models/qwen2_5_vl_3b_instruct/quantized/`, starting from the full-sensitivity `fp8_autoquant_all_layers_fp8` baseline (e.g., a full-coverage `fp8_autoquant_all_layers_fp8_coco2017` checkpoint, with optional future extensions to top-X% all-layers variants).
- A small helper function or script (possibly inside `qwen2_5_vl_3b_autoquant_fp8_schemes.py` or a sibling module) that wraps `export_hf_checkpoint` for this model and can **slice the all-layers sensitivity baseline** into a concrete quantization config (which layers remain FP8 vs BF16/FP16) before export.
- Basic validation that the exported all-layers checkpoint loads as an HF model and can run simple text (and optionally multimodal) inference.

Stage 2: LM-only scheme exports (later)

- HF checkpoint directories for the LM-only coverage schemes (e.g., `fp8_autoquant_top10`, `fp8_autoquant_top20`, …, `fp8_autoquant_top100`) under `models/qwen2_5_vl_3b_instruct/quantized/`, following the same naming convention (e.g., `fp8_autoquant_topXX_coco2017`), each derived by **slicing the LM-only full-sensitivity baseline** according to its target top-`XX%` coverage.
- Reuse or extend the Stage 1 export helper so it can operate on the LM-only sensitivity manifest: select the top-`XX%` most sensitive LM blocks from `sensitivity_ranking`, enable their FP8 quantizers, revert the rest to BF16/FP16, and then export.
- Optional sanity checks that LM-only checkpoints load and run basic text generation; these can wait until after all-layers analysis and export plumbing are in place.

## TODOs

Stage 1: all-layers FP8 analysis

1. **Obtain full-sensitivity analysis for all selected layers**
   - [ ] Job-004-104-001 Run the all-layers AutoQuant baseline (`fp8_autoquant_all_layers_fp8`) so that **all model layers selected by the FP8_ALL_LAYERS_CFG config** participate in the search, and write a self-contained manifest + state + Markdown report under a stable `tmp/` subdirectory and the baseline checkpoint’s `layer-sensitivity/` folder.
   - [ ] Job-004-104-002 Verify that the baseline manifest exposes a usable `sensitivity_ranking` over all selected layers (LM + vision + any other quantizable blocks), and that the exported HF checkpoint (`fp8_autoquant_all_layers_fp8_coco2017`) is loadable.
2. **Derive and export top-X% quantized schemes from the all-layers baseline**
   - [ ] Job-004-104-003 Implement a slicer/helper that, given the all-layers baseline manifest, **quantizes layers in order of increasing sensitivity**: for each coverage point (10%, 20%, 30%, …, 100%), select the 10/20/…/100% of selected layers with the **lowest** sensitivity scores to keep in FP8, and treat the remaining layers as BF16/FP16.
   - [ ] Job-004-104-004 For each chosen coverage point, apply this slicing rule to construct a concrete quantization config (or equivalent ModelOpt representation) and export a scheme-specific HF checkpoint directory (e.g., `fp8_autoquant_all_layers_top10_coco2017`, `fp8_autoquant_all_layers_top20_coco2017`), making each directory self-contained (weights + hf_quant_config + layer-sensitivity artifacts + per-scheme coverage manifest).
   - [ ] Job-004-104-005 Implement light sanity checks (e.g., short text-only and/or multimodal generation) for a small subset of all-layers schemes (such as 10%, 50%, and 100% coverage) to confirm that models load and run end-to-end under PyTorch/Transformers, even if quality is not yet evaluated.

Stage 2: LM-only schemes (deferred)

1. **Obtain full-sensitivity analysis for LM-selected layers only**
   - [ ] Job-004-104-006 Run an LM-only AutoQuant baseline (e.g., `fp8_autoquant_top100` or a dedicated LM-only full-coverage scheme) so that **all LM blocks selected by ModelOpt’s default FP8 config and disabled-layer patterns** participate in the search, and write the manifest + state + Markdown report under a stable `tmp/` subdirectory and a corresponding LM-only baseline checkpoint’s `layer-sensitivity/` folder.
   - [ ] Job-004-104-007 Verify that the LM-only manifest exposes a usable `sensitivity_ranking` over the selected LM layers (excluding vision and other components) and that the LM-only baseline checkpoint is loadable.
2. **Derive and export top-X% LM-only quantized schemes**
   - [ ] Job-004-104-008 Extend the slicer/helper so it can operate on the LM-only sensitivity baseline: for each LM-only scheme (`fp8_autoquant_top10`, `fp8_autoquant_top20`, …, `fp8_autoquant_top100`), select the 10/20/…/100% of selected LM layers with the **lowest** sensitivity scores to keep in FP8 and treat the remaining LM layers as BF16/FP16.
   - [ ] Job-004-104-009 For each chosen LM-only coverage point, apply this slicing rule to construct the corresponding quantization config and export an HF checkpoint directory (e.g., `fp8_autoquant_top10_coco2017`, `fp8_autoquant_top20_coco2017`), again ensuring that each directory is self-contained and includes both the LM-only layer-sensitivity artifacts and a per-scheme coverage manifest.
   - [ ] Job-004-104-010 Implement light text-only sanity checks for a few LM-only schemes (e.g., 10%, 50%, and 100% coverage) to confirm that they run in PyTorch/Transformers and provide basic outputs suitable for later quality analysis.

## Notes

- For LM-only coverage schemes, exports remain focused on the language model component and keep the vision tower in BF16/FP16, consistent with the LM-only AutoQuant setup; all-layers variants explicitly relax this constraint and may quantize non-LM modules as part of the analysis.
- Treat these checkpoints as **experimental artifacts** for comparing schemes and configs; it is acceptable if some exported models have poor quality, as long as they are loadable for analysis.
