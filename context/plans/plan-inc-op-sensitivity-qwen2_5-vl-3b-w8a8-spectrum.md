# Plan: INC Op Sensitivity and Quantization Spectrum for Qwen2.5-VL-3B-Instruct

## HEADER
- **Purpose**: Use Intel Neural Compressor to analyze per-layer sensitivity of Qwen2.5-VL-3B-Instruct and derive a family of quantized models ranging from fp16/bf16 baseline to increasingly aggressive W8A8 INT8 configurations with different numbers of quantized layers.
- **Status**: Draft
- **Date**: 2025-12-08
- **Dependencies**:
  - `models/qwen2_5_vl_3b_instruct/` (baseline checkpoint bootstrap + quantized variants)
  - `scripts/qwen/run_qwen2_5_vl_3b_sanity.py`
  - `context/plans/plan-quantize-qwen2_5-vl-3b-w8a8-modelopt.md`
  - `extern/neural-compressor/` (INC source and docs)
  - `context/summaries/inc-kb/intro-inc-w8a8-quantization-fp8-int8.md`
  - `context/summaries/inc-kb/howto-inc-layer-sensitivity-for-mixed-precision.md`
- **Target**: Auto-quantize-model developers and AI assistants designing mixed-precision quantization schemes for Qwen2.5-VL-3B-Instruct and similar LLM/VLMs.

---

## 1. Purpose and Outcome

We want a repeatable workflow that uses Intel Neural Compressor (INC) to:
- Run **op/layer sensitivity analysis** on the Qwen2.5-VL-3B-Instruct model (or its language-model submodule), using MSE-based and/or Hessian-based strategies.
- Translate the sensitivity information into several **quantization configurations** that span a spectrum:
  - **Min quantization**: fp16/bf16 baseline (no int8), plus possibly a “light quantization” variant where only the least sensitive layers are W8A8.
  - **Intermediate**: multiple W8A8 INT8 variants with increasing numbers of quantized layers, guided by sensitivity (e.g., bottom 25%, 50%, 75% of layers by sensitivity).
  - **Max quantization**: most aggressive W8A8 INT8 configuration that still meets a specified accuracy / quality budget, optionally close to full W8A8.
- Evaluate and compare these checkpoints against:
  - A small but meaningful **sanity/quality suite** (text-only prompts, image+text prompts).
  - Simple resource metrics (VRAM usage, latency per token or per sample where feasible).

The outcome should be:
- A set of documented INC configs and scripts to:
  - Run sensitivity analysis on Qwen2.5-VL-3B-Instruct.
  - Generate multiple quantized HF-style checkpoints along the fp16→W8A8 spectrum.
- Notes on how to reuse this analysis for other LLMs (and potentially non-LLMs such as ViT) in this repo.

This complements the existing ModelOpt-based W8A8 pipeline by giving us an INC-based, sensitivity-driven view of where quantization hurts and where we can be more aggressive.

## 2. Implementation Approach

### 2.1 High-level flow

1. **Baseline and environment validation**
   - Confirm the bf16/fp16 Qwen2.5-VL-3B-Instruct checkpoint is available under `models/qwen2_5_vl_3b_instruct/checkpoints/Qwen2.5-VL-3B-Instruct`.
   - Verify INC is installed in the `rtx5090` Pixi environment and can import successfully.
   - Run `scripts/qwen/run_qwen2_5_vl_3b_sanity.py` against the baseline model to establish reference behavior and outputs.

2. **Decide on the INC integration path for Qwen2.5-VL-3B**
   - Choose whether to:
     - Work directly with the HF `Qwen2_5_VLForConditionalGeneration` PyTorch model using INC’s 3.x PyTorch APIs, or
     - Export a language-model-only ONNX graph and use INC’s ONNX Runtime adaptor for sensitivity and int8 QDQ generation.
   - Based on that choice, define how to:
     - Isolate the **language-model component** (and optionally ignore or keep the vision encoder fp16).
     - Wrap the model in an INC `Model` object and provide calibration + evaluation hooks.

3. **Design calibration and evaluation for INC**
   - Reuse the existing COCO2017 caption text (and optionally images) as calibration inputs:
     - For PyTorch: implement a `DataLoader` that yields representative text prompts (and possibly image+text pairs if we include vision).
     - For ONNX: build a calibration dataloader that matches the exported graph inputs (token IDs, attention masks, etc.).
   - Define a lightweight evaluation function for INC:
     - For LLM/VLM: compute a simple metric such as average log-likelihood on a small text set, or compare perplexity surrogates on a held-out subset.
     - Ensure evaluation is fast enough to run many times during tuning.

4. **Implement MSE-based op sensitivity analysis (MSE_V2)**
   - Configure `PostTrainingQuantConfig` with:
     - `quant_level=1`, `tuning_criterion.strategy="mse_v2"`, and a reasonable `confidence_batches` (e.g., 2–4 for LLM).
     - `op_type_dict` / `op_name_dict` constraints to:
       - Force critical ops (embeddings, final LM head, LayerNorm, Softmax, vision blocks) to stay in higher precision.
       - Allow attention and MLP blocks to be candidates for INT8.
   - Run a **pure sensitivity analysis pass**:
     - Either run INC’s tuning process and capture the MSE-based fallback logs, or
     - Directly call `adaptor.calculate_op_sensitivity(...)` on a constructed `tune_cfg` to obtain a ranked list of ops by MSE.
   - Persist the sensitivity ranking (op names, types, scores) to a JSON/Markdown artifact under `datasets/quantize-calib` or `context/summaries`.

5. **(Optional) Implement Hessian-based sensitivity analysis (HAWQ_V2)**
   - Configure `TuningCriterion(strategy="hawq_v2", strategy_kwargs={"hawq_v2_loss": loss_fn})` for the PyTorch model.
   - Run a short HAWQ_V2 session on a small calibration subset to obtain:
     - `op_to_traces` (Hessian trace per op) via `calculate_hessian_trace`.
   - Compare Hessian rankings with MSE rankings to validate consistency, and record any notable discrepancies.

6. **Define quantization spectrum (configs) from sensitivity**
   - Based on the ranked op list, choose several quantization cut points, for example:
     - **Variant A (minimal)**: all ops left in fp16/bf16, or only the least sensitive 10–20% of ops quantized to W8A8.
     - **Variant B (moderate)**: quantize the bottom 40–50% of ops by sensitivity score.
     - **Variant C (aggressive)**: quantize the bottom 70–80% of ops, leaving only the top ~20–30% in higher precision.
     - **Variant D (max W8A8)**: attempt near-full W8A8 (except clearly fragile ops such as embeddings/LayerNorm/Softmax).
   - For each variant:
     - Generate a specific `PostTrainingQuantConfig` / `tune_cfg` that encodes which ops are INT8 vs fp16/bf16.
     - Document the mapping (which layer groups or blocks are quantized) in a small summary file.

7. **Run INC PTQ to produce quantized checkpoints for each variant**
   - For each configuration (A–D):
     - Run INC PTQ (PyTorch or ONNX) with the fixed per-op precision assignments, skipping extra tuning if needed to save time.
     - Export the quantized model to a well-defined directory under `models/qwen2_5_vl_3b_instruct/quantized/inc/`, e.g.:
       - `models/qwen2_5_vl_3b_instruct/quantized/inc/fp16_baseline/`
       - `models/qwen2_5_vl_3b_instruct/quantized/inc/w8a8_mild/`
       - `models/qwen2_5_vl_3b_instruct/quantized/inc/w8a8_medium/`
       - `models/qwen2_5_vl_3b_instruct/quantized/inc/w8a8_aggressive/`
   - If ONNX-based, produce QDQ ONNX models; if PyTorch-based, save quantized HF-style checkpoints that `transformers` can load.

8. **Evaluate and compare quantized variants**
   - Extend `scripts/qwen/run_qwen2_5_vl_3b_sanity.py` (or add a sibling script) to:
     - Accept a model directory argument for each variant.
     - Run a fixed set of text-only and image+text prompts for all variants and the baseline.
   - For each variant:
     - Record qualitative outputs and any obvious regressions.
     - Optionally, log simple metrics (e.g., average log-prob on short text tasks) and rough latency/VRAM measurements.
   - Summarize trade-offs (accuracy/quality vs quantization level) in a Markdown note under `context/summaries/inc-kb/`.

9. **Generalize and document for other models (e.g., ViT)**
   - Write a short guide describing how to:
     - Reuse the same INC sensitivity pipeline on non-LLM models in this repo (e.g., ViT-like models or YOLO/ONNX flows).
     - Adjust `op_type_dict` and evaluation functions appropriately.
   - Link this guide from `context/summaries/inc-kb/howto-inc-layer-sensitivity-for-mixed-precision.md` or a sibling hint.

### 2.2 Sequence diagram (steady-state usage)

```mermaid
sequenceDiagram
    participant Dev as Developer
    participant Pixi as Pixi RTX5090 Env
    participant INC as Intel Neural Compressor
    participant Qwen as Qwen2.5-VL-3B Model
    participant Calib as Calib/Eval Data
    participant Repo as Repo Checkpoints & Summaries

    Dev->>Pixi: pixi run -e rtx5090 python inc_sensitivity_qwen.py --mode=analyze
    Pixi->>Qwen: Load bf16/fp16 Qwen2.5-VL-3B checkpoint
    Pixi->>INC: Build PostTrainingQuantConfig (mse_v2 / hawq_v2)
    INC->>Calib: Iterate over calibration batches
    INC-->>Pixi: Return per-op sensitivity rankings (MSE / Hessian trace)
    Pixi->>Repo: Save rankings to context/summaries/inc-kb and/or datasets/quantize-calib

    Dev->>Pixi: pixi run -e rtx5090 python inc_sensitivity_qwen.py --mode=quantize --profile=medium
    Pixi->>INC: Construct quant config using sensitivity-based layer selection
    INC->>Qwen: Run PTQ to produce W8A8 variant (medium)
    INC->>Repo: Export quantized checkpoint under models/qwen2_5_vl_3b_instruct/quantized/inc/...

    Dev->>Pixi: pixi run -e rtx5090 python scripts/qwen/run_qwen2_5_vl_3b_sanity.py --model-dir <quantized-path>
    Pixi->>Qwen: Load quantized variant and run text/image+text prompts
    Qwen-->>Repo: Write outputs and simple metrics
    Dev->>Repo: Compare baseline vs multiple quantized variants and choose appropriate trade-off
```

## 3. Files to Modify or Add

- **scripts/qwen/run_qwen2_5_vl_3b_sanity.py**
  - Ensure it can easily switch between baseline and INC-quantized checkpoints via `--model-dir` and optionally label outputs with the variant name.
- **scripts/qwen/inc_qwen2_5_vl_3b_sensitivity.py** (new)
  - Python driver to:
    - Load the Qwen2.5-VL model (or language-model submodule).
    - Build calibration/eval dataloaders.
    - Run INC MSE_V2 and optionally HAWQ_V2 sensitivity analysis.
    - Save per-op rankings and helper summaries.
- **scripts/qwen/inc_qwen2_5_vl_3b_quantize.py** (new)
  - Python driver to:
    - Read saved sensitivity rankings.
    - Construct multiple INC PTQ configs representing different quantization levels (A–D).
    - Run PTQ for each variant and export quantized models.
- **context/summaries/inc-kb/howto-inc-layer-sensitivity-for-mixed-precision.md**
  - Optionally update with a short section linking to Qwen-specific scripts and sample commands.
- **context/summaries/inc-kb/** (new artifacts)
  - Small JSON/Markdown files capturing sensitivity rankings and quantization profiles, e.g.:
    - `qwen2_5_vl_3b_mse_sensitivity.json`
    - `qwen2_5_vl_3b_hawq_trace.json`
    - `qwen2_5_vl_3b_quant_profiles.md`
- **models/qwen2_5_vl_3b_instruct/quantized/inc/** (new directories)
  - Export locations for INC-quantized variants (mild, medium, aggressive, etc.), keeping them separate from ModelOpt outputs.
- **datasets/vlm-quantize-calib/** (optional additions)
  - Any helper scripts or metadata needed to reuse COCO-based calibration for INC (e.g. cached tokenized prompts).

## 4. TODOs (Implementation Steps)

- [ ] **Verify baseline Qwen2.5-VL environment** Run the existing sanity script on the bf16/fp16 checkpoint under the `rtx5090` environment and capture baseline outputs/notes.
- [ ] **Choose INC integration path for Qwen** Decide between PyTorch-INC vs ONNX-INC for sensitivity and quantization, and document the choice in this plan.
- [ ] **Implement INC calibration/eval loaders for Qwen** Build calibration and evaluation dataloaders using COCO captions (and optionally images) that are compatible with the chosen INC adaptor.
- [ ] **Implement MSE_V2 sensitivity script** Add `inc_qwen2_5_vl_3b_sensitivity.py` to run INC with `strategy="mse_v2"` and persist an ordered list of ops with their sensitivity scores.
- [ ] **(Optional) Implement HAWQ_V2 sensitivity script** Extend the sensitivity driver to optionally run `strategy="hawq_v2"` with a suitable loss function and save Hessian trace rankings.
- [ ] **Define quantization profiles A–D from sensitivity** Design multiple quantization cut points (minimal → aggressive) and encode them into reusable INC config fragments or a small profile file.
- [ ] **Implement INC quantization driver** Add `inc_qwen2_5_vl_3b_quantize.py` to read sensitivity profiles, construct per-profile configs, run PTQ, and export quantized checkpoints under `models/qwen2_5_vl_3b_instruct/quantized/inc/`.
- [ ] **Extend sanity-check script for multiple variants** Update or wrap `run_qwen2_5_vl_3b_sanity.py` so it can easily evaluate baseline and all INC variants with consistent prompts and produce labeled outputs.
- [ ] **Compare variants and summarize trade-offs** Run the evaluation suite across all variants, record qualitative/quantitative differences, and write a concise summary in `context/summaries/inc-kb/qwen2_5_vl_3b_quant_profiles.md`.
- [ ] **Generalize guidance for other models** Add a short section or hint in the INC KB explaining how to adapt the Qwen sensitivity+quantization pipeline to non-LLM models (e.g., ViT-like architectures) in this repo.

