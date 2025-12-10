# Sprint: AutoQuant FP8 Qwen2.5-VL accuracy comparison

## Scope

This sprint tracks the work required to compare FP8 AutoQuant mixed-precision schemes for Qwen2.5-VL-3B-Instruct against the original FP16/BF16 model without relying on full-scale LLM benchmarks (e.g., MMLU). The focus is on small, reproducible metrics runs over COCO2017-derived subsets and per-layer sensitivity-aware schemes.

Related plan and tasks:
- Plan: `context/plans/plan-modelopt-autoquant-fp8-qwen2_5-vl-mixed-schemes.md`
- Working task: `context/tasks/working/modelopt-autoquant-fp8-qwen2_5-vl-mixed-schemes/`
- Subtask: `subtask-004-104-export-hf-checkpoints-per-scheme.md`

## Goals

- Produce quantitative comparisons between:
  - FP16/BF16 base Qwen2.5-VL-3B-Instruct.
  - All-layers AutoQuant FP8 schemes (e.g., top-10%, top-20%, …, top-100% of selected layers quantized by sensitivity).
  - Later: LM-only schemes (e.g., `fp8_autoquant_top10`…`fp8_autoquant_top100`).
- Use small COCO2017-derived subsets to measure:
  - Perplexity deltas.
  - Logit MSE / KL divergence vs FP16.
  - Optional small VLM image+text probes.
- Keep all scripts and artifacts discoverable under:
  - `models/qwen2_5_vl_3b_instruct/helpers/autoquant_eval/`
  - `tmp/modelopt-autoquant-fp8/`

## Tasks

### 1. Build small evaluation subsets

- [x] Sprint-001: Create a 100-sample COCO2017 VLM eval subset
  - Script: `models/qwen2_5_vl_3b_instruct/helpers/autoquant_eval/build_coco2017_vlm_eval_subset.py`
  - Input: `datasets/vlm-quantize-calib/coco2017_vlm_calib.db`
  - Output: `datasets/vlm-quantize-calib/coco2017_vlm_eval_100.jsonl` (image_relpath + caption pairs).
- [ ] Sprint-002: Define a small, fixed text-only eval subset (if needed) separate from calibration text for more robust perplexity comparisons.

### 2. Implement comparison scripts (FP16 vs quantized)

- [x] Sprint-003: Add an all-layers top-10% vs FP16 comparison script
  - Script: `models/qwen2_5_vl_3b_instruct/helpers/autoquant_eval/compare_qwen2_5_vl_3b_top10_vs_fp16.py`
  - Metrics (see `context/hints/about-evaluating-quantized-llms-without-full-benchmarks.md` for details):
    - Perplexity on `datasets/vlm-quantize-calib/coco2017_captions.txt` using TorchMetrics `Perplexity` (teacher-forced causal LM NLL → perplexity).
    - Last-token logit comparison: MSE and KL divergence `KL(p_fp16 || p_quant)` between FP16 and quantized next-token distributions.
  - Outputs:
    - JSON: `tmp/modelopt-autoquant-fp8/eval-all-layers-top10/metrics.json`
    - Markdown: `tmp/modelopt-autoquant-fp8/eval-all-layers-top10/summary.md`
- [ ] Sprint-004: Generalize the comparison script to accept arbitrary quantized scheme dirs (e.g., top-20%, top-50%, top-100%) so we can sweep coverage points without duplicating code, reusing the same metric set (TorchMetrics perplexity + logit MSE/KL) defined in the hint.
- [ ] Sprint-005: Add an image+text evaluation script that:
  - Reads `datasets/vlm-quantize-calib/coco2017_vlm_eval_100.jsonl`.
  - Builds VLM inputs (image + caption) via `qwen_vl_utils.process_vision_info` and `AutoProcessor`.
  - Computes logit-level metrics or simple response similarity metrics for a small set of prompts, following the same principles as the text-only metrics in the hint (e.g., distribution KL or embedding similarity).

### 3. LM-only schemes (later)

- [ ] Sprint-006: Run an LM-only full-sensitivity AutoQuant baseline and export LM-only top-X% schemes (as per Subtask 4.4 Stage 2).
- [ ] Sprint-007: Add an LM-only comparison script mirroring Sprint-003:
  - FP16 vs LM-only `fp8_autoquant_top10` / `top50` / `top100`.
  - Perplexity and logit metrics on the same text eval subset, following the metric definitions in `context/hints/about-evaluating-quantized-llms-without-full-benchmarks.md` (TorchMetrics perplexity + logit MSE/KL).

### 4. Reporting and documentation

- [ ] Sprint-008: Add a short report under `models/qwen2_5_vl_3b_instruct/reports/` summarizing:
  - Perplexity/logit metric trends vs coverage for all-layers schemes.
  - Any qualitative observations from the VLM eval subset.
- [ ] Sprint-009: Cross-link this sprint and the evaluation helpers from:
  - `context/plans/plan-modelopt-autoquant-fp8-qwen2_5-vl-mixed-schemes.md`
  - Relevant KB hints under `context/hints/` (e.g., the quantized LLM evaluation hint).

## Notes

- All evaluation scripts should be run via the RTX 5090 vLLM Pixi environment:
  - `pixi run -e rtx5090-vllm python ...`
- The intent is to keep evaluation runs light-weight and repeatable, so they can be used interactively during scheme design without requiring large benchmark suites.
