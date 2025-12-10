# AutoQuant Evaluation Helpers for Qwen2.5-VL-3B-Instruct

This directory contains helper scripts for evaluating ModelOpt AutoQuant FP8 schemes for Qwen2.5-VL-3B-Instruct by comparing quantized checkpoints against the original FP16/BF16 Hugging Face checkpoint.

The focus is on lightweight metrics (perplexity and logit-level comparisons) over small COCO2017-derived subsets, not full benchmark suites like MMLU.

## Contents

- `build_coco2017_vlm_eval_subset.py`
  - Builds a small evaluation subset of COCO2017 image+caption pairs by sampling from the existing calibration database:
    - Input: `datasets/vlm-quantize-calib/coco2017_vlm_calib.db`
    - Output: `datasets/vlm-quantize-calib/coco2017_vlm_eval_100.jsonl`
  - Each JSONL row has:
    - `split`: COCO split name (`train2017` or `val2017`).
    - `image_relpath`: relative path to the image under `datasets/coco2017/source-data`.
    - `caption`: the associated caption text.

- `compare_qwen2_5_vl_3b_top10_vs_fp16.py`
  - Compares the FP16/BF16 base model against the 10%-split all-layers FP8 AutoQuant checkpoint using:
    - Perplexity (`torchmetrics.text.Perplexity`) on COCO2017 captions.
    - Logit MSE and KL divergence for the last-token distribution on the same inputs.
  - Defaults:
    - FP16 model: `models/qwen2_5_vl_3b_instruct/checkpoints/Qwen2.5-VL-3B-Instruct`
    - Quantized model: `models/qwen2_5_vl_3b_instruct/quantized/fp8_autoquant_all_layers_top10_coco2017`
    - Eval data (text-only): `datasets/vlm-quantize-calib/coco2017_captions.txt`
    - Output metrics:
      - JSON: `tmp/modelopt-autoquant-fp8/eval-all-layers-top10/metrics.json`
      - Markdown summary: `tmp/modelopt-autoquant-fp8/eval-all-layers-top10/summary.md`

- `text_eval_common.py`
  - Shared utilities for text-only evaluation:
    - Captions dataset + dataloader.
    - Perplexity computation using TorchMetrics.
    - Logit MSE / KL computation for last-token distributions.

- `compare_qwen2_5_vl_3b_schemes_vs_fp16.py`
  - Generalized text-only comparison script that evaluates multiple quantized schemes against the FP16/BF16 baseline on a shared caption set.
  - Reuses the same perplexity + logit MSE/KL metrics as the top-10 script.
  - Defaults:
    - FP16 model: `models/qwen2_5_vl_3b_instruct/checkpoints/Qwen2.5-VL-3B-Instruct`
    - Quantized schemes: `models/qwen2_5_vl_3b_instruct/quantized/fp8_autoquant_all_layers_top10_coco2017` (extend via `--quant-model-dirs`)
    - Eval data (text-only): `datasets/vlm-quantize-calib/coco2017_captions.txt`
    - Output metrics:
      - JSON: `tmp/modelopt-autoquant-fp8/eval-all-layers-schemes/metrics.json`
      - Markdown summary: `tmp/modelopt-autoquant-fp8/eval-all-layers-schemes/summary.md`

- `compare_qwen2_5_vl_3b_vlm_eval_top10_vs_fp16.py`
  - Image+text comparison script using the 100-sample COCO2017 VLM eval subset.
  - For each (image, caption) pair:
    - Builds chat-style VLM inputs via `qwen_vl_utils.process_vision_info` and `AutoProcessor`.
    - Runs both FP16 and quantized models and compares last-token logits.
  - Metrics:
    - Logit MSE (last token).
    - Logit KL divergence `KL(p_fp16 || p_quant)` on the last-token distribution.
  - Defaults:
    - FP16 model: `models/qwen2_5_vl_3b_instruct/checkpoints/Qwen2.5-VL-3B-Instruct`
    - Quantized model: `models/qwen2_5_vl_3b_instruct/quantized/fp8_autoquant_all_layers_top10_coco2017`
    - Eval subset (image+text): `datasets/vlm-quantize-calib/coco2017_vlm_eval_100.jsonl`
    - COCO root: `datasets/coco2017/source-data`
    - Output metrics:
      - JSON: `tmp/modelopt-autoquant-fp8/eval-all-layers-top10-vlm/metrics.json`
      - Markdown summary: `tmp/modelopt-autoquant-fp8/eval-all-layers-top10-vlm/summary.md`

## Usage examples

Build the 100-sample COCO2017 VLM eval subset:

```bash
pixi run -e rtx5090-vllm python \
  models/qwen2_5_vl_3b_instruct/helpers/autoquant_eval/build_coco2017_vlm_eval_subset.py \
  --num-samples 100
```

Run FP16 vs 10%-split quantized comparison on COCO captions:

```bash
pixi run -e rtx5090-vllm python \
  models/qwen2_5_vl_3b_instruct/helpers/autoquant_eval/compare_qwen2_5_vl_3b_top10_vs_fp16.py
```

Sweep multiple schemes (e.g., top-10/top-50/top-100) on COCO captions:

```bash
pixi run -e rtx5090-vllm python \
  models/qwen2_5_vl_3b_instruct/helpers/autoquant_eval/compare_qwen2_5_vl_3b_schemes_vs_fp16.py \
  --quant-model-dirs \
    models/qwen2_5_vl_3b_instruct/quantized/fp8_autoquant_all_layers_top10_coco2017 \
    models/qwen2_5_vl_3b_instruct/quantized/fp8_autoquant_all_layers_top50_coco2017 \
    models/qwen2_5_vl_3b_instruct/quantized/fp8_autoquant_all_layers_top100_coco2017 \
  --scheme-names top10 top50 top100
```

Run FP16 vs 10%-split quantized comparison on the VLM eval subset:

```bash
pixi run -e rtx5090-vllm python \
  models/qwen2_5_vl_3b_instruct/helpers/autoquant_eval/compare_qwen2_5_vl_3b_vlm_eval_top10_vs_fp16.py
```

The structure is intended to be extended with additional comparison scripts for other coverage points (e.g., top-20%, top-50%, LM-only schemes) and for VLM image+text evaluation using the JSONL subset built above.
