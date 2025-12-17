# auto-quantize-model

A project to test and evaluate different DNN model quantization tools and techniques.

## Overview

This repository serves as a testing ground for various neural network quantization methods, including:

- **QAT (Quantization-Aware Training)**: Training models with quantization in mind
- **PTQ (Post-Training Quantization)**: Quantizing pre-trained models
- **Automatic Mixed-Precision Scheme Selection**: Intelligently selecting precision for different layers
- **LLM Quantization**: Weight and activation quantization (WxAy) for large language models

## Purpose

The goal is to compare and benchmark different quantization approaches to understand their trade-offs in terms of:
- Model accuracy
- Inference speed
- Memory footprint
- Ease of implementation

## Getting Started

Coming soon...

## Calibration datasets (COCO captions)

Our per-layer quantization sensitivity runs for Qwen (LM-only) use a **text-only**
calibration dataset built from **COCO 2017 captions**.

- **Captions files (text-only):** `datasets/vlm-quantize-calib/coco2017_captions_{small,medium,large}.txt`
- **Hydra config (defaults):** `conf/dataset/vlm_coco2017_captions.yaml` (sets `dataset.root`, `dataset.size`, `dataset.captions_path`)
- **Used by the runner:** `scripts/qwen/qwen3_lm_sensitivity.py` → `src/auto_quantize_model/qwen/autoquant_sensitivity.py` (`CocoCaptionsDataset`)
- **How the subset is built:** `scripts/build_vlm_quantize_calib_coco2017_db.py` (also writes `datasets/vlm-quantize-calib/coco2017_vlm_calib*.db`)
  - See `datasets/vlm-quantize-calib/README.md` for details and rebuild commands.

## VS Code settings

If VS Code warns that it’s “unable to watch for file changes” (common on Linux
when this repo contains many vendored files under `extern/`), exclude the heavy
directories from file watching by adding this to `.vscode/settings.json`:

```json
{
	"files.watcherExclude": {
		"**/extern/**": true,
		"**/custom-build/**": true
	}
}
```

## License

TBD
