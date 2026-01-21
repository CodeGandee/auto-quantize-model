# auto-quantize-model

A project to test and evaluate different DNN model quantization tools and techniques.

## Overview

This repository serves as a testing ground for various neural network quantization methods, including:

- **QAT (Quantization-Aware Training)**: Training models with quantization in mind
- **PTQ (Post-Training Quantization)**: Quantizing pre-trained models
- **Automatic Mixed-Precision Scheme Selection**: Intelligently selecting precision for different layers
- **LLM Quantization**: Weight and activation quantization (WxAy) for large language models

This repo currently focuses on two concrete tracks:

- **YOLOv10 W4A16 QAT stability validation** (EMA + post-hoc Quantization Correction)
- **ModelOpt AutoQuant layer sensitivity** for VLM/LLM-style models (Qwen3-VL, YOLOv10m Torch→ONNX proxy)

## Purpose

The goal is to compare and benchmark different quantization approaches to understand their trade-offs in terms of:
- Model accuracy
- Inference speed
- Memory footprint
- Ease of implementation

## Getting Started

### 1) Environment

This repo is Pixi-managed.

```bash
pixi install
```

### 2) Docs

```bash
pixi run mkdocs serve
```

### 3) Run an end-to-end layer sensitivity smoke test (Qwen3-VL-4B)

This tutorial pack is self-contained (it generates a tiny synthetic COCO-like input) and verifies results against a tracked expected report:

```bash
bash docs/tutorial/howto/tut-qwen3-vl-4b-layer-sensitivity/run_demo.sh
```

## What’s Going On Under the Hood

### QAT: YOLOv10 W4A16 stability (EMA + QC)

This track validates whether **EMA + post-hoc Quantization Correction (QC)** mitigates the “early peak then collapse” instability observed in low-bit YOLO QAT.

- **Model family**: YOLOv10 (`models/yolo10/`)
- **Quantization regime**: W4A16 (weight-only int4, higher-precision activations during training)
- **Method variants**:
  - baseline QAT
  - EMA-stabilized QAT
  - EMA + QC (1-epoch post-hoc correction stage)
- **Outputs**: Each run writes a machine-readable `run_summary.json` plus a human-readable `summary.md`, designed for side-by-side comparison.

Entry points:

- Runner: `scripts/cv-models/run_yolov10_w4a16_qat_validation.py`
- Summarizer: `scripts/cv-models/summarize_yolov10_w4a16_qat_validation.py`
- Core logic: `src/auto_quantize_model/cv_models/yolov10_w4a16_validation.py`
- Manual runbook: `tests/manual/yolov10_w4a16_ema_qc_validation/README.md` (uses Pixi `cu128`)

Design notes and rationale live in:

- `specs/001-yolov10-qat-validation/research.md`

### Layer sensitivity: ModelOpt AutoQuant (Qwen3-VL)

Layer sensitivity here means: run **NVIDIA ModelOpt AutoQuant** over a model with a given calibration dataset and quantization format(s), then record how sensitive each layer is under the search/scoring procedure.

Key ideas:

- A run selects a **model**, a **calibration dataset**, and a **quantization format** (e.g., INT8/FP8 variants).
- AutoQuant produces:
  - `*_quant_manifest.json` (raw results + per-layer candidate stats)
  - `layer-sensitivity-report.{md,json}` (derived reports)
  - `*_autoquant_state.pt` (raw state, typically not committed)
  - `composed-config.yaml` (reproducibility)

Entry points:

- Tutorial pack (recommended starting point): `docs/tutorial/howto/tut-qwen3-vl-4b-layer-sensitivity/`
- All-layers driver (vision + text): `models/qwen3_vl_4b_instruct/helpers/qwen3_vl_4b_autoquant_all_layers/run_qwen3_vl_4b_autoquant_all_layers.py`
- LM-only driver (text tower): `models/qwen3_vl_4b_instruct/helpers/qwen3_vl_4b_autoquant_int8_lm/run_qwen3_vl_4b_autoquant_int8_lm.py`
- Hydra runner (LM-only sweeps): `scripts/qwen/qwen3_lm_sensitivity.py`
- Core helpers:
  - `src/auto_quantize_model/modelopt_autoquant.py` (manifest + report writers)
  - `src/auto_quantize_model/modelopt_configs.py` (custom quantization configs)
  - `src/auto_quantize_model/qwen/autoquant_sensitivity.py` (Qwen calibration loaders + LM extraction + AutoQuant wrapper)

## Calibration datasets (COCO captions)

Our per-layer quantization sensitivity runs for Qwen (LM-only) use a **text-only**
calibration dataset built from **COCO 2017 captions**.

- **Captions files (text-only):** `datasets/vlm-quantize-calib/coco2017_captions_{small,medium,large}.txt`
- **Hydra config (defaults):** `conf/dataset/vlm_coco2017_captions.yaml` (sets `dataset.root`, `dataset.size`, `dataset.captions_path`)
- **Used by the runner:** `scripts/qwen/qwen3_lm_sensitivity.py` → `src/auto_quantize_model/qwen/autoquant_sensitivity.py` (`CocoCaptionsDataset`)
- **How the subset is built:** `scripts/build_vlm_quantize_calib_coco2017_db.py` (also writes `datasets/vlm-quantize-calib/coco2017_vlm_calib*.db`)
  - See `datasets/vlm-quantize-calib/README.md` for details and rebuild commands.

For **all-layers** Qwen3-VL runs (vision + text), we also use the matching small/medium/large **VLM SQLite DBs** under `datasets/vlm-quantize-calib/` (image relpaths + captions).

## Repo layout (high level)

- `src/auto_quantize_model/`: reusable quantization + reporting helpers
- `scripts/`: runnable experiment entrypoints (Hydra and CLIs)
- `conf/`: Hydra/OmegaConf configs
- `models/`: model-family assets, bootstraps, helpers, and curated reports (external checkpoints are linked, not committed)
- `tests/`: unit/integration/manual tests

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
