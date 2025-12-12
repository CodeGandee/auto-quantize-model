# Model Assets

## HEADER
- **Purpose**: Central index for model-specific assets and bootstraps
- **Status**: Active
- **Date**: 2025-12-05
- **Dependencies**: Git, curl or wget, internet access
- **Target**: AI assistants and developers

## Layout

This directory groups model families used by the project. Each subdirectory owns its own bootstrap script, source checkout, and checkpoints, or contains symlinks to external storage:

- `yolo11/` — Ultralytics YOLO11:
  - See `models/yolo11/README.md` for details.
  - Bootstrap script: `./models/yolo11/bootstrap.sh`
- `yolo10/` — YOLOv10:
  - See `models/yolo10/README.md` for details.
  - Bootstrap script: `./models/yolo10/bootstrap.sh`
- `qwen2_5_vl_3b_instruct/` — Qwen2.5-VL 3B Instruct (HF snapshot, external weights):
  - See `models/qwen2_5_vl_3b_instruct/README.md` for details.
  - Bootstrap script: `./models/qwen2_5_vl_3b_instruct/bootstrap.sh`
  - Contains `checkpoints/Qwen2.5-VL-3B-Instruct` symlink pointing to a local HF snapshot (e.g. `/data2/llm-models/Qwen2.5-VL-3B-Instruct`), ignored by Git.
- `qwen3_vl_4b_instruct/` — Qwen3-VL 4B Instruct (HF snapshot, external weights):
  - Tracks a local HF snapshot via `checkpoints/Qwen3-VL-4B-Instruct` symlink pointing to a host-specific directory (e.g. `/workspace/llm-models/Qwen3-VL-4B-Instruct`), ignored by Git.

All large artifacts (source clones, checkpoints, temporary files, and external HF snapshots) are managed per-model and are not committed to the repository. Use the per-model README files and local symlink conventions for exact setup and bootstrap instructions.

## Quantization sensitivity helpers

For ModelOpt AutoQuant per-layer sensitivity analysis on the Qwen VLMs:

- Qwen2.5-VL-3B LM-only FP8/INT8: `models/qwen2_5_vl_3b_instruct/helpers/qwen2_5_vl_3b_autoquant_fp8_schemes.py`
- Qwen3-VL-4B all-layers FP8/INT8: `models/qwen3_vl_4b_instruct/helpers/qwen3_vl_4b_autoquant_all_layers/run_qwen3_vl_4b_autoquant_all_layers.py`
- Qwen3-VL-4B LM-only INT8: `models/qwen3_vl_4b_instruct/helpers/qwen3_vl_4b_autoquant_int8_lm/run_qwen3_vl_4b_autoquant_int8_lm.py`

See the per-model READMEs for example commands and expected output layout.
