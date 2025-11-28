# YOLO11 Model Assets

## HEADER
- **Purpose**: Manage YOLO11 source and pretrained checkpoints for this project
- **Status**: Active
- **Date**: 2025-11-28
- **Dependencies**: Git, curl or wget, internet access
- **Target**: AI assistants and developers

## Content

This directory contains helper tooling and local assets for working with Ultralytics YOLO11 models:

- `bootstrap.sh` — bootstrap script that:
  - Clones the Ultralytics `ultralytics` repository into `src/` with `--depth 1`.
  - Downloads YOLO11 nano/small/medium/large/xlarge checkpoints (`yolo11n/s/m/l/x.pt`) into `checkpoints/`.
  - Ensures `src/`, `checkpoints/`, and `tmp/` are ignored by Git via `.gitignore`.
- `src/` — YOLO11 source code cloned from the Ultralytics GitHub repository (not committed to this repo).
- `checkpoints/` — downloaded YOLO11 pretrained weights (not committed to this repo).
- `tmp/` — optional temporary download/extraction area used by scripts; removed after successful bootstrap.

To (re)initialize the YOLO11 assets, run from the project root:

```bash
./models/yolo11/bootstrap.sh
```

