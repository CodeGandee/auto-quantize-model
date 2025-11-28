# YOLOv10 Model Assets

## HEADER
- **Purpose**: Manage YOLOv10 source and pretrained checkpoints for this project
- **Status**: Active
- **Date**: 2025-11-28
- **Dependencies**: Git, curl or wget, internet access
- **Target**: AI assistants and developers

## Content

This directory contains helper tooling and local assets for working with YOLOv10 models:

- `bootstrap.sh` — bootstrap script that:
  - Clones the official YOLOv10 repository into `src/` with `--depth 1`.
  - Downloads YOLOv10 nano/small/medium/base/large/xlarge checkpoints (`yolov10n/s/m/b/l/x.pt`) into `checkpoints/`.
  - Ensures `src/`, `checkpoints/`, and `tmp/` are ignored by Git via `.gitignore`.
- `src/` — YOLOv10 source code cloned from the YOLOv10 GitHub repository (not committed to this repo).
- `checkpoints/` — downloaded YOLOv10 pretrained weights (not committed to this repo).
- `tmp/` — optional temporary download/extraction area used by scripts; removed after successful bootstrap.

To (re)initialize the YOLOv10 assets, run from the project root:

```bash
./models/yolo10/bootstrap.sh
```

