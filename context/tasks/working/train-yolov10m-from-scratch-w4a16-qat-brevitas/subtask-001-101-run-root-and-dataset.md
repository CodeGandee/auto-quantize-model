# Subtask 1.1: Run-root + dataset conversion + provenance

## Scope

- Define a stable run-root layout under:
  - `tmp/yolov10m_scratch_fp16_vs_w4a16_qat_brevitas/<run-id>/`
- Create a **run-local YOLO-format COCO2017 dataset** without writing into the shared `datasets/coco2017/source-data/` tree:
  - Convert COCO JSON → YOLO labels under the run root.
  - Symlink images from `datasets/coco2017/source-data/{train2017,val2017}` into the run dataset root.
  - Ensure empty label files exist for images with no annotations.
- Write dataset YAML used by Ultralytics and record provenance (paths + counts) as JSON.

## Planned outputs

- `tmp/.../<run-id>/dataset/coco_yolo/`:
  - `images/train2017/` (symlinks)
  - `images/val2017/` (symlinks)
  - `labels/train2017/*.txt`
  - `labels/val2017/*.txt`
  - `coco2017-yolo.yaml` (Ultralytics dataset yaml)
- `tmp/.../<run-id>/dataset/provenance.json` with:
  - COCO root, annotation JSON paths, image counts, and conversion settings.

## TODOs

- [ ] Job-001-101-001 Define the run-root directory contract and subfolders (fp16/ qat-w4a16/ onnx/ eval/ logs/ summary/ dataset/).
- [ ] Job-001-101-002 Implement a helper that creates a run-local YOLO-format COCO dataset using Ultralytics `convert_coco`.
- [ ] Job-001-101-003 Add provenance + counts writer (JSON) under the run root.
- [ ] Job-001-101-004 Verify dataset YAML resolves correctly in a minimal Ultralytics dataloader smoke (no training).

## Notes

- Prefer deterministic behavior: fixed seed + sorted file ordering wherever we select subsets.
- Keep all generated dataset artifacts under `tmp/` (never commit).

