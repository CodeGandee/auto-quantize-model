# VLM Quantization Calibration Subset (COCO2017)

This directory stores lightweight metadata for a vision-language (VLM)
calibration subset built from the COCO 2017 captions dataset.

The goal is to provide a small, reusable list of `(image, caption)` pairs
for post-training quantization (PTQ) of VLMs such as Qwen2.5-VL, without
copying any image data into the repository.

## Contents

- `coco2017_vlm_calib.db` — SQLite database containing:
  - Table `vlm_calib_samples` with:
    - `split`: `train2017` or `val2017`.
    - `image_relpath`: image path relative to the COCO 2017 root
      (for this repo, typically `datasets/coco2017/source-data`).
    - `image_id`: COCO image id.
    - `caption_id`: COCO caption annotation id.
    - `caption`: the human-written caption text.
  - Table `meta` with small key/value metadata, including:
    - `source_dataset = "coco2017_captions"`.
    - `coco_root_relative = "../coco2017/source-data"`.
    - `num_samples`: number of rows in `vlm_calib_samples`.
    - `random_seed`: seed used when sampling.
- `coco2017_captions.txt` — newline-delimited captions corresponding to the
  selected subset, one caption per line, suitable for generic text-only PTQ
  loaders (e.g., ModelOpt's `get_dataset_dataloader` via the `text` dataset
  script).

No images are stored in this directory; all paths are references into the
external COCO 2017 dataset.

## Building the DB from COCO2017

Assuming `datasets/coco2017/source-data` is a symlink to your local COCO
2017 directory (see `datasets/coco2017/README.md`), you can rebuild the DB
with:

```bash
pixi run python scripts/build_vlm_quantize_calib_coco2017_db.py \
  --coco-root datasets/coco2017/source-data \
  --out datasets/vlm-quantize-calib/coco2017_vlm_calib.db \
  --max-samples 4096 \
  --captions-text-out datasets/vlm-quantize-calib/coco2017_captions.txt
```

You can adjust `--max-samples` and `--seed` to change the subset size and
sampling.
