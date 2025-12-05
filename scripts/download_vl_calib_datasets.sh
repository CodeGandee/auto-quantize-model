#!/usr/bin/env bash
set -euo pipefail

# Download VL calibration datasets via OpenXLab / OpenDataLab.
# Targets:
#   - ScienceQA (images + questions)
#   - COCO Caption annotations (Karpathy splits)
#   - Flickr30k (if OpenXLab direct download is supported)

BASE_DIR="/data2/datasets"

echo "[INFO] Using base download directory: ${BASE_DIR}"
mkdir -p "${BASE_DIR}"

###############################################################################
# ScienceQA (OpenDataLab/ScienceQA)
###############################################################################

echo "[INFO] Downloading ScienceQA (OpenDataLab/ScienceQA) ..."
mkdir -p "${BASE_DIR}/scienceqa"
openxlab dataset download \
  --dataset-repo OpenDataLab/ScienceQA \
  --source-path "/raw/ScienceQA.tar.gz" \
  --target-path "${BASE_DIR}/scienceqa"

echo "[INFO] ScienceQA download step finished (check ${BASE_DIR}/scienceqa)."

###############################################################################
# COCO Caption annotations (OpenDataLab/COCOCaption)
###############################################################################

echo "[INFO] Downloading COCOCaption annotations (OpenDataLab/COCOCaption) ..."
mkdir -p "${BASE_DIR}/coco_caption"
openxlab dataset download \
  --dataset-repo OpenDataLab/COCOCaption \
  --source-path "/raw" \
  --target-path "${BASE_DIR}/coco_caption"

echo "[INFO] COCOCaption annotations download step finished (check ${BASE_DIR}/coco_caption)."

###############################################################################
# Flickr30k (OpenDataLab/Flickr30k) — may require manual download
###############################################################################

echo "[INFO] Attempting to download Flickr30k (OpenDataLab/Flickr30k) ..."
mkdir -p "${BASE_DIR}/flickr30k"
if ! openxlab dataset download \
  --dataset-repo OpenDataLab/Flickr30k \
  --source-path "/raw" \
  --target-path "${BASE_DIR}/flickr30k"; then
  echo "[WARN] Flickr30k direct download is not available via OpenXLab or download failed." >&2
  echo "[WARN] Please download Flickr30k manually from its homepage:" >&2
  echo "[WARN]   https://shannon.cs.illinois.edu/DenotationGraph/" >&2
  echo "[WARN] and place the data under: ${BASE_DIR}/flickr30k" >&2
fi

echo "[INFO] All download steps completed (with possible warnings above)."

