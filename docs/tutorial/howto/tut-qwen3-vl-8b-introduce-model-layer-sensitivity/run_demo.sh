#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"

SNAPSHOT=false
DEVICE="cuda:0"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--snapshot-report] [--device <torch-device>]

End-to-end tutorial pack runner for:
  - introducing Qwen3-VL-8B-Instruct into this workspace (via a local symlink), and
  - running ModelOpt AutoQuant per-layer sensitivity (all-layers INT8 + LM-only INT8).

Options:
  --snapshot-report    Overwrite expected_report/ with sanitized artifacts from this run.
  --device DEVICE      Torch device (default: cuda:0). Use 'cpu' only if you accept a very slow run.

Environment variables:
  HF_SNAPSHOTS_ROOT    Directory containing local model snapshots (default: /data1/huangzhe/llm-models).
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --snapshot-report)
      SNAPSHOT=true
      shift
      ;;
    --device)
      DEVICE="${2:-}"
      if [[ -z "${DEVICE}" ]]; then
        echo "[ERROR] --device requires an argument" >&2
        exit 2
      fi
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown arg: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

MODEL_DIR="$REPO_ROOT/models/qwen3_vl_8b_instruct/checkpoints/Qwen3-VL-8B-Instruct"
if [[ ! -d "$MODEL_DIR" ]]; then
  SNAPSHOTS_ROOT="${HF_SNAPSHOTS_ROOT:-/data1/huangzhe/llm-models}"
  CANDIDATE="${SNAPSHOTS_ROOT}/Qwen3-VL-8B-Instruct"
  if [[ -d "$CANDIDATE" ]]; then
    echo "[INFO] Creating local model link:"
    echo "  $MODEL_DIR -> $CANDIDATE"
    mkdir -p "$(dirname "$MODEL_DIR")"
    ln -s "$CANDIDATE" "$MODEL_DIR"
  fi
fi

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "[ERROR] Missing model directory: $MODEL_DIR" >&2
  echo "Hint: create a symlink like:" >&2
  echo "  MODELS_ROOT=/path/to/local/model-snapshots" >&2
  echo "  ln -s \"${MODELS_ROOT}/Qwen3-VL-8B-Instruct\" \\" >&2
  echo "    models/qwen3_vl_8b_instruct/checkpoints/Qwen3-VL-8B-Instruct" >&2
  echo "Or run: ./models/qwen3_vl_8b_instruct/bootstrap.sh --yes" >&2
  exit 1
fi

WORKSPACE_DIR="$REPO_ROOT/tmp/tutorial_workspace_qwen3_vl_8b_intro_layer_sensitivity_$(date +%s)"
mkdir -p "$WORKSPACE_DIR"

INPUTS_DIR="$SCRIPT_DIR/inputs"
CAPTIONS_TXT="$WORKSPACE_DIR/coco2017_captions_small.txt"
cp "$INPUTS_DIR/coco2017_captions_small.txt" "$CAPTIONS_TXT"

COCO_ROOT="$WORKSPACE_DIR/coco2017/source-data"
IMAGE_PATH="$COCO_ROOT/train2017/000000000001.jpg"
VLM_DB="$WORKSPACE_DIR/vlm_calib_small.db"

mkdir -p "$(dirname "$IMAGE_PATH")"

echo "[INFO] Preparing minimal calibration inputs in: $WORKSPACE_DIR"
pixi run python - "$COCO_ROOT" "$IMAGE_PATH" "$VLM_DB" <<'PY'
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

from PIL import Image

coco_root = Path(sys.argv[1])
image_path = Path(sys.argv[2])
vlm_db = Path(sys.argv[3])

_ = coco_root
Image.new("RGB", (8, 8), color=(0, 0, 0)).save(image_path, format="JPEG")

if vlm_db.exists():
    vlm_db.unlink()
conn = sqlite3.connect(str(vlm_db))
try:
    conn.execute(
        """
        CREATE TABLE vlm_calib_samples (
            id INTEGER PRIMARY KEY,
            image_relpath TEXT NOT NULL,
            caption TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "INSERT INTO vlm_calib_samples (id, image_relpath, caption) VALUES (?, ?, ?)",
        (1, "train2017/000000000001.jpg", "A black square."),
    )
    conn.commit()
finally:
    conn.close()

print("Wrote:", image_path)
print("Wrote:", vlm_db)
PY

if [[ "${DEVICE}" != "cpu" ]]; then
  pixi run python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available but a CUDA device was requested.")
PY
fi

OUT_ALL_INT8="$WORKSPACE_DIR/all_layers_int8"
OUT_LM_INT8="$WORKSPACE_DIR/lm_only_int8"
mkdir -p "$OUT_ALL_INT8" "$OUT_LM_INT8"

echo "[INFO] Running all-layers INT8..."
pixi run python \
  "$REPO_ROOT/models/qwen3_vl_4b_instruct/helpers/qwen3_vl_4b_autoquant_all_layers/run_qwen3_vl_4b_autoquant_all_layers.py" \
  --quant-format int8 \
  --model-dir "$MODEL_DIR" \
  --output-dir "$OUT_ALL_INT8" \
  --vlm-calib-db "$VLM_DB" \
  --coco-root "$COCO_ROOT" \
  --max-calib-samples 1 \
  --calib-seq-len 64 \
  --batch-size 1 \
  --device "$DEVICE" \
  --auto-quantize-score-size 1 \
  2>&1 | tee "$OUT_ALL_INT8/run.log"

echo "[INFO] Running LM-only INT8..."
pixi run python \
  "$REPO_ROOT/models/qwen3_vl_4b_instruct/helpers/qwen3_vl_4b_autoquant_int8_lm/run_qwen3_vl_4b_autoquant_int8_lm.py" \
  --model-dir "$MODEL_DIR" \
  --output-dir "$OUT_LM_INT8" \
  --captions-path "$CAPTIONS_TXT" \
  --max-calib-samples 1 \
  --calib-seq-len 64 \
  --batch-size 1 \
  --device "$DEVICE" \
  --auto-quantize-score-size 1 \
  2>&1 | tee "$OUT_LM_INT8/run.log"

SUMMARIZER="$SCRIPT_DIR/scripts/summarize_manifest.py"
SANITIZER="$SCRIPT_DIR/scripts/sanitize_artifacts.py"
SUMMARY_DIR="$WORKSPACE_DIR/summaries"
mkdir -p "$SUMMARY_DIR"

echo "[INFO] Writing sanitized summaries..."
pixi run python "$SUMMARIZER" "$OUT_ALL_INT8/int8_autoquant_all_layers_int8_quant_manifest.json" "$SUMMARY_DIR/all_layers_int8"
pixi run python "$SUMMARIZER" "$OUT_LM_INT8/int8_autoquant_lm_default_quant_manifest.json" "$SUMMARY_DIR/lm_only_int8"

EXPECTED_DIR="$SCRIPT_DIR/expected_report"
if $SNAPSHOT; then
  echo "[INFO] Snapshotting expected report..."
  rm -rf "$EXPECTED_DIR/all_layers_int8" "$EXPECTED_DIR/lm_only_int8"
  mkdir -p "$EXPECTED_DIR/all_layers_int8" "$EXPECTED_DIR/lm_only_int8"
  cp -f "$SUMMARY_DIR/all_layers_int8/summary.json" "$EXPECTED_DIR/all_layers_int8/summary.json"
  cp -f "$SUMMARY_DIR/all_layers_int8/summary.md" "$EXPECTED_DIR/all_layers_int8/summary.md"
  cp -f "$SUMMARY_DIR/lm_only_int8/summary.json" "$EXPECTED_DIR/lm_only_int8/summary.json"
  cp -f "$SUMMARY_DIR/lm_only_int8/summary.md" "$EXPECTED_DIR/lm_only_int8/summary.md"

  pixi run python "$SANITIZER" "$OUT_ALL_INT8" "$EXPECTED_DIR/all_layers_int8"
  pixi run python "$SANITIZER" "$OUT_LM_INT8" "$EXPECTED_DIR/lm_only_int8"

  echo "[INFO] Updated expected_report/ from: $WORKSPACE_DIR"
  exit 0
fi

if [[ -d "$EXPECTED_DIR/all_layers_int8" && -d "$EXPECTED_DIR/lm_only_int8" ]]; then
  echo "[INFO] Verifying summaries against expected_report/..."
  diff -u "$EXPECTED_DIR/all_layers_int8/summary.json" "$SUMMARY_DIR/all_layers_int8/summary.json"
  diff -u "$EXPECTED_DIR/all_layers_int8/summary.md" "$SUMMARY_DIR/all_layers_int8/summary.md"
  diff -u "$EXPECTED_DIR/lm_only_int8/summary.json" "$SUMMARY_DIR/lm_only_int8/summary.json"
  diff -u "$EXPECTED_DIR/lm_only_int8/summary.md" "$SUMMARY_DIR/lm_only_int8/summary.md"
else
  echo "[WARN] expected_report/ is missing; run with --snapshot-report to create it." >&2
fi

echo "[INFO] Done."
echo "[INFO] Workspace: $WORKSPACE_DIR"
echo "[INFO] Outputs: $OUT_ALL_INT8 and $OUT_LM_INT8"
