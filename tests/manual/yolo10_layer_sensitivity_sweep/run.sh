#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
RUN_ROOT="${REPO_ROOT}/tmp/yolo10_layer_sensitivity_sweep/${RUN_ID}"
OUT_ROOT="${RUN_ROOT}/outputs"
LOG_ROOT="${RUN_ROOT}/logs"

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"

echo "[INFO] Bootstrapping YOLOv10 assets (if missing)..."
bash "${REPO_ROOT}/models/yolo10/bootstrap.sh"

echo "[INFO] Running YOLOv10 layer sensitivity sweep..."
pixi run -e rtx5090-vllm python "${SCRIPT_DIR}/scripts/run_layer_sensitivity_sweep.py" \
  --output-root "${OUT_ROOT}" \
  --log-root "${LOG_ROOT}" \
  --max-calib-samples 100 \
  --batch-size 1 \
  --imgsz 640 \
  --auto-quantize-score-size 16 \
  "$@"

echo "[INFO] Done. Outputs: ${OUT_ROOT}"
