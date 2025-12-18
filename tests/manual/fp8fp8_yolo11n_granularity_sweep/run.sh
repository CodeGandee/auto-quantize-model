#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
RUN_ROOT="${REPO_ROOT}/tmp/fp8fp8_yolo11n_granularity_sweep/${RUN_ID}"
OUT_ROOT="${RUN_ROOT}/outputs"
LOG_ROOT="${RUN_ROOT}/logs"

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"

echo "[INFO] Bootstrapping YOLO11 assets (if missing)..."
bash "${REPO_ROOT}/models/yolo11/bootstrap.sh"

echo "[INFO] Running FP8/FP8 granularity sweep..."
pixi run -e rtx5090-vllm python "${SCRIPT_DIR}/scripts/run_granularity_sweep.py" \
  --output-root "${OUT_ROOT}" \
  --log-root "${LOG_ROOT}" \
  --max-calib-samples 16 \
  --batch-size 1 \
  --imgsz 640 \
  --effective-bits 11.0 \
  --auto-quantize-score-size 16

echo "[INFO] Done. Outputs: ${OUT_ROOT}"
