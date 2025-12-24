#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/tmp/yolov10m_scratch_fp16_vs_w4a16_qat_brevitas/${RUN_ID}}"

COCO_ROOT="${COCO_ROOT:-${REPO_ROOT}/datasets/coco2017/source-data}"
IMG_SIZE="${IMG_SIZE:-640}"
EPOCHS="${EPOCHS:-300}"
BATCH="${BATCH:-32}"
FRACTION="${FRACTION:-1.0}"
WORKERS="${WORKERS:-8}"
DEVICE="${DEVICE:-0}"
SAVE_PERIOD="${SAVE_PERIOD:-5}"

MAX_EVAL_IMAGES="${MAX_EVAL_IMAGES:-100}"
EVAL_PROVIDERS="${EVAL_PROVIDERS:-CUDAExecutionProvider CPUExecutionProvider}"

RUN_FP16="${RUN_FP16:-1}"
RUN_QAT="${RUN_QAT:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

mkdir -p "${RUN_ROOT}/logs"

echo "[INFO] RUN_ROOT=${RUN_ROOT}"
echo "[INFO] COCO_ROOT=${COCO_ROOT}"
echo "[INFO] IMG_SIZE=${IMG_SIZE} EPOCHS=${EPOCHS} BATCH=${BATCH} FRACTION=${FRACTION} SAVE_PERIOD=${SAVE_PERIOD}"
echo "[INFO] DEVICE=${DEVICE} WORKERS=${WORKERS}"

if [[ "${RUN_FP16}" == "1" ]]; then
  echo "[INFO] Training baseline FP16..."
  pixi run -e rtx5090 python scripts/cv-models/train_yolov10m_scratch_fp16_vs_w4a16_qat_brevitas.py fp16 \
    --run-root "${RUN_ROOT}" \
    --coco-root "${COCO_ROOT}" \
    --imgsz "${IMG_SIZE}" \
    --epochs "${EPOCHS}" \
    --batch "${BATCH}" \
    --fraction "${FRACTION}" \
    --device "${DEVICE}" \
    --workers "${WORKERS}" \
    --save-period "${SAVE_PERIOD}" \
    |& tee "${RUN_ROOT}/logs/train_fp16.log"
fi

if [[ "${RUN_QAT}" == "1" ]]; then
  echo "[INFO] Training QAT W4A16..."
  pixi run -e rtx5090 python scripts/cv-models/train_yolov10m_scratch_fp16_vs_w4a16_qat_brevitas.py qat-w4a16 \
    --run-root "${RUN_ROOT}" \
    --coco-root "${COCO_ROOT}" \
    --imgsz "${IMG_SIZE}" \
    --epochs "${EPOCHS}" \
    --batch "${BATCH}" \
    --fraction "${FRACTION}" \
    --device "${DEVICE}" \
    --workers "${WORKERS}" \
    --save-period "${SAVE_PERIOD}" \
    |& tee "${RUN_ROOT}/logs/train_qat_w4a16.log"
fi

baseline_onnx="${RUN_ROOT}/onnx/yolov10m-scratch-fp16.onnx"
qat_onnx="${RUN_ROOT}/onnx/yolov10m-scratch-w4a16-qcdq-qat.opt.onnx"
if [[ ! -f "${qat_onnx}" ]]; then
  qat_onnx="${RUN_ROOT}/onnx/yolov10m-scratch-w4a16-qcdq-qat.onnx"
fi

if [[ "${RUN_EVAL}" == "1" ]]; then
  if [[ -n "${baseline_onnx}" && -f "${baseline_onnx}" ]]; then
    echo "[INFO] COCO eval (baseline)..."
    pixi run -e rtx5090 python scripts/cv-models/eval_yolov10m_onnx_coco.py \
      --onnx-path "${baseline_onnx}" \
      --data-root "${COCO_ROOT}" \
      --max-images "${MAX_EVAL_IMAGES}" \
      --providers ${EVAL_PROVIDERS} \
      --warmup-runs 10 \
      --skip-latency 10 \
      --imgsz "${IMG_SIZE}" \
      --out "${RUN_ROOT}/eval/fp16/metrics.json" \
      |& tee "${RUN_ROOT}/logs/eval_fp16.log"
  fi

  if [[ -n "${qat_onnx}" && -f "${qat_onnx}" ]]; then
    echo "[INFO] COCO eval (QAT W4A16)..."
    pixi run -e rtx5090 python scripts/cv-models/eval_yolov10m_onnx_coco.py \
      --onnx-path "${qat_onnx}" \
      --data-root "${COCO_ROOT}" \
      --max-images "${MAX_EVAL_IMAGES}" \
      --providers ${EVAL_PROVIDERS} \
      --warmup-runs 10 \
      --skip-latency 10 \
      --imgsz "${IMG_SIZE}" \
      --out "${RUN_ROOT}/eval/qat-w4a16/metrics.json" \
      |& tee "${RUN_ROOT}/logs/eval_qat_w4a16.log"
  fi
fi

echo "[INFO] Writing summary..."
pixi run -e rtx5090 python scripts/cv-models/write_yolov10m_scratch_train_summary.py \
  --run-root "${RUN_ROOT}" \
  |& tee "${RUN_ROOT}/logs/summary.log"

echo "[INFO] Done."
echo "[INFO] Summary: ${RUN_ROOT}/summary/summary.md"
echo "[INFO] TensorBoard: pixi run -e rtx5090 tensorboard --logdir ${RUN_ROOT}"
