#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/tmp/yolov10m_brevitas_w4a8_w4a16/${RUN_ID}}"

IMG_SIZE="${IMG_SIZE:-640}"
MAX_EVAL_IMAGES="${MAX_EVAL_IMAGES:-100}"

CALIB_LIST="${CALIB_LIST:-${REPO_ROOT}/datasets/quantize-calib/quant100.txt}"
COCO_ROOT="${COCO_ROOT:-${REPO_ROOT}/datasets/coco2017/source-data}"

RUN_QAT="${RUN_QAT:-0}"
QAT_MODE="${QAT_MODE:-w4a8}"

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/onnx"

echo "[INFO] RUN_ROOT=${RUN_ROOT}"
echo "[INFO] IMG_SIZE=${IMG_SIZE} MAX_EVAL_IMAGES=${MAX_EVAL_IMAGES}"
echo "[INFO] CALIB_LIST=${CALIB_LIST}"
echo "[INFO] COCO_ROOT=${COCO_ROOT}"

notes_path="${RUN_ROOT}/notes.md"
cat > "${notes_path}" <<EOF
# Notes

- Run root: \`${RUN_ROOT}\`
- Env: \`pixi run -e rtx5090\`

EOF

echo "[INFO] Exporting baseline ONNX..."
pixi run -e rtx5090 python scripts/cv-models/quantize_yolov10m_brevitas_w4.py baseline \
  --run-root "${RUN_ROOT}" \
  --imgsz "${IMG_SIZE}" \
  --no-prefer-fp16 \
  |& tee "${RUN_ROOT}/logs/baseline_export.log"

baseline_onnx="${RUN_ROOT}/onnx/yolov10m-baseline-fp16.onnx"
if [[ ! -f "${baseline_onnx}" ]]; then
  baseline_onnx="${RUN_ROOT}/onnx/yolov10m-baseline-fp32.onnx"
fi
if [[ ! -f "${baseline_onnx}" ]]; then
  echo "[ERROR] Baseline ONNX not found under ${RUN_ROOT}/onnx" >&2
  exit 1
fi

echo "[INFO] Random-tensor smoke (baseline)..."
pixi run -e rtx5090 python models/cv-models/helpers/run_random_onnx_inference.py \
  --model "${baseline_onnx}" \
  --output-root "${RUN_ROOT}/smoke" \
  --providers CUDAExecutionProvider CPUExecutionProvider \
  |& tee "${RUN_ROOT}/logs/baseline_smoke.log"

echo "[INFO] COCO subset eval (baseline)..."
pixi run -e rtx5090 python scripts/cv-models/eval_yolov10m_onnx_coco.py \
  --onnx-path "${baseline_onnx}" \
  --data-root "${COCO_ROOT}" \
  --max-images "${MAX_EVAL_IMAGES}" \
  --providers CUDAExecutionProvider CPUExecutionProvider \
  --warmup-runs 10 \
  --skip-latency 10 \
  --imgsz "${IMG_SIZE}" \
  --out "${RUN_ROOT}/baseline-coco/metrics.json" \
  |& tee "${RUN_ROOT}/logs/baseline_eval.log"

echo "[INFO] Exporting PTQ W8A16 QCDQ ONNX..."
pixi run -e rtx5090 python scripts/cv-models/quantize_yolov10m_brevitas_w4.py ptq \
  --mode w8a16 \
  --run-root "${RUN_ROOT}" \
  --imgsz "${IMG_SIZE}" \
  --opset 13 \
  --no-export-fp16-input \
  |& tee "${RUN_ROOT}/logs/ptq_w8a16_export.log"

w8a16_onnx="${RUN_ROOT}/onnx/yolov10m-w8a16-qcdq-ptq-opt.onnx"
if [[ ! -f "${w8a16_onnx}" ]]; then
  w8a16_onnx="${RUN_ROOT}/onnx/yolov10m-w8a16-qcdq-ptq.onnx"
fi

echo "[INFO] Random-tensor smoke (PTQ W8A16)..."
pixi run -e rtx5090 python models/cv-models/helpers/run_random_onnx_inference.py \
  --model "${w8a16_onnx}" \
  --output-root "${RUN_ROOT}/smoke" \
  --providers CUDAExecutionProvider CPUExecutionProvider \
  |& tee "${RUN_ROOT}/logs/ptq_w8a16_smoke.log"

echo "[INFO] COCO subset eval (PTQ W8A16)..."
pixi run -e rtx5090 python scripts/cv-models/eval_yolov10m_onnx_coco.py \
  --onnx-path "${w8a16_onnx}" \
  --data-root "${COCO_ROOT}" \
  --max-images "${MAX_EVAL_IMAGES}" \
  --providers CUDAExecutionProvider CPUExecutionProvider \
  --warmup-runs 10 \
  --skip-latency 10 \
  --imgsz "${IMG_SIZE}" \
  --out "${RUN_ROOT}/ptq-w8a16-coco/metrics.json" \
  |& tee "${RUN_ROOT}/logs/ptq_w8a16_eval.log"

echo "[INFO] Exporting PTQ W8A8 QCDQ ONNX (calibration from ${CALIB_LIST})..."
pixi run -e rtx5090 python scripts/cv-models/quantize_yolov10m_brevitas_w4.py ptq \
  --mode w8a8 \
  --run-root "${RUN_ROOT}" \
  --imgsz "${IMG_SIZE}" \
  --opset 13 \
  --calib-list "${CALIB_LIST}" \
  --calib-device "cuda:0" \
  --calib-batch-size 4 \
  |& tee "${RUN_ROOT}/logs/ptq_w8a8_export.log"

w8a8_onnx="${RUN_ROOT}/onnx/yolov10m-w8a8-qcdq-ptq-opt.onnx"
if [[ ! -f "${w8a8_onnx}" ]]; then
  w8a8_onnx="${RUN_ROOT}/onnx/yolov10m-w8a8-qcdq-ptq.onnx"
fi

echo "[INFO] Random-tensor smoke (PTQ W8A8)..."
pixi run -e rtx5090 python models/cv-models/helpers/run_random_onnx_inference.py \
  --model "${w8a8_onnx}" \
  --output-root "${RUN_ROOT}/smoke" \
  --providers CUDAExecutionProvider CPUExecutionProvider \
  |& tee "${RUN_ROOT}/logs/ptq_w8a8_smoke.log"

echo "[INFO] COCO subset eval (PTQ W8A8)..."
pixi run -e rtx5090 python scripts/cv-models/eval_yolov10m_onnx_coco.py \
  --onnx-path "${w8a8_onnx}" \
  --data-root "${COCO_ROOT}" \
  --max-images "${MAX_EVAL_IMAGES}" \
  --providers CUDAExecutionProvider CPUExecutionProvider \
  --warmup-runs 10 \
  --skip-latency 10 \
  --imgsz "${IMG_SIZE}" \
  --out "${RUN_ROOT}/ptq-w8a8-coco/metrics.json" \
  |& tee "${RUN_ROOT}/logs/ptq_w8a8_eval.log"

echo "[INFO] Exporting PTQ W4A16(-like) QCDQ ONNX..."
pixi run -e rtx5090 python scripts/cv-models/quantize_yolov10m_brevitas_w4.py ptq \
  --mode w4a16 \
  --run-root "${RUN_ROOT}" \
  --imgsz "${IMG_SIZE}" \
  --opset 13 \
  |& tee "${RUN_ROOT}/logs/ptq_w4a16_export.log"

w4a16_onnx="${RUN_ROOT}/onnx/yolov10m-w4a16-qcdq-ptq-opt.onnx"
if [[ ! -f "${w4a16_onnx}" ]]; then
  w4a16_onnx="${RUN_ROOT}/onnx/yolov10m-w4a16-qcdq-ptq.onnx"
fi

echo "[INFO] Random-tensor smoke (PTQ W4A16)..."
pixi run -e rtx5090 python models/cv-models/helpers/run_random_onnx_inference.py \
  --model "${w4a16_onnx}" \
  --output-root "${RUN_ROOT}/smoke" \
  --providers CUDAExecutionProvider CPUExecutionProvider \
  |& tee "${RUN_ROOT}/logs/ptq_w4a16_smoke.log"

echo "[INFO] COCO subset eval (PTQ W4A16)..."
pixi run -e rtx5090 python scripts/cv-models/eval_yolov10m_onnx_coco.py \
  --onnx-path "${w4a16_onnx}" \
  --data-root "${COCO_ROOT}" \
  --max-images "${MAX_EVAL_IMAGES}" \
  --providers CUDAExecutionProvider CPUExecutionProvider \
  --warmup-runs 10 \
  --skip-latency 10 \
  --imgsz "${IMG_SIZE}" \
  --out "${RUN_ROOT}/ptq-w4a16-coco/metrics.json" \
  |& tee "${RUN_ROOT}/logs/ptq_w4a16_eval.log"

echo "[INFO] Exporting PTQ W4A8 QCDQ ONNX (calibration from ${CALIB_LIST})..."
pixi run -e rtx5090 python scripts/cv-models/quantize_yolov10m_brevitas_w4.py ptq \
  --mode w4a8 \
  --run-root "${RUN_ROOT}" \
  --imgsz "${IMG_SIZE}" \
  --opset 13 \
  --calib-list "${CALIB_LIST}" \
  --calib-device "cuda:0" \
  --calib-batch-size 4 \
  |& tee "${RUN_ROOT}/logs/ptq_w4a8_export.log"

w4a8_onnx="${RUN_ROOT}/onnx/yolov10m-w4a8-qcdq-ptq-opt.onnx"
if [[ ! -f "${w4a8_onnx}" ]]; then
  w4a8_onnx="${RUN_ROOT}/onnx/yolov10m-w4a8-qcdq-ptq.onnx"
fi

echo "[INFO] Random-tensor smoke (PTQ W4A8)..."
pixi run -e rtx5090 python models/cv-models/helpers/run_random_onnx_inference.py \
  --model "${w4a8_onnx}" \
  --output-root "${RUN_ROOT}/smoke" \
  --providers CUDAExecutionProvider CPUExecutionProvider \
  |& tee "${RUN_ROOT}/logs/ptq_w4a8_smoke.log"

echo "[INFO] COCO subset eval (PTQ W4A8)..."
pixi run -e rtx5090 python scripts/cv-models/eval_yolov10m_onnx_coco.py \
  --onnx-path "${w4a8_onnx}" \
  --data-root "${COCO_ROOT}" \
  --max-images "${MAX_EVAL_IMAGES}" \
  --providers CUDAExecutionProvider CPUExecutionProvider \
  --warmup-runs 10 \
  --skip-latency 10 \
  --imgsz "${IMG_SIZE}" \
  --out "${RUN_ROOT}/ptq-w4a8-coco/metrics.json" \
  |& tee "${RUN_ROOT}/logs/ptq_w4a8_eval.log"

if [[ "${RUN_QAT}" == "1" ]]; then
  echo "[INFO] Running optional QAT (${QAT_MODE})..."
  pixi run -e rtx5090 python scripts/cv-models/quantize_yolov10m_brevitas_w4.py qat \
    --mode "${QAT_MODE}" \
    --run-root "${RUN_ROOT}" \
    --imgsz "${IMG_SIZE}" \
    --opset 13 \
    --calib-list "${CALIB_LIST}" \
    --calib-device "cuda:0" \
    --calib-batch-size 4 \
    --coco-root "${COCO_ROOT}" \
    --train-list "${CALIB_LIST}" \
    --val-max-images 20 \
    --epochs 1 \
    --batch 2 \
    --device 0 \
    --seed 0 \
    |& tee "${RUN_ROOT}/logs/qat_${QAT_MODE}.log"

  qat_onnx="${RUN_ROOT}/onnx/yolov10m-${QAT_MODE}-qcdq-qat-pl-opt.onnx"
  if [[ ! -f "${qat_onnx}" ]]; then
    qat_onnx="${RUN_ROOT}/onnx/yolov10m-${QAT_MODE}-qcdq-qat-pl.onnx"
  fi

  echo "[INFO] Random-tensor smoke (QAT ${QAT_MODE})..."
  pixi run -e rtx5090 python models/cv-models/helpers/run_random_onnx_inference.py \
    --model "${qat_onnx}" \
    --output-root "${RUN_ROOT}/smoke" \
    --providers CUDAExecutionProvider CPUExecutionProvider \
    |& tee "${RUN_ROOT}/logs/qat_${QAT_MODE}_smoke.log"

  echo "[INFO] COCO subset eval (QAT ${QAT_MODE})..."
  pixi run -e rtx5090 python scripts/cv-models/eval_yolov10m_onnx_coco.py \
    --onnx-path "${qat_onnx}" \
    --data-root "${COCO_ROOT}" \
    --max-images "${MAX_EVAL_IMAGES}" \
    --providers CUDAExecutionProvider CPUExecutionProvider \
    --warmup-runs 10 \
    --skip-latency 10 \
    --imgsz "${IMG_SIZE}" \
    --out "${RUN_ROOT}/qat-${QAT_MODE}-pl-coco/metrics.json" \
    |& tee "${RUN_ROOT}/logs/qat_${QAT_MODE}_eval.log"
fi

echo "[INFO] Writing summary.md..."
pixi run -e rtx5090 python scripts/cv-models/write_yolov10m_brevitas_summary.py \
  --run-root "${RUN_ROOT}" \
  |& tee "${RUN_ROOT}/logs/summary.log"

echo "[INFO] Done."
