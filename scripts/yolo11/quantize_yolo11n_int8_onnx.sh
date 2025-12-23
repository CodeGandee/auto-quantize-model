#!/usr/bin/env bash
set -euo pipefail

# Quantize YOLO11n ONNX to INT8 Q/DQ with NVIDIA ModelOpt.
#
# This script wraps the ModelOpt ONNX PTQ CLI to produce:
#   models/yolo11/onnx/yolo11n-int8-qdq-proto.onnx
# using the existing YOLO11n ONNX export and calibration tensor.
#
# Quantization parameters:
#   - quantize_mode=int8
#       Use ModelOpt's INT8 post-training quantization path. The tool
#       will insert Q/DQ nodes following TensorRT-friendly patterns and
#       optionally convert some weights/activations to FP16 where
#       appropriate for mixed-precision execution.
#   - calibration_method=max
#       Use per-tensor max-absolute calibration for INT8 scales
#       (simple, robust default recommended by ModelOpt for first-pass
#       CNN quantization).
#   - calibration_data=datasets/quantize-calib/calib_yolo11_640.npy
#       Prebuilt calibration tensor with shape (100, 3, 640, 640)
#       generated from COCO2017 train images using YOLO-style
#       letterbox + [0,1] scaling.
#   - calibration_eps=\"cuda:0 cpu\"
#       Run calibration with ONNX Runtime's CUDAExecutionProvider on
#       GPU 0, with CPUExecutionProvider as a fallback. This exercises
#       GPU-backed kernels during calibration while remaining robust if
#       some ops must fall back to CPU.
#
# Usage:
#   pixi run bash scripts/yolo11/quantize_yolo11n_int8_onnx.sh
#
# Next step (TensorRT engine from QDQ ONNX):
#   For pre-quantized QDQ models, TensorRT should obey the Q/DQ-defined
#   precisions when building INT8 or mixed-precision engines. When
#   converting the resulting ONNX to TensorRT with `trtexec`, prefer:
#
#   pixi run trtexec \
#     --onnx=models/yolo11/onnx/yolo11n-int8-qdq-proto.onnx \
#     --saveEngine=models/yolo11/onnx/yolo11n-int8-qdq-proto-obey.plan \
#     --int8 \
#     --fp16 \
#     --precisionConstraints=obey
#
# Using `--precisionConstraints=obey` ensures the engine honors the
# ModelOpt QDQ quantization scheme instead of silently ignoring it,
# which is critical to avoid accuracy collapse in some networks.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ONNX_PATH="${REPO_ROOT}/models/yolo11/onnx/yolo11n.onnx"
CALIB_PATH="${REPO_ROOT}/datasets/quantize-calib/calib_yolo11_640.npy"
OUTPUT_PATH="${REPO_ROOT}/models/yolo11/onnx/yolo11n-int8-qdq-proto.onnx"

if [[ ! -f "${ONNX_PATH}" ]]; then
  echo "Error: ONNX model not found at ${ONNX_PATH}" >&2
  echo "Hint: run the YOLO11 ONNX export helper first." >&2
  exit 1
fi

if [[ ! -f "${CALIB_PATH}" ]]; then
  echo "Error: calibration tensor not found at ${CALIB_PATH}" >&2
  echo "Hint: run scripts/yolo11/make_yolo11_calib_npy.py to generate it." >&2
  exit 1
fi

read -r -a CALIBRATION_EPS_ARR <<< "cuda:0 cpu"

echo "Quantizing YOLO11n ONNX with ModelOpt..."
echo "  ONNX input  : ${ONNX_PATH}"
echo "  Calibration : ${CALIB_PATH}"
echo "  Output      : ${OUTPUT_PATH}"

python -m modelopt.onnx.quantization \
  --onnx_path="${ONNX_PATH}" \
  --quantize_mode=int8 \
  --calibration_data_path "${CALIB_PATH}" \
  --calibration_method=max \
  --output_path="${OUTPUT_PATH}" \
  --calibration_eps "${CALIBRATION_EPS_ARR[@]}"

echo "Done. Quantized model written to:"
echo "  ${OUTPUT_PATH}"
