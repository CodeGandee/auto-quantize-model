#!/usr/bin/env bash
set -euo pipefail

# Run text-only and VLM comparison for all AutoQuant FP8 all-layers top-XX schemes
# for Qwen2.5-VL-3B-Instruct.
#
# This script:
#   1. Discovers all `fp8_autoquant_all_layers_top*_coco2017` checkpoints under:
#        models/qwen2_5_vl_3b_instruct/quantized/
#   2. Runs the text-only multi-scheme comparison:
#        compare_qwen2_5_vl_3b_schemes_vs_fp16.py
#      over all discovered schemes in a single pass.
#   3. Runs the VLM image+text comparison:
#        compare_qwen2_5_vl_3b_vlm_eval_top10_vs_fp16.py
#      once per scheme, using the COCO2017 VLM eval subset.
#   4. Stores metrics and SQLite DBs under:
#        tmp/modelopt-autoquant-fp8/
#
# Usage (from repo root, RTX 5090 env):
#   pixi run -e rtx5090-vllm bash \
#     models/qwen2_5_vl_3b_instruct/helpers/autoquant_eval/run_all_autoquant_fp8_schemes_eval.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

QUANT_ROOT="${REPO_ROOT}/models/qwen2_5_vl_3b_instruct/quantized"
TEXT_EVAL_SCRIPT="${REPO_ROOT}/models/qwen2_5_vl_3b_instruct/helpers/autoquant_eval/compare_qwen2_5_vl_3b_schemes_vs_fp16.py"
VLM_EVAL_SCRIPT="${REPO_ROOT}/models/qwen2_5_vl_3b_instruct/helpers/autoquant_eval/compare_qwen2_5_vl_3b_vlm_eval_top10_vs_fp16.py"

TEXT_OUT_DIR="${REPO_ROOT}/tmp/modelopt-autoquant-fp8/eval-all-layers-schemes-top10-100"
VLM_OUT_ROOT="${REPO_ROOT}/tmp/modelopt-autoquant-fp8"

if [[ ! -d "${QUANT_ROOT}" ]]; then
  echo "Error: quantized model directory not found:" >&2
  echo "  ${QUANT_ROOT}" >&2
  exit 1
fi

mapfile -t SCHEME_DIRS < <(printf '%s\n' "${QUANT_ROOT}"/fp8_autoquant_all_layers_top*_coco2017 2>/dev/null | sort)

if [[ "${#SCHEME_DIRS[@]}" -eq 0 ]]; then
  echo "Error: no fp8_autoquant_all_layers_top*_coco2017 schemes found under:" >&2
  echo "  ${QUANT_ROOT}" >&2
  exit 1
fi

SCHEME_NAMES=()
for dir in "${SCHEME_DIRS[@]}"; do
  base="$(basename "${dir}")"
  # Expect names like fp8_autoquant_all_layers_top10_coco2017 -> scheme name top10
  scheme="${base#fp8_autoquant_all_layers_}"   # top10_coco2017
  scheme="${scheme%_coco2017}"                # top10
  SCHEME_NAMES+=("${scheme}")
done

echo "[INFO] Found ${#SCHEME_DIRS[@]} AutoQuant FP8 all-layers schemes:"
for i in "${!SCHEME_DIRS[@]}"; do
  printf '  - %s (%s)\n' "${SCHEME_NAMES[$i]}" "${SCHEME_DIRS[$i]}"
done

echo
echo "[INFO] Running text-only multi-scheme comparison..."
pixi run -e rtx5090-vllm python "${TEXT_EVAL_SCRIPT}" \
  --quant-model-dirs "${SCHEME_DIRS[@]}" \
  --scheme-names "${SCHEME_NAMES[@]}" \
  --out-dir "${TEXT_OUT_DIR}"

echo
echo "[INFO] Running VLM image+text comparison for each scheme..."
for i in "${!SCHEME_DIRS[@]}"; do
  scheme_name="${SCHEME_NAMES[$i]}"
  scheme_dir="${SCHEME_DIRS[$i]}"

  out_dir="${VLM_OUT_ROOT}/eval-all-layers-${scheme_name}-vlm-gpu"
  echo "[INFO]  - Scheme ${scheme_name}: ${scheme_dir}"
  echo "         Output: ${out_dir}"

  pixi run -e rtx5090-vllm python "${VLM_EVAL_SCRIPT}" \
    --quant-model-dir "${scheme_dir}" \
    --device cuda \
    --batch-size 1 \
    --max-samples 100 \
    --max-batches 100 \
    --out-dir "${out_dir}"
done

echo
echo "[INFO] Completed AutoQuant FP8 all-layers scheme evaluations."
echo "  Text-only metrics: ${TEXT_OUT_DIR}"
echo "  VLM metrics per scheme under: ${VLM_OUT_ROOT}/eval-all-layers-*-vlm-gpu"

