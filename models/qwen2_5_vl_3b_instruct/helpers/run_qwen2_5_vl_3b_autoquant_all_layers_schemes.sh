#!/usr/bin/env bash
set -euo pipefail

# Sweep all existing fp8_autoquant_all_layers_topXX_coco2017 coverage
# manifests and re-run ModelOpt AutoQuant per scheme to produce distinct
# FP8 checkpoints.
#
# This wrapper discovers coverage manifests created by
# scripts/qwen/slice_qwen2_5_vl_3b_autoquant_all_layers_schemes.py and
# invokes:
#   qwen2_5_vl_3b_autoquant_fp8_all_layers_per_scheme.py
# once per scheme.
#
# Usage (from repo root, RTX 5090 env):
#   pixi run -e rtx5090-vllm bash \
#     scripts/qwen/run_qwen2_5_vl_3b_autoquant_all_layers_schemes.sh
#
# Optional flags:
#   --only top10,top50,top100   # restrict schemes by their "topXX" token
#   --dry-run                   # print commands without executing

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL_DIR="${REPO_ROOT}/models/qwen2_5_vl_3b_instruct/checkpoints/Qwen2.5-VL-3B-Instruct"
BASELINE_DIR="${REPO_ROOT}/models/qwen2_5_vl_3b_instruct/quantized/fp8_autoquant_all_layers_fp8_coco2017"
BASELINE_MANIFEST="${BASELINE_DIR}/layer-sensitivity/fp8_autoquant_all_layers_fp8_quant_manifest.json"
QUANT_ROOT="${REPO_ROOT}/models/qwen2_5_vl_3b_instruct/quantized"

DRY_RUN=0
ONLY_FILTER=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --only)
      if [[ $# -lt 2 ]]; then
        echo "Error: --only requires a comma-separated list (e.g. top10,top50)" >&2
        exit 1
      fi
      ONLY_FILTER="$2"
      shift 2
      ;;
    *)
      echo "Error: unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "Error: base Qwen2.5-VL-3B-Instruct checkpoint not found:" >&2
  echo "  ${MODEL_DIR}" >&2
  echo "Hint: run models/qwen2_5_vl_3b_instruct/bootstrap.sh first." >&2
  exit 1
fi

if [[ ! -d "${BASELINE_DIR}" ]]; then
  echo "Error: baseline all-layers AutoQuant directory not found:" >&2
  echo "  ${BASELINE_DIR}" >&2
  exit 1
fi

if [[ ! -f "${BASELINE_MANIFEST}" ]]; then
  echo "Error: baseline manifest not found at:" >&2
  echo "  ${BASELINE_MANIFEST}" >&2
  exit 1
fi

mapfile -t COVERAGE_FILES < <(
  find "${QUANT_ROOT}" -maxdepth 3 -type f -name "fp8_autoquant_all_layers_top*_coco2017_coverage_from_baseline.json" | sort
)

if [[ "${#COVERAGE_FILES[@]}" -eq 0 ]]; then
  echo "Error: no coverage manifests found under:" >&2
  echo "  ${QUANT_ROOT}" >&2
  exit 1
fi

if [[ -n "${ONLY_FILTER}" ]]; then
  IFS=',' read -r -a ONLY_TOKENS <<< "${ONLY_FILTER}"
  FILTERED=()
  for cov in "${COVERAGE_FILES[@]}"; do
    base="$(basename "${cov}")"
    keep=0
    for tok in "${ONLY_TOKENS[@]}"; do
      tok_trimmed="${tok//[[:space:]]/}"
      if [[ -n "${tok_trimmed}" && "${base}" == *"${tok_trimmed}"* ]]; then
        keep=1
        break
      fi
    done
    if [[ "${keep}" -eq 1 ]]; then
      FILTERED+=("${cov}")
    fi
  done
  COVERAGE_FILES=("${FILTERED[@]}")
fi

if [[ "${#COVERAGE_FILES[@]}" -eq 0 ]]; then
  echo "Error: no coverage manifests remain after applying --only filter." >&2
  exit 1
fi

echo "[INFO] Found ${#COVERAGE_FILES[@]} coverage manifests:"
for cov in "${COVERAGE_FILES[@]}"; do
  echo "  - ${cov}"
done

DRIVER="${REPO_ROOT}/scripts/qwen/qwen2_5_vl_3b_autoquant_fp8_all_layers_per_scheme.py"

if [[ ! -f "${DRIVER}" ]]; then
  echo "Error: per-scheme AutoQuant driver not found at:" >&2
  echo "  ${DRIVER}" >&2
  exit 1
fi

echo
echo "[INFO] Starting per-scheme AutoQuant runs..."

for cov in "${COVERAGE_FILES[@]}"; do
  scheme_dir="$(cd "$(dirname "${cov}")/.." && pwd)"
  scheme_name="$(basename "${scheme_dir}")"

  # Default naming: append _v2 so we do not overwrite the original
  # sliced directories produced by the coverage helper.
  out_dir="${QUANT_ROOT}/${scheme_name}_v2"

  echo
  echo "[INFO] Scheme: ${scheme_name}"
  echo "       Coverage manifest: ${cov}"
  echo "       Output directory:  ${out_dir}"

  cmd=(
    pixi run -e rtx5090-vllm python "${DRIVER}"
    --model-dir "${MODEL_DIR}"
    --baseline-dir "${BASELINE_DIR}"
    --baseline-manifest "${BASELINE_MANIFEST}"
    --coverage-manifest "${cov}"
    --scheme-name "${scheme_name}"
    --out-dir "${out_dir}"
  )

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[DRY-RUN] ${cmd[*]}"
  else
    "${cmd[@]}"
  fi
done

echo
echo "[INFO] Completed per-scheme AutoQuant all-layers runs."
