#!/usr/bin/env bash

set -euo pipefail

# Ensure we kill all child processes (process group) on interrupt
trap 'trap - SIGINT SIGTERM; echo "[build-vllm] Interrupted. Killing all processes..."; kill 0' SIGINT SIGTERM

# Simple helper to build vLLM from source.
# - Clones/updates vLLM under extern/vllm
# - Checks out a specific tag (default: v0.10.1)
# - Either:
#     * builds a wheel into an output directory (default: \$PWD/build-vllm), or
#     * installs vLLM as an editable package (with -e / --editable)
#
# Usage (from repo root, inside your desired Python env, e.g. Pixi):
#   pixi run bash extern/build-vllm.sh                 # build wheel only
#   pixi run bash extern/build-vllm.sh -o out/dir      # build wheel into out/dir
#   pixi run bash extern/build-vllm.sh -e              # editable install (no wheel)
#
# Environment variables:
#   VLLM_TAG        vLLM git tag to build (default: v0.10.1)
#   PYTHON          Python executable to use (default: python)
#   VLLM_BUILD_DIR  Output directory for wheels (wheel mode only)
#                   (default: \$PWD/build-vllm; overridden by -o/--output-dir)

MODE="wheel"
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -e|--editable)
      MODE="editable"
      shift
      ;;
    -h|--help)
      cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Build vLLM from source under extern/vllm.

Modes:
  (default)   Build a wheel into an output directory (no install)
  -e, --editable
              Install vLLM as an editable package in the current Python env
              (uses "pip install -e ." with --no-deps).

Options:
  -o, --output-dir DIR
              Wheel output dir in wheel mode. Overrides VLLM_BUILD_DIR.

Environment:
  VLLM_TAG        vLLM git tag to build (default: v0.10.1)
  PYTHON          Python executable to use (default: python)
  VLLM_BUILD_DIR  Wheel output dir in wheel mode (default: \$PWD/build-vllm)
  TORCH_CUDA_ARCH_LIST
                  CUDA arch list. If unset, auto-detected from the host GPU
                  via nvidia-smi (compute_cap, RTX 3090+). Falls back to 8.6
                  (RTX 3090, sm_86) if detection fails.
EOF
      exit 0
      ;;
    -o|--output-dir)
      if [[ $# -lt 2 ]]; then
        echo "[build-vllm] ERROR: -o/--output-dir requires a directory argument." >&2
        exit 1
      fi
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      # Ignore unknown positional arguments for now.
      break
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

VLLM_TAG="${VLLM_TAG:-v0.10.1}"
PYTHON_BIN="${PYTHON:-python}"
VLLM_DIR="${SCRIPT_DIR}/vllm"

if [ -n "${OUTPUT_DIR}" ]; then
  BUILD_DIR="${OUTPUT_DIR}"
else
  # Default to caller's current directory so wheels are easy to find.
  BUILD_DIR="${VLLM_BUILD_DIR:-$(pwd)/build-vllm}"
fi

echo "[build-vllm] Repo root: ${ROOT_DIR}"
echo "[build-vllm] Using Python: ${PYTHON_BIN}"
echo "[build-vllm] vLLM tag: ${VLLM_TAG}"
echo "[build-vllm] vLLM source dir: ${VLLM_DIR}"
echo "[build-vllm] Wheel output dir: ${BUILD_DIR}"
echo "[build-vllm] Mode: ${MODE}"

# Detect CUDA arch list from the current GPU (RTX 3090 and up), while
# respecting the maximum architecture supported by the local CUDA toolkit
# (nvcc). We pick the highest compute capability that is:
#   - supported by at least one GPU on this host, and
#   - not higher than what nvcc advertises via its supported sm_XX targets.
detect_cuda_arch_list() {
  local fallback="8.6"

  # If nvidia-smi is not available, fall back to RTX 3090 (sm_86).
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[build-vllm] WARNING: nvidia-smi not found; falling back to TORCH_CUDA_ARCH_LIST=${fallback}." >&2
    echo "${fallback}"
    return 0
  fi

  # Query compute capability for all GPUs and pick the highest (e.g., 8.6, 8.9, 9.0, 12.0).
  local caps
  caps="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | awk 'NF' | sort -uV || true)"
  if [ -z "${caps}" ]; then
    echo "[build-vllm] WARNING: Failed to read compute_cap from nvidia-smi; falling back to TORCH_CUDA_ARCH_LIST=${fallback}." >&2
    echo "${fallback}"
    return 0
  fi

  local best_cap
  best_cap="$(printf '%s\n' "${caps}" | tail -n1 | tr -d ' ')"

  # Basic sanity check on the detected capability string.
  if ! [[ "${best_cap}" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo "[build-vllm] WARNING: Unexpected compute_cap value '${best_cap}'; falling back to TORCH_CUDA_ARCH_LIST=${fallback}." >&2
    echo "${fallback}"
    return 0
  fi

  # Try to detect the maximum architecture supported by the local nvcc.
  local nvcc_bin="${NVCC_BIN:-nvcc}"
  if ! command -v "${nvcc_bin}" >/dev/null 2>&1; then
    if [ -x /usr/local/cuda/bin/nvcc ]; then
      nvcc_bin="/usr/local/cuda/bin/nvcc"
    else
      nvcc_bin=""
    fi
  fi

  local nvcc_caps=""
  if [ -n "${nvcc_bin}" ]; then
    nvcc_caps="$("${nvcc_bin}" -h 2>&1 | grep -Eo 'sm_[0-9]+' | sort -u || true)"
  fi

  if [ -z "${nvcc_caps}" ]; then
    # If we can't read nvcc capabilities, just return the GPU's cap.
    echo "[build-vllm] INFO: Unable to detect nvcc-supported SMs; using GPU compute_cap=${best_cap}." >&2
    echo "${best_cap}"
    return 0
  fi

  # Convert nvcc sm_XX / sm_XXX tokens into version-like strings (e.g., 8.6, 8.9, 9.0, 12.0).
  local converted_caps=""
  local sm
  while read -r sm; do
    [ -z "${sm}" ] && continue
    local digits="${sm#sm_}"
    # Two-digit (e.g., 86 -> 8.6, 89 -> 8.9, 90 -> 9.0)
    if [ "${#digits}" -eq 2 ]; then
      converted_caps+="$((10#${digits:0:1})).$((10#${digits:1:1})) "
    # Three-digit (e.g., 120 -> 12.0)
    elif [ "${#digits}" -eq 3 ]; then
      converted_caps+="$((10#${digits:0:2})).$((10#${digits:2:1})) "
    fi
  done <<< "${nvcc_caps}"

  if [ -z "${converted_caps}" ]; then
    echo "[build-vllm] INFO: Parsed no nvcc SM versions; using GPU compute_cap=${best_cap}." >&2
    echo "${best_cap}"
    return 0
  fi

  # Select the highest nvcc-supported capability that does not exceed the GPU's capability.
  local selected=""
  local cap
  for cap in $(printf '%s\n' ${converted_caps} | sort -uV); do
    # If min(cap, best_cap) == cap, then cap <= best_cap.
    if [ "$(printf '%s\n' "${cap}" "${best_cap}" | sort -V | head -n1)" != "${cap}" ]; then
      # cap > best_cap, skip
      continue
    fi
    selected="${cap}"
  done

  if [ -z "${selected}" ]; then
    # GPU arch is lower than any nvcc-supported arch (unlikely for 3090+); fall back to GPU cap.
    echo "[build-vllm] INFO: No nvcc SM <= GPU compute_cap=${best_cap}; using GPU compute_cap." >&2
    echo "${best_cap}"
    return 0
  fi

  echo "[build-vllm] INFO: GPU compute_cap=${best_cap}, nvcc max SM-derived cap=${selected}; using ${selected}." >&2
  echo "${selected}"
}

# Configure build parallelism so we don't OOM during heavy CUDA builds.
# We auto-select MAX_JOBS based on CPU cores and ~90% of system RAM,
# assuming ~3.5 GiB peak usage per heavy CUDA compilation job.
configure_build_parallelism() {
  local cores
  if command -v nproc >/dev/null 2>&1; then
    cores="$(nproc --all)"
  else
    cores=8
  fi

  if [ -z "${MAX_JOBS:-}" ]; then
    local mem_kb mem90_kb per_job_kb jobs_from_mem max_jobs
    if [ -r /proc/meminfo ]; then
      mem_kb="$(awk '/MemTotal:/ {print $2}' /proc/meminfo || echo "")"
    else
      mem_kb=""
    fi

    if [ -n "${mem_kb:-}" ]; then
      # Use 90% of total memory.
      mem90_kb=$((mem_kb * 9 / 10))
      # Approximate 3.5 GiB per heavy CUDA job: 3.5 * 1024 * 1024 ≈ 3670016 KiB.
      per_job_kb=3670016
      jobs_from_mem=$((mem90_kb / per_job_kb))
      if [ "${jobs_from_mem}" -lt 1 ]; then
        jobs_from_mem=1
      fi
      if [ "${jobs_from_mem}" -gt "${cores}" ]; then
        max_jobs="${cores}"
      else
        max_jobs="${jobs_from_mem}"
      fi
    else
      # Fallback: conservative half of cores (at least 1).
      max_jobs=$((cores / 2))
      if [ "${max_jobs}" -lt 1 ]; then
        max_jobs=1
      fi
    fi

    export MAX_JOBS="${max_jobs}"
    echo "[build-vllm] MAX_JOBS not set; auto-selected ${MAX_JOBS} (cores=${cores}, mem_kb=${mem_kb:-unknown})."
  else
    echo "[build-vllm] MAX_JOBS already set to '${MAX_JOBS}'."
  fi

  if [ -z "${NVCC_THREADS:-}" ]; then
    export NVCC_THREADS=1
    echo "[build-vllm] NVCC_THREADS not set; defaulting to ${NVCC_THREADS}."
  else
    echo "[build-vllm] NVCC_THREADS already set to '${NVCC_THREADS}'."
  fi
}

if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
  detected_arch="$(detect_cuda_arch_list)"
  export TORCH_CUDA_ARCH_LIST="${detected_arch}"
  echo "[build-vllm] TORCH_CUDA_ARCH_LIST not set; detected '${TORCH_CUDA_ARCH_LIST}' from GPU/nvcc capabilities."
else
  echo "[build-vllm] TORCH_CUDA_ARCH_LIST already set to '${TORCH_CUDA_ARCH_LIST}'."
fi

# Apply parallelism limits before starting any heavy build work.
configure_build_parallelism

mkdir -p "${BUILD_DIR}"

if ! command -v git >/dev/null 2>&1; then
  echo "[build-vllm] ERROR: git is required but not found in PATH." >&2
  exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[build-vllm] ERROR: Python executable '${PYTHON_BIN}' not found." >&2
  exit 1
fi

if [ ! -d "${VLLM_DIR}/.git" ]; then
  echo "[build-vllm] Cloning vLLM into ${VLLM_DIR}..."
  git clone https://github.com/vllm-project/vllm.git "${VLLM_DIR}"
fi

cd "${VLLM_DIR}"

echo "[build-vllm] Fetching tags..."
git fetch --tags --quiet

echo "[build-vllm] Checking out tag ${VLLM_TAG}..."
git checkout "${VLLM_TAG}"

echo "[build-vllm] Updating submodules..."
git submodule update --init --recursive

echo "[build-vllm] Python version in this env:"
"${PYTHON_BIN}" -V

echo "[build-vllm] Torch version in this env (if available):"
"${PYTHON_BIN}" - << 'EOF' || true
try:
    import torch
    print("torch", torch.__version__, "cuda", getattr(torch.version, "cuda", None))
except Exception as exc:
    print("torch not importable:", exc)
EOF

if [ "${MODE}" = "editable" ]; then
  echo "[build-vllm] Installing vLLM in editable mode (no deps, no build isolation, verbose pip)..."
  echo "[build-vllm]   Command: ${PYTHON_BIN} -m pip install -v -e . --no-deps --no-build-isolation"
  set -x
  "${PYTHON_BIN}" -m pip install -v -e . --no-deps --no-build-isolation
  set +x
  echo "[build-vllm] Editable install complete."
else
  echo "[build-vllm] Building vLLM wheel (no install, no build isolation, verbose pip)..."
  echo "[build-vllm]   Command: ${PYTHON_BIN} -m pip wheel -v . -w \"${BUILD_DIR}\" --no-deps --no-build-isolation"

  # Capture full build logs into the wheel output directory for debugging.
  # This keeps stdout/stderr visible to the caller while also writing a log file.
  build_log="${BUILD_DIR}/vllm-build-$(date +%Y%m%d-%H%M%S).log"
  echo "[build-vllm] Build log will be written to: ${build_log}"

  set +e
  {
    set -x
    "${PYTHON_BIN}" -m pip wheel -v . -w "${BUILD_DIR}" --no-deps --no-build-isolation
  } 2>&1 | tee "${build_log}"
  build_rc=${PIPESTATUS[0]}
  set +x
  set -e

  if [ "${build_rc}" -ne 0 ]; then
    echo "[build-vllm] ERROR: vLLM wheel build failed with exit code ${build_rc}. See log: ${build_log}" >&2
    exit "${build_rc}"
  fi

  echo "[build-vllm] Build complete. Wheels in: ${BUILD_DIR}"
  ls -1 "${BUILD_DIR}"/vllm-*.whl 2>/dev/null || {
    echo "[build-vllm] WARNING: No vllm-*.whl files found in ${BUILD_DIR}." >&2
  }
fi
