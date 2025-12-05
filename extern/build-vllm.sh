#!/usr/bin/env bash

set -euo pipefail

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
                  CUDA arch list; defaults to "8.6" (RTX 3090, sm_86) if unset.
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

# Restrict CUDA arch list to this host's GPUs (RTX 3090, sm_86) by default.
if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
  export TORCH_CUDA_ARCH_LIST="8.6"
  echo "[build-vllm] TORCH_CUDA_ARCH_LIST not set; defaulting to 8.6 (RTX 3090, sm_86)."
else
  echo "[build-vllm] TORCH_CUDA_ARCH_LIST already set to '${TORCH_CUDA_ARCH_LIST}'."
fi

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

  set -x
  "${PYTHON_BIN}" -m pip wheel -v . -w "${BUILD_DIR}" --no-deps --no-build-isolation
  set +x

  echo "[build-vllm] Build complete. Wheels in: ${BUILD_DIR}"
  ls -1 "${BUILD_DIR}"/vllm-*.whl 2>/dev/null || {
    echo "[build-vllm] WARNING: No vllm-*.whl files found in ${BUILD_DIR}." >&2
  }
fi
