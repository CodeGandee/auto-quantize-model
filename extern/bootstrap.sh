#!/usr/bin/env bash

set -euo pipefail

# Bootstrap script to populate extern/ with shallow clones of the
# external tool repositories referenced in this project.
#
# Usage (from repo root or any directory):
#   bash extern/bootstrap.sh
#
# Environment:
#   CLONE_DEPTH   Git clone depth (default: 1)
#
# Notes:
#   - Existing directories are left untouched (no update/overwrite).
#   - All cloned repos are ignored by Git via extern/.gitignore.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DEPTH="${CLONE_DEPTH:-1}"

if ! command -v git >/dev/null 2>&1; then
  echo "[extern/bootstrap] ERROR: git is required but not found in PATH." >&2
  exit 1
fi

echo "[extern/bootstrap] Using clone depth: ${DEPTH}"
echo "[extern/bootstrap] Working directory: ${SCRIPT_DIR}"

clone_repo() {
  local dir="$1"
  local url="$2"

  if [ -d "${dir}/.git" ]; then
    echo "[extern/bootstrap] Skipping ${dir}: already a Git checkout."
    return 0
  fi

  if [ -d "${dir}" ]; then
    echo "[extern/bootstrap] Skipping ${dir}: directory exists but is not a Git repo."
    echo "  (Remove it manually if you want to re-clone.)"
    return 0
  fi

  echo "[extern/bootstrap] Cloning ${url} -> ${dir} (depth=${DEPTH})..."
  git clone --depth="${DEPTH}" "${url}" "${dir}"
}

clone_repo "TensorRT-Model-Optimizer" "https://github.com/NVIDIA/TensorRT-Model-Optimizer.git"
clone_repo "neural-compressor"        "https://github.com/intel/neural-compressor.git"
clone_repo "nncf"                     "https://github.com/openvinotoolkit/nncf.git"
clone_repo "openvino"                 "https://github.com/openvinotoolkit/openvino.git"
clone_repo "vllm"                     "https://github.com/vllm-project/vllm.git"
clone_repo "onnxruntime"              "https://github.com/microsoft/onnxruntime.git"

echo "[extern/bootstrap] Done."
