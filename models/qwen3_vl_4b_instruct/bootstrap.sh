#!/usr/bin/env bash
set -euo pipefail

# Bootstrap script for Qwen3-VL-4B-Instruct:
# 1. Resolve a local snapshot root (HF_SNAPSHOTS_ROOT or default).
# 2. Create checkpoints/Qwen3-VL-4B-Instruct symlink pointing to the snapshot.
# 3. Ensure checkpoints/ and heavy artifacts are ignored by Git via .gitignore.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ROOT="${SCRIPT_DIR}"
CHECKPOINTS_DIR="${MODEL_ROOT}/checkpoints"
MODEL_NAME="Qwen3-VL-4B-Instruct"

# Prefer HF_SNAPSHOTS_ROOT when set; fall back to a common local path.
SNAPSHOT_ROOT="${HF_SNAPSHOTS_ROOT:-/workspace/llm-models}"
SNAPSHOT_PATH="${SNAPSHOT_ROOT}/${MODEL_NAME}"

echo "[Qwen3-VL-4B] Bootstrap starting..."
echo "[Qwen3-VL-4B] Expecting local snapshot at: ${SNAPSHOT_PATH}"

if [ ! -d "${SNAPSHOT_PATH}" ]; then
  echo "[Qwen3-VL-4B] ERROR: Snapshot directory not found: ${SNAPSHOT_PATH}" >&2
  echo "[Qwen3-VL-4B] Please download the model manually (e.g. via ModelScope or Hugging Face CLI)" >&2
  echo "[Qwen3-VL-4B] into ${SNAPSHOT_PATH} and re-run this bootstrap script." >&2
  echo "[Qwen3-VL-4B] You can also set HF_SNAPSHOTS_ROOT to change the base directory." >&2
  exit 1
fi

mkdir -p "${CHECKPOINTS_DIR}"
TARGET_LINK="${CHECKPOINTS_DIR}/${MODEL_NAME}"

if [ -L "${TARGET_LINK}" ]; then
  CURRENT_TARGET="$(readlink "${TARGET_LINK}")"
  if [ "${CURRENT_TARGET}" = "${SNAPSHOT_PATH}" ]; then
    echo "[Qwen3-VL-4B] Existing symlink is correct, nothing to do."
  else
    echo "[Qwen3-VL-4B] Updating existing symlink from ${CURRENT_TARGET} to ${SNAPSHOT_PATH}..."
    rm -f "${TARGET_LINK}"
    ln -s "${SNAPSHOT_PATH}" "${TARGET_LINK}"
  fi
else
  if [ -e "${TARGET_LINK}" ]; then
    echo "[Qwen3-VL-4B] WARNING: ${TARGET_LINK} exists and is not a symlink; not overwriting." >&2
  else
    echo "[Qwen3-VL-4B] Creating symlink ${TARGET_LINK} -> ${SNAPSHOT_PATH}..."
    ln -s "${SNAPSHOT_PATH}" "${TARGET_LINK}"
  fi
fi

echo "[Qwen3-VL-4B] Ensuring .gitignore rules for checkpoints and artifacts..."
GITIGNORE_PATH="${MODEL_ROOT}/.gitignore"
touch "${GITIGNORE_PATH}"

ensure_gitignore_pattern() {
  local pattern="$1"
  if ! grep -qxF "${pattern}" "${GITIGNORE_PATH}"; then
    echo "${pattern}" >> "${GITIGNORE_PATH}"
  fi
}

ensure_gitignore_pattern "checkpoints/"
ensure_gitignore_pattern "quantized/"
ensure_gitignore_pattern "*.onnx"
ensure_gitignore_pattern "*.onnx_data"

echo "[Qwen3-VL-4B] Bootstrap completed successfully."

