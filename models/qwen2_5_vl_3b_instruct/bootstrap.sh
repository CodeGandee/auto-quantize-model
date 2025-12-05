#!/usr/bin/env bash
set -euo pipefail

# Bootstrap script for Qwen2.5-VL-3B-Instruct:
# 1. Resolve a local snapshot root (HF_SNAPSHOTS_ROOT or default).
# 2. Create checkpoints/Qwen2.5-VL-3B-Instruct symlink pointing to the snapshot.
# 3. Ensure checkpoints/ is ignored by Git via .gitignore.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ROOT="${SCRIPT_DIR}"
CHECKPOINTS_DIR="${MODEL_ROOT}/checkpoints"
MODEL_NAME="Qwen2.5-VL-3B-Instruct"

# Prefer HF_SNAPSHOTS_ROOT when set; fall back to a common local path.
SNAPSHOT_ROOT="${HF_SNAPSHOTS_ROOT:-/data2/llm-models}"
SNAPSHOT_PATH="${SNAPSHOT_ROOT}/${MODEL_NAME}"

echo "[Qwen2.5-VL-3B] Bootstrap starting..."
echo "[Qwen2.5-VL-3B] Expecting local snapshot at: ${SNAPSHOT_PATH}"

if [ ! -d "${SNAPSHOT_PATH}" ]; then
  echo "[Qwen2.5-VL-3B] ERROR: Snapshot directory not found: ${SNAPSHOT_PATH}" >&2
  echo "[Qwen2.5-VL-3B] This model is distributed via ModelScope:" >&2
  echo "[Qwen2.5-VL-3B]   https://modelscope.cn/models/Qwen/Qwen2.5-VL-3B-Instruct" >&2
  echo "[Qwen2.5-VL-3B] Please download the model manually, for example:" >&2
  echo "[Qwen2.5-VL-3B]   pip install modelscope" >&2
  echo "[Qwen2.5-VL-3B]   modelscope download --model Qwen/Qwen2.5-VL-3B-Instruct --local_dir \"${SNAPSHOT_PATH}\"" >&2
  echo "[Qwen2.5-VL-3B] or download from the ModelScope web UI into ${SNAPSHOT_PATH}." >&2
  echo "[Qwen2.5-VL-3B] You can also set HF_SNAPSHOTS_ROOT to change the base directory." >&2
  echo "[Qwen2.5-VL-3B] Re-run this bootstrap script after the download completes." >&2
  exit 1
fi

mkdir -p "${CHECKPOINTS_DIR}"
TARGET_LINK="${CHECKPOINTS_DIR}/${MODEL_NAME}"

if [ -L "${TARGET_LINK}" ]; then
  # Existing symlink: check if it already points to the desired location.
  CURRENT_TARGET="$(readlink "${TARGET_LINK}")"
  if [ "${CURRENT_TARGET}" = "${SNAPSHOT_PATH}" ]; then
    echo "[Qwen2.5-VL-3B] Existing symlink is correct, nothing to do."
  else
    echo "[Qwen2.5-VL-3B] Updating existing symlink from ${CURRENT_TARGET} to ${SNAPSHOT_PATH}..."
    rm -f "${TARGET_LINK}"
    ln -s "${SNAPSHOT_PATH}" "${TARGET_LINK}"
  fi
else
  if [ -e "${TARGET_LINK}" ]; then
    echo "[Qwen2.5-VL-3B] WARNING: ${TARGET_LINK} exists and is not a symlink; not overwriting." >&2
  else
    echo "[Qwen2.5-VL-3B] Creating symlink ${TARGET_LINK} -> ${SNAPSHOT_PATH}..."
    ln -s "${SNAPSHOT_PATH}" "${TARGET_LINK}"
  fi
fi

echo "[Qwen2.5-VL-3B] Ensuring .gitignore rules for checkpoints..."
GITIGNORE_PATH="${MODEL_ROOT}/.gitignore"
touch "${GITIGNORE_PATH}"

ensure_gitignore_pattern() {
  local pattern="$1"
  if ! grep -qxF "${pattern}" "${GITIGNORE_PATH}"; then
    echo "${pattern}" >> "${GITIGNORE_PATH}"
  fi
}

ensure_gitignore_pattern "checkpoints/"

echo "[Qwen2.5-VL-3B] Bootstrap completed successfully."
