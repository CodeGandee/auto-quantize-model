#!/usr/bin/env bash
set -euo pipefail

# Bootstrap script for YOLO11:
# 1. Clone YOLO11 source (Ultralytics) into models/yolo11/src with depth=1.
# 2. Download YOLO11 checkpoints (nano/small/medium/large/xlarge) into models/yolo11/checkpoints.
# 3. Ensure models/yolo11/.gitignore ignores downloaded content.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YOLO11_ROOT="${SCRIPT_DIR}"
SRC_DIR="${YOLO11_ROOT}/src"
CHECKPOINTS_DIR="${YOLO11_ROOT}/checkpoints"
TMP_DIR="${YOLO11_ROOT}/tmp"

YOLO11_REPO_URL="https://github.com/ultralytics/ultralytics.git"
ASSETS_BASE_URL="https://github.com/ultralytics/assets/releases/download/v8.3.0"

MODELS=("n" "s" "m" "l" "x")  # nano, small, medium, large, xlarge

echo "[YOLO11] Bootstrap starting..."

download_file() {
  local url="$1"
  local dest="$2"

  mkdir -p "$(dirname "$dest")"

  if command -v curl >/dev/null 2>&1; then
    curl -L -o "$dest" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$dest" "$url"
  else
    echo "Error: neither curl nor wget is available for downloading files." >&2
    exit 1
  fi
}

echo "[YOLO11] Checking out YOLO11 source to ${SRC_DIR}..."
if [ -d "${SRC_DIR}/.git" ]; then
  echo "[YOLO11] Existing git repository found at ${SRC_DIR}, leaving as-is."
else
  if [ -d "${SRC_DIR}" ] && [ ! -d "${SRC_DIR}/.git" ]; then
    echo "[YOLO11] Warning: ${SRC_DIR} exists but is not a git repo. Skipping clone to avoid overwriting." >&2
  else
    rm -rf "${SRC_DIR}"
    git clone --depth 1 "${YOLO11_REPO_URL}" "${SRC_DIR}"
  fi
fi

echo "[YOLO11] Downloading checkpoints to ${CHECKPOINTS_DIR}..."
mkdir -p "${CHECKPOINTS_DIR}"

for size in "${MODELS[@]}"; do
  model_file="yolo11${size}.pt"
  url="${ASSETS_BASE_URL}/${model_file}"
  dest="${CHECKPOINTS_DIR}/${model_file}"

  if [ -f "${dest}" ]; then
    echo "[YOLO11] Checkpoint ${model_file} already exists, skipping."
  else
    echo "[YOLO11] Downloading ${model_file}..."
    download_file "${url}" "${dest}"
  fi
done

# If a temporary directory was used for archives, clean it up.
if [ -d "${TMP_DIR}" ]; then
  echo "[YOLO11] Cleaning up temporary directory ${TMP_DIR}..."
  rm -rf "${TMP_DIR}"
fi

echo "[YOLO11] Ensuring .gitignore rules for downloaded assets..."
GITIGNORE_PATH="${YOLO11_ROOT}/.gitignore"
touch "${GITIGNORE_PATH}"

ensure_gitignore_pattern() {
  local pattern="$1"
  if ! grep -qxF "${pattern}" "${GITIGNORE_PATH}"; then
    echo "${pattern}" >> "${GITIGNORE_PATH}"
  fi
}

ensure_gitignore_pattern "src/"
ensure_gitignore_pattern "checkpoints/"
ensure_gitignore_pattern "tmp/"

echo "[YOLO11] Bootstrap completed successfully."
