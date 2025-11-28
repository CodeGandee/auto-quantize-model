#!/usr/bin/env bash
set -euo pipefail

# Bootstrap script for YOLOv10:
# 1. Clone YOLOv10 source into models/yolo10/src with depth=1.
# 2. Download YOLOv10 checkpoints (nano/small/medium/large/xlarge/base) into models/yolo10/checkpoints.
# 3. Ensure models/yolo10/.gitignore ignores downloaded content and temp files.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YOLO10_ROOT="${SCRIPT_DIR}"
SRC_DIR="${YOLO10_ROOT}/src"
CHECKPOINTS_DIR="${YOLO10_ROOT}/checkpoints"
TMP_DIR="${YOLO10_ROOT}/tmp"

YOLO10_REPO_URL="https://github.com/THU-MIG/yolov10.git"
ASSETS_BASE_URL="https://github.com/jameslahm/yolov10/releases/download/v1.0"

# nano, small, medium, base, large, xlarge
MODELS=("n" "s" "m" "b" "l" "x")

echo "[YOLO10] Bootstrap starting..."

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

echo "[YOLO10] Checking out YOLOv10 source to ${SRC_DIR}..."
if [ -d "${SRC_DIR}/.git" ]; then
  echo "[YOLO10] Existing git repository found at ${SRC_DIR}, leaving as-is."
else
  if [ -d "${SRC_DIR}" ] && [ ! -d "${SRC_DIR}/.git" ]; then
    echo "[YOLO10] Warning: ${SRC_DIR} exists but is not a git repo. Skipping clone to avoid overwriting." >&2
  else
    rm -rf "${SRC_DIR}"
    git clone --depth 1 "${YOLO10_REPO_URL}" "${SRC_DIR}"
  fi
fi

echo "[YOLO10] Downloading checkpoints to ${CHECKPOINTS_DIR}..."
mkdir -p "${CHECKPOINTS_DIR}"

for size in "${MODELS[@]}"; do
  model_file="yolov10${size}.pt"
  url="${ASSETS_BASE_URL}/${model_file}"
  dest="${CHECKPOINTS_DIR}/${model_file}"

  if [ -f "${dest}" ]; then
    echo "[YOLO10] Checkpoint ${model_file} already exists, skipping."
  else
    echo "[YOLO10] Downloading ${model_file}..."
    download_file "${url}" "${dest}"
  fi
done

if [ -d "${TMP_DIR}" ]; then
  echo "[YOLO10] Cleaning up temporary directory ${TMP_DIR}..."
  rm -rf "${TMP_DIR}"
fi

echo "[YOLO10] Ensuring .gitignore rules for downloaded assets..."
GITIGNORE_PATH="${YOLO10_ROOT}/.gitignore"
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

echo "[YOLO10] Bootstrap completed successfully."

