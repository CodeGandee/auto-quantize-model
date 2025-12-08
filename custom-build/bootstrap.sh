#!/usr/bin/env bash
set -euo pipefail

# Bootstrap helper to link a locally built vLLM wheel (built from
# extern/vllm) into this repository under custom-build/.
#
# Behavior:
#   - Auto-discovers a vllm-*.whl in common locations:
#       * tmp/vllm-build/ (repo-local build-vllm output)
#       * extern/vllm/tmp/vllm-build/ (legacy location)
#       * ${VLLM_WHEEL_DIR}/ (if set)
#       * ${VLLM_WHEEL_PATH} (if set to an explicit file)
#       * /workspace/python-pkgs/
#   - Or uses an explicit wheel path provided via --path.
#   - Creates/updates a symlink in custom-build/ with the wheel's basename.
#   - Supports --clean to remove existing vLLM wheel links.
#
# Usage (from repo root or anywhere):
#   bash custom-build/bootstrap.sh
#   bash custom-build/bootstrap.sh --path /path/to/vllm-*.whl
#   bash custom-build/bootstrap.sh --clean

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

ASSUME_YES=false
CLEAN_ONLY=false
EXPLICIT_PATH=""

usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Link a locally built vLLM wheel into this repo under:
  ${SCRIPT_DIR}/vllm-*.whl

Options:
  -p, --path PATH   Use PATH as the vLLM wheel file (must exist).
  -y, --yes         Automatically confirm replacing an existing link/path.
      --clean       Remove existing vLLM wheel links from this directory and exit.
  -h, --help        Show this help and exit.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--path)
      if [[ $# -lt 2 ]]; then
        echo "Error: --path requires an argument." >&2
        exit 2
      fi
      EXPLICIT_PATH="$2"
      shift 2
      ;;
    -y|--yes)
      ASSUME_YES=true
      shift
      ;;
    --clean)
      CLEAN_ONLY=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if $CLEAN_ONLY; then
  echo "Cleaning vLLM wheel links in ${SCRIPT_DIR}..."
  mapfile -t LINKS < <(find "${SCRIPT_DIR}" -maxdepth 1 -type l -name 'vllm-*.whl' -print 2>/dev/null || true)
  if ((${#LINKS[@]} == 0)); then
    echo "No vLLM wheel symlinks to remove."
    exit 0
  fi
  for link in "${LINKS[@]}"; do
    echo "Removing symlink: ${link}"
    rm -f -- "${link}"
  done
  echo "Cleanup complete."
  exit 0
fi

discover_wheel() {
  local candidate=""

  if [[ -n "${EXPLICIT_PATH}" ]]; then
    candidate="${EXPLICIT_PATH}"
  elif [[ -n "${VLLM_WHEEL_PATH:-}" ]]; then
    candidate="${VLLM_WHEEL_PATH}"
  fi

  if [[ -n "${candidate}" ]]; then
    echo "${candidate}"
    return 0
  fi

  local search_dirs=()
  search_dirs+=("${ROOT_DIR}/tmp/vllm-build")
  search_dirs+=("${ROOT_DIR}/extern/vllm/tmp/vllm-build")

  if [[ -n "${VLLM_WHEEL_DIR:-}" ]]; then
    search_dirs+=("${VLLM_WHEEL_DIR}")
  fi

  search_dirs+=("/workspace/python-pkgs")

  for d in "${search_dirs[@]}"; do
    if [[ -d "${d}" ]]; then
      local latest
      latest="$(ls -1t "${d}"/vllm-*.whl 2>/dev/null | head -n1 || true)"
      if [[ -n "${latest}" ]]; then
        echo "${latest}"
        return 0
      fi
    fi
  done

  echo ""
}

WHEEL_PATH="$(discover_wheel)"

if [[ -z "${WHEEL_PATH}" ]]; then
  echo "Error: could not auto-discover a vllm-*.whl." >&2
  echo "Hint: build a wheel with pixi (e.g. 'pixi run build-vllm-wheel')" >&2
  echo "      or pass an explicit path with --path /path/to/vllm-*.whl." >&2
  exit 1
fi

if [[ ! -f "${WHEEL_PATH}" ]]; then
  echo "Error: wheel file does not exist: ${WHEEL_PATH}" >&2
  exit 1
fi

LINK_NAME="$(basename "${WHEEL_PATH}")"
LINK_PATH="${SCRIPT_DIR}/${LINK_NAME}"

echo "Using vLLM wheel: ${WHEEL_PATH}"
echo "Repo link path:   ${LINK_PATH}"

if [[ -e "${LINK_PATH}" || -L "${LINK_PATH}" ]]; then
  if $ASSUME_YES; then
    rm -rf -- "${LINK_PATH}"
  else
    echo "Path already exists at ${LINK_PATH}"
    read -r -p "Replace it with a symlink to ${WHEEL_PATH}? [y/N]: " answer
    case "${answer,,}" in
      y|yes)
        rm -rf -- "${LINK_PATH}"
        ;;
      *)
        echo "Aborting without changes."
        exit 0
        ;;
    esac
  fi
fi

ln -s -- "${WHEEL_PATH}" "${LINK_PATH}"
echo "Bootstrap complete: ${LINK_PATH} -> ${WHEEL_PATH}"

