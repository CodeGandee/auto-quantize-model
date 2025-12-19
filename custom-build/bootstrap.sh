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
#   - Optionally links an ONNX Runtime GPU wheel built from extern/onnxruntime
#     (via extern/build-onnxruntime-cuda-12_8.sh), typically staged under tmp/
#     or /workspace/source-builds.
#   - Supports --clean to remove existing wheel links.
#
# Usage (from repo root or anywhere):
#   bash custom-build/bootstrap.sh
#   bash custom-build/bootstrap.sh --vllm-path /path/to/vllm-*.whl
#   bash custom-build/bootstrap.sh --onnxruntime-path /path/to/onnxruntime_gpu-*.whl
#   bash custom-build/bootstrap.sh --clean

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

ASSUME_YES=false
CLEAN_ONLY=false
ARTIFACT="all"
VLLM_EXPLICIT_PATH=""
ONNXR_EXPLICIT_PATH=""

usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Link locally built wheels into this repo under:
  ${SCRIPT_DIR}/*.whl

Options:
  -a, --artifact NAME
                   What to link: vllm, onnxruntime, or all (default: all).
  -p, --path PATH   Alias for --vllm-path (backwards compatible).
      --vllm-path PATH
                   Use PATH as the vLLM wheel file (must exist).
      --onnxruntime-path PATH
                   Use PATH as the ONNX Runtime GPU wheel file (must exist).
  -y, --yes         Automatically confirm replacing an existing link/path.
      --clean       Remove existing wheel links from this directory and exit.
  -h, --help        Show this help and exit.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -a|--artifact)
      if [[ $# -lt 2 ]]; then
        echo "Error: --artifact requires an argument." >&2
        exit 2
      fi
      ARTIFACT="$2"
      shift 2
      ;;
    -p|--path)
      if [[ $# -lt 2 ]]; then
        echo "Error: --path requires an argument." >&2
        exit 2
      fi
      VLLM_EXPLICIT_PATH="$2"
      shift 2
      ;;
    --vllm-path)
      if [[ $# -lt 2 ]]; then
        echo "Error: --vllm-path requires an argument." >&2
        exit 2
      fi
      VLLM_EXPLICIT_PATH="$2"
      shift 2
      ;;
    --onnxruntime-path)
      if [[ $# -lt 2 ]]; then
        echo "Error: --onnxruntime-path requires an argument." >&2
        exit 2
      fi
      ONNXR_EXPLICIT_PATH="$2"
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

case "${ARTIFACT}" in
  all|vllm|onnxruntime) ;;
  *)
    echo "Error: invalid --artifact: ${ARTIFACT} (expected: all|vllm|onnxruntime)" >&2
    exit 2
    ;;
esac

if $CLEAN_ONLY; then
  echo "Cleaning wheel links in ${SCRIPT_DIR}..."
  mapfile -t LINKS < <(find "${SCRIPT_DIR}" -maxdepth 1 -type l \( -name 'vllm-*.whl' -o -name 'onnxruntime_gpu-*.whl' \) -print 2>/dev/null || true)
  if ((${#LINKS[@]} == 0)); then
    echo "No wheel symlinks to remove."
    exit 0
  fi
  for link in "${LINKS[@]}"; do
    echo "Removing symlink: ${link}"
    rm -f -- "${link}"
  done
  echo "Cleanup complete."
  exit 0
fi

discover_vllm_wheel() {
  local candidate=""

  if [[ -n "${VLLM_EXPLICIT_PATH}" ]]; then
    candidate="${VLLM_EXPLICIT_PATH}"
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

discover_onnxruntime_wheel() {
  local candidate=""

  if [[ -n "${ONNXR_EXPLICIT_PATH}" ]]; then
    candidate="${ONNXR_EXPLICIT_PATH}"
  elif [[ -n "${ONNXR_WHEEL_PATH:-}" ]]; then
    candidate="${ONNXR_WHEEL_PATH}"
  fi

  if [[ -n "${candidate}" ]]; then
    echo "${candidate}"
    return 0
  fi

  local source_builds_root="${SOURCE_BUILDS_ROOT:-/workspace/source-builds}"
  local search_dirs=()
  search_dirs+=("${ROOT_DIR}/tmp")

  if [[ -n "${ONNXR_WHEEL_DIR:-}" ]]; then
    search_dirs+=("${ONNXR_WHEEL_DIR}")
  fi

  # Prefer /workspace/source-builds, which may contain timestamped build roots.
  if [[ -d "${source_builds_root}" ]]; then
    local latest
    latest="$(ls -1t "${source_builds_root}"/*/build/Linux/*/*/dist/onnxruntime_gpu-*.whl 2>/dev/null | head -n1 || true)"
    if [[ -n "${latest}" ]]; then
      echo "${latest}"
      return 0
    fi
  fi

  # Fall back to scanning common repo-local build outputs.
  local d
  for d in "${search_dirs[@]}"; do
    if [[ -d "${d}" ]]; then
      local latest
      latest="$(find "${d}" -maxdepth 6 -type f -name 'onnxruntime_gpu-*.whl' -print 2>/dev/null | xargs -r ls -1t 2>/dev/null | head -n1 || true)"
      if [[ -n "${latest}" ]]; then
        echo "${latest}"
        return 0
      fi
    fi
  done

  echo ""
}

link_wheel() {
  local wheel_path="$1"
  local label="$2"

  if [[ -z "${wheel_path}" ]]; then
    echo "Error: could not auto-discover ${label} wheel." >&2
    return 1
  fi

  if [[ ! -f "${wheel_path}" ]]; then
    echo "Error: wheel file does not exist: ${wheel_path}" >&2
    return 1
  fi

  local link_name
  link_name="$(basename "${wheel_path}")"
  local link_path="${SCRIPT_DIR}/${link_name}"

  echo "Using ${label} wheel: ${wheel_path}"
  echo "Repo link path:     ${link_path}"

  if [[ -e "${link_path}" || -L "${link_path}" ]]; then
    if $ASSUME_YES; then
      rm -rf -- "${link_path}"
    else
      echo "Path already exists at ${link_path}"
      read -r -p "Replace it with a symlink to ${wheel_path}? [y/N]: " answer
      case "${answer,,}" in
        y|yes)
          rm -rf -- "${link_path}"
          ;;
        *)
          echo "Skipping ${label} without changes."
          return 0
          ;;
      esac
    fi
  fi

  ln -s -- "${wheel_path}" "${link_path}"
  echo "Linked: ${link_path} -> ${wheel_path}"
}

if [[ "${ARTIFACT}" == "all" || "${ARTIFACT}" == "vllm" ]]; then
  VLLM_WHEEL_PATH_FOUND="$(discover_vllm_wheel)"
  if [[ -z "${VLLM_WHEEL_PATH_FOUND}" ]]; then
    echo "Warning: could not auto-discover a vllm-*.whl." >&2
    echo "  Hint: build a wheel with pixi (e.g. 'pixi run build-vllm-wheel')" >&2
    echo "        or pass an explicit path with --vllm-path /path/to/vllm-*.whl." >&2
    if [[ "${ARTIFACT}" == "vllm" ]]; then
      exit 1
    fi
  else
    link_wheel "${VLLM_WHEEL_PATH_FOUND}" "vLLM"
  fi
fi

if [[ "${ARTIFACT}" == "all" || "${ARTIFACT}" == "onnxruntime" ]]; then
  ONNXR_WHEEL_PATH_FOUND="$(discover_onnxruntime_wheel)"
  if [[ -z "${ONNXR_WHEEL_PATH_FOUND}" ]]; then
    echo "Warning: could not auto-discover an onnxruntime_gpu-*.whl." >&2
    echo "  Hint: build a wheel with:" >&2
    echo "        pixi run -e rtx5090 bash extern/build-onnxruntime-cuda-12_8.sh -o tmp/onnxruntime-build" >&2
    echo "        or pass an explicit path with --onnxruntime-path /path/to/onnxruntime_gpu-*.whl." >&2
    if [[ "${ARTIFACT}" == "onnxruntime" ]]; then
      exit 1
    fi
  else
    link_wheel "${ONNXR_WHEEL_PATH_FOUND}" "ONNX Runtime GPU"
  fi
fi

echo "Bootstrap complete."
