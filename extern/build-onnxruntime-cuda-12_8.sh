#!/usr/bin/env bash

set -euo pipefail

# Helper script to configure and build ONNX Runtime with CUDA Execution Provider
# (CUDA toolkit 12.8.1) from the local checkout under extern/onnxruntime.
#
# This does NOT install anything into your Python environment by default. It
# builds a Python wheel that you can install manually.
#
# Usage (from repo root, inside your desired Python env, e.g. Pixi):
#   pixi run bash extern/build-onnxruntime-cuda-12_8.sh -o tmp/onnxruntime
#
# Required args:
#   -o, --output-dir   Directory to write all build artifacts/logs under
#                      (recommended: a task-specific subdir of tmp/).
#
# Environment variables:
#   CUDA_HOME      Path to CUDA 12.8.1 toolkit.
#                 If unset, tries to auto-detect a Pixi/conda CUDA layout under
#                 $CONDA_PREFIX/targets/*, otherwise falls back to
#                 /usr/local/cuda-12.8.1 (if present).
#   CUDNN_HOME     Path to cuDNN installation matching CUDA.
#                 If unset, tries to auto-detect under the Pixi/conda env first,
#                 and then /usr (Debian/Ubuntu).
#   ONNXR_BUILD_DIR
#                  Optional explicit build directory for ONNX Runtime.
#                  If unset, defaults to:
#                    <output-dir>/build/Linux/Release-cuda1281
#   ONNXR_CONFIG   Build config (default: Release)
#   ONNXR_CUDA_VERSION
#                  CUDA version passed to build.sh (default: 12.8)
#   ONNXR_CUDA_ARCHS
#                  CMAKE_CUDA_ARCHITECTURES (default: 120)
#   PYTHON         Python executable to use (default: python)
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ORT_DIR="${ROOT_DIR}/extern/onnxruntime"

CONDA_PREFIX="${CONDA_PREFIX:-}"
OUTPUT_DIR=""
PYTHON_BIN="${PYTHON:-python}"

print_help() {
  cat <<'EOF'
Build ONNX Runtime from extern/onnxruntime with CUDA EP (CUDA toolkit 12.8.1).

Usage:
  extern/build-onnxruntime-cuda-12_8.sh -o <output-dir> [-- <extra build.sh args...>]

Args:
  -o, --output-dir <dir>   Write build artifacts/logs under this directory
  -h, --help               Show this help

Examples:
  pixi run -e rtx5090 bash extern/build-onnxruntime-cuda-12_8.sh -o tmp/onnxruntime
  pixi run -e rtx5090 bash extern/build-onnxruntime-cuda-12_8.sh -o tmp/onnxruntime -- --parallel 12
EOF
}

EXTRA_BUILD_SH_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    -o|--output-dir)
      OUTPUT_DIR="${2:-}"
      shift 2
      ;;
    -h|--help)
      print_help
      exit 0
      ;;
    --)
      shift
      EXTRA_BUILD_SH_ARGS+=("$@")
      break
      ;;
    *)
      echo "[build-onnxruntime] ERROR: unknown argument: $1" >&2
      echo "  Run with --help for usage." >&2
      exit 2
      ;;
  esac
done

if [ -z "${OUTPUT_DIR}" ]; then
  echo "[build-onnxruntime] ERROR: missing required --output-dir/-o." >&2
  echo "  Example: pixi run -e rtx5090 bash extern/build-onnxruntime-cuda-12_8.sh -o tmp/onnxruntime" >&2
  exit 2
fi

case "${OUTPUT_DIR}" in
  /*) ;;
  *) OUTPUT_DIR="${ROOT_DIR}/${OUTPUT_DIR}" ;;
esac
OUTPUT_DIR="$(mkdir -p "${OUTPUT_DIR}" && cd "${OUTPUT_DIR}" && pwd)"

default_cuda_home() {
  local arch
  arch="$(uname -m)"

  find_cuda_home_in_prefix() {
    local prefix="$1"
    [ -z "${prefix}" ] && return 1

    if [ -d "${prefix}/targets" ]; then
      local preferred="${prefix}/targets/${arch}-linux"
      # Conda-forge CUDA layout:
      # - nvcc lives in <prefix>/bin
      # - headers/libs live in <prefix>/targets/<arch>-linux
      # Prefer returning <prefix> as CUDA_HOME so nvcc can locate its includes
      # via ../targets/<arch>-linux/include.
      if [ -x "${prefix}/bin/nvcc" ] && [ -f "${preferred}/include/cuda.h" ]; then
        echo "${prefix}"
        return 0
      fi

      [ -f "${preferred}/include/cuda.h" ] && { echo "${preferred}"; return 0; }

      local d
      for d in "${prefix}"/targets/*; do
        if [ -x "${prefix}/bin/nvcc" ] && [ -f "${d}/include/cuda.h" ]; then
          echo "${prefix}"
          return 0
        fi
        [ -f "${d}/include/cuda.h" ] && { echo "${d}"; return 0; }
      done
    fi

    # Non-conda layout fallback.
    [ -f "${prefix}/include/cuda.h" ] && { echo "${prefix}"; return 0; }
    return 1
  }

  local candidates=()
  [ -n "${CUDA_HOME:-}" ] && candidates+=("${CUDA_HOME}")
  [ -n "${CUDAToolkit_ROOT:-}" ] && candidates+=("${CUDAToolkit_ROOT}")
  [ -n "${CUDA_TOOLKIT_ROOT_DIR:-}" ] && candidates+=("${CUDA_TOOLKIT_ROOT_DIR}")
  [ -n "${CONDA_PREFIX:-}" ] && candidates+=("${CONDA_PREFIX}")

  if command -v nvcc >/dev/null 2>&1; then
    local nvcc_path nvcc_prefix
    nvcc_path="$(command -v nvcc)"
    if command -v readlink >/dev/null 2>&1; then
      nvcc_path="$(readlink -f "${nvcc_path}" || echo "${nvcc_path}")"
    fi
    nvcc_prefix="$(cd "$(dirname "${nvcc_path}")/.." && pwd)"
    candidates+=("${nvcc_prefix}")
  fi

  # System fallbacks (only used if Pixi/conda detection fails).
  candidates+=("/usr/local/cuda-12.8.1" "/usr/local/cuda")

  local c resolved
  for c in "${candidates[@]}"; do
    if resolved="$(find_cuda_home_in_prefix "${c}")"; then
      echo "${resolved}"
      return 0
    fi
  done

  echo ""
}

cuda_include_dir() {
  local arch
  arch="$(uname -m)"

  if [ -f "${CUDA_HOME}/include/cuda.h" ]; then
    echo "${CUDA_HOME}/include"
    return 0
  fi

  if [ -f "${CUDA_HOME}/targets/${arch}-linux/include/cuda.h" ]; then
    echo "${CUDA_HOME}/targets/${arch}-linux/include"
    return 0
  fi

  if [ -d "${CUDA_HOME}/targets" ]; then
    local d
    for d in "${CUDA_HOME}"/targets/*/include/cuda.h; do
      [ -f "${d}" ] && { echo "$(dirname "${d}")"; return 0; }
    done
  fi

  echo ""
}

default_cudnn_home() {
  if [ -n "${CUDNN_HOME:-}" ]; then
    echo "${CUDNN_HOME}"
    return 0
  fi

  # Prefer cuDNN from the active Python env if present (e.g. PyPI nvidia-cudnn-cu12).
  if command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    local python_cudnn_root=""
    python_cudnn_root="$("${PYTHON_BIN}" - <<'PY' 2>/dev/null || true
import importlib.util
import pathlib

spec = importlib.util.find_spec("nvidia.cudnn")
if not spec or not spec.submodule_search_locations:
  raise SystemExit(1)
root = pathlib.Path(list(spec.submodule_search_locations)[0])
if not (root / "include" / "cudnn.h").exists():
  raise SystemExit(1)
print(str(root))
PY
)"
    if [ -n "${python_cudnn_root}" ] && [ -f "${python_cudnn_root}/include/cudnn.h" ]; then
      echo "${python_cudnn_root}"
      return 0
    fi
  fi

  local arch
  arch="$(uname -m)"

  find_cudnn_home_in_prefix() {
    local prefix="$1"
    [ -z "${prefix}" ] && return 1
    [ -f "${prefix}/include/cudnn.h" ] && { echo "${prefix}"; return 0; }

    if [ -d "${prefix}/targets" ]; then
      local preferred="${prefix}/targets/${arch}-linux"
      [ -f "${preferred}/include/cudnn.h" ] && { echo "${preferred}"; return 0; }

      local d
      for d in "${prefix}"/targets/*; do
        [ -f "${d}/include/cudnn.h" ] && { echo "${d}"; return 0; }
      done
    fi
    return 1
  }

  # Prefer cuDNN from the current Pixi/conda env if available.
  local candidates=()
  [ -n "${CUDA_HOME:-}" ] && candidates+=("${CUDA_HOME}")
  [ -n "${CONDA_PREFIX:-}" ] && candidates+=("${CONDA_PREFIX}")

  local c resolved
  for c in "${candidates[@]}"; do
    if resolved="$(find_cudnn_home_in_prefix "${c}")"; then
      echo "${resolved}"
      return 0
    fi
  done

  # Debian/Ubuntu typically install cuDNN headers to /usr/include.
  if [ -f "/usr/include/cudnn.h" ]; then
    echo "/usr"
    return 0
  fi

  echo ""
}

CUDA_HOME="$(default_cuda_home)"
CUDNN_HOME="$(default_cudnn_home)"
ONNXR_CONFIG="${ONNXR_CONFIG:-Release}"
ONNXR_BUILD_DIR="${ONNXR_BUILD_DIR:-${OUTPUT_DIR}/build/Linux/Release-cuda1281}"
ONNXR_CUDA_VERSION="${ONNXR_CUDA_VERSION:-12.8}"
ONNXR_CUDA_ARCHS="${ONNXR_CUDA_ARCHS:-120}"

ONNXR_LOG_FILE="${ONNXR_LOG_FILE:-${OUTPUT_DIR}/logs/build-$(date -u +%Y%m%dT%H%M%SZ).log}"

mkdir -p "$(dirname "${ONNXR_LOG_FILE}")"
exec > >(tee -a "${ONNXR_LOG_FILE}") 2>&1

echo "[build-onnxruntime] Repo root: ${ROOT_DIR}"
echo "[build-onnxruntime] ORT source dir: ${ORT_DIR}"
echo "[build-onnxruntime] CONDA_PREFIX: ${CONDA_PREFIX:-<unset>}"
echo "[build-onnxruntime] Output dir: ${OUTPUT_DIR}"
echo "[build-onnxruntime] CUDA_HOME: ${CUDA_HOME}"
echo "[build-onnxruntime] CUDNN_HOME: ${CUDNN_HOME:-<unset>}"
echo "[build-onnxruntime] Config: ${ONNXR_CONFIG}"
echo "[build-onnxruntime] Build dir: ${ONNXR_BUILD_DIR}"
echo "[build-onnxruntime] CUDA version: ${ONNXR_CUDA_VERSION}"
echo "[build-onnxruntime] CUDA archs: ${ONNXR_CUDA_ARCHS}"
echo "[build-onnxruntime] Python: ${PYTHON_BIN}"
echo "[build-onnxruntime] Log file: ${ONNXR_LOG_FILE}"
if [ ${#EXTRA_BUILD_SH_ARGS[@]} -gt 0 ]; then
  echo "[build-onnxruntime] Extra build.sh args: ${EXTRA_BUILD_SH_ARGS[*]}"
fi

if [ ! -d "${ORT_DIR}" ]; then
  echo "[build-onnxruntime] ERROR: extern/onnxruntime not found. Run extern/bootstrap.sh first." >&2
  exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[build-onnxruntime] ERROR: Python executable '${PYTHON_BIN}' not found." >&2
  exit 1
fi

CUDA_INCLUDE_DIR="$(cuda_include_dir)"
if [ -z "${CUDA_INCLUDE_DIR}" ]; then
  echo "[build-onnxruntime] ERROR: CUDA headers not found under CUDA_HOME='${CUDA_HOME}'." >&2
  echo "  Hint: in Pixi/conda CUDA toolkits, CUDA headers are often under:" >&2
  echo "    \${CONDA_PREFIX}/targets/<arch>-linux/include/cuda.h" >&2
  exit 1
fi
echo "[build-onnxruntime] CUDA include dir: ${CUDA_INCLUDE_DIR}"

if [ -z "${CUDNN_HOME}" ]; then
  echo "[build-onnxruntime] ERROR: Could not auto-detect cuDNN. Set CUDNN_HOME explicitly." >&2
  echo "  Examples:" >&2
  echo "    export CUDNN_HOME=/usr" >&2
  echo "    export CUDNN_HOME=/usr/local/cuda-12.8.1" >&2
  exit 1
fi

if [ ! -f "${CUDNN_HOME}/include/cudnn.h" ]; then
  echo "[build-onnxruntime] ERROR: cuDNN headers not found at '${CUDNN_HOME}/include/cudnn.h'." >&2
  exit 1
fi

export CUDA_HOME
export CUDNN_HOME
export PATH="${CONDA_PREFIX:+${CONDA_PREFIX}/bin:}${CUDA_HOME}/bin:${PATH}"

# Force the host C++ compiler to use C++20 so that
# CMake try-compile checks (e.g., Abseil's C++20 probe)
# see a C++20-capable configuration even if the default
# standard is older.
# Also disable certain warnings as errors:
# - unused-parameter: CUDA 12.8's cuda_fp4.hpp has unused parameters
# - stringop-overflow: GCC 14.3.0's stricter checks flag false positives in CUTLASS code
export CXXFLAGS="${CXXFLAGS:-} -std=c++20 -Wno-error=unused-parameter -Wno-error=stringop-overflow"

# Fix linker to use conda sysroot instead of system /lib64 paths
# This prevents "cannot find /lib64/libm.so.6" errors
# The conda cross-compilation toolchain has a sysroot with all system libraries
CONDA_SYSROOT=""
if [ -n "${CONDA_PREFIX}" ]; then
  for d in "${CONDA_PREFIX}"/*-conda-linux-gnu/sysroot; do
    [ -d "${d}" ] && { CONDA_SYSROOT="${d}"; break; }
  done
fi
if [ -z "${CONDA_SYSROOT}" ]; then
  for d in "${CUDA_HOME}"/*-conda-linux-gnu/sysroot; do
    [ -d "${d}" ] && { CONDA_SYSROOT="${d}"; break; }
  done
fi
if [ -n "${CONDA_SYSROOT}" ]; then
  export LDFLAGS="${LDFLAGS:-} -Wl,--sysroot=${CONDA_SYSROOT} -L${CONDA_SYSROOT}/lib64"
fi

echo "[build-onnxruntime] Python version in this env:"
"${PYTHON_BIN}" -V

echo "[build-onnxruntime] Entering ${ORT_DIR} ..."
cd "${ORT_DIR}"

echo "[build-onnxruntime] Running build.sh with CUDA EP..."
set -x
./build.sh \
  --config "${ONNXR_CONFIG}" \
  --build_dir "${ONNXR_BUILD_DIR}" \
  --update --build --parallel 6 \
  --build_wheel \
  --use_cuda \
  --cuda_home "${CUDA_HOME}" \
  --cudnn_home "${CUDNN_HOME}" \
  --cuda_version "${ONNXR_CUDA_VERSION}" \
  --cmake_extra_defines "CMAKE_CUDA_ARCHITECTURES=${ONNXR_CUDA_ARCHS}" \
  --cmake_extra_defines "CMAKE_CXX_STANDARD=20" \
  --cmake_extra_defines "CMAKE_CXX_STANDARD_REQUIRED=ON" \
  --cmake_extra_defines "CMAKE_FIND_USE_CMAKE_SYSTEM_PATH=OFF" \
  --cmake_extra_defines "CMAKE_DISABLE_FIND_PACKAGE_Protobuf=ON" \
  --cmake_extra_defines "CMAKE_DISABLE_FIND_PACKAGE_dlpack=ON" \
  --cmake_extra_defines "CMAKE_DISABLE_FIND_PACKAGE_pybind11=ON" \
  --skip_tests \
  "${EXTRA_BUILD_SH_ARGS[@]}"
set +x

echo "[build-onnxruntime] Build complete. Wheel should be under:"
echo "  ${ONNXR_BUILD_DIR}/${ONNXR_CONFIG}/dist"
echo "[build-onnxruntime] Wheels found:"
find "${ONNXR_BUILD_DIR}" -maxdepth 5 -type f -path "*/dist/*.whl" -print || true
