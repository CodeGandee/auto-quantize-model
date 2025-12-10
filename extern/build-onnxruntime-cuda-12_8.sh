#!/usr/bin/env bash

set -euo pipefail

# Helper script to configure and build ONNX Runtime with CUDA Execution Provider
# (CUDA toolkit 12.8.1) from the local checkout under extern/onnxruntime.
#
# This does NOT install anything into your Python environment by default. It
# builds a Python wheel that you can install manually.
#
# Usage (from repo root, inside your desired Python env, e.g. Pixi):
#   pixi run bash extern/build-onnxruntime-cuda-12_8.sh
#
# Environment variables:
#   CUDA_HOME      Path to CUDA 12.8.1 toolkit (default: /usr/local/cuda-12.8.1)
#   CUDNN_HOME     Path to cuDNN installation matching CUDA (default: $CUDA_HOME)
#   ONNXR_BUILD_DIR
#                  Build directory for ONNX Runtime
#                  (default: extern/onnxruntime/build/Linux/Release-cuda1281)
#   ONNXR_CONFIG   Build config (default: Release)
#   ONNXR_CUDA_VERSION
#                  CUDA version passed to build.sh (default: 12.8)
#   ONNXR_CUDA_ARCHS
#                  CMAKE_CUDA_ARCHITECTURES (default: 80;86;89)
#   PYTHON         Python executable to use (default: python)
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ORT_DIR="${ROOT_DIR}/extern/onnxruntime"

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8.1}"
CUDNN_HOME="${CUDNN_HOME:-${CUDA_HOME}}"
ONNXR_CONFIG="${ONNXR_CONFIG:-Release}"
ONNXR_BUILD_DIR="${ONNXR_BUILD_DIR:-${ORT_DIR}/build/Linux/Release-cuda1281}"
ONNXR_CUDA_VERSION="${ONNXR_CUDA_VERSION:-12.8}"
ONNXR_CUDA_ARCHS="${ONNXR_CUDA_ARCHS:-120}"
PYTHON_BIN="${PYTHON:-python}"

echo "[build-onnxruntime] Repo root: ${ROOT_DIR}"
echo "[build-onnxruntime] ORT source dir: ${ORT_DIR}"
echo "[build-onnxruntime] CUDA_HOME: ${CUDA_HOME}"
echo "[build-onnxruntime] CUDNN_HOME: ${CUDNN_HOME}"
echo "[build-onnxruntime] Config: ${ONNXR_CONFIG}"
echo "[build-onnxruntime] Build dir: ${ONNXR_BUILD_DIR}"
echo "[build-onnxruntime] CUDA version: ${ONNXR_CUDA_VERSION}"
echo "[build-onnxruntime] CUDA archs: ${ONNXR_CUDA_ARCHS}"
echo "[build-onnxruntime] Python: ${PYTHON_BIN}"

if [ ! -d "${ORT_DIR}" ]; then
  echo "[build-onnxruntime] ERROR: extern/onnxruntime not found. Run extern/bootstrap.sh first." >&2
  exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[build-onnxruntime] ERROR: Python executable '${PYTHON_BIN}' not found." >&2
  exit 1
fi

export CUDA_HOME
export CUDNN_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"

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
CONDA_SYSROOT="${CUDA_HOME}/x86_64-conda-linux-gnu/sysroot"
export LDFLAGS="${LDFLAGS:-} -Wl,--sysroot=${CONDA_SYSROOT} -L${CONDA_SYSROOT}/lib64"

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
  --skip_tests
set +x

echo "[build-onnxruntime] Build complete. Wheel should be under:"
echo "  ${ONNXR_BUILD_DIR}/dist"
