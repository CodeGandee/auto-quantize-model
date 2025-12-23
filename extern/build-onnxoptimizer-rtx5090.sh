#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ONNXOPT_DIR="${ROOT_DIR}/extern/onnxoptimizer"
WHEEL_DIR="${ROOT_DIR}/custom-build"

usage() {
  cat <<'EOF'
Build and install onnxoptimizer in the current Pixi environment.

Usage:
  bash extern/build-onnxoptimizer-rtx5090.sh [--clean] [--update]

Options:
  --clean   Remove local build dir and any previously built onnxoptimizer wheels in custom-build/
  --update  Fetch latest origin/main (shallow) before building

Environment:
  MAX_JOBS        Build parallelism (default: nproc)
  CMAKE_ARGS      Extra CMake args appended after defaults
  ONNXOPT_CMAKE_ARGS  Override defaults entirely
EOF
}

CLEAN=0
UPDATE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean) CLEAN=1; shift ;;
    --update) UPDATE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if ! command -v python >/dev/null 2>&1; then
  echo "python not found; run via Pixi, e.g.:" >&2
  echo "  pixi run -e rtx5090 bash extern/build-onnxoptimizer-rtx5090.sh" >&2
  exit 1
fi

mkdir -p "${ROOT_DIR}/extern" "${WHEEL_DIR}"

if [[ ! -d "${ONNXOPT_DIR}/.git" ]]; then
  git clone --depth=1 https://github.com/onnx/optimizer.git "${ONNXOPT_DIR}"
fi

if [[ "${UPDATE}" -eq 1 ]]; then
  git -C "${ONNXOPT_DIR}" fetch --depth=1 origin main
  git -C "${ONNXOPT_DIR}" checkout -f origin/main
fi

git -C "${ONNXOPT_DIR}" submodule update --init --recursive

if [[ "${CLEAN}" -eq 1 ]]; then
  rm -rf "${ONNXOPT_DIR}"/.setuptools-cmake-build* || true
  rm -f "${WHEEL_DIR}"/onnxoptimizer-*.whl || true
fi

# Known-good flags for this container:
# - protobuf needs Abseil headers; force fetching a compatible version.
# - zlib headers may be missing even when libz is present; build without zlib.
DEFAULT_CMAKE_ARGS="-Dprotobuf_FORCE_FETCH_DEPENDENCIES=ON -Dprotobuf_WITH_ZLIB=OFF"
if [[ -n "${ONNXOPT_CMAKE_ARGS:-}" ]]; then
  export CMAKE_ARGS="${ONNXOPT_CMAKE_ARGS}"
else
  export CMAKE_ARGS="${DEFAULT_CMAKE_ARGS} ${CMAKE_ARGS:-}"
fi

export MAX_JOBS="${MAX_JOBS:-$(nproc)}"

python -m pip wheel --no-deps -w "${WHEEL_DIR}" "${ONNXOPT_DIR}"
python -m pip install --no-deps --force-reinstall "${WHEEL_DIR}"/onnxoptimizer-*.whl

python -c "import onnxoptimizer; print('onnxoptimizer', onnxoptimizer.__version__)"
