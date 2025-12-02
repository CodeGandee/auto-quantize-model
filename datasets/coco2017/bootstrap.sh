#!/usr/bin/env bash
set -euo pipefail

# Bootstrap script for COCO 2017 dataset:
# - Discover a default dataset directory from ENV / YAML.
# - Ask the user to confirm or override that location.
# - Create a symlink "source-data" pointing to the chosen directory.
# - Optionally clean up the symlink with --clean.

require_cmd() {
  for cmd in "$@"; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
      echo "Error: required command not found in PATH: $cmd" >&2
      exit 127
    fi
  done
}

require_cmd yq ln mkdir

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG="${SCRIPT_DIR}/bootstrap.yaml"

if [[ ! -f "${CFG}" ]]; then
  echo "Error: missing config file: ${CFG}" >&2
  exit 1
fi

DATA_ROOT_ENV="$(yq -r '.env.data_root_env' "${CFG}")"
DEFAULT_DATA_ROOT="$(yq -r '.env.default_data_root' "${CFG}")"
SRC_SUBDIR="$(yq -r '.dataset.source_subdir' "${CFG}")"
LINK_NAME="$(yq -r '.dataset.repo_link_name' "${CFG}")"

LINK_PATH="${SCRIPT_DIR}/${LINK_NAME}"

ASSUME_YES=false
CLEAN_ONLY=false
DATASET_DIR=""

usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Interactive helper to link the local COCO 2017 dataset into this repo.

It will create:
  ${LINK_PATH} -> /absolute/path/to/your/coco2017

Options:
  -p, --path PATH   Use PATH as the coco2017 dataset directory (must exist).
  -y, --yes         Automatically confirm replacing an existing link/path.
      --clean       Remove the source-data symlink (if present) and exit.
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
      DATASET_DIR="$2"
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
  echo "Cleaning dataset link: ${LINK_PATH}"
  if [[ -L "$LINK_PATH" ]]; then
    rm -f -- "$LINK_PATH"
    echo "Removed symlink ${LINK_PATH}"
  elif [[ -e "$LINK_PATH" ]]; then
    echo "exists but is not a symlink: ${LINK_PATH}" >&2
    echo "Not removing it; please inspect manually." >&2
  else
    echo "already clean; ${LINK_PATH} does not exist."
  fi
  exit 0
fi

if [[ -z "${DATASET_DIR}" ]]; then
  # Discover a candidate directory from env/default.
  set +u
  ENV_RAW="${!DATA_ROOT_ENV-}"
  set -u

  # Treat DEFAULT_DATA_ROOT / ENV_RAW as full dataset directory for COCO2017.
  BASE="${ENV_RAW:-${DEFAULT_DATA_ROOT}}"
  CANDIDATE="${BASE}"

  echo "Dataset discovery:"
  echo "  env var name   : ${DATA_ROOT_ENV}"
  echo "  env var value  : ${ENV_RAW:-<unset>}"
  echo "  default root   : ${DEFAULT_DATA_ROOT}"
  echo "  candidate path : ${CANDIDATE}"
  echo "  repo link path : ${LINK_PATH}"

  if [[ -d "${CANDIDATE}" ]]; then
    read -r -p "Use this dataset directory? [Y/n]: " answer
    case "${answer,,}" in
      ""|y|yes)
        DATASET_DIR="${CANDIDATE}"
        ;;
      *)
        :
        ;;
    esac
  else
    echo "Candidate directory does not exist: ${CANDIDATE}"
  fi
fi

if [[ -z "${DATASET_DIR}" ]]; then
  echo "Enter the physical path to your COCO 2017 dataset directory."
  read -r -p "COCO 2017 dataset path: " DATASET_DIR
fi

if [[ -z "${DATASET_DIR}" ]]; then
  echo "Error: dataset path is required; none provided." >&2
  exit 1
fi

if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "Error: directory does not exist: ${DATASET_DIR}" >&2
  exit 1
fi

echo "Using dataset directory: ${DATASET_DIR}"
echo "Repo link path:        ${LINK_PATH}"

mkdir -p "${SCRIPT_DIR}"

if [[ -e "${LINK_PATH}" || -L "${LINK_PATH}" ]]; then
  if $ASSUME_YES; then
    rm -rf -- "${LINK_PATH}"
  else
    echo "Path already exists at ${LINK_PATH}"
    read -r -p "Replace it with a symlink to ${DATASET_DIR}? [y/N]: " answer
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

ln -s -- "${DATASET_DIR}" "${LINK_PATH}"
echo "Dataset bootstrap: ${LINK_PATH} -> ${DATASET_DIR}"
echo "Bootstrap completed."
