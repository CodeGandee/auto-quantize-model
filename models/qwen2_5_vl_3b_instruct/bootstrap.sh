#!/usr/bin/env bash
set -euo pipefail

# Bootstrap script for Qwen2.5-VL-3B-Instruct:
# - Discover a default snapshot directory from ENV / YAML.
# - Ask the user to confirm or override that location if not found.
# - Create a symlink in checkpoints/ pointing to the chosen directory.
# - Record the chosen directory back into boostrap.yaml for future runs.

require_cmd() {
  for cmd in "$@"; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
      echo "Error: required command not found in PATH: $cmd" >&2
      exit 127
    fi
  done
}

require_cmd yq ln mkdir grep date python3

record_choice() {
  local snapshot_path="$1"
  local cfg_path="$2"
  local ts
  ts="$(date -Iseconds)"

  # Edit the YAML file in-place without requiring a specific yq flavor.
  # Only updates/creates these keys under the top-level `choices:` block:
  #   snapshot_path
  #   updated_at
  python3 - "${cfg_path}" "${snapshot_path}" "${ts}" <<'PY'
import re
import sys

cfg, snapshot_path, ts = sys.argv[1:]

with open(cfg, "r", encoding="utf-8") as f:
  lines = f.read().splitlines(True)

out = []
in_choices = False
seen_choices = False
seen_path = False
seen_ts = False

for line in lines:
  if re.match(r"^choices:\s*$", line):
    in_choices = True
    seen_choices = True
    out.append(line)
    continue

  if in_choices and re.match(r"^\S", line):
    # leaving choices block
    if not seen_path:
      out.append(f'  snapshot_path: "{snapshot_path}"\n')
      seen_path = True
    if not seen_ts:
      out.append(f'  updated_at: "{ts}"\n')
      seen_ts = True
    in_choices = False

  if in_choices and re.match(r"^\s{2}snapshot_path:\s*", line):
    out.append(f'  snapshot_path: "{snapshot_path}"\n')
    seen_path = True
    continue
  if in_choices and re.match(r"^\s{2}updated_at:\s*", line):
    out.append(f'  updated_at: "{ts}"\n')
    seen_ts = True
    continue

  out.append(line)

if not seen_choices:
  # Ensure file ends with a newline before appending.
  if out and not out[-1].endswith("\n"):
    out[-1] = out[-1] + "\n"
  out.append("\nchoices:\n")
  out.append(f'  snapshot_path: "{snapshot_path}"\n')
  out.append(f'  updated_at: "{ts}"\n')
else:
  # choices block exists and file ended while still inside it.
  if in_choices:
    if not seen_path:
      out.append(f'  snapshot_path: "{snapshot_path}"\n')
    if not seen_ts:
      out.append(f'  updated_at: "{ts}"\n')

with open(cfg, "w", encoding="utf-8") as f:
  f.write("".join(out))
PY
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ROOT="${SCRIPT_DIR}"

CFG="${SCRIPT_DIR}/boostrap.yaml"
if [[ ! -f "${CFG}" ]]; then
  echo "Error: missing config file: ${CFG}" >&2
  echo "This repo expects ${CFG} to exist (it records your local model path)." >&2
  exit 1
fi

MODEL_NAME="$(yq -r '.model.model_name' "${CFG}")"
SNAPSHOTS_ROOT_ENV="$(yq -r '.env.snapshots_root_env' "${CFG}")"
DEFAULT_SNAPSHOTS_ROOT="$(yq -r '.env.default_snapshots_root' "${CFG}")"
CHECKPOINTS_SUBDIR="$(yq -r '.link.checkpoints_subdir' "${CFG}")"
LINK_NAME="$(yq -r '.link.repo_link_name' "${CFG}")"
SAVED_SNAPSHOT_PATH="$(yq -r '.choices.snapshot_path // ""' "${CFG}")"

CHECKPOINTS_DIR="${MODEL_ROOT}/${CHECKPOINTS_SUBDIR}"
TARGET_LINK="${CHECKPOINTS_DIR}/${LINK_NAME}"

ASSUME_YES=false
CLEAN_ONLY=false
SNAPSHOT_PATH=""

usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Interactive helper to link the local model snapshot into this repo.

It will create:
  ${TARGET_LINK} -> /absolute/path/to/your/${MODEL_NAME}

Options:
  -p, --path PATH   Use PATH as the model snapshot directory (must exist).
  -y, --yes         Automatically confirm replacing an existing link/path.
      --clean       Remove the checkpoints symlink (if present) and exit.
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
      SNAPSHOT_PATH="$2"
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
  echo "Cleaning model link: ${TARGET_LINK}"
  if [[ -L "${TARGET_LINK}" ]]; then
    rm -f -- "${TARGET_LINK}"
    echo "Removed symlink ${TARGET_LINK}"
  elif [[ -e "${TARGET_LINK}" ]]; then
    echo "exists but is not a symlink: ${TARGET_LINK}" >&2
    echo "Not removing it; please inspect manually." >&2
  else
    echo "already clean; ${TARGET_LINK} does not exist."
  fi
  exit 0
fi

if [[ -z "${SNAPSHOT_PATH}" ]]; then
  set +u
  ENV_RAW="${!SNAPSHOTS_ROOT_ENV-}"
  set -u

  CANDIDATE=""
  if [[ -n "${SAVED_SNAPSHOT_PATH}" ]]; then
    CANDIDATE="${SAVED_SNAPSHOT_PATH}"
  else
    BASE="${ENV_RAW:-${DEFAULT_SNAPSHOTS_ROOT}}"
    CANDIDATE="${BASE}/${MODEL_NAME}"
  fi

  echo "Model discovery:"
  echo "  model name      : ${MODEL_NAME}"
  echo "  env var name    : ${SNAPSHOTS_ROOT_ENV}"
  echo "  env var value   : ${ENV_RAW:-<unset>}"
  echo "  default root    : ${DEFAULT_SNAPSHOTS_ROOT}"
  echo "  saved path      : ${SAVED_SNAPSHOT_PATH:-<unset>}"
  echo "  candidate path  : ${CANDIDATE}"
  echo "  repo link path  : ${TARGET_LINK}"

  if [[ -d "${CANDIDATE}" ]]; then
    read -r -p "Use this model directory? [Y/n]: " answer
    case "${answer,,}" in
      ""|y|yes)
        SNAPSHOT_PATH="${CANDIDATE}"
        ;;
      *)
        :
        ;;
    esac
  else
    echo "Candidate directory does not exist: ${CANDIDATE}"
  fi
fi

if [[ -z "${SNAPSHOT_PATH}" ]]; then
  echo "Enter the physical path to your local model directory for ${MODEL_NAME}."
  echo "Example ModelScope download (optional):"
  echo "  pip install modelscope"
  echo "  modelscope download --model Qwen/${MODEL_NAME} --local_dir \"/path/to/${MODEL_NAME}\""
  read -r -p "${MODEL_NAME} path: " SNAPSHOT_PATH
fi

if [[ -z "${SNAPSHOT_PATH}" ]]; then
  echo "Error: model path is required; none provided." >&2
  exit 1
fi

if [[ ! -d "${SNAPSHOT_PATH}" ]]; then
  echo "Error: directory does not exist: ${SNAPSHOT_PATH}" >&2
  exit 1
fi

mkdir -p "${CHECKPOINTS_DIR}"

echo "Using model directory: ${SNAPSHOT_PATH}"
echo "Repo link path:       ${TARGET_LINK}"

if [[ -e "${TARGET_LINK}" || -L "${TARGET_LINK}" ]]; then
  if $ASSUME_YES; then
    rm -rf -- "${TARGET_LINK}"
  else
    echo "Path already exists at ${TARGET_LINK}"
    read -r -p "Replace it with a symlink to ${SNAPSHOT_PATH}? [y/N]: " answer
    case "${answer,,}" in
      y|yes)
        rm -rf -- "${TARGET_LINK}"
        ;;
      *)
        echo "Aborting without changes."
        exit 0
        ;;
    esac
  fi
fi

ln -s -- "${SNAPSHOT_PATH}" "${TARGET_LINK}"
echo "Model bootstrap: ${TARGET_LINK} -> ${SNAPSHOT_PATH}"

echo "Recording selected path into ${CFG}..."
record_choice "${SNAPSHOT_PATH}" "${CFG}"

echo "Ensuring .gitignore rules for checkpoints..."
GITIGNORE_PATH="${MODEL_ROOT}/.gitignore"
touch "${GITIGNORE_PATH}"

ensure_gitignore_pattern() {
  local pattern="$1"
  if ! grep -qxF "${pattern}" "${GITIGNORE_PATH}"; then
    echo "${pattern}" >> "${GITIGNORE_PATH}"
  fi
}

ensure_gitignore_pattern "${CHECKPOINTS_SUBDIR}/"

echo "Bootstrap completed."
