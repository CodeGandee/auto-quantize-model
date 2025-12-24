#!/usr/bin/env bash
set -euo pipefail

# Bootstrap script for YOLOv10:
# - YAML-driven configuration (boostrap.yaml)
# - Discover default assets dir (saved choice -> env -> yaml default)
# - If default is not usable, prompt user to locate it
# - Clone source + download checkpoints into the chosen assets dir
# - Persist choice back to boostrap.yaml for future runs

require_cmd() {
  for cmd in "$@"; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
      echo "Error: required command not found in PATH: $cmd" >&2
      exit 127
    fi
  done
}

require_cmd yq python3 git ln mkdir grep date

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YOLO10_ROOT="${SCRIPT_DIR}"

CFG="${SCRIPT_DIR}/boostrap.yaml"
if [[ ! -f "${CFG}" ]]; then
  echo "Error: missing config file: ${CFG}" >&2
  exit 1
fi

ASSETS_DIR_ENV="$(yq -r '.env.assets_dir_env' "${CFG}")"
DEFAULT_ASSETS_DIR_REL="$(yq -r '.env.default_assets_dir' "${CFG}")"
SRC_SUBDIR="$(yq -r '.layout.src_subdir' "${CFG}")"
CHECKPOINTS_SUBDIR="$(yq -r '.layout.checkpoints_subdir' "${CFG}")"
TMP_SUBDIR="$(yq -r '.layout.tmp_subdir' "${CFG}")"
YOLO10_REPO_URL="$(yq -r '.source.repo_url' "${CFG}")"
ASSETS_BASE_URL="$(yq -r '.checkpoints.assets_base_url' "${CFG}")"
MODELS_CSV="$(yq -r '.checkpoints.sizes | join(",")' "${CFG}")"
FILE_TEMPLATE="$(yq -r '.checkpoints.file_template' "${CFG}")"
SAVED_ASSETS_DIR="$(yq -r '.choices.assets_dir // ""' "${CFG}")"

ASSUME_YES=false
CLEAN_ONLY=false
ASSETS_DIR=""

record_choice() {
  local assets_dir="$1"
  local cfg_path="$2"
  local ts
  ts="$(date -Iseconds)"

  python3 - "${cfg_path}" "${assets_dir}" "${ts}" <<'PY'
import re
import sys

cfg, assets_dir, ts = sys.argv[1:]

with open(cfg, "r", encoding="utf-8") as f:
    lines = f.read().splitlines(True)

out = []
in_choices = False
seen_choices = False
seen_dir = False
seen_ts = False

for line in lines:
    if re.match(r"^choices:\s*$", line):
        in_choices = True
        seen_choices = True
        out.append(line)
        continue

    if in_choices and re.match(r"^\S", line):
        if not seen_dir:
            out.append(f'  assets_dir: "{assets_dir}"\n')
            seen_dir = True
        if not seen_ts:
            out.append(f'  updated_at: "{ts}"\n')
            seen_ts = True
        in_choices = False

    if in_choices and re.match(r"^\s{2}assets_dir:\s*", line):
        out.append(f'  assets_dir: "{assets_dir}"\n')
        seen_dir = True
        continue
    if in_choices and re.match(r"^\s{2}updated_at:\s*", line):
        out.append(f'  updated_at: "{ts}"\n')
        seen_ts = True
        continue

    out.append(line)

if not seen_choices:
    if out and not out[-1].endswith("\n"):
        out[-1] = out[-1] + "\n"
    out.append("\nchoices:\n")
    out.append(f'  assets_dir: "{assets_dir}"\n')
    out.append(f'  updated_at: "{ts}"\n')
else:
    if in_choices:
        if not seen_dir:
            out.append(f'  assets_dir: "{assets_dir}"\n')
        if not seen_ts:
            out.append(f'  updated_at: "{ts}"\n')

with open(cfg, "w", encoding="utf-8") as f:
    f.write("".join(out))
PY
}

usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Bootstrap YOLOv10 assets (source + checkpoints).

By default it stores assets under this folder:
  ${YOLO10_ROOT}

Options:
  -p, --path PATH   Assets directory (stores src/, checkpoints/, tmp/).
  -y, --yes         Automatically confirm replacing an existing link/path.
      --clean       Remove tmp/ and exit.
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
      ASSETS_DIR="$2"
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
  echo "Cleaning temporary directory ${YOLO10_ROOT}/${TMP_SUBDIR}"
  rm -rf -- "${YOLO10_ROOT}/${TMP_SUBDIR}" || true
  exit 0
fi

if [[ -z "${ASSETS_DIR}" ]]; then
  set +u
  ENV_RAW="${!ASSETS_DIR_ENV-}"
  set -u

  CANDIDATE=""
  if [[ -n "${SAVED_ASSETS_DIR}" ]]; then
    CANDIDATE="${SAVED_ASSETS_DIR}"
  elif [[ -n "${ENV_RAW}" ]]; then
    CANDIDATE="${ENV_RAW}"
  else
    CANDIDATE="$(cd "${YOLO10_ROOT}/${DEFAULT_ASSETS_DIR_REL}" && pwd)"
  fi

  echo "Assets discovery:"
  echo "  env var name    : ${ASSETS_DIR_ENV}"
  echo "  env var value   : ${ENV_RAW:-<unset>}"
  echo "  default assets  : ${DEFAULT_ASSETS_DIR_REL} (relative)"
  echo "  saved assets    : ${SAVED_ASSETS_DIR:-<unset>}"
  echo "  candidate path  : ${CANDIDATE}"

  if [[ -d "${CANDIDATE}" ]]; then
    read -r -p "Use this assets directory? [Y/n]: " answer
    case "${answer,,}" in
      ""|y|yes)
        ASSETS_DIR="${CANDIDATE}"
        ;;
      *)
        :
        ;;
    esac
  else
    echo "Candidate directory does not exist: ${CANDIDATE}"
  fi
fi

if [[ -z "${ASSETS_DIR}" ]]; then
  echo "Enter the directory where YOLOv10 assets should be stored (will create if missing)."
  read -r -p "YOLOv10 assets dir: " ASSETS_DIR
fi

if [[ -z "${ASSETS_DIR}" ]]; then
  echo "Error: assets dir is required; none provided." >&2
  exit 1
fi

mkdir -p "${ASSETS_DIR}"

SRC_DIR="${ASSETS_DIR}/${SRC_SUBDIR}"
CHECKPOINTS_DIR="${ASSETS_DIR}/${CHECKPOINTS_SUBDIR}"
TMP_DIR="${ASSETS_DIR}/${TMP_SUBDIR}"

REPO_SRC_PATH="${YOLO10_ROOT}/${SRC_SUBDIR}"
REPO_CHECKPOINTS_PATH="${YOLO10_ROOT}/${CHECKPOINTS_SUBDIR}"
REPO_TMP_PATH="${YOLO10_ROOT}/${TMP_SUBDIR}"

ensure_repo_path() {
  local repo_path="$1"
  local physical_path="$2"

  if [[ "${ASSETS_DIR}" = "${YOLO10_ROOT}" ]]; then
    return 0
  fi

  mkdir -p "${physical_path}"

  if [[ -L "${repo_path}" ]]; then
    local cur
    cur="$(readlink "${repo_path}")"
    if [[ "${cur}" = "${physical_path}" ]]; then
      return 0
    fi
  fi

  if [[ -e "${repo_path}" && ! -L "${repo_path}" ]]; then
    if [[ -d "${repo_path}" && -z "$(ls -A "${repo_path}" 2>/dev/null || true)" ]]; then
      rm -rf -- "${repo_path}"
    else
      echo "Error: ${repo_path} exists and is not a symlink (and not empty)." >&2
      echo "To use an external assets dir, please move it aside first." >&2
      exit 1
    fi
  fi

  mkdir -p "$(dirname "${repo_path}")"
  ln -s -- "${physical_path}" "${repo_path}"
}

ensure_repo_path "${REPO_SRC_PATH}" "${SRC_DIR}"
ensure_repo_path "${REPO_CHECKPOINTS_PATH}" "${CHECKPOINTS_DIR}"
ensure_repo_path "${REPO_TMP_PATH}" "${TMP_DIR}"

echo "[YOLO10] Bootstrap starting..."
echo "[YOLO10] Assets dir:      ${ASSETS_DIR}"
echo "[YOLO10] Source dir:      ${SRC_DIR}"
echo "[YOLO10] Checkpoints dir: ${CHECKPOINTS_DIR}"

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

IFS=',' read -r -a MODELS <<<"${MODELS_CSV}"
for size in "${MODELS[@]}"; do
  model_file="${FILE_TEMPLATE//\{size\}/${size}}"
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

echo "[YOLO10] Recording selected assets dir into ${CFG}..."
record_choice "${ASSETS_DIR}" "${CFG}"

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

