#!/usr/bin/env bash
set -euo pipefail

# Bootstrap script for cv-models:
# - Discover a default ONNX root directory from ENV / YAML.
# - Create per-model symlinks under models/cv-models/*/checkpoints/.
# - Extract the expected checkpoint filenames from each model's README.md.
# - Record the chosen root directory back into boostrap.yaml for future runs.

require_cmd() {
  for cmd in "$@"; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
      echo "Error: required command not found in PATH: $cmd" >&2
      exit 127
    fi
  done
}

require_cmd ln mkdir grep date python3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CV_MODELS_ROOT="${SCRIPT_DIR}"

CFG="${SCRIPT_DIR}/boostrap.yaml"
if [[ ! -f "${CFG}" ]]; then
  echo "Error: missing config file: ${CFG}" >&2
  exit 1
fi

yaml_get() {
  local path="$1"
  python3 - "${CFG}" "${path}" <<'PY'
import sys

import yaml

cfg_path, dotted_path = sys.argv[1], sys.argv[2]
with open(cfg_path, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f) or {}

cur = data
for part in dotted_path.split("."):
    if isinstance(cur, dict) and part in cur:
        cur = cur[part]
    else:
        cur = None
        break

if cur is None:
    print("")
elif isinstance(cur, bool):
    print("true" if cur else "false")
else:
    print(cur)
PY
}

yaml_get_default() {
  local path="$1"
  local default_value="$2"
  python3 - "${CFG}" "${path}" "${default_value}" <<'PY'
import sys

import yaml

cfg_path, dotted_path, default_value = sys.argv[1], sys.argv[2], sys.argv[3]
with open(cfg_path, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f) or {}

cur = data
for part in dotted_path.split("."):
    if isinstance(cur, dict) and part in cur:
        cur = cur[part]
    else:
        cur = None
        break

if cur is None:
    print(default_value)
elif isinstance(cur, bool):
    print("true" if cur else "false")
else:
    print(cur)
PY
}

MODELS_ROOT_ENV="$(yaml_get 'env.models_root_env')"
DEFAULT_MODELS_ROOT="$(yaml_get 'env.default_models_root')"
CHECKPOINTS_SUBDIR="$(yaml_get 'layout.checkpoints_subdir')"
SAVED_MODELS_ROOT="$(yaml_get_default 'choices.models_root' '')"

ASSUME_YES=false
CLEAN_ONLY=false
ALLOW_MISSING=false
FILTER_SUBSTR=""
ONLY_DIRS_CSV=""
MODELS_ROOT=""

usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Bootstrap cv-model ONNX symlinks under:
  ${CV_MODELS_ROOT}/*/${CHECKPOINTS_SUBDIR}/*.onnx

It will look for checkpoint filenames in each model's README.md line like:
  - \`checkpoints/<name>.onnx\` -> \`/some/path/<name>.onnx\`

Options:
  -p, --path PATH        Root directory containing ONNX files.
  -f, --filter SUBSTR    Only bootstrap model dirs whose name contains SUBSTR (e.g. "yolo").
      --only DIRS        Comma-separated list of model directory names (overrides --filter).
  -y, --yes              Automatically confirm replacing an existing link/path.
      --allow-missing    Create symlinks even if the target file does not exist.
      --clean            Remove existing checkpoints symlinks and exit.
  -h, --help             Show this help and exit.
EOF
}

record_choice() {
  local models_root="$1"
  local cfg_path="$2"
  local ts
  ts="$(date -Iseconds)"

  python3 - "${cfg_path}" "${models_root}" "${ts}" <<'PY'
import re
import sys

cfg, models_root, ts = sys.argv[1:]

with open(cfg, "r", encoding="utf-8") as f:
    lines = f.read().splitlines(True)

out = []
in_choices = False
seen_choices = False
seen_root = False
seen_ts = False

for line in lines:
    if re.match(r"^choices:\s*$", line):
        in_choices = True
        seen_choices = True
        out.append(line)
        continue

    if in_choices and re.match(r"^\S", line):
        if not seen_root:
            out.append(f'  models_root: "{models_root}"\n')
            seen_root = True
        if not seen_ts:
            out.append(f'  updated_at: "{ts}"\n')
            seen_ts = True
        in_choices = False

    if in_choices and re.match(r"^\s{2}models_root:\s*", line):
        out.append(f'  models_root: "{models_root}"\n')
        seen_root = True
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
    out.append(f'  models_root: "{models_root}"\n')
    out.append(f'  updated_at: "{ts}"\n')
else:
    if in_choices:
        if not seen_root:
            out.append(f'  models_root: "{models_root}"\n')
        if not seen_ts:
            out.append(f'  updated_at: "{ts}"\n')

with open(cfg, "w", encoding="utf-8") as f:
    f.write("".join(out))
PY
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--path)
      if [[ $# -lt 2 ]]; then
        echo "Error: --path requires an argument." >&2
        exit 2
      fi
      MODELS_ROOT="$2"
      shift 2
      ;;
    -f|--filter)
      if [[ $# -lt 2 ]]; then
        echo "Error: --filter requires an argument." >&2
        exit 2
      fi
      FILTER_SUBSTR="$2"
      shift 2
      ;;
    --only)
      if [[ $# -lt 2 ]]; then
        echo "Error: --only requires an argument." >&2
        exit 2
      fi
      ONLY_DIRS_CSV="$2"
      shift 2
      ;;
    -y|--yes)
      ASSUME_YES=true
      shift
      ;;
    --allow-missing)
      ALLOW_MISSING=true
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
  echo "Cleaning checkpoints symlinks under ${CV_MODELS_ROOT}"
  find "${CV_MODELS_ROOT}" \
    -mindepth 3 -maxdepth 3 \
    -path "*/${CHECKPOINTS_SUBDIR}/*" \
    -type l \
    -print \
    -exec rm -f -- {} \;
  exit 0
fi

if [[ -z "${MODELS_ROOT}" ]]; then
  set +u
  ENV_RAW="${!MODELS_ROOT_ENV-}"
  set -u

  CANDIDATE=""
  if [[ -n "${SAVED_MODELS_ROOT}" ]]; then
    CANDIDATE="${SAVED_MODELS_ROOT}"
  else
    CANDIDATE="${ENV_RAW:-${DEFAULT_MODELS_ROOT}}"
  fi

  echo "Models discovery:"
  echo "  env var name    : ${MODELS_ROOT_ENV}"
  echo "  env var value   : ${ENV_RAW:-<unset>}"
  echo "  default root    : ${DEFAULT_MODELS_ROOT}"
  echo "  saved root      : ${SAVED_MODELS_ROOT:-<unset>}"
  echo "  candidate root  : ${CANDIDATE}"

  if [[ -d "${CANDIDATE}" ]]; then
    read -r -p "Use this models root? [Y/n]: " answer
    case "${answer,,}" in
      ""|y|yes)
        MODELS_ROOT="${CANDIDATE}"
        ;;
      *)
        :
        ;;
    esac
  else
    echo "Candidate directory does not exist: ${CANDIDATE}"
  fi
fi

while [[ -z "${MODELS_ROOT}" ]]; do
  read -r -p "Enter ONNX models root directory (absolute path): " input
  if [[ -z "${input}" ]]; then
    echo "Empty path; please try again." >&2
    continue
  fi
  if [[ ! -d "${input}" ]]; then
    echo "Not a directory: ${input}" >&2
    continue
  fi
  MODELS_ROOT="${input}"
done

if [[ "${MODELS_ROOT}" != /* ]]; then
  echo "Error: models root must be an absolute path: ${MODELS_ROOT}" >&2
  exit 2
fi

python3 - "${CV_MODELS_ROOT}" "${MODELS_ROOT}" "${CHECKPOINTS_SUBDIR}" "${FILTER_SUBSTR}" "${ONLY_DIRS_CSV}" "${ASSUME_YES}" "${ALLOW_MISSING}" <<'PY'
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

cv_models_root = Path(sys.argv[1]).resolve()
models_root = Path(sys.argv[2]).resolve()
checkpoints_subdir = sys.argv[3]
filter_substr = sys.argv[4]
only_dirs_csv = sys.argv[5]
assume_yes = sys.argv[6].lower() == "true"
allow_missing = sys.argv[7].lower() == "true"


def iter_model_dirs() -> list[Path]:
    candidates = []
    for child in cv_models_root.iterdir():
        if not child.is_dir():
            continue
        if child.name == "helpers":
            continue
        candidates.append(child)
    return sorted(candidates, key=lambda p: p.name)


def selected(model_dir: Path) -> bool:
    if only_dirs_csv.strip():
        allowed = {s.strip() for s in only_dirs_csv.split(",") if s.strip()}
        return model_dir.name in allowed
    if filter_substr.strip():
        return filter_substr in model_dir.name
    return True


def parse_checkpoint_name(readme_path: Path) -> str:
    text = readme_path.read_text(encoding="utf-8")
    # Expected line:
    # - `checkpoints/foo.onnx` -> `/workspace/.../foo.onnx`
    match = re.search(r"`checkpoints/([^`]+)`\s*->\s*`[^`]+`", text)
    if not match:
        raise ValueError(f"Could not find checkpoints mapping in {readme_path}")
    checkpoint_name = match.group(1).strip()
    if not checkpoint_name:
        raise ValueError(f"Empty checkpoint name in {readme_path}")
    return checkpoint_name


def ensure_symlink(link_path: Path, target_path: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        if link_path.is_symlink():
            current = Path(os.readlink(link_path))
            resolved_current = (link_path.parent / current).resolve() if not current.is_absolute() else current.resolve()
            if resolved_current == target_path.resolve():
                print(f"OK  {link_path} -> {target_path}")
                return
        if not assume_yes:
            answer = input(f"Replace existing path? {link_path} [y/N]: ").strip().lower()
            if answer not in ("y", "yes"):
                print(f"SKIP {link_path}")
                return
        if link_path.is_dir() and not link_path.is_symlink():
            raise RuntimeError(f"Refusing to replace directory: {link_path}")
        link_path.unlink()
    link_path.symlink_to(target_path)
    print(f"LINK {link_path} -> {target_path}")


errors: list[str] = []
selected_dirs = [d for d in iter_model_dirs() if selected(d)]

if not selected_dirs:
    print("No model directories selected; nothing to do.")
    sys.exit(0)

for model_dir in selected_dirs:
    readme = model_dir / "README.md"
    if not readme.is_file():
        errors.append(f"Missing README.md: {readme}")
        continue

    try:
        checkpoint_name = parse_checkpoint_name(readme)
    except Exception as exc:
        errors.append(f"{model_dir.name}: {exc}")
        continue

    checkpoints_dir = model_dir / checkpoints_subdir
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    link_path = checkpoints_dir / checkpoint_name
    target_path = models_root / checkpoint_name

    if not target_path.exists() and not allow_missing:
        errors.append(f"{model_dir.name}: target missing: {target_path}")
        continue

    try:
        ensure_symlink(link_path, target_path)
    except Exception as exc:
        errors.append(f"{model_dir.name}: failed to link {link_path}: {exc}")

if errors:
    print("\nErrors:", file=sys.stderr)
    for e in errors:
        print(f"- {e}", file=sys.stderr)
    sys.exit(1)
PY

record_choice "${MODELS_ROOT}" "${CFG}"
echo "Saved models root to ${CFG}"
