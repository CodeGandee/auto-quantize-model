#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"

SNAPSHOT=false
DEVICE="cuda:0"
DATASET_SIZE="medium"
MODES_CSV="all_layers,lm_only"
QUANT_PAIRS_CSV="wint4_afp16,wint4_aint8"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

End-to-end tutorial pack runner for:
  - running ModelOpt AutoQuant per-layer sensitivity for Qwen3-VL-4B-Instruct,
  - across both all-layers and LM-only sensitivity modes.

Options:
  --snapshot-report
      Overwrite expected_report/ with sanitized artifacts from this run.
  --device <torch-device>
      Torch device (default: cuda:0). Use 'cpu' only if you accept a very slow run.
  --dataset-size <small|medium|large>
      Dataset preset (default: medium).
  --modes <all_layers,lm_only>
      Comma-separated modes to run (default: all_layers,lm_only).
  --quant-pairs <wint4_afp16,wint4_aint8>
      Comma-separated quant pairs to run (default: wint4_afp16,wint4_aint8).

Environment variables:
  HF_SNAPSHOTS_ROOT
      Directory containing local model snapshots (default: /data1/huangzhe/llm-models).
EOF
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --snapshot-report)
        SNAPSHOT=true
        shift
        ;;
      --device)
        DEVICE="${2:-}"
        if [[ -z "${DEVICE}" ]]; then
          echo "[ERROR] --device requires an argument" >&2
          exit 2
        fi
        shift 2
        ;;
      --dataset-size)
        DATASET_SIZE="${2:-}"
        if [[ -z "${DATASET_SIZE}" ]]; then
          echo "[ERROR] --dataset-size requires an argument" >&2
          exit 2
        fi
        shift 2
        ;;
      --modes)
        MODES_CSV="${2:-}"
        if [[ -z "${MODES_CSV}" ]]; then
          echo "[ERROR] --modes requires an argument" >&2
          exit 2
        fi
        shift 2
        ;;
      --quant-pairs)
        QUANT_PAIRS_CSV="${2:-}"
        if [[ -z "${QUANT_PAIRS_CSV}" ]]; then
          echo "[ERROR] --quant-pairs requires an argument" >&2
          exit 2
        fi
        shift 2
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        echo "[ERROR] Unknown arg: $1" >&2
        usage >&2
        exit 2
        ;;
    esac
  done
}

split_csv() {
  local csv="$1"
  local -a out=()
  local IFS=","
  read -r -a out <<<"$csv"
  printf '%s\n' "${out[@]}"
}

has_value() {
  local needle="$1"
  shift
  local item
  for item in "$@"; do
    if [[ "$item" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

resolve_dataset_preset() {
  case "$DATASET_SIZE" in
    small)
      MAX_CALIB_SAMPLES=16
      NUM_CALIB_BATCHES=2
      BATCH_SIZE=8
      ;;
    medium)
      MAX_CALIB_SAMPLES=128
      NUM_CALIB_BATCHES=16
      BATCH_SIZE=8
      ;;
    large)
      MAX_CALIB_SAMPLES=512
      NUM_CALIB_BATCHES=64
      BATCH_SIZE=8
      ;;
    *)
      echo "[ERROR] Unknown --dataset-size: $DATASET_SIZE (expected small|medium|large)" >&2
      exit 2
      ;;
  esac

  CALIB_SEQ_LEN=512
  SCORE_SIZE=128

  COCO_ROOT="$REPO_ROOT/datasets/coco2017/source-data"
  VLM_DB="$REPO_ROOT/datasets/vlm-quantize-calib/coco2017_vlm_calib_${DATASET_SIZE}.db"
  CAPTIONS_TXT="$REPO_ROOT/datasets/vlm-quantize-calib/coco2017_captions_${DATASET_SIZE}.txt"
}

ensure_assets() {
  MODEL_DIR="$REPO_ROOT/models/qwen3_vl_4b_instruct/checkpoints/Qwen3-VL-4B-Instruct"
  if [[ ! -d "$MODEL_DIR" ]]; then
    SNAPSHOTS_ROOT="${HF_SNAPSHOTS_ROOT:-/data1/huangzhe/llm-models}"
    CANDIDATE="${SNAPSHOTS_ROOT}/Qwen3-VL-4B-Instruct"
    if [[ -d "$CANDIDATE" ]]; then
      echo "[INFO] Creating local model link:"
      echo "  $MODEL_DIR -> $CANDIDATE"
      mkdir -p "$(dirname "$MODEL_DIR")"
      ln -s "$CANDIDATE" "$MODEL_DIR"
    fi
  fi

  if [[ ! -d "$MODEL_DIR" ]]; then
    echo "[ERROR] Missing model directory: $MODEL_DIR" >&2
    echo "Hint: create a symlink like:" >&2
    echo "  MODELS_ROOT=/path/to/local/model-snapshots" >&2
    echo "  ln -s \"${MODELS_ROOT}/Qwen3-VL-4B-Instruct\" \\" >&2
    echo "    models/qwen3_vl_4b_instruct/checkpoints/Qwen3-VL-4B-Instruct" >&2
    echo "Or run: ./models/qwen3_vl_4b_instruct/bootstrap.sh --yes" >&2
    exit 1
  fi

  if [[ ! -d "$COCO_ROOT" ]]; then
    echo "[ERROR] Missing COCO root: $COCO_ROOT" >&2
    echo "Hint: run: ./datasets/coco2017/bootstrap.sh --path /absolute/path/to/coco2017" >&2
    exit 1
  fi
  if [[ ! -f "$VLM_DB" ]]; then
    echo "[ERROR] Missing VLM calibration DB: $VLM_DB" >&2
    echo "Hint: check datasets/vlm-quantize-calib/ and dataset-size presets." >&2
    exit 1
  fi
  if [[ ! -f "$CAPTIONS_TXT" ]]; then
    echo "[ERROR] Missing captions file: $CAPTIONS_TXT" >&2
    echo "Hint: check datasets/vlm-quantize-calib/ and dataset-size presets." >&2
    exit 1
  fi

  if [[ "$DEVICE" == cuda:* || "$DEVICE" == cuda ]]; then
    pixi run python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available but a CUDA device was requested.")
PY
  fi
}

find_manifest() {
  local out_dir="$1"
  shopt -s nullglob
  local -a manifests=("$out_dir"/*_quant_manifest.json)
  shopt -u nullglob
  if [[ "${#manifests[@]}" -lt 1 ]]; then
    echo "[ERROR] No *_quant_manifest.json found under: $out_dir" >&2
    exit 1
  fi
  echo "${manifests[0]}"
}

run_scenario_all_layers() {
  local quant_pair="$1"
  local out_dir="$2"
  mkdir -p "$out_dir"

  echo "[INFO] Running all-layers: $quant_pair"
  pixi run python \
    "$REPO_ROOT/models/qwen3_vl_4b_instruct/helpers/qwen3_vl_4b_autoquant_all_layers/run_qwen3_vl_4b_autoquant_all_layers.py" \
    --model-dir "$MODEL_DIR" \
    --output-dir "$out_dir" \
    --quant-pair "$quant_pair" \
    --dataset-size "$DATASET_SIZE" \
    --vlm-calib-db "$VLM_DB" \
    --coco-root "$COCO_ROOT" \
    --max-calib-samples "$MAX_CALIB_SAMPLES" \
    --num-calib-batches "$NUM_CALIB_BATCHES" \
    --calib-seq-len "$CALIB_SEQ_LEN" \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --auto-quantize-score-size "$SCORE_SIZE" \
    2>&1 | tee "$out_dir/run.log"
}

run_scenario_lm_only() {
  local quant_pair="$1"
  local out_dir="$2"
  mkdir -p "$out_dir"

  echo "[INFO] Running lm-only: $quant_pair"
  pixi run python scripts/qwen/qwen3_lm_sensitivity.py \
    model.path="$MODEL_DIR" \
    model.name=qwen3_vl_4b_instruct \
    model.variant=3-vl-4b-instruct \
    dataset.size="$DATASET_SIZE" \
    autoquant.device="$DEVICE" \
    autoquant.batch_size="$BATCH_SIZE" \
    autoquant.score_size="$SCORE_SIZE" \
    quant_pair="$quant_pair" \
    runner.output_dir="$out_dir" \
    hydra.run.dir="$out_dir" \
    hydra.job.chdir=false \
    2>&1 | tee "$out_dir/run.log"
}

summarize_scenario() {
  local mode="$1"
  local quant_pair="$2"
  local manifest_json="$3"
  local summary_dir="$4"

  pixi run python "$SUMMARIZER" "$manifest_json" "$summary_dir" \
    --scenario-id "${mode}/${quant_pair}" \
    --mode "$mode" \
    --quant-pair "$quant_pair" \
    --dataset-size "$DATASET_SIZE"
}

assert_non_degenerate() {
  local summary_json="$1"
  pixi run python - "$summary_json" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("has_nonzero_sensitivity") is True:
    raise SystemExit(0)
raise SystemExit(
    "Detected degenerate sensitivity (all sensitivities are exactly 0.0). "
    "Try increasing calibration budget (e.g., --dataset-size medium) and "
    "see the tutorial README for remediation guidance."
)
PY
}

snapshot_cleanup_stale_modes() {
  local expected_dir="$1"
  shift
  local -a selected_modes=("$@")

  if [[ ! -d "$expected_dir" ]]; then
    return 0
  fi

  local mode_dir
  for mode_dir in "$expected_dir"/*; do
    if [[ ! -d "$mode_dir" ]]; then
      continue
    fi
    local mode_name
    mode_name="$(basename "$mode_dir")"
    if [[ "$mode_name" != "all_layers" && "$mode_name" != "lm_only" ]]; then
      rm -rf "$mode_dir"
      continue
    fi
    if ! has_value "$mode_name" "${selected_modes[@]}"; then
      rm -rf "$mode_dir"
    fi
  done
}

snapshot_cleanup_stale_pairs() {
  local expected_dir="$1"
  shift
  local -a selected_pairs=("$@")

  local mode
  for mode in "all_layers" "lm_only"; do
    if [[ ! -d "$expected_dir/$mode" ]]; then
      continue
    fi
    local dir
    for dir in "$expected_dir/$mode"/*; do
      if [[ ! -d "$dir" ]]; then
        continue
      fi
      local pair_name
      pair_name="$(basename "$dir")"
      if ! has_value "$pair_name" "${selected_pairs[@]}"; then
        rm -rf "$dir"
      fi
    done
  done
}

snapshot_or_verify() {
  local mode="$1"
  local quant_pair="$2"
  local out_dir="$3"
  local summary_dir="$4"

  local expected_case_dir="$EXPECTED_DIR/$mode/$quant_pair"
  mkdir -p "$EXPECTED_DIR"

  assert_non_degenerate "$summary_dir/summary.json"

  if $SNAPSHOT; then
    rm -rf "$expected_case_dir"
    mkdir -p "$expected_case_dir"
    cp -f "$summary_dir/summary.json" "$expected_case_dir/summary.json"
    cp -f "$summary_dir/summary.md" "$expected_case_dir/summary.md"
    pixi run python "$SANITIZER" "$out_dir" "$expected_case_dir"
    return 0
  fi

  if [[ ! -d "$expected_case_dir" ]]; then
    echo "[WARN] expected_report case missing: $expected_case_dir" >&2
    echo "[WARN] Run with --snapshot-report to create/refresh expected_report/." >&2
    return 0
  fi

  diff -u "$expected_case_dir/summary.json" "$summary_dir/summary.json"
  diff -u "$expected_case_dir/summary.md" "$summary_dir/summary.md"
}

parse_args "$@"
cd "$REPO_ROOT"

SUMMARIZER="$SCRIPT_DIR/scripts/summarize_manifest.py"
SANITIZER="$SCRIPT_DIR/scripts/sanitize_artifacts.py"
EXPECTED_DIR="$SCRIPT_DIR/expected_report"

resolve_dataset_preset
ensure_assets

mapfile -t MODES < <(split_csv "$MODES_CSV")
mapfile -t QUANT_PAIRS < <(split_csv "$QUANT_PAIRS_CSV")

WORKSPACE_DIR="$REPO_ROOT/tmp/tutorial_workspace_qwen3_vl_4b_layer_sensitivity_$(date +%Y%m%d_%H%M%S)"
OUTPUTS_DIR="$WORKSPACE_DIR/outputs"
SUMMARIES_DIR="$WORKSPACE_DIR/summaries"
mkdir -p "$OUTPUTS_DIR" "$SUMMARIES_DIR"

echo "[INFO] Workspace: $WORKSPACE_DIR"
echo "[INFO] Dataset preset: $DATASET_SIZE (seq_len=$CALIB_SEQ_LEN batch_size=$BATCH_SIZE num_batches=$NUM_CALIB_BATCHES score_size=$SCORE_SIZE)"

if $SNAPSHOT; then
  snapshot_cleanup_stale_modes "$EXPECTED_DIR" "${MODES[@]}"
fi

for mode in "${MODES[@]}"; do
  case "$mode" in
    all_layers|lm_only)
      ;;
    *)
      echo "[ERROR] Unknown mode: $mode (expected all_layers or lm_only)" >&2
      exit 2
      ;;
  esac

  for quant_pair in "${QUANT_PAIRS[@]}"; do
    out_dir="$OUTPUTS_DIR/$mode/$quant_pair"
    summary_dir="$SUMMARIES_DIR/$mode/$quant_pair"
    mkdir -p "$out_dir" "$summary_dir"

    if [[ "$mode" == "all_layers" ]]; then
      run_scenario_all_layers "$quant_pair" "$out_dir"
    else
      run_scenario_lm_only "$quant_pair" "$out_dir"
    fi

    manifest_json="$(find_manifest "$out_dir")"
    echo "[INFO] Summarizing manifest: $manifest_json"
    summarize_scenario "$mode" "$quant_pair" "$manifest_json" "$summary_dir"

    if ! $SNAPSHOT; then
      echo "[INFO] Verifying: $mode/$quant_pair"
    else
      echo "[INFO] Snapshotting: $mode/$quant_pair"
    fi
    snapshot_or_verify "$mode" "$quant_pair" "$out_dir" "$summary_dir"
  done
done

if $SNAPSHOT; then
  snapshot_cleanup_stale_pairs "$EXPECTED_DIR" "${QUANT_PAIRS[@]}"
  echo "[INFO] Updated expected_report/ from: $WORKSPACE_DIR"
else
  echo "[INFO] Verification complete."
fi

echo "[INFO] Done."
