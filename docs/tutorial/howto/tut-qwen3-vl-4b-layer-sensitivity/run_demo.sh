#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"

cd "$REPO_ROOT"

pixi run python -m auto_quantize_model.qwen.cli_tutorial_pack_runner \
  --model-id "qwen3_vl_4b_instruct" \
  --expected-report-dir "$SCRIPT_DIR/expected_report" \
  "$@"
