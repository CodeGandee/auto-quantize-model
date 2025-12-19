#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../../.." && pwd)

OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/tmp/cv-models-random-infer}"
DEFAULT_IMAGE_SIZE="${DEFAULT_IMAGE_SIZE:-}"
DEFAULT_BATCH="${DEFAULT_BATCH:-}"
DEFAULT_CHANNELS="${DEFAULT_CHANNELS:-}"
USE_CPU="${USE_CPU:-}"

EXTRA_ARGS=()
if [ -n "$DEFAULT_IMAGE_SIZE" ]; then
  EXTRA_ARGS+=("--default-image-size" "$DEFAULT_IMAGE_SIZE")
fi
if [ -n "$DEFAULT_BATCH" ]; then
  EXTRA_ARGS+=("--default-batch" "$DEFAULT_BATCH")
fi
if [ -n "$DEFAULT_CHANNELS" ]; then
  EXTRA_ARGS+=("--default-channels" "$DEFAULT_CHANNELS")
fi
if [ "$USE_CPU" = "1" ]; then
  EXTRA_ARGS+=("--use-cpu")
fi

failures=0
while IFS= read -r -d '' model_path; do
  echo "Running random inference for $model_path"
  if ! pixi run -e rtx5090 python "$SCRIPT_DIR/run_random_onnx_inference.py" \
    --model "$model_path" \
    --output-root "$OUTPUT_ROOT" \
    "${EXTRA_ARGS[@]}"; then
    echo "Failed: $model_path" >&2
    failures=$((failures + 1))
  fi
done < <(
  find "$ROOT_DIR/models/cv-models" \
    -path "*/checkpoints/*.onnx" \
    \( -type l -o -type f \) \
    -print0
)

if [ "$failures" -ne 0 ]; then
  echo "Completed with $failures failures." >&2
  exit 1
fi

printf "All models completed. Outputs in %s\n" "$OUTPUT_ROOT"
