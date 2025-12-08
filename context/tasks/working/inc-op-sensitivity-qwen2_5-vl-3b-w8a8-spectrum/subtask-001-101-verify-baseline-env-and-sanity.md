# Subtask 1.1: Verify baseline Qwen2.5-VL environment and sanity checks

## Scope

Ensure that the bf16/fp16 Qwen2.5-VL-3B-Instruct checkpoint is correctly bootstrapped on this machine, that the `rtx5090` Pixi environment can load and run it, and that the existing sanity script produces reasonable text-only and image+text outputs. This subtask does not involve any quantization or INC integration; it only validates the baseline.

## Planned outputs

- Confirmed location and layout of the baseline Qwen2.5-VL-3B-Instruct checkpoint under `models/qwen2_5_vl_3b_instruct/checkpoints/Qwen2.5-VL-3B-Instruct`.
- Verified that the `rtx5090` Pixi environment can load the model on GPU (or CPU as a fallback).
- Saved baseline sanity outputs (text-only and image+text) in `tmp/qwen2_5_vl_3b_sanity/` (or a variant-specific directory) for later comparison with quantized models.
- Short note summarizing any environment quirks (e.g., memory usage, device_map choices).

## TODOs

- [x] Job-001-101-001 Verify that `models/qwen2_5_vl_3b_instruct/checkpoints/Qwen2.5-VL-3B-Instruct` exists and points to a valid HF snapshot (use `bootstrap.sh` if needed).
- [x] Job-001-101-002 Run `pixi run -e rtx5090 python scripts/qwen/run_qwen2_5_vl_3b_sanity.py` against the baseline checkpoint and confirm it completes without errors.
- [x] Job-001-101-003 Inspect the generated `tmp/qwen2_5_vl_3b_sanity/` outputs (text-only and image+text) to ensure they are qualitatively reasonable.
- [x] Job-001-101-004 Record any notable hardware / environment details (GPU type, VRAM usage, device_map settings) in a short note or in the main plan.

## Notes

- Baseline checkpoint symlink is present at `models/qwen2_5_vl_3b_instruct/checkpoints/Qwen2.5-VL-3B-Instruct` → `/workspace/llm-models/Qwen2.5-VL-3B-Instruct`, with standard HF files (`config.json`, `generation_config.json`, tokenizer and safetensors shards).
- `pixi run -e rtx5090 python scripts/qwen/run_qwen2_5_vl_3b_sanity.py` loads the model with `device_map="auto"` on GPU and completes both text-only and image+text sanity checks without errors.
- Sanity outputs are saved under `tmp/qwen2_5_vl_3b_sanity/` (`text_only.txt`, `image_text.txt`, and the input image copy), and the responses are coherent and on-topic for the prompts.
- Hardware: `NVIDIA GeForce RTX 5090` with ~31 GB VRAM; CUDA is available (`torch.cuda.is_available() == True`), so subsequent INC runs can assume a single-GPU, CUDA-enabled environment.
