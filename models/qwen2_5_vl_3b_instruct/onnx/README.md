# Qwen2.5-VL-3B-Instruct ONNX Artifacts

This directory contains ONNX exports for the Qwen2.5‑VL‑3B‑Instruct model, used for Intel Neural Compressor sensitivity analysis and ONNX Runtime–based experiments.

All models are exported from the local HF checkpoint:

- `models/qwen2_5_vl_3b_instruct/checkpoints/Qwen2.5-VL-3B-Instruct/`

and rely on the same config/tokenizer.

## Files

- `qwen2_5_vl_3b_vision_672_fp32.onnx`
- `qwen2_5_vl_3b_vision_672_fp32.onnx_data`
  - Vision encoder ONNX with fixed 672×672 RGB input.
  - Export script: `models/qwen2_5_vl_3b_instruct/helpers/convert_to_onnx_vision_672_fp32.py`
  - Typical I/O:
    - Input: `pixel_values` — `float32` tensor of shape `(1, 3, 672, 672)`.
    - Output: `vision_features` — `float32` tensor of shape `(576, 2048)` for a single 672×672 image (576 image tokens, hidden size 2048).
  - Notes:
    - Uses custom `build_patches_and_grid` to construct `grid_thw` and match Qwen2.5‑VL’s vision patching scheme.
    - External data is aggregated into `qwen2_5_vl_3b_vision_672_fp32.onnx_data`.

- `qwen2_5_vl_3b_text_fp16.onnx`
- `qwen2_5_vl_3b_text_fp16.onnx_data`
  - Text-only language model ONNX (no vision inputs).
  - Export script: `models/qwen2_5_vl_3b_instruct/helpers/convert_to_onnx_text_fp16.py`
  - Typical I/O:
    - Inputs:
      - `input_ids` — `int64` tensor of shape `(batch, seq)`.
      - `attention_mask` — `int64` or `bool` tensor of shape `(batch, seq)`.
    - Output:
      - `logits` — `float16` tensor of shape `(batch, seq, vocab_size)`.
  - Notes:
    - This export is used for text-only sensitivity analysis and as a baseline decoder without image conditioning.
    - Dynamic axes are configured on batch and sequence dimensions.

- `qwen2_5_vl_3b_text_with_image_fp16.onnx`
- `qwen2_5_vl_3b_text_with_image_fp16.onnx_data`
  - Text decoder ONNX with an additional input for pre-aligned image embeddings.
  - Export script: `models/qwen2_5_vl_3b_instruct/helpers/convert_to_onnx_text_with_image_fp16.py`
  - Wrapper module: `Qwen25VLTextWithImageWrapper`
    - Embeds `input_ids` via the model’s token embedding matrix.
    - For positions where `input_ids == config.image_token_id`, replaces token embeddings with `image_embeds`.
    - Forwards fused embeddings through `Qwen2_5_VLForConditionalGeneration` with `inputs_embeds`.
  - Typical I/O:
    - Inputs:
      - `input_ids` — `int64` tensor `(batch, seq)`.
      - `attention_mask` — `int64` or `bool` tensor `(batch, seq)`.
      - `image_embeds` — `float16` tensor `(batch, seq, hidden_size)`; non-zero entries should live at image-token positions.
    - Output:
      - `logits` — `float16` tensor `(batch, seq, vocab_size)`.
  - Notes:
    - Intended to be composed with the vision ONNX offline:
      - Run `qwen2_5_vl_3b_vision_672_fp32.onnx` to get vision features `(B, T_img, D)`.
      - Pack these into `image_embeds (B, S, D)` at a fixed image-token window; set `input_ids` at those positions to `image_token_id`.
      - Run `qwen2_5_vl_3b_text_with_image_fp16.onnx` to obtain multimodal logits.
    - Export uses `torch._dynamo.config.patch(fake_tensor_cache_enabled=False)` and the `torch.export`-based ONNX path, with external data aggregated into `_data`.

## Usage Notes

- All ONNX models were exported using Pixi environments:
  - Text-only and text+image exporters are intended to run in `rtx5090-vllm`.
  - Vision exporter can run in either `rtx5090` (GPU) or `rtx5090-vllm` (CPU) as configured in the helper script.
- These artifacts are designed for:
  - ONNX Runtime (CPU or CUDA) sensitivity analysis with Intel Neural Compressor.
  - Prototype multimodal inference by composing vision and text+image ONNX graphs with a small host-side orchestrator.

