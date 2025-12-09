# Subtask 4.4: Export HF checkpoints per AutoQuant FP8 scheme

## Scope

Using the quantized models and manifests produced by the AutoQuant driver, export one Hugging Face-style checkpoint directory per scheme under `models/qwen2_5_vl_3b_instruct/quantized/`, following a consistent naming convention and ensuring that each checkpoint is self-contained and ready for vLLM or other consumers.

## Planned outputs

- One HF checkpoint directory per scheme (e.g., `fp8_autoquant_top25_coco2017`, `fp8_autoquant_top50_coco2017`, `fp8_autoquant_full_coco2017`) under `models/qwen2_5_vl_3b_instruct/quantized/`.
- A small helper function or script (possibly inside `qwen2_5_vl_3b_autoquant_fp8_schemes.py` or a sibling module) that wraps `export_hf_checkpoint` for this model.
- Basic validation that each exported checkpoint loads as an HF model and produces reasonable text outputs.

## TODOs

- [ ] Job-004-104-001 Decide on an exact directory naming convention for scheme-specific checkpoints under `models/qwen2_5_vl_3b_instruct/quantized/` (e.g., `fp8_autoquant_<schemename>_coco2017`).
- [ ] Job-004-104-002 Implement or reuse a helper function that calls ModelOpt’s `export_hf_checkpoint` or equivalent API to export the quantized LM-only model to an HF-style directory.
- [ ] Job-004-104-003 Integrate the export helper into the AutoQuant driver or create a standalone export script that takes a quantization manifest and outputs a checkpoint directory.
- [ ] Job-004-104-004 For each defined scheme from Subtask 4.2, run the export flow and create the corresponding HF checkpoint directories.
- [ ] Job-004-104-005 Implement a light sanity check (e.g., a short text generation script) that loads each exported checkpoint with the HF `AutoModel`/`AutoTokenizer` stack and verifies that basic inference works.

## Notes

- Keep exports focused on the language model component; ensure that any assumptions about the vision tower (e.g., unquantized BF16/FP16) are clearly documented for potential future integration.

