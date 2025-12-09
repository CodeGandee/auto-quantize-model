# Subtask 4.3: Implement Qwen2.5-VL-3B LM-only AutoQuant FP8 driver

## Scope

Implement the main AutoQuant driver script `scripts/qwen/qwen2_5_vl_3b_autoquant_fp8_schemes.py` that loads the Qwen2.5-VL-3B language model only, constructs a calibration forward loop over text data, invokes ModelOpt’s AutoQuant to derive mixed-precision FP8 configurations according to the scheme definitions, and emits both a quantized model and a machine-readable quantization manifest.

## Planned outputs

- A new Python script `scripts/qwen/qwen2_5_vl_3b_autoquant_fp8_schemes.py` with a clear CLI:
  - Required flags such as `--scheme-name`, `--model-dir`, and `--output-dir`.
  - Optional flags for AutoQuant parameters (e.g., `--effective-bits`, `--auto-quantize-method`, `--num-score-steps`) with sensible defaults pulled from the scheme catalog.
- Implementation of LM-only Qwen2.5-VL-3B loading (reusing existing LM-only extraction patterns from the FP8 baseline path where possible).
- A calibration forward loop over COCO captions (text-only) that is compatible with ModelOpt AutoQuant expectations.
- Generation of a quantization manifest (e.g., JSON) that records, per layer, whether it is FP8 or BF16/FP16 plus any relevant scores or metadata.

## TODOs

- [ ] Job-004-103-001 Sketch the CLI and main entrypoint for `scripts/qwen/qwen2_5_vl_3b_autoquant_fp8_schemes.py`, including flags for `--scheme-name`, `--model-dir`, `--output-dir`, and optional overrides for AutoQuant settings.
- [ ] Job-004-103-002 Implement LM-only loading for Qwen2.5-VL-3B by reusing or adapting the pattern from the existing FP8 LM-only checkpoint path (e.g., detaching the vision tower and freezing its weights).
- [ ] Job-004-103-003 Implement a calibration dataset loader that reads COCO captions from `datasets/vlm-quantize-calib/coco2017_captions.txt`, tokenizes them with the Qwen2.5-VL tokenizer, and builds a simple text-only data loader with a configurable number of samples.
- [ ] Job-004-103-004 Implement the calibration forward loop callable that runs the LM on a batch of tokens (no images) and is suitable for passing to `mtq.auto_quantize`.
- [ ] Job-004-103-005 Wire up ModelOpt AutoQuant invocation inside the driver, using scheme-specific defaults from Subtask 4.2 but allowing CLI overrides for experimentation.
- [ ] Job-004-103-006 Design and implement a quantization manifest format (e.g., JSON mapping layer names to quantization dtypes and any scores) and write it to `--output-dir` alongside any temporary artifacts.
- [ ] Job-004-103-007 Add basic logging and argument validation to the script, including clear error messages when required paths or dependencies are missing.

## Notes

- Follow existing patterns for Qwen2.5-VL sanity scripts and FP8 tools when choosing how to load the model, tokenizer, and calibration data.
- Keep the driver script focused on AutoQuant and manifest generation; checkpoint export is handled in a separate subtask.

