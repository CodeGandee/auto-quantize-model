# Subtask 4.5: Probe vLLM compatibility for AutoQuant FP8 schemes

## Scope

Evaluate whether each exported AutoQuant FP8 scheme checkpoint can be loaded and used by vLLM in the existing FP8 environment, and document any incompatibilities, error messages, or required workarounds. This subtask focuses on wiring vLLM to the new checkpoints and capturing structured results.

## Planned outputs

- For each scheme, a recorded result indicating whether vLLM loads and runs successfully, including:
  - Success/failure status.
  - Any initialization or runtime error traces (e.g., missing tensors, unexpected dtypes).
  - Notes about required flags or configuration tweaks.
- Optional: a small helper script or extension to `scripts/qwen/run_qwen2_5_vl_3b_vllm_fp8.py` that can iterate over multiple scheme directories and emit a summary JSON or Markdown report.

## TODOs

- [ ] Job-004-105-001 Review `scripts/qwen/run_qwen2_5_vl_3b_vllm_fp8.py` to understand how it currently loads the FP8 LM-only checkpoint and how model paths and quantization settings are specified.
- [ ] Job-004-105-002 Extend the script (or add a new helper) to accept either a `--scheme-name` argument or an explicit `--model-dir` pointing to a scheme-specific checkpoint.
- [ ] Job-004-105-003 Implement a loop or small driver that attempts to load each scheme’s checkpoint in vLLM, capturing success/failure and any exceptions.
- [ ] Job-004-105-004 Define and implement a structured output format (e.g., JSON or Markdown table) that records compatibility results per scheme and saves it under `models/qwen2_5_vl_3b_instruct/reports/` or `tmp/`.
- [ ] Job-004-105-005 Run the compatibility checks for all schemes on the RTX 5090 environment and store the resulting logs and summary artifacts.

## Notes

- Treat vLLM compatibility as a practical constraint: if certain AutoQuant settings consistently break compatibility, capture that insight for potential changes in earlier subtasks.

