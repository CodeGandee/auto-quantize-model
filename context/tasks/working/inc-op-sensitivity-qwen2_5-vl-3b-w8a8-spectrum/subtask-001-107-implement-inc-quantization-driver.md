# Subtask 1.7: Implement INC quantization driver for Qwen2.5-VL-3B profiles

## Scope

Implement a quantization driver script (e.g., `scripts/qwen/inc_qwen2_5_vl_3b_quantize.py`) that reads the sensitivity-derived profiles and invokes INC PTQ to produce quantized checkpoints for each profile under a consistent directory structure.

## Planned outputs

- A driver script that:
  - Loads the baseline Qwen2.5-VL model.
  - Applies a chosen profile to construct a `PostTrainingQuantConfig` (per-profile op precision).
  - Runs PTQ with INC to generate an INT8 W8A8 model for that profile.
  - Exports each quantized model under `models/qwen2_5_vl_3b_instruct/quantized/inc/<profile-name>/`.

## TODOs

- [ ] Job-001-107-001 Design the command-line interface or entrypoint for the quantization driver (e.g., `--profile minimal|medium|aggressive`).
- [ ] Job-001-107-002 Implement logic to map profile definitions into INC op-wise precision settings (`op_type_dict` / `op_name_dict` or explicit `tune_cfg`).
- [ ] Job-001-107-003 Run PTQ for each profile and verify that quantized checkpoints are saved correctly and can be loaded.
- [ ] Job-001-107-004 Capture logs and basic metadata (e.g., quantization config, runtime) for each profile under a reproducible path (e.g., `tmp/qwen2_5_vl_3b_inc_quant/`).

## Notes

- Consider adding a “dry-run” mode that validates profile mapping without actually running PTQ, to speed up iteration on configuration errors.

