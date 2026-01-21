# Contract: `run_demo.sh` CLI (tutorial pack)

**Branch**: `002-revise-qwen3-vl-tutorial`  
**Date**: 2026-01-21  
**Applies to**: `/data1/huangzhe/code/auto-quantize-model/docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/run_demo.sh`

## Purpose

Define a stable, user-facing interface for running and verifying the tutorial pack.

## Command

From `/data1/huangzhe/code/auto-quantize-model`:

```bash
bash /data1/huangzhe/code/auto-quantize-model/docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/run_demo.sh [options]
```

## Options

| Option | Required | Description |
|--------|----------|-------------|
| `--snapshot-report` | No | Overwrite `expected_report/` with sanitized artifacts from the current run. |
| `--device <torch-device>` | No | Torch device to use (default: `cuda:0`). |
| `--dataset-size <small\|medium\|large>` | No | Dataset preset (default: `medium`). |
| `--modes <all_layers,lm_only>` | No | Comma-separated modes to run; defaults to `all_layers,lm_only`. |
| `--quant-pairs <wint4_afp16,wint4_aint8>` | No | Comma-separated quant pairs to run; defaults to `wint4_afp16,wint4_aint8`. |

Notes:

- Default execution with no options runs and verifies all 4 scenarios (2 modes × 2 quant pairs).
- If dataset assets for the chosen dataset-size are missing, the script fails fast with actionable guidance.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_SNAPSHOTS_ROOT` | No | Local directory containing model snapshots; used to auto-create the checkpoint link when possible. |

## Outputs

- Workspace outputs are written under `/data1/huangzhe/code/auto-quantize-model/tmp/` in a new per-run directory.
- Verification compares only sanitized summaries per scenario against `expected_report/`.
