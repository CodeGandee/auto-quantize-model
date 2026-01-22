# How to Run Qwen3-VL-4B Layer Sensitivity (Tutorial Pack)

This tutorial pack runs a reproducible per-layer sensitivity analysis for
`Qwen3-VL-4B-Instruct` using NVIDIA ModelOpt AutoQuant.

Default behavior:

- Runs and verifies **4 scenarios** (2 modes × 2 quant pairs).
- Uses the **medium** dataset preset by default.
- Verifies by diffing **only `summary.json`** against `expected_report/`.

## Prerequisites

- Pixi env installed: `pixi install`
- CUDA GPU recommended (CPU is allowed but likely very slow)
- Local model snapshot link exists:
  - `models/qwen3_vl_4b_instruct/checkpoints/Qwen3-VL-4B-Instruct`
- Repo dataset assets exist:
  - `datasets/coco2017/source-data/`
  - `datasets/vlm-quantize-calib/coco2017_vlm_calib_<size>.db`
  - `datasets/vlm-quantize-calib/coco2017_captions_<size>.txt`

## One-click run (verify mode)

From repo root:

```bash
bash docs/tutorial/howto/tut-qwen3-vl-4b-layer-sensitivity/run_demo.sh
```

Subset examples:

```bash
# All-layers only (both quant pairs)
bash docs/tutorial/howto/tut-qwen3-vl-4b-layer-sensitivity/run_demo.sh --modes all_layers

# LM-only only, one quant pair
bash docs/tutorial/howto/tut-qwen3-vl-4b-layer-sensitivity/run_demo.sh \
  --modes lm_only \
  --quant-pairs wint4_afp16
```

## Refresh expected outputs (snapshot mode)

When intentional code changes alter outputs, refresh the tracked snapshots:

```bash
bash docs/tutorial/howto/tut-qwen3-vl-4b-layer-sensitivity/run_demo.sh --snapshot-report
```

## What `run_demo.sh` does

`run_demo.sh` is the tutorial pack’s public interface. It delegates to the
shared runner under `src/auto_quantize_model/qwen/` which:

1. Ensures required assets exist (checkpoint link + datasets) and fails fast with actionable guidance.
2. Creates a fresh workspace under `tmp/`.
3. Runs each selected scenario:
   - **all_layers**: VLM all-layers sensitivity (vision + text) per quant-pair.
   - **lm_only**: LM-only sensitivity (text tower) per quant-pair via the Hydra runner.
4. Generates a schema-locked `summary.json` per scenario.
5. Either verifies `summary.json` against `expected_report/` or snapshots `expected_report/` (if `--snapshot-report`).
   Snapshot mode writes **summary-only** snapshots and removes stale scenarios not selected.

## Outputs

Run workspace (untracked, created under `tmp/`):

```text
tmp/tutorial_workspace_qwen3_vl_4b_layer_sensitivity_<timestamp>/
├── outputs/
│   ├── all_layers/<quant_pair>/...
│   └── lm_only/<quant_pair>/...
```

Expected snapshots (tracked, sanitized):

```text
docs/tutorial/howto/tut-qwen3-vl-4b-layer-sensitivity/expected_report/
└── outputs/
    ├── all_layers/<quant_pair>/summary.json
    └── lm_only/<quant_pair>/summary.json
```

Markdown reports (generated for all scenarios):

```text
tmp/tutorial_workspace_qwen3_vl_4b_layer_sensitivity_<timestamp>/
└── outputs/<mode>/<quant_pair>/layer-sensitivity-report.md
```

This file includes both:

- the stable tutorial summary table (previously `summary.md`), and
- the per-layer sensitivity table.

Snapshot mode also writes a sanitized copy of this report under:

```text
docs/tutorial/howto/tut-qwen3-vl-4b-layer-sensitivity/expected_report/
└── outputs/<mode>/<quant_pair>/layer-sensitivity-report.md
```

## Troubleshooting

### Missing checkpoint link

Create the repo-local symlink:

```bash
ln -s /absolute/path/to/Qwen3-VL-4B-Instruct \
  models/qwen3_vl_4b_instruct/checkpoints/Qwen3-VL-4B-Instruct
```

Or use:

```bash
./models/qwen3_vl_4b_instruct/bootstrap.sh --yes
```

### Missing dataset assets

Create the COCO2017 link:

```bash
./datasets/coco2017/bootstrap.sh --path /absolute/path/to/coco2017
```

Ensure the tracked calibration assets exist:

- `datasets/vlm-quantize-calib/coco2017_vlm_calib_<size>.db`
- `datasets/vlm-quantize-calib/coco2017_captions_<size>.txt`

### “All-zero sensitivities” (degenerate summaries)

Verification fails if any scenario summary reports `has_nonzero_sensitivity=false`
(i.e., every per-layer sensitivity is exactly `0.0`).

Remediation:

- Prefer `--dataset-size medium` (default) or `--dataset-size large`.
- Avoid tiny calibration budgets when validating LM-only sensitivity.

## References

- Runner: `docs/tutorial/howto/tut-qwen3-vl-4b-layer-sensitivity/run_demo.sh`
- Shared CLI: `src/auto_quantize_model/qwen/cli_tutorial_pack_runner.py`
- Shared runner: `src/auto_quantize_model/qwen/tutorial_pack_runner.py`
- Summary builder: `src/auto_quantize_model/qwen/tutorial_pack_summary.py`
- All-layers runner: `models/qwen3_vl_4b_instruct/helpers/qwen3_vl_4b_autoquant_all_layers/run_qwen3_vl_4b_autoquant_all_layers.py`
- LM-only runner: `scripts/qwen/qwen3_lm_sensitivity.py`
