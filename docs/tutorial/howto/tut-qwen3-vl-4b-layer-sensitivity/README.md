# How to Run Qwen3-VL-4B Layer Sensitivity with NVIDIA ModelOpt (Tutorial Pack)

## Question
How do I run an end-to-end per-layer sensitivity analysis for `Qwen3-VL-4B-Instruct` using NVIDIA ModelOpt AutoQuant (all-layers + LM-only) in this repo?

## Prerequisites

- [ ] **Environment:** `pixi` is installed and `pixi install` has been run for this repo.
- [ ] **GPU:** CUDA is available (recommended). CPU runs may work but can be extremely slow.
- [ ] **Model snapshot:** `models/qwen3_vl_4b_instruct/checkpoints/Qwen3-VL-4B-Instruct` exists (usually a symlink).
  - If missing: `./models/qwen3_vl_4b_instruct/bootstrap.sh --yes`

## Implementation Idea

*   **Approach:**
    1. Create a minimal, self-contained calibration setup (tiny COCO-like image root + 1-row SQLite DB + 1-line captions file).
    2. Run the all-layers sensitivity driver (vision + text) and the LM-only driver (text tower) using `pixi run`.
    3. Snapshot and/or verify the results against `expected_report/` (sanitized for portability).

## Step-by-Step with Code

### Step 1: Environment + Model Link

The ModelOpt drivers load the HF snapshot from `models/qwen3_vl_4b_instruct/checkpoints/Qwen3-VL-4B-Instruct`. This tutorial assumes that directory exists locally (it is intentionally ignored by git).

```bash
# From repo root:
pixi install
./models/qwen3_vl_4b_instruct/bootstrap.sh --yes
test -d models/qwen3_vl_4b_instruct/checkpoints/Qwen3-VL-4B-Instruct
```

### Step 2: Prepare Minimal Calibration Inputs (Smoke-Test)

The all-layers driver expects a COCO-style root plus a small SQLite DB mapping `image_relpath` → caption. The LM-only driver uses a captions text file (one caption per line).

This tutorial pack provides a minimal captions file in `inputs/` and generates the tiny DB + image in a gitignored workspace under `tmp/`.

```bash
# One-click runner handles input setup automatically.
# (See run_demo.sh for the embedded input generator.)
```

### Step 3: Run Drivers + Verify Outputs

Run the tutorial pack from the repo root. It will:

- create a fresh workspace under `tmp/tutorial_workspace_qwen3_vl_4b_layer_sensitivity_<epoch>/`
- run all-layers INT8
- run LM-only INT8
- generate sanitized summaries and compare them against `expected_report/` (if present)

```bash
bash docs/tutorial/howto/tut-qwen3-vl-4b-layer-sensitivity/run_demo.sh
```

To also run the FP8 all-layers pass:

```bash
bash docs/tutorial/howto/tut-qwen3-vl-4b-layer-sensitivity/run_demo.sh --with-fp8
```

### Complete Runnable Script

This tutorial pack is designed to be executed via its one-click runner:

```bash
#!/usr/bin/env bash
set -euo pipefail

# From repo root:
bash docs/tutorial/howto/tut-qwen3-vl-4b-layer-sensitivity/run_demo.sh
```

### [Optional] Maintenance Mode: Update the Golden Expected Report

When legitimate code changes alter outputs, regenerate the tracked expected report snapshot:

```bash
bash docs/tutorial/howto/tut-qwen3-vl-4b-layer-sensitivity/run_demo.sh --snapshot-report
```

## Input and Output

### Input

*   `model_dir` (path): `models/qwen3_vl_4b_instruct/checkpoints/Qwen3-VL-4B-Instruct`
*   `device` (str): default `cuda:0` (override with `--device`)
*   `captions` (file): `docs/tutorial/howto/tut-qwen3-vl-4b-layer-sensitivity/inputs/coco2017_captions_small.txt`
*   `vlm_calib_db` (sqlite db): generated into the workspace (1 row)
*   `coco_root` (dir): generated into the workspace (1 image)

### Output

The workspace contains full driver outputs plus sanitized summaries:

```text
<REPO_ROOT>/tmp/tutorial_workspace_qwen3_vl_4b_layer_sensitivity_<epoch>/
├── all_layers_int8/
│   ├── int8_autoquant_all_layers_int8_quant_manifest.json
│   ├── layer-sensitivity-report.json
│   └── layer-sensitivity-report.md
├── lm_only_int8/
│   ├── int8_autoquant_lm_default_quant_manifest.json
│   ├── layer-sensitivity-report.json
│   └── layer-sensitivity-report.md
└── summaries/
    ├── all_layers_int8/summary.json
    └── lm_only_int8/summary.json
```

## Appendix: Key Parameters and Files

### Key Parameters (Table)

| Name | Value | Explanation |
|------|-------|-------------|
| `MODEL_DIR` | `models/qwen3_vl_4b_instruct/checkpoints/Qwen3-VL-4B-Instruct` | HF snapshot directory (local symlink) |
| `DEVICE` | `cuda:0` | Torch device used by both drivers (override via `--device`) |
| `max_calib_samples` | `1` | Smoke-test calibration size |
| `calib_seq_len` | `64` | Smoke-test sequence length |
| `batch_size` | `1` | Smoke-test batch size |
| `auto_quantize_score_size` | `1` | Smoke-test score size |
| `WORKSPACE_DIR` | `tmp/tutorial_workspace_qwen3_vl_4b_layer_sensitivity_<epoch>/` | Gitignored outputs for the run |

### Input Files

*   `docs/tutorial/howto/tut-qwen3-vl-4b-layer-sensitivity/inputs/coco2017_captions_small.txt`: 1-line captions file used by the LM-only driver.

### Output Files

*   `tmp/tutorial_workspace_qwen3_vl_4b_layer_sensitivity_<epoch>/all_layers_int8/*`: all-layers INT8 run outputs.
*   `tmp/tutorial_workspace_qwen3_vl_4b_layer_sensitivity_<epoch>/lm_only_int8/*`: LM-only INT8 run outputs.
*   `docs/tutorial/howto/tut-qwen3-vl-4b-layer-sensitivity/expected_report/*`: tracked, sanitized “golden” outputs (updated via `--snapshot-report`).

## References

### Relevant Source Code

*   `docs/tutorial/howto/tut-qwen3-vl-4b-layer-sensitivity/run_demo.sh`: one-click tutorial runner (workspace creation, driver invocations, verification).
*   `models/qwen3_vl_4b_instruct/helpers/qwen3_vl_4b_autoquant_all_layers/run_qwen3_vl_4b_autoquant_all_layers.py`: all-layers (vision + text) AutoQuant driver.
*   `models/qwen3_vl_4b_instruct/helpers/qwen3_vl_4b_autoquant_int8_lm/run_qwen3_vl_4b_autoquant_int8_lm.py`: LM-only INT8 AutoQuant driver.
*   `docs/tutorial/howto/tut-qwen3-vl-4b-layer-sensitivity/scripts/sanitize_artifacts.py`: sanitizes run artifacts for `expected_report/`.
*   `docs/tutorial/howto/tut-qwen3-vl-4b-layer-sensitivity/scripts/summarize_manifest.py`: generates stable summaries used for verification.
