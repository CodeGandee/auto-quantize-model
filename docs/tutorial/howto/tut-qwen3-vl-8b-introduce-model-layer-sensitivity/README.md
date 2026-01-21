# How to Introduce Qwen3-VL-8B and Run Layer Sensitivity (Tutorial Pack)

## Question
How do I introduce `Qwen3-VL-8B-Instruct` into this repo (via a local checkpoint link) and run an end-to-end per-layer sensitivity analysis using NVIDIA ModelOpt AutoQuant (all-layers + LM-only) in the Pixi default env?

## Prerequisites

- [ ] **Environment:** `pixi` is installed and `pixi install` has been run for this repo.
- [ ] **GPU:** CUDA is available (recommended). CPU runs may work but can be extremely slow.
- [ ] **Local model snapshot exists:** you have a local directory containing the model files for `Qwen3-VL-8B-Instruct`.
  - On this host we keep snapshots in `/data1/huangzhe/llm-models/` (override via `HF_SNAPSHOTS_ROOT`).

## Implementation Idea

*   **Approach:**
    1. Create a stable “checkpoint link” inside the repo at `models/qwen3_vl_8b_instruct/checkpoints/Qwen3-VL-8B-Instruct` (a symlink; ignored by git).
    2. Build a tiny calibration setup (synthetic COCO-like image root + 1-row SQLite DB + 1-line captions file).
    3. Run the all-layers INT8 sensitivity driver (vision + text) and the LM-only INT8 driver (text tower) using `pixi run`.
    4. Verify outputs against `expected_report/` (sanitized and tracked), or regenerate the expected report snapshot when behavior changes intentionally.

## Step-by-Step with Code

### Step 1: Introduce the model (create the checkpoint link)

This repo does not commit model weights. Instead, it expects a local checkpoint directory and a symlink from:

- `models/qwen3_vl_8b_instruct/checkpoints/Qwen3-VL-8B-Instruct`

to wherever you store the snapshot on your machine.

If your snapshots live in `/data1/huangzhe/llm-models` (default on this host):

```bash
ln -s /data1/huangzhe/llm-models/Qwen3-VL-8B-Instruct \
  models/qwen3_vl_8b_instruct/checkpoints/Qwen3-VL-8B-Instruct
```

Or use the interactive helper:

```bash
./models/qwen3_vl_8b_instruct/bootstrap.sh --yes
```

### Step 2: Prepare minimal calibration inputs (smoke-test)

The all-layers driver expects a COCO-style root plus a small SQLite DB mapping `image_relpath` → caption. The LM-only driver uses a captions text file (one caption per line).

This tutorial pack provides a minimal captions file in `inputs/` and generates the tiny DB + image in a gitignored workspace under `tmp/`.

### Step 3: Run the tutorial end-to-end (what `run_demo.sh` does)

`run_demo.sh` is intentionally a robust, non-destructive orchestrator. It:

1. Ensures the `models/qwen3_vl_8b_instruct/checkpoints/Qwen3-VL-8B-Instruct` link exists.
2. Creates a fresh workspace directory under `tmp/`.
3. Creates a synthetic “COCO-like” image root with 1 JPEG.
4. Creates a 1-row SQLite DB table `vlm_calib_samples` containing `(image_relpath, caption)`.
5. Runs two ModelOpt AutoQuant drivers:
   - all-layers INT8 (vision + text)
   - LM-only INT8 (text tower only)
6. Builds stable, sanitized summaries and either:
   - verifies them against `expected_report/`, or
   - snapshots `expected_report/` if you pass `--snapshot-report`.

Run it:

```bash
# From repo root:
bash docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/run_demo.sh
```

#### 3.1 All-layers INT8 (vision + text)

The runner executes this (smoke-test settings):

> **Why does the 8B tutorial call a “4B” script?**
>
> This is intentional: the all-layers driver is parameterized by `--model-dir`, so it can run with
> `Qwen3-VL-8B-Instruct` as long as you pass the 8B checkpoint path. The filename still contains “4b”
> because it was originally introduced for the 4B model, but it is model-agnostic at runtime.
>
> One cosmetic caveat: the script’s log line prints `Loading Qwen3-VL-4B-Instruct ...` even when the
> `--model-dir` points to the 8B checkpoint.

```bash
pixi run python \
  models/qwen3_vl_4b_instruct/helpers/qwen3_vl_4b_autoquant_all_layers/run_qwen3_vl_4b_autoquant_all_layers.py \
  --quant-format int8 \
  --model-dir models/qwen3_vl_8b_instruct/checkpoints/Qwen3-VL-8B-Instruct \
  --output-dir <WORKSPACE>/all_layers_int8 \
  --vlm-calib-db <WORKSPACE>/vlm_calib_small.db \
  --coco-root <WORKSPACE>/coco2017/source-data \
  --max-calib-samples 1 \
  --calib-seq-len 64 \
  --batch-size 1 \
  --device cuda:0 \
  --auto-quantize-score-size 1
```

Outputs land under `<WORKSPACE>/all_layers_int8/`, including:

- `*_quant_manifest.json`
- `layer-sensitivity-report.{md,json}`
- `composed-config.yaml`

#### 3.1.1 Under the hood: `run_qwen3_vl_4b_autoquant_all_layers.py` (step by step)

Even though the file is named “4b”, the driver is parameterized via `--model-dir` and can run with the 8B checkpoint. At a high level it:

1. **Selects a scheme** from `--quant-format` (`int8` or `fp8`) and applies optional overrides (score size / effective bits).
2. **Loads the HF model + processor/tokenizer** using `transformers`.
3. **Builds the VLM calibration loader** from a COCO root + a SQLite DB table named `vlm_calib_samples`.
4. **Runs ModelOpt AutoQuant** (`modelopt.torch.quantization.auto_quantize`) to explore mixed-precision candidates under an effective-bits constraint.
5. **Writes artifacts**:
   - `<scheme>_autoquant_state.pt`
   - `<scheme>_quant_manifest.json` (includes `layer_sensitivity` extracted from candidate stats)
   - `layer-sensitivity-report.{md,json}`
   - `composed-config.yaml` (a small reproducibility summary)

Source: `models/qwen3_vl_4b_instruct/helpers/qwen3_vl_4b_autoquant_all_layers/run_qwen3_vl_4b_autoquant_all_layers.py`.

#### 3.2 LM-only INT8 (text tower)

The runner executes this (smoke-test settings):

```bash
pixi run python \
  models/qwen3_vl_4b_instruct/helpers/qwen3_vl_4b_autoquant_int8_lm/run_qwen3_vl_4b_autoquant_int8_lm.py \
  --model-dir models/qwen3_vl_8b_instruct/checkpoints/Qwen3-VL-8B-Instruct \
  --output-dir <WORKSPACE>/lm_only_int8 \
  --captions-path <WORKSPACE>/coco2017_captions_small.txt \
  --max-calib-samples 1 \
  --calib-seq-len 64 \
  --batch-size 1 \
  --device cuda:0 \
  --auto-quantize-score-size 1
```

This uses the LM-only path in `auto_quantize_model.qwen.autoquant_sensitivity` to keep the vision tower out of the sensitivity loop.

### Complete Runnable Script

This tutorial pack is executed via its one-click runner:

```bash
# From repo root:
bash docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/run_demo.sh
```

### [Optional] Maintenance Mode: Update the Golden Expected Report

When legitimate code changes alter outputs, regenerate the tracked expected report snapshot:

```bash
bash docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/run_demo.sh --snapshot-report
```

## Input and Output

### Input

*   `model_dir` (path): `models/qwen3_vl_8b_instruct/checkpoints/Qwen3-VL-8B-Instruct`
*   `device` (str): default `cuda:0` (override with `--device`)
*   `captions` (file): `docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/inputs/coco2017_captions_small.txt`
*   `vlm_calib_db` (sqlite db): generated into the workspace (1 row)
*   `coco_root` (dir): generated into the workspace (1 image)

### Output

The workspace contains full driver outputs plus sanitized summaries:

```text
<REPO_ROOT>/tmp/tutorial_workspace_qwen3_vl_8b_intro_layer_sensitivity_<epoch>/
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
| `MODEL_DIR` | `models/qwen3_vl_8b_instruct/checkpoints/Qwen3-VL-8B-Instruct` | HF snapshot directory (local symlink) |
| `HF_SNAPSHOTS_ROOT` | `/data1/huangzhe/llm-models` | Local snapshot root used by `run_demo.sh` when creating the link |
| `DEVICE` | `cuda:0` | Torch device used by both drivers (override via `--device`) |
| `max_calib_samples` | `1` | Smoke-test calibration size |
| `calib_seq_len` | `64` | Smoke-test sequence length |
| `batch_size` | `1` | Smoke-test batch size |
| `auto_quantize_score_size` | `1` | Smoke-test score size |
| `WORKSPACE_DIR` | `tmp/tutorial_workspace_qwen3_vl_8b_intro_layer_sensitivity_<epoch>/` | Gitignored outputs for the run |

### Input Files

*   `docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/inputs/coco2017_captions_small.txt`: 1-line captions file for LM-only smoke testing.

### Output Files

Tracked golden outputs (sanitized):

*   `docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/expected_report/all_layers_int8/summary.json`: stable keys for verification.
*   `docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/expected_report/lm_only_int8/summary.json`: stable keys for verification.
*   `docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/expected_report/**/layer-sensitivity-report.{md,json}`: detailed sensitivity artifacts (sanitized).
*   `docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/expected_report/**/quant_manifest.json`: sanitized quantization manifest.

Non-tracked run outputs:

*   `tmp/tutorial_workspace_qwen3_vl_8b_intro_layer_sensitivity_<epoch>/...`: full raw run outputs, including `run.log`.

## References

### Relevant Source Code

*   `models/qwen3_vl_8b_instruct/bootstrap.sh`: interactive helper to create/update the local checkpoint link.
*   `models/qwen3_vl_4b_instruct/helpers/qwen3_vl_4b_autoquant_all_layers/run_qwen3_vl_4b_autoquant_all_layers.py`: all-layers AutoQuant driver (parameterized by `--model-dir`).
*   `models/qwen3_vl_4b_instruct/helpers/qwen3_vl_4b_autoquant_int8_lm/run_qwen3_vl_4b_autoquant_int8_lm.py`: LM-only INT8 AutoQuant driver (parameterized by `--model-dir`).
*   `src/auto_quantize_model/qwen/autoquant_sensitivity.py`: core LM-only sensitivity runner used by the LM-only driver.
