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
# One-click runner handles input setup automatically by:
# - copying the tracked captions file from inputs/
# - generating a tiny synthetic COCO-like root with a single JPEG
# - generating a 1-row SQLite calibration DB (`vlm_calib_small.db`)
```

### Step 3: Run Drivers + Verify Outputs (What `run_demo.sh` Does)

The tutorial is executed by `run_demo.sh`, which is intentionally written as a
robust, non-destructive orchestrator. It runs in a fresh workspace under `tmp/`
and never writes into the tracked tutorial directory unless you explicitly pass
`--snapshot-report`.

At a high level, `run_demo.sh` does:

1. Creates a workspace directory.
2. Prepares minimal calibration inputs.
3. Runs the all-layers INT8 driver.
4. Runs the LM-only INT8 driver.
5. Generates sanitized summaries.
6. Verifies against `expected_report/` (or snapshots it).

```bash
# From repo root:
bash docs/tutorial/howto/tut-qwen3-vl-4b-layer-sensitivity/run_demo.sh
```

#### 3.1 Workspace + inputs

`run_demo.sh` creates a fresh, gitignored workspace:

- `tmp/tutorial_workspace_qwen3_vl_4b_layer_sensitivity_<epoch>/`

Inside that workspace it generates:

- `coco2017/source-data/train2017/000000000001.jpg` (a tiny synthetic image)
- `vlm_calib_small.db` (SQLite DB with one row in `vlm_calib_samples`)
- `coco2017_captions_small.txt` (copied from `inputs/`)

#### 3.2 All-layers INT8 (vision + text)

This is the core all-layers call (smoke-test settings):

```bash
pixi run python \
  models/qwen3_vl_4b_instruct/helpers/qwen3_vl_4b_autoquant_all_layers/run_qwen3_vl_4b_autoquant_all_layers.py \
  --quant-format int8 \
  --model-dir models/qwen3_vl_4b_instruct/checkpoints/Qwen3-VL-4B-Instruct \
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

#### 3.2.1 Under the hood: `run_qwen3_vl_4b_autoquant_all_layers.py` (step by step)

This is what happens inside `models/qwen3_vl_4b_instruct/helpers/qwen3_vl_4b_autoquant_all_layers/run_qwen3_vl_4b_autoquant_all_layers.py` when you run the all-layers driver:

1. **Parse CLI args and pick a scheme**
   - `--quant-format {int8,fp8}` selects one of the hard-coded scheme configs:
     - `AUTOQUANT_INT8_ALL_LAYERS` → `INT8_ALL_LAYERS_CFG`, `effective_bits=8.0`
     - `AUTOQUANT_FP8_ALL_LAYERS` → `FP8_ALL_LAYERS_CFG`, `effective_bits=11.0`
   - Optional overrides `--effective-bits` and `--auto-quantize-score-size` rewrite the chosen scheme.
   - If `--output-dir` is omitted, the script picks a default under `tmp/` based on `--quant-format` and `--max-calib-samples`.

2. **Report-only short path**
   - If `--report-only` is set, the driver does not run AutoQuant:
     - it reads `<output_dir>/<scheme.name>_quant_manifest.json`
     - it regenerates `layer-sensitivity-report.{md,json}` from that manifest

3. **Load the HF model and preprocessing stack**
   - Loads `Qwen3-VL-4B-Instruct` via `transformers.AutoModelForImageTextToText.from_pretrained(...)` with `trust_remote_code=True`.
   - Uses `torch_dtype=bfloat16` on CUDA and `float32` on CPU.
   - Loads `AutoTokenizer` and sets `padding_side="left"` (needed for batching/padding behavior).
   - Loads `AutoProcessor` for multimodal preprocessing.

4. **Build the VLM calibration samples (COCO-like)**
   - Reads `(image_relpath, caption)` rows from the SQLite DB table `vlm_calib_samples` (up to `--max-calib-samples`).
   - Resolves each image path as `<coco_root>/<image_relpath>`.
   - For each sample, constructs a “chat template” message containing one image + one caption, then:
     - `tokenizer.apply_chat_template(..., add_generation_prompt=True)` produces the text prompt
     - `qwen_vl_utils.process_vision_info(messages)` extracts the image/video inputs
     - `processor(...)` converts (text + image/video inputs) into tensor inputs
   - Adds `labels = input_ids.clone()` so the driver can compute a causal LM loss during gradient-based scoring.
   - The loader is a simple Python list of samples (each item treated as its own batch), to avoid image collation edge-cases.

5. **Resolve quantization formats and run AutoQuant**
   - For the selected scheme, each format name is resolved to a ModelOpt config:
     - first from `auto_quantize_model.modelopt_configs.CUSTOM_QUANT_CONFIGS` (repo-defined configs like `INT8_ALL_LAYERS_CFG`)
     - otherwise from `modelopt.torch.quantization` presets (if present)
   - Calls `modelopt.torch.quantization.auto_quantize(...)` with:
     - `constraints={"effective_bits": scheme.auto_quantize_bits}`
     - `forward_step(model, batch)` that moves tensors to the target device
     - a standard causal LM cross-entropy loss over `logits` vs `labels`
     - `num_score_steps` derived from `--auto-quantize-score-size` and `--batch-size`
   - The result is a quantized model + an AutoQuant `state_dict` containing (among other things) per-layer `candidate_stats` and the chosen “best” solution.

6. **Write artifacts**
   - Saves the raw AutoQuant state: `<scheme.name>_autoquant_state.pt`
   - Builds a manifest JSON:
     - enumerates quantized linear layers using `modelopt.torch.quantization.utils.is_quantized_linear`
     - extracts `candidate_stats` into `layer_sensitivity` and a ranked list of layers
     - records dataset + quantization metadata and a small composed-config YAML for reproducibility
   - Writes:
     - `<scheme.name>_quant_manifest.json`
     - `layer-sensitivity-report.md` and `layer-sensitivity-report.json` via `auto_quantize_model.modelopt_autoquant.write_layer_sensitivity_md/json`

The tutorial’s expected outputs under `expected_report/all_layers_int8/` are a sanitized snapshot of these artifacts (paths replaced with `<ABSOLUTE_PATH>`).

#### 3.3 LM-only INT8 (text tower)

This runs the LM-only driver over the captions file:

```bash
pixi run python \
  models/qwen3_vl_4b_instruct/helpers/qwen3_vl_4b_autoquant_int8_lm/run_qwen3_vl_4b_autoquant_int8_lm.py \
  --model-dir models/qwen3_vl_4b_instruct/checkpoints/Qwen3-VL-4B-Instruct \
  --output-dir <WORKSPACE>/lm_only_int8 \
  --captions-path <WORKSPACE>/coco2017_captions_small.txt \
  --max-calib-samples 1 \
  --calib-seq-len 64 \
  --batch-size 1 \
  --device cuda:0 \
  --auto-quantize-score-size 1
```

Outputs land under `<WORKSPACE>/lm_only_int8/` with the same artifact shapes.

#### 3.4 Summaries, sanitization, and verification

`run_demo.sh` produces stable summaries used for verification:

- `summaries/all_layers_int8/summary.{json,md}`
- `summaries/lm_only_int8/summary.{json,md}`

It compares these against `expected_report/**/summary.{json,md}`. In snapshot
mode it also sanitizes and copies the detailed run artifacts into
`expected_report/` (paths replaced with `<ABSOLUTE_PATH>`).

#### 3.5 Flags

To also run the FP8 all-layers pass:

```bash
bash docs/tutorial/howto/tut-qwen3-vl-4b-layer-sensitivity/run_demo.sh --with-fp8
```

### Complete Runnable Script

This tutorial pack is executed via its one-click runner:

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
