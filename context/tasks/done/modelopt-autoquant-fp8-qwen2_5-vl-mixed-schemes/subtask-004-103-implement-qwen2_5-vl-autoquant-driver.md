# Subtask 4.3: Implement Qwen2.5-VL-3B LM-only AutoQuant FP8 driver

## Scope

Implement the main AutoQuant driver script `scripts/qwen/qwen2_5_vl_3b_autoquant_fp8_schemes.py` that loads the Qwen2.5-VL-3B language model only, constructs a calibration forward loop over text data, invokes ModelOpt’s AutoQuant to derive mixed-precision FP8 configurations according to the scheme definitions, and emits both a quantized model and a machine-readable quantization manifest. The driver is intended as a **research tool** for exploring different quantization configs and sensitivity settings, not as a production accuracy-maximizing pipeline.

## Planned outputs

- A new Python script `scripts/qwen/qwen2_5_vl_3b_autoquant_fp8_schemes.py` with a clear CLI:
  - Required flags such as `--scheme-name`, `--model-dir`, and `--output-dir`.
  - Optional flags for AutoQuant parameters (e.g., `--effective-bits`, `--auto-quantize-method`, `--num-score-steps`) with sensible defaults pulled from the scheme catalog, plus support for selecting **custom quantization configs** beyond the default FP8 scheme.
- Implementation of LM-only Qwen2.5-VL-3B loading (reusing existing LM-only extraction patterns from the FP8 baseline path where possible).
- A calibration forward loop over COCO captions (text-only) that is compatible with ModelOpt AutoQuant expectations.
- Generation of a quantization manifest (e.g., JSON) that records, per layer, whether it is FP8 or BF16/FP16 plus any relevant scores or metadata, including **layer sensitivity statistics**.
- Ability to run **pure PyTorch inference** on the quantized LM (no vLLM required), so we can compare outputs across schemes and configs for error analysis.

## TODOs

- [x] Job-004-103-001 Sketch the CLI and main entrypoint for `scripts/qwen/qwen2_5_vl_3b_autoquant_fp8_schemes.py`, including flags for `--scheme-name`, `--model-dir`, `--output-dir`, and optional overrides for AutoQuant settings.
- [x] Job-004-103-002 Implement LM-only loading for Qwen2.5-VL-3B by reusing or adapting the pattern from the existing FP8 LM-only checkpoint path (e.g., detaching the vision tower and freezing its weights).
- [x] Job-004-103-003 Implement a calibration dataset loader that reads COCO captions from `datasets/vlm-quantize-calib/coco2017_captions.txt`, tokenizes them with the Qwen2.5-VL tokenizer, and builds a simple text-only data loader with a configurable number of samples.
- [x] Job-004-103-004 Implement the calibration forward loop callable that runs the LM on a batch of tokens (no images) and is suitable for passing to `mtq.auto_quantize`.
- [x] Job-004-103-005 Wire up ModelOpt AutoQuant invocation inside the driver, using scheme-specific defaults from Subtask 4.2 but allowing CLI overrides for experimentation.
- [x] Job-004-103-006 Design and implement a quantization manifest format (e.g., JSON mapping layer names to quantization dtypes and any scores) and write it to `--output-dir` alongside any temporary artifacts.
- [x] Job-004-103-007 Add basic logging and argument validation to the script, including clear error messages when required paths or dependencies are missing.

## Notes

- Follow existing patterns for Qwen2.5-VL sanity scripts and FP8 tools when choosing how to load the model, tokenizer, and calibration data.
- Keep the driver script focused on AutoQuant and manifest generation; checkpoint export is handled in a separate subtask.

## Summary

This subtask adds the LM-only AutoQuant FP8 driver:

- Implemented `scripts/qwen/qwen2_5_vl_3b_autoquant_fp8_schemes.py` with a CLI that accepts `--scheme-name`, `--model-dir`, `--output-dir`, `--captions-path`, calibration sizes, and optional overrides for effective bits, method, and score size.
- Added an `AUTOQUANT_FP8_SCHEMES` catalog (three schemes initially, now extended with an `fp8_autoquant_all_layers_fp8` variant) consistent with Subtask 4.2 and exposed via CLI choices.
- Implemented LM-only extraction for Qwen2.5-VL using `get_language_model_from_vl` in the RTX 5090 vLLM env, keeping the vision tower unmodified while focusing AutoQuant on the language model module.
- Built a COCO captions-based text-only calibration dataset and dataloader, plus the `forward_step` / `loss_func` wiring required by `mtq.auto_quantize`.
- Wired ModelOpt AutoQuant for FP8 formats on the LM-only module and produced a JSON manifest per run that records which layers are quantized along with the scheme metadata and AutoQuant state keys, written to `--output-dir`.
- Added support for custom quantization configs through `auto_quantize_model.modelopt_configs.CUSTOM_QUANT_CONFIGS`, including an `FP8_ALL_LAYERS_CFG` that removes default name-pattern exclusions and enables all-layer FP8 sensitivity analysis.
- Implemented a VLM-aware calibration path for the all-layers scheme using COCO2017 image+caption pairs and `CocoVlmDataset`, so both language and vision blocks receive non-zero AutoQuant sensitivity scores.
- Added a `--report-only` CLI flag and a `layer-sensitivity-report.md` generator that reads the manifest’s `layer_sensitivity` and `autoquant_state` sections and emits a sorted Markdown table summarizing effective bits, sensitivity scores, and costs per layer without rerunning AutoQuant.

## Test plan: sanity-check AutoQuant layer sensitivity

Goal: verify that `qwen2_5_vl_3b_autoquant_fp8_schemes.py` runs end-to-end on a small calibration set, produces a manifest with quantized LM layers, and exposes AutoQuant’s per-layer sensitivity state (via `candidate_stats` / `best` keys).

### 1. Environment

- Use the RTX 5090 vLLM Pixi env for all runs:
  - `pixi run -e rtx5090-vllm python ...`
- Ensure the base checkpoint is bootstrapped:
  - `models/qwen2_5_vl_3b_instruct/bootstrap.sh`

### 2. Prepare a small calibration dataset

Option A (reuse existing captions file):

- Use `datasets/vlm-quantize-calib/coco2017_captions.txt` directly.
- Limit calibration size via CLI:
  - `--max-calib-samples 64`
  - `--batch-size 4`

Option B (tiny custom file for faster checks):

- Create `tmp/autoquant-sanity/coco2017_captions_tiny.txt` with ~16 short, diverse captions (one per line).
- Pass:
  - `--captions-path tmp/autoquant-sanity/coco2017_captions_tiny.txt`
  - `--max-calib-samples 16`
  - `--batch-size 4`

### 3. Run the driver for a single scheme

Example (top-50 scheme, small run):

```bash
pixi run -e rtx5090-vllm python scripts/qwen/qwen2_5_vl_3b_autoquant_fp8_schemes.py \
  --scheme-name fp8_autoquant_top50 \
  --model-dir models/qwen2_5_vl_3b_instruct/checkpoints/Qwen2.5-VL-3B-Instruct \
  --output-dir tmp/autoquant-sanity/fp8_autoquant_top50 \
  --captions-path datasets/vlm-quantize-calib/coco2017_captions.txt \
  --max-calib-samples 64 \
  --batch-size 4 \
  --calib-seq-len 256
```

Checks:

- Script exits with status 0.
- `tmp/autoquant-sanity/fp8_autoquant_top50/fp8_autoquant_top50_quant_manifest.json` exists.

### 4. Inspect manifest for layer coverage and AutoQuant state

Open the manifest and verify:

- `scheme.name` matches the selected scheme (`fp8_autoquant_top50`).
- `num_quantized_layers` is > 0 (and not excessively large for the small LM-only run).
- `layers` keys look like LM transformer modules (e.g., contain `model.layers` / `decoder.layers`), and no obvious vision-only prefixes appear.
- `autoquant_state.keys` contains at least:
  - `"candidate_stats"`
  - `"best"`

These indicate that:

- AutoQuant has run and selected recipes per layer (candidate stats + best solution).
- The script has successfully exposed the presence of per-layer sensitivity information (even if detailed scores are not yet dumped into the manifest).

Quick CLI probe (optional):

```bash
pixi run -e rtx5090-vllm python - << 'EOF'
import json, pathlib
manifest_path = pathlib.Path("tmp/autoquant-sanity/fp8_autoquant_top50/fp8_autoquant_top50_quant_manifest.json")
m = json.loads(manifest_path.read_text())
print("Scheme:", m["scheme"]["name"])
print("Quantized layers:", m["num_quantized_layers"])
print("Has candidate_stats:", "candidate_stats" in m["autoquant_state"]["keys"])
print("Has best:", "best" in m["autoquant_state"]["keys"])
EOF
```

Expected output:

- A non-zero `Quantized layers` count.
- `Has candidate_stats: True`
- `Has best: True`

### 5. Compare schemes (optional)

Repeat step 3 for `fp8_autoquant_top25` and `fp8_autoquant_full` (same small calibration settings) and compare:

- `num_quantized_layers` across schemes:
  - `top25` ≤ `top50` ≤ `full` (or at least non-increasing in conservativeness).
- Presence of `candidate_stats` / `best` keys for all schemes.

This provides a lightweight sanity check that:

- Scheme selection affects how many LM layers AutoQuant chooses to quantize.
- Underlying layer sensitivity analysis is being run and recorded by ModelOpt, and the driver script is correctly surfacing its state keys.
