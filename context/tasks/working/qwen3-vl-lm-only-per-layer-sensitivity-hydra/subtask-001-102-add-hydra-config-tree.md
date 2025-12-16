# Subtask 1.2: Add Hydra config tree for Qwen3 LM-only sensitivity

## Scope

Create a Hydra configuration composition for Qwen3‑VL **LM-only** per-layer sensitivity runs so we can sweep:

- Calibration subset size (`small|medium|large`)
- Precision pair(s) (weight/activation)
- AutoQuant scoring settings and output layout

This subtask focuses on `conf/` structure and composition (defaults + overrides), not the Python runner implementation.

## Planned outputs

- Model config for Qwen3‑VL checkpoint:
  - `conf/model/qwen3_vl_4b_instruct/arch/qwen3_vl_4b_instruct.default.yaml`
- Dataset config(s) for captions calibration subsets:
  - `conf/dataset/vlm_coco2017_captions.yaml` (with a `size` field and interpolated path)
- New Hydra config group to select precision pair or format-set (based on Subtask 1.1 decision):
  - `conf/quant_pair/*.yaml` (or `conf/quant_formats/*.yaml`)
- AutoQuant defaults config:
  - `conf/autoquant/gradient_default.yaml` (method, effective bits, score size, etc.)
- A dedicated task config wiring defaults together for the runner:
  - `conf/preset/qwen3_lm_sensitivity.yaml` (or a similar top-level config name)

## Implementation notes (what was added)

- Dedicated config entrypoint: `conf/preset/qwen3_lm_sensitivity.yaml` (we do not reuse `conf/config.yaml` to avoid YOLO defaults).
- Model arch + infer preset:
  - `conf/model/qwen3_vl_4b_instruct/arch/qwen3_vl_4b_instruct.default.yaml`
  - `conf/model/qwen3_vl_4b_instruct/infer/qwen3_vl_4b_instruct.default.yaml`
- Captions dataset with `size` selector + `size_to_max_samples` mapping:
  - `conf/dataset/vlm_coco2017_captions.yaml`
- Precision pair selector group (one pair per run, as decided in Subtask 1.1):
  - `conf/quant_pair/wfp4_afp8.yaml`, `conf/quant_pair/wfp4_afp16.yaml`
  - `conf/quant_pair/wfp8_afp8.yaml`, `conf/quant_pair/wfp8_afp16.yaml`
  - `conf/quant_pair/wint8_afp16.yaml`, `conf/quant_pair/wint8_afp8.yaml`
  - Legacy/published parity: `conf/quant_pair/wint8_aint8.yaml`
- AutoQuant defaults:
  - `conf/autoquant/gradient_default.yaml`
- Output layout presets (iteration vs publish):
  - `conf/output_layout/tmp.yaml`
  - `conf/output_layout/publish.yaml`

## TODOs

- [x] Job-001-102-001 Choose Hydra config entrypoint conventions
  - Decide `config_path` / `config_name` for the runner (e.g., `conf` + `qwen3_lm_sensitivity`).
  - Decide whether the runner uses `conf/config.yaml` or a dedicated config file (prefer dedicated to avoid YOLO defaults).

- [x] Job-001-102-002 Add Qwen3 model arch config
  - Create `conf/model/qwen3_vl_4b_instruct/arch/qwen3_vl_4b_instruct.default.yaml` mirroring the existing Qwen2.5 pattern.
  - Include: `name`, `family`, `variant`, `format`, `path`, and `dtype` (and any additional metadata the runner will need).

- [x] Job-001-102-003 Add captions calibration dataset config
  - Create `conf/dataset/vlm_coco2017_captions.yaml` with:
    - `root` (base dataset dir via `${hydra:runtime.cwd}`)
    - `size: small|medium|large`
    - `captions_path: ${...}/coco2017_captions_${dataset.size}.txt` (or equivalent)
    - Optional: `max_samples` and `seq_len` defaults (runner may override)

- [x] Job-001-102-004 Add precision-pair config group
  - Create a `conf/quant_pair/` group (or equivalent) with one config per intended pair:
    - `wfp4_afp16`, `wfp4_afp8`, `wfp8_afp16`, `wfp8_afp8`, `wint8_afp16`, `wint8_afp8`
  - Each config should be purely declarative (names, mapping keys), with the actual config dict selection handled by code or by referencing `CUSTOM_QUANT_CONFIGS` names.

- [x] Job-001-102-005 Add AutoQuant settings group
  - Create `conf/autoquant/gradient_default.yaml` with:
    - `method` (`gradient` vs `kl_div`)
    - `effective_bits`
    - `score_size` (in samples)
    - `calib_steps` / `batch_size` / `device`
    - Any fixed knobs we want standardized across runs

- [x] Job-001-102-006 Add output layout config
  - Decide whether output layout is configured via:
    - A small config group (e.g., `conf/output_layout/tmp.yaml` vs `conf/output_layout/publish.yaml`), or
    - Fields under the task config (e.g., `output.publish_dir`, `output.use_hydra_run_dir`)
  - Ensure we can write to:
    - Hydra run dir under `tmp/model-experiments/...` for iteration
    - Published location under `models/qwen3_vl_4b_instruct/layer-analysis/...` for committed artifacts

- [x] Job-001-102-007 Create the dedicated task config wiring defaults
  - Add `conf/preset/qwen3_lm_sensitivity.yaml` with defaults similar to:
    - model: qwen3_vl_4b_instruct.default
    - dataset: vlm_coco2017_captions
    - quant_pair: <default pair>
    - autoquant: gradient_default
    - output layout: <default>
  - Ensure overrides are ergonomic and composable for multirun.

- [x] Job-001-102-008 Add example overrides (for later docs)
  - Record canonical example commands to be copied into docs in Subtask 1.6:
    - Single run for one pair and one dataset size
    - Multirun over pairs × sizes

## Status

Completed.

## What was done

- Added a dedicated Hydra entrypoint config to avoid coupling to unrelated defaults:
  - `conf/preset/qwen3_lm_sensitivity.yaml` (includes `hydra.run.dir` and `hydra.sweep.dir` under `tmp/model-experiments/...`).
- Added Qwen3-VL model configs:
  - `conf/model/qwen3_vl_4b_instruct/arch/qwen3_vl_4b_instruct.default.yaml`
  - `conf/model/qwen3_vl_4b_instruct/infer/qwen3_vl_4b_instruct.default.yaml`
- Added captions calibration dataset config with a `size` selector and sample-count mapping:
  - `conf/dataset/vlm_coco2017_captions.yaml`
- Added a `quant_pair` group for one-pair-per-run sweeps and a legacy parity pair:
  - `conf/quant_pair/*.yaml` (including `wint8_aint8` for published INT8 LM-only folder compatibility)
- Added AutoQuant and output layout groups:
  - `conf/autoquant/gradient_default.yaml`
  - `conf/output_layout/{tmp,publish}.yaml`
