# Qwen3 LM-only sensitivity config reference

The Qwen3-VL LM-only sensitivity runner is configured via:

- Entry-point defaults: `conf/preset/qwen3_lm_sensitivity.yaml`
- Groups under `conf/`:
  - `conf/model/`
  - `conf/dataset/`
  - `conf/quant_pair/`
  - `conf/autoquant/`
  - `conf/output_layout/`

The runner code is `scripts/qwen/qwen3_lm_sensitivity.py`.

## Top-level runner controls

In `conf/preset/qwen3_lm_sensitivity.yaml`:

- `runner.report_only`: when `true`, do not run AutoQuant; regenerate reports from an existing manifest.
- `runner.output_dir`: optional explicit output directory override (bypasses `output_layout` resolution).
- `output_layout`: selects where outputs are written (`tmp` vs `publish`).

## Model

Selected via the `model/...@model` default.

Common fields:

- `model.name`: used in output naming.
- `model.path`: HF checkpoint directory.
- `model.dtype`: used by the loader (LM compute dtype, not activation quantization).

Override example:

```bash
pixi run -e <env> python scripts/qwen/qwen3_lm_sensitivity.py \
  model.path=/abs/path/to/Qwen3-VL-4B-Instruct
```

## Dataset

Selected via `dataset=vlm_coco2017_captions` by default.

Common fields:

- `dataset.size`: `small|medium|large`
- `dataset.captions_path`: resolved from `dataset.root` and `dataset.size` by default
- `dataset.max_calib_samples`: optional override; if null, the runner uses `dataset.size_to_max_samples[dataset.size]`
- `dataset.calib_seq_len`: tokenization max length

Override example:

```bash
pixi run -e <env> python scripts/qwen/qwen3_lm_sensitivity.py \
  dataset.size=medium \
  dataset.max_calib_samples=64 \
  dataset.calib_seq_len=256
```

## Quant pair

Select a single weight/activation precision pair per run via:

```bash
quant_pair=<name>
```

Common fields in each `conf/quant_pair/*.yaml`:

- `quant_pair.name`: identifier used in output naming and defaults.
- `quant_pair.weight`: `fp4|fp8|int8`
- `quant_pair.activation`: `fp8|fp16|int8` (the `int8` activation is for legacy parity only).
- `quant_pair.format_name`: ModelOpt format name (either built-in `mtq.*` or a key in `CUSTOM_QUANT_CONFIGS`).
- `quant_pair.experimental`: when true, the runner prints a warning.

Optional compatibility overrides (used for matching published folder layouts):

- `quant_pair.scheme_name`: overrides the AutoQuant scheme name (affects manifest/state filenames).
- `quant_pair.publish_pair_dir`: overrides the `weight-<w>-act-<a>` folder name in publish mode.
- `quant_pair.publish_run_dir`: overrides the run directory name in publish mode.
- `quant_pair.coverage_mode` / `quant_pair.coverage_fraction`: stored in the scheme metadata (does not affect execution).

## AutoQuant

Selected via `autoquant=gradient_default` by default.

Common fields:

- `autoquant.method`: `gradient` or `kl_div`
- `autoquant.device`: typically `cuda`
- `autoquant.batch_size`: calibration batch size
- `autoquant.effective_bits`: AutoQuant constraint (passed as `constraints={"effective_bits": ...}`)
- `autoquant.score_size`: score-size in samples (converted to score-steps in code)

Override example:

```bash
pixi run -e <env> python scripts/qwen/qwen3_lm_sensitivity.py \
  autoquant.effective_bits=7.0 \
  autoquant.score_size=256 \
  autoquant.batch_size=4
```

## Output layout resolution

The runner resolves the final output directory as:

1. If `runner.output_dir` is set: use that.
2. Else if `output_layout.mode == tmp`: use the Hydra run directory (the process `chdir`s there).
3. Else if `output_layout.mode == publish`: write under:
   - `${output_layout.root_dir}/weight-<weight>-act-<activation>/<run_dir>`
   - with optional overrides from `quant_pair.publish_pair_dir` / `quant_pair.publish_run_dir`.

## Adding a new quant pair

1. Ensure the ModelOpt format is resolvable:
   - Add a custom config to `src/auto_quantize_model/modelopt_configs.py`, or
   - Reference a built-in preset exposed by `modelopt.torch.quantization` (e.g. `FP8_DEFAULT_CFG`).
2. Add a new `conf/quant_pair/<new_name>.yaml` pointing at that `format_name`.
3. Run:

```bash
pixi run -e <env> python scripts/qwen/qwen3_lm_sensitivity.py \
  quant_pair=<new_name> \
  output_layout=tmp
```
