# Layer sensitivity analysis (Hydra)

This repo uses **Hydra** to configure and run **per-layer sensitivity** analysis with NVIDIA ModelOpt AutoQuant. The key idea is:

- Each run selects a **model**, **calibration dataset**, and a **weight/activation precision pair**.
- AutoQuant produces a **quantization manifest** plus **per-layer sensitivity** reports (Markdown + JSON).
- Hydra makes it easy to sweep configurations (`-m`) without editing Python code.

## What you get per run

Each run writes:

- `*_quant_manifest.json`: raw AutoQuant outputs + per-layer sensitivity data.
- `per-layer-sensitivity.md`: human-readable report.
- `per-layer-sensitivity.json`: machine-readable report.
- `*_autoquant_state.pt`: raw AutoQuant state (typically ignored by Git under `models/...`).

Notes:

- The manifest JSON includes `scheme`, `model`, and `dataset` metadata. The Markdown report also includes a `## Dataset` section (paths, size, and calibration sample counts).

## Common Hydra patterns

Single run:

```bash
pixi run -e <env> python scripts/qwen/qwen3_lm_sensitivity.py \
  output_layout=tmp \
  quant_pair=wfp8_afp8 \
  dataset.size=small
```

Sweep multiple values (Hydra multirun):

```bash
pixi run -e <env> python scripts/qwen/qwen3_lm_sensitivity.py -m \
  output_layout=tmp \
  quant_pair=wfp4_afp8,wfp8_afp8,wint8_afp16 \
  dataset.size=small,medium,large
```

Override a single field:

```bash
pixi run -e <env> python scripts/qwen/qwen3_lm_sensitivity.py \
  autoquant.effective_bits=7.0 \
  autoquant.score_size=256
```

## Output layouts

The runner supports two output modes:

- `output_layout=tmp`: write into the Hydra run directory under `tmp/model-experiments/...` (good for iteration).
- `output_layout=publish`: write into a stable, publishable folder under `models/<model>/layer-analysis/...`.

See the per-run resolution logic and override fields in the config reference page.

If you want a fully deterministic output directory (e.g. matching a repo-specific publish layout), prefer `runner.output_dir=/abs/or/rel/path` over `output_layout=publish`.

## Regenerate reports without rerunning AutoQuant

If you already have a `*_quant_manifest.json`, you can regenerate the reports:

```bash
pixi run -e <env> python scripts/qwen/qwen3_lm_sensitivity.py \
  runner.report_only=true \
  output_layout=publish \
  quant_pair=wint8_aint8 \
  dataset.size=small
```
