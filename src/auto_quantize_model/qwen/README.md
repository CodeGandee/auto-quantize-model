# Qwen Utilities

## Tutorial Pack Runner

The shared Qwen3-VL tutorial-pack runner lives in:

- `src/auto_quantize_model/qwen/tutorial_pack_runner.py` (orchestration + snapshot/verify)
- `src/auto_quantize_model/qwen/cli_tutorial_pack_runner.py` (CLI frontend)
- `src/auto_quantize_model/qwen/tutorial_pack_registry.py` (model registry)

### How to add a new Qwen3-VL tutorial pack

1) Add a new model entry to the registry:

- Edit `src/auto_quantize_model/qwen/tutorial_pack_registry.py`
- Add a `ModelConfig` entry with:
  - `model_id` (stable identifier; used by the wrapper)
  - `workspace_slug` (workspace name prefix under `tmp/`)
  - `checkpoint_dir` and `bootstrap_script` (must exist locally; runner does not auto-create links)
  - `all_layers_script` and `lm_only_script` (entrypoints for scenario execution)
  - `lm_only_model_name` and `lm_only_model_variant` (Hydra overrides for LM-only mode)
  - `allowed_quant_pairs` (validated against `--quant-pairs`)

2) Create the tutorial folder under `docs/tutorial/howto/<your-pack>/` with:

- `run_demo.sh` delegating to the shared CLI:
  - `pixi run python -m auto_quantize_model.qwen.cli_tutorial_pack_runner --model-id <id> --expected-report-dir <pack>/expected_report "$@"`
- `expected_report/` directory containing summary-only snapshots:
  - `expected_report/<mode>/<quant_pair>/summary.json`
  - (no markdown is required for verification; markdown reports are optional and workspace-local)

3) Validate (manual, GPU):

- `bash docs/tutorial/howto/<your-pack>/run_demo.sh` (verify mode)
- `bash docs/tutorial/howto/<your-pack>/run_demo.sh --snapshot-report` (refresh snapshots)
