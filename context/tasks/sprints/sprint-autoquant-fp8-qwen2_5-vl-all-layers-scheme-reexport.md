# Sprint: AutoQuant FP8 Qwen2.5-VL all-layers scheme re-export (per-coverage checkpoints)

## Scope

This sprint tracks the work required to fix the known issue where all all-layers AutoQuant FP8 coverage schemes (`fp8_autoquant_all_layers_top10_coco2017`, `..._top20_...`, `..._top100_...`) share identical HF weights, despite having different coverage manifests. The goal is to:

- Re-run AutoQuant (ModelOpt) per coverage scheme using `disabled_layers` to produce **distinct, per-coverage FP8 checkpoints**.
- Make those checkpoints self-contained and compatible with the existing evaluation scripts.
- Preserve the current all-layers baseline and manifests as the canonical source of sensitivity information.

We will follow **Option A** (per-scheme AutoQuant runs) as described in `context/issues/known/bugfix-autoquant-fp8-qwen2_5-vl-schemes-identical.md`.

Related context:
- Plan: `context/plans/plan-modelopt-autoquant-fp8-qwen2_5-vl-mixed-schemes.md`
- Working task: `context/tasks/working/modelopt-autoquant-fp8-qwen2_5-vl-mixed-schemes/subtask-004-104-export-hf-checkpoints-per-scheme.md`
- Known bug: `context/issues/known/bugfix-autoquant-fp8-qwen2_5-vl-schemes-identical.md`
- ModelOpt layer-wise bits hint: `context/hints/howto-modelopt-layer-wise-quant-bits.md`

## Goals

- Ensure that each all-layers AutoQuant FP8 coverage scheme (`top10`, `top20`, …, `top100`) has:
  - Distinct quantized weights (different FP8/BF16 layouts per layer group).
  - A stable, self-contained HF checkpoint directory under `models/qwen2_5_vl_3b_instruct/quantized/`.
  - Per-scheme manifests linking back to the original all-layers sensitivity baseline.
- Keep the existing evaluation helpers under `models/qwen2_5_vl_3b_instruct/helpers/autoquant_eval/` usable without structural changes (only checkpoint contents change).
- Provide a repeatable script/driver that can regenerate per-coverage schemes if the baseline AutoQuant run is updated.

## Tasks

### 1. Baseline inspection and constraints

- [ ] Sprint-A01: Confirm that the all-layers baseline is the single source of sensitivity truth
  - Verify `fp8_autoquant_all_layers_fp8_coco2017` contents:
    - `layer-sensitivity/fp8_autoquant_all_layers_fp8_quant_manifest.json`
    - `fp8_autoquant_all_layers_fp8_autoquant_state.pt`
    - `per-layer-sensitivity.md`
  - Ensure `sensitivity_ranking` in the manifest matches expectations from Subtask 4.4 (least-sensitive layers first).
- [x] Sprint-A02: Document the mapping from coverage manifests to `disabled_layers`
  - For each `fp8_autoquant_all_layers_topXX_coco2017/layer-sensitivity/*_coverage_from_baseline.json`, validate that:
    - `selected_layers` + `dropped_layers` partitions the same set of quant_recipe keys as the baseline.
  - Decide and document the rule for `disabled_layers` when re-running AutoQuant:
    - Example: For a `topXX` scheme derived from the all-layers baseline, treat all `dropped_layers` as `disabled_layers` when running an FP8 AutoQuant scheme with `FP8_ALL_LAYERS_CFG`.
  - Current implementation: `scripts/qwen/qwen2_5_vl_3b_autoquant_fp8_all_layers_per_scheme.py` loads the baseline manifest and per-scheme coverage manifest, validates the partition (with warnings on mismatch), and passes `coverage_manifest["dropped_layers"]` directly as the `disabled_layers` argument to `mtq.auto_quantize`.

### 2. Design a per-scheme AutoQuant driver (Option A)

- [x] Sprint-A03: Design CLI and workflow for a new driver script
  - New script (proposal): `scripts/qwen/qwen2_5_vl_3b_autoquant_fp8_all_layers_per_scheme.py`
  - Inputs:
    - `--baseline-dir` (e.g., `models/qwen2_5_vl_3b_instruct/quantized/fp8_autoquant_all_layers_fp8_coco2017`)
    - `--coverage-manifest` for a specific scheme (e.g., `fp8_autoquant_all_layers_top10_coco2017_coverage_from_baseline.json`)
    - `--scheme-name` / `--out-dir` for the new HF checkpoint
    - Model / data args as in `qwen2_5_vl_3b_autoquant_fp8_schemes.py` (captions vs VLM calib).
  - Behavior:
    - Build `disabled_layers` from `dropped_layers` in the coverage manifest.
    - Call `mtq.auto_quantize` with:
      - `quantization_formats=[FP8_ALL_LAYERS_CFG]`
      - `constraints` derived from the baseline (e.g., baseline `effective_bits`).
      - `disabled_layers` for the current scheme.
    - Export a new HF checkpoint to `--out-dir` via `export_hf_checkpoint`.
    - Copy per-scheme manifests into `layer-sensitivity/` under `--out-dir`.
  - Implemented CLI matches this design and adds `--effective-bits`, `--auto-quantize-score-size`, and `--overwrite` for reproducible reruns.
- [x] Sprint-A04: Define calibration strategy for per-scheme runs
  - Decide whether to:
    - Reuse the existing VLM calibration DB (`coco2017_vlm_calib.db`) for all schemes, or
    - Use a lighter text-only calibration for scheme derivation, with VLM handled at evaluation time.
  - Make sure `num_calib_steps` and `num_score_steps` are tuned so the total runtime across all `topXX` schemes is tractable on RTX 5090.
  - Current decision: reuse the existing VLM calibration DB (`datasets/vlm-quantize-calib/coco2017_vlm_calib.db`) for all all-layers schemes via `build_vlm_calib_dataloader`, with `num_calib_steps = len(calib_batches)` and `num_score_steps` derived from a logical score-size-in-samples / batch-size rule and clamped to the available calibration batches.

### 3. Implement per-scheme AutoQuant and export

- [x] Sprint-A05: Implement the per-scheme AutoQuant driver (single scheme)
  - Implement `qwen2_5_vl_3b_autoquant_fp8_all_layers_per_scheme.py`:
    - Load base Qwen2.5-VL checkpoint and extract LM/VLM as appropriate.
    - Load the baseline manifest and coverage manifest.
    - Construct `disabled_layers` list from coverage manifest `dropped_layers` (e.g., using pattern keys like `language_model.layers.0.mlp.gate_proj.quant_recipe` and ModelOpt’s grouping rules).
    - Call `mtq.auto_quantize` with:
      - `quantization_formats=[FP8_ALL_LAYERS_CFG]` (or a small set if needed).
      - `constraints={"effective_bits": baseline_bits}`.
      - `disabled_layers` for the scheme.
      - `data_loader`, `forward_step`, `loss_func` as in the baseline driver.
    - Export a **scheme-specific HF checkpoint** (config + weights + tokenizer + processor) to `--out-dir`.
    - Copy or regenerate:
      - `*_quant_manifest.json` for the scheme.
      - AutoQuant state (`*_autoquant_state.pt`).
      - `per-layer-sensitivity.md` and coverage manifest into `layer-sensitivity/`.
  - Implemented: the new driver runs `mtq.auto_quantize` on the full Qwen2.5-VL model with `quantization_formats=[FP8_ALL_LAYERS_CFG]`, `disabled_layers` from the coverage manifest, and a VLM calibration stream; it exports a self-contained HF checkpoint plus scheme-specific `*_quant_manifest.json`, `*_autoquant_state.pt`, `per-layer-sensitivity.md`, and copies of the baseline and coverage manifests into `--out-dir/layer-sensitivity/`.
- [x] Sprint-A06: Add a wrapper script to sweep all all-layers coverage points
  - New script (or extend existing one):
    - `scripts/qwen/run_qwen2_5_vl_3b_autoquant_all_layers_schemes.sh`
  - Responsibilities:
    - Enumerate all existing coverage manifests under `fp8_autoquant_all_layers_topXX_coco2017/layer-sensitivity/`.
    - For each `topXX`, call the per-scheme AutoQuant driver with appropriate `--coverage-manifest` and `--out-dir` (e.g., `models/qwen2_5_vl_3b_instruct/quantized/fp8_autoquant_all_layers_topXX_coco2017_new`).
    - Optionally support `--dry-run` or `--only top10,top50,top100` for partial regeneration.
  - Implemented: the wrapper discovers `*_coverage_from_baseline.json` files, infers the scheme directory name, and invokes the per-scheme driver via `pixi run -e rtx5090-vllm python ...`; output directories default to `<scheme_name>_v2` so original sliced checkpoints remain available, and it supports `--dry-run` and `--only` filters.

### 4. Validation and migration

- [ ] Sprint-A07: Sanity-check per-scheme checkpoints
  - For a small subset of schemes (e.g., `top10`, `top50`, `top100`):
    - Load each new checkpoint via `Qwen2_5_VLForConditionalGeneration.from_pretrained`.
    - Run:
      - Short text-only generation (reuse `run_qwen2_5_vl_3b_sanity.py`).
      - A few VLM prompts (reuse existing VLM sanity script).
    - Confirm:
      - No shape/dtype errors.
      - Model forward passes are stable and reasonably fast on RTX 5090.
- [ ] Sprint-A08: Check weight diversity between schemes
  - Reuse or adapt the debug helper in `tmp/autoquant-schemes-debug/src/check_scheme_weight_diffs.py` to:
    - Compare representative layers across `top10`, `top50`, `top100` new schemes.
    - Assert `max|top10 - top50| > 0` (and similarly for other pairs), confirming that weights truly differ between coverage points.
  - Capture a short Markdown or JSON summary under `tmp/modelopt-autoquant-fp8/` for regression tracking.
- [ ] Sprint-A09: Re-run evaluation scripts against new checkpoints
  - Point:
    - Text-only multi-scheme comparison: `compare_qwen2_5_vl_3b_schemes_vs_fp16.py`
    - VLM eval per scheme: `compare_qwen2_5_vl_3b_vlm_eval_top10_vs_fp16.py` (or a generalized variant)
  - Confirm:
    - Perplexity and logit MSE/KL differ meaningfully across `topXX`.
    - Trends match expectations (more aggressive coverage → higher degradation).

### 5. Cleanup and documentation

- [ ] Sprint-A10: Decide on directory naming and migration strategy
  - Decide whether to:
    - Overwrite existing `fp8_autoquant_all_layers_topXX_coco2017` dirs, or
    - Write new ones (e.g., `fp8_autoquant_all_layers_topXX_coco2017_v2`) and keep the old ones for historical reference.
  - Update any scripts that assume the old layout to point to the new checkpoint dirs.
- [ ] Sprint-A11: Update documentation and known-issue status
  - Update:
    - `context/issues/known/bugfix-autoquant-fp8-qwen2_5-vl-schemes-identical.md` with:
      - The chosen fix path (Option A).
      - Pointers to the new driver and wrapper scripts.
      - Where to find regenerated checkpoints and metrics.
    - `context/tasks/working/modelopt-autoquant-fp8-qwen2_5-vl-mixed-schemes/subtask-004-104-export-hf-checkpoints-per-scheme.md` to mention:
      - The existence of per-scheme AutoQuant re-runs for all-layers coverage.
  - Optionally add a short report under `models/qwen2_5_vl_3b_instruct/reports/` summarizing:
    - Per-scheme effective bits vs coverage.
    - High-level accuracy trends across new all-layers schemes.

## Notes

- This sprint assumes the all-layers AutoQuant baseline (`fp8_autoquant_all_layers_fp8`) is already computed and stable.
- AutoQuant re-runs per coverage scheme can be expensive; plan for long-running jobs and consider:
  - Reducing `num_score_steps` for non-critical schemes.
  - Prioritizing a subset of coverage points (e.g., 10%, 50%, 100%) for early validation.
- All Python-based drivers should be run via the RTX 5090 ModelOpt Pixi environment (e.g., `pixi run -e rtx5090-vllm python ...`) to ensure the correct ModelOpt/Transformers stack is used.
