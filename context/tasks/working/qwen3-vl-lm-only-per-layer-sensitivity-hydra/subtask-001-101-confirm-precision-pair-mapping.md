# Subtask 1.1: Confirm precision-pair mapping to ModelOpt configs

## Scope

Define and validate a concrete mapping from the requested precision pairs:

- **Weights**: FP4, FP8, INT8
- **Activations**: FP8, FP16

to ModelOpt quantization format config(s) usable by `modelopt.torch.quantization.auto_quantize`, specifically for **Qwen3‑VL LM-only** sensitivity runs.

This subtask is about *deciding and encoding the mapping* (and feasibility), not about building the full Hydra runner or publishing large artifact grids.

## Planned outputs

- A written mapping table (in this subtask doc and/or a referenced module) covering each target (W,A) pair.
- A decision on what “FP4” means in this repo for this workflow (NVFP4 vs MXFP4).
- A decision on what “activation FP16” means operationally (typically: **disable input quantizers**; keep compute dtype as fp16/bf16).
- Any missing quantization configs added to `src/auto_quantize_model/modelopt_configs.py` (with names suitable for Hydra references).
- A small “capability check” function or runtime validation strategy (fail early if a requested pair/config is not supported).

## Decisions and mapping

### Canonical meaning of FP4

For this workflow, **FP4 weights** default to **MXFP4** (dynamic per-block FP4) because the TRT-LLM-aligned ModelOpt build in our Pixi env exposes `W4A8_MXFP4_FP8_CFG`, while `W4A8_NVFP4_FP8_CFG` is not consistently available.

We still support NVFP4 where possible (e.g., weight-only) via optional custom configs, but the default FP4+FP8 activation pair uses MXFP4.

### Meaning of “activation FP16”

“Activation FP16” means **no activation quantization** (disable `*input_quantizer`). The model execution dtype may still be `bf16` for practicality on recent GPUs/checkpoints; the key point is that activations are not quantized.

### One pair per run (default sensitivity semantics)

We run **one (W,A) pair per AutoQuant run** (one quantization format in `quantization_formats`). AutoQuant will still include its implicit “NONE(…)” option, which matches the current report writers’ assumption that there is exactly one non-NONE candidate format per layer.

### Mapping table (W,A → ModelOpt config name)

| Weights | Activations | ModelOpt config name | Source | Notes |
|---|---|---|---|---|
| FP4 | FP8 | `W4A8_MXFP4_FP8_CFG` | built-in (`mtq.*`) | FP4 weights (MXFP4) + FP8 input quantizer |
| FP4 | FP16 | `MXFP4_WEIGHT_ONLY_CFG` | custom (`CUSTOM_QUANT_CONFIGS`) | disable `*input_quantizer` |
| FP8 | FP8 | `FP8_DEFAULT_CFG` | built-in (`mtq.*`) | standard FP8 W8A8-style FP8 quantization |
| FP8 | FP16 | `FP8_WEIGHT_ONLY_CFG` | custom (`CUSTOM_QUANT_CONFIGS`) | disable `*input_quantizer` |
| INT8 | FP16 | `INT8_WEIGHT_ONLY_CFG` | custom (`CUSTOM_QUANT_CONFIGS`) | built-in name not present in all ModelOpt builds |
| INT8 | FP8 | `INT8_WEIGHT_FP8_ACT_CFG` | custom (`CUSTOM_QUANT_CONFIGS`) | **experimental** hybrid (may not be supported by all backends) |

### Capability checks

- Use `auto_quantize_model.modelopt_configs.resolve_quant_config(format_name)` to fail fast when a format name is unknown or resolves to a non-dict config.
- FP4-related custom configs are registered only when the underlying ModelOpt build exposes the required base presets (e.g., `MXFP4_DEFAULT_CFG` / `NVFP4_DEFAULT_CFG`).

## TODOs

- [x] Job-001-101-001 Inventory existing ModelOpt formats relevant to this task
  - Read `extern/TensorRT-Model-Optimizer/modelopt/torch/quantization/config.py` to enumerate built-in configs for:
    - FP8 (`FP8_DEFAULT_CFG`, `FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG`, …)
    - INT8 (`INT8_DEFAULT_CFG`, `INT8_WEIGHT_ONLY_CFG`, …)
    - FP4 (`NVFP4_*`, `MXFP4_*`, and `W4A8_*` hybrids)
  - Cross-check what the repo already exposes via `src/auto_quantize_model/modelopt_configs.py` (`CUSTOM_QUANT_CONFIGS`).

- [x] Job-001-101-002 Decide the canonical “FP4” interpretation
  - Choose **NVFP4** (default) vs **MXFP4** for “FP4 weights” in this task.
  - Document rationale and how to switch (if we support both).

- [x] Job-001-101-003 Define the mapping for each (W,A) pair
  - Produce an explicit mapping table for:
    - `W=FP4, A=FP8` (likely `W4A8_NVFP4_FP8_CFG` or `W4A8_MXFP4_FP8_CFG`)
    - `W=FP4, A=FP16` (requires an FP4 **weight-only** config; add custom if missing)
    - `W=FP8, A=FP8` (`FP8_DEFAULT_CFG` or repo’s “all layers” variant if needed)
    - `W=FP8, A=FP16` (FP8 **weight-only** config; built-in or custom)
    - `W=INT8, A=FP16` (`INT8_WEIGHT_ONLY_CFG`)
    - `W=INT8, A=FP8` (confirm whether a mixed INT8-weight + FP8-activation config is supported; if not, mark unsupported/experimental)

- [x] Job-001-101-004 Decide “one pair per run” vs “multi-format candidate set”
  - Clarify intended semantics for sensitivity:
    - Option A: run sensitivity **separately** per (W,A) pair (simpler reporting/layout).
    - Option B: run AutoQuant with a **candidate set** including multiple (W,A) formats at once (richer per-layer comparison).
  - Choose one as the default for Hydra UX; document how the other could be supported later.

- [x] Job-001-101-005 Implement missing configs in `src/auto_quantize_model/modelopt_configs.py`
  - Add new config dicts (deep-copied and minimally edited) for any missing pairs (e.g., FP4 weight-only, FP8 weight-only).
  - Register them in `CUSTOM_QUANT_CONFIGS` under stable names suitable for Hydra (e.g., `NVFP4_WEIGHT_ONLY_CFG`).

- [x] Job-001-101-006 Add early validation / capability checks
  - Define how the runner will fail fast when the requested mapping is unsupported:
    - Validate that referenced config names exist (built-in `mtq.*` or `CUSTOM_QUANT_CONFIGS`).
    - Optionally validate the config structure (e.g., presence/absence of `*input_quantizer` for “activation fp16”).

- [x] Job-001-101-007 Smoke-test mapping without GPUs (structure-level)
  - Add a small, non-GPU check (script or unit test) that:
    - Imports `modelopt.torch.quantization as mtq`.
    - Loads each mapped config dict.
    - Optionally runs a tiny `mtq.quantize`/`mtq.auto_quantize` “dry” on a toy `nn.Linear` if feasible without CUDA (only to validate config shape is accepted).

## Notes

- “Activation FP16” likely means **no activation quantization** (input quantizers disabled) while the model compute dtype is `fp16` or `bf16`. Decide whether to treat `bf16` as acceptable for this category (common for Qwen checkpoints).
- If `W=INT8, A=FP8` is unsupported, consider whether we:
  - Drop it entirely, or
  - Keep it as an “experimental custom config” behind an explicit Hydra flag.
