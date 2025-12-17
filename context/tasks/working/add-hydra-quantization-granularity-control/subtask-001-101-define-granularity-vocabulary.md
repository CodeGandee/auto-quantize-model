# Subtask 1.1: Define granularity vocabulary and ModelOpt mapping

## Scope

Define the canonical user-facing vocabulary for quantization granularity controls in this repo (weights + activations) and map each option to the specific ModelOpt quantizer attributes (`axis`, `block_sizes`, `type`) we will use in Hydra overlays. This subtask is primarily about **nailing semantics and naming** so later implementation is consistent and sweepable.

In scope:
- Decide the canonical names we will expose in Hydra (`quant_granularity.name`) for:
  - **Weights granularity** (e.g., per-output-channel, per-input-channel/per-column, per-group/blockwise, per-tensor).
  - **Activations granularity** (e.g., per-token dynamic, per-tensor, optional per-hidden-channel).
- Define how each option maps to ModelOpt:
  - For weights: `axis` vs `block_sizes` rules and recommended defaults (group sizes like 64/128).
  - For activations: dynamic per-token conventions (`type="dynamic"` + token pattern) and any safe static alternatives.
- Define the “recipe match” mapping for Qwen3 target recipe (`context/design/qwen3-target-recipe.md`).

Out of scope:
- Implementing the Hydra config files or code overlay (those are later subtasks).

## Planned outputs

- A documented mapping table (inside this subtask file) from our vocabulary → ModelOpt quant_cfg overrides.
- A final list of “initial supported” overlay options we will implement in `conf/quant_granularity/`.
- A clear decision on how we represent:
  - “per-column” for Linear weights (`axis=1` for `(out, in)`).
  - “per-channel” for Linear weights (`axis=0`).
  - “per-token dynamic activations” (ModelOpt-friendly representation).

## TODOs

- [ ] Job-001-101-001 Review `context/design/qwen3-modelopt-production-options.md` and `context/hints/about-modelopt-quantization-granularity.md` to list the ModelOpt knobs we must control (`axis`, `block_sizes`, `type`, and any constraints).
- [ ] Job-001-101-002 Define the weights granularity vocabulary and mapping:
  - Per-tensor: `axis: null`, no `block_sizes`.
  - Per-output-channel: `axis: 0`, no `block_sizes`.
  - Per-input-channel (“per-column”): `axis: 1`, no `block_sizes`.
  - Per-group: `block_sizes: {-1: <group_size>}` (confirm axis key and typical group sizes).
- [ ] Job-001-101-003 Define the activations granularity vocabulary and mapping:
  - Per-token dynamic: `type: dynamic` plus the ModelOpt per-token pattern used in our build (likely `block_sizes: {-1: null}`).
  - Per-tensor/static: `axis: null` and remove `block_sizes` (confirm whether `type` should be `static`/omitted).
  - Optional per-hidden-channel: decide if we support it now; if so, define a safe mapping and constraints.
- [ ] Job-001-101-004 Define the “recipe match” overlay for Qwen3 target recipe:
  - Weights: int4 per-channel (Linear weight `(out, in)` → `axis=0`).
  - Activations: int8 per-token dynamic (`type=dynamic` + token pattern).
  - Document how this interacts with `quant_pair.format_name` (dtype selection remains in the base format).
- [ ] Job-001-101-005 Decide the minimal initial overlay set to implement as YAML files (default + recipe match + a few alternates) and record the chosen set here so subtask 1.2 can create them.

## Notes

- Keep the vocabulary stable: output directories and manifests will use `quant_granularity.name`, so renames later are painful.
- Prefer overlay options that are shape/layout-agnostic and common across ModelOpt presets; avoid per-layer special-casing in the first iteration.

