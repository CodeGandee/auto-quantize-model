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

- [x] Job-001-101-001 Review `context/design/qwen3-modelopt-production-options.md` and `context/hints/about-modelopt-quantization-granularity.md` to list the ModelOpt knobs we must control (`axis`, `block_sizes`, `type`, and any constraints).
- [x] Job-001-101-002 Define the weights granularity vocabulary and mapping:
  - Per-tensor: `axis: null`, no `block_sizes`.
  - Per-output-channel: `axis: 0`, no `block_sizes`.
  - Per-input-channel (“per-column”): `axis: 1`, no `block_sizes`.
  - Per-group: `block_sizes: {-1: <group_size>}` (confirm axis key and typical group sizes).
- [x] Job-001-101-003 Define the activations granularity vocabulary and mapping:
  - Per-token dynamic: `type: dynamic` plus the ModelOpt per-token pattern used in our build (likely `block_sizes: {-1: null}`).
  - Per-tensor/static: `axis: null` and remove `block_sizes` (confirm whether `type` should be `static`/omitted).
  - Optional per-hidden-channel: decide if we support it now; if so, define a safe mapping and constraints.
- [x] Job-001-101-004 Define the “recipe match” overlay for Qwen3 target recipe:
  - Weights: int4 per-channel (Linear weight `(out, in)` → `axis=0`).
  - Activations: int8 per-token dynamic (`type=dynamic` + token pattern).
  - Document how this interacts with `quant_pair.format_name` (dtype selection remains in the base format).
- [x] Job-001-101-005 Decide the minimal initial overlay set to implement as YAML files (default + recipe match + a few alternates) and record the chosen set here so subtask 1.2 can create them.

## Notes

- Keep the vocabulary stable: output directories and manifests will use `quant_granularity.name`, so renames later are painful.
- Prefer overlay options that are shape/layout-agnostic and common across ModelOpt presets; avoid per-layer special-casing in the first iteration.

## Canonical vocabulary + ModelOpt mapping

### ModelOpt knobs we control

These are applied at the quantizer-attribute level inside a ModelOpt config dict:

- `axis` (static per-channel/per-axis scaling).
  - `axis: null` → per-tensor (single scale).
  - Linear weights are `(out_features, in_features)`:
    - `axis: 0` → per-output-channel (row-wise).
    - `axis: 1` → per-input-channel / “per-column”.
- `block_sizes` (block/group quantization and some per-token patterns).
  - Group-wise weights typically use `block_sizes: {-1: <group_size>}` (block on last dim).
  - Per-token activations are commonly expressed as `type: dynamic` with `block_sizes: {-1: null}`.
  - `block_sizes` may contain special string keys like `type`, `scale_bits`, `scale_block_sizes`.
- Mutual exclusion: `axis` and `block_sizes` cannot coexist on the same quantizer.

### Weights granularity vocabulary (`quant_granularity.name`)

| Option name | ModelOpt overrides (applied to `"*weight_quantizer"`) | Meaning |
|---|---|---|
| `w_per_tensor` | `{"axis": null}` | Per-tensor weights (single scale) |
| `w_per_out_channel` | `{"axis": 0}` | Per-output-channel weights (Linear rows) |
| `w_per_in_channel` | `{"axis": 1}` | Per-input-channel / per-column weights (Linear cols) |
| `w_group64` | `{"block_sizes": {-1: 64}}` | Group-wise weights, group size 64 on last dim |
| `w_group128` | `{"block_sizes": {-1: 128}}` | Group-wise weights, group size 128 on last dim |

### Activations granularity vocabulary (`quant_granularity.name`)

| Option name | ModelOpt overrides (applied to `"*input_quantizer"`) | Meaning |
|---|---|---|
| `a_per_token_dynamic` | `{"type": "dynamic", "block_sizes": {-1: null}}` | Dynamic per-token (ModelOpt standard pattern) |
| `a_per_tensor_static` | `{"type": "static", "axis": null}` | Static per-tensor activations |

Decision: **do not** add a per-hidden-channel activation overlay in the initial set. (For LM inputs, this would likely be `axis: -1`, but it is more shape/layout-dependent and can be added later.)

### “Recipe match” for Qwen3 target recipe

The llm-compressor target recipe from `context/design/qwen3-target-recipe.md` maps onto ModelOpt as:

- Weights: per-channel (Linear `(out, in)` → `axis: 0`)
- Activations: per-token dynamic (`type: dynamic`, `block_sizes: {-1: null}`)

This is implemented as the `quant_granularity.name=recipe_match_channel_token` overlay, which only changes granularity/dynamic behavior. Dtype/bitwidth selection remains owned by `quant_pair.format_name`.

## Initial overlay set to implement (Subtask 1.2)

Implement these in `conf/quant_granularity/`:

- `default` (no-op)
- `recipe_match_channel_token`
- `w_per_tensor`
- `w_per_out_channel`
- `w_per_in_channel`
- `w_group64`
- `w_group128`
- `a_per_token_dynamic`
- `a_per_tensor_static`
