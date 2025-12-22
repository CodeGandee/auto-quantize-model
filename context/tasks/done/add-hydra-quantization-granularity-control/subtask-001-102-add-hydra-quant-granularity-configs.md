# Subtask 1.2: Add Hydra `quant_granularity` config group (initial overlays)

## Scope

Add a new Hydra config group `conf/quant_granularity/` that defines named quant_cfg overlays (granularity controls) to be applied on top of a base ModelOpt quantization format selected by `quant_pair.format_name`.

In scope:
- Define a stable Hydra schema:
  - `name: <string>`
  - `quant_cfg_overrides: { ... }` with keys such as `*weight_quantizer`, `*input_quantizer`, and optionally `default`.
- Create `conf/quant_granularity/default.yaml` (no-op overlay).
- Create a small initial set of overlays based on Subtask 1.1 decisions (including a “recipe match” option).
- Wire `quant_granularity` into `conf/preset/qwen3_lm_sensitivity.yaml` defaults so existing runs keep working (default overlay selected).

Out of scope:
- Implementing the overlay merge logic and runner integration (later subtasks).

## Planned outputs

- `conf/quant_granularity/default.yaml` (no overrides)
- Additional `conf/quant_granularity/*.yaml` overlays:
  - Recipe-match overlay (W per-channel, A per-token dynamic)
  - A couple of alternative weight granularities (per-column, per-group sizes)
  - Optional activation-only overlay(s)
- Updated `conf/preset/qwen3_lm_sensitivity.yaml` defaults to include `quant_granularity=default`.

## TODOs

- [x] Job-001-102-001 Create `conf/quant_granularity/default.yaml` with a consistent schema and `quant_cfg_overrides: {}`.
- [x] Job-001-102-002 Add `conf/quant_granularity/recipe_match_channel_token.yaml` (or final chosen name) encoding the Qwen3 target recipe granularity mapping from Subtask 1.1.
- [x] Job-001-102-003 Add 2–4 additional overlays for sweep coverage (based on Subtask 1.1 decisions), for example:
  - `w_per_out_channel`, `w_per_in_channel`
  - `w_group64`, `w_group128`
  - `a_per_token_dynamic` (if needed as a standalone overlay)
- [x] Job-001-102-004 Update `conf/preset/qwen3_lm_sensitivity.yaml` to include `quant_granularity: default` in defaults composition.
- [x] Job-001-102-005 Add a short README note (either in the main task file or in `models/qwen3_vl_4b_instruct/layer-analysis/README.md`) describing the purpose and schema of `conf/quant_granularity/` (full usage docs are in Subtask 1.6).

## Implemented overlays

Created in `conf/quant_granularity/`:

- `default.yaml` (no-op)
- `recipe_match_channel_token.yaml`
- `w_per_tensor.yaml`
- `w_per_out_channel.yaml`
- `w_per_in_channel.yaml`
- `w_group64.yaml`
- `w_group128.yaml`
- `a_per_token_dynamic.yaml`
- `a_per_tensor_static.yaml`

## Notes

- Keep overrides narrowly scoped to granularity/dynamic/static settings (do not override dtype/num_bits unless explicitly intended).
- Use YAML `null` for `None`, and avoid OmegaConf interpolation inside override dicts unless necessary.
