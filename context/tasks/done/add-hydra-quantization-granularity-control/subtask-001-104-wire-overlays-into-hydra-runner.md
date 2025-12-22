# Subtask 1.4: Wire granularity overlays into the Qwen3 Hydra runner + manifest

## Scope

Update the Hydra runner (`scripts/qwen/qwen3_lm_sensitivity.py`) and related glue so that:
- Hydra composes a `quant_granularity` config.
- The runner resolves the base format config from `quant_pair.format_name`.
- The runner applies the overlay to produce an effective ModelOpt quantization config dict.
- AutoQuant runs using that effective config dict.
- The manifest JSON records the base format + overlay for reproducibility.

In scope:
- Code path changes required so `mtq.auto_quantize` receives the effective config dict(s) (not just names).
- Update scheme naming if needed to avoid collisions in filenames.
- Add manifest metadata: base format name, granularity name, and applied overrides (and optionally the final resolved `quant_cfg` if size is acceptable).

Out of scope:
- Output directory template changes and publish layout naming (handled in Subtask 1.5).

## Planned outputs

- `scripts/qwen/qwen3_lm_sensitivity.py` reads `cfg.quant_granularity` and applies overlays.
- `src/auto_quantize_model/...` changes (as needed) so quantization formats can be passed as explicit dicts, not only names.
- Manifest JSON includes a dedicated section (e.g., `manifest["quantization"]`) describing:
  - base `format_name`
  - `quant_granularity.name`
  - `quant_cfg_overrides`

## TODOs

- [x] Job-001-104-001 Decide how the runner passes “effective quantization formats” to the AutoQuant call:
  - extend `AutoQuantSchemeConfig` to carry overrides, or
  - apply overrides directly in the runner and pass dicts downstream, bypassing name resolution.
- [x] Job-001-104-002 Implement the runner-side overlay application using the helper from Subtask 1.3.
- [x] Job-001-104-003 Ensure the effective config dict is used by `mtq.auto_quantize` (not only base names).
- [x] Job-001-104-004 Record base + overlay metadata into the manifest JSON (and ensure report-only regeneration preserves it).
- [x] Job-001-104-005 Add a small “print the effective config summary” log line (key fields only) to make it easy to debug sweeps from console logs.

## Implementation notes

- Runner applies overlays in `scripts/qwen/qwen3_lm_sensitivity.py` and passes the resolved dict via `quantization_formats=[...]`.
- Downstream support added in `src/auto_quantize_model/qwen/autoquant_sensitivity.py` (optional `quantization_formats` override).
- Manifest writes a new top-level key `manifest["quantization"]` with base format + overlay metadata and a small effective-quantizer summary.
- Scheme naming was left unchanged; collisions are avoided by including `quant_granularity.name` in run/publish directory layouts (Subtask 1.5).

## Notes

- Keep manifests stable: downstream scripts may rely on manifest keys, so put new metadata under a new top-level key rather than reshaping existing ones.
- Ensure YAML → Python conversion is handled before passing to ModelOpt (OmegaConf structures can surprise you).
