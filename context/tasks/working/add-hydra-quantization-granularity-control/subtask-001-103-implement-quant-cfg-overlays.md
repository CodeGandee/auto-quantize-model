# Subtask 1.3: Implement ModelOpt quant_cfg overlay + normalization helper

## Scope

Implement a small helper that takes:
- a resolved base ModelOpt quantization config dict (e.g., from `resolve_quant_config(format_name)`), and
- a Hydra-provided overlay dict (`quant_cfg_overrides`)

and returns a **deep-copied**, ModelOpt-compatible “effective” config dict with overrides applied safely.

In scope:
- Deep copy base config dict before mutation.
- Apply overrides to:
  - `cfg["quant_cfg"]["*weight_quantizer"]`
  - `cfg["quant_cfg"]["*input_quantizer"]`
  - optionally `cfg["quant_cfg"]["default"]` when necessary for “all layers” configs.
- Enforce ModelOpt constraints:
  - `axis` and `block_sizes` cannot coexist within the same quantizer attributes.
- Normalize `block_sizes` structures that come from YAML/OmegaConf:
  - Convert keys `"-1"` / `"-2"` to integers `-1` / `-2`.
  - Preserve special string keys like `"type"`, `"scale_bits"`, `"scale_block_sizes"`.
- Fail fast with clear errors for invalid overlay structures.

Out of scope:
- Wiring into Hydra runner or output naming (later subtasks).

## Planned outputs

- A dedicated helper (new module or added to an existing module) that:
  - Applies `quant_cfg_overrides` to a base config dict and returns a new dict.
  - Can be unit-tested without importing heavy model code.
- Minimal docstring + usage examples in the helper module.

## TODOs

- [ ] Job-001-103-001 Choose the code location for the helper:
  - new module (e.g., `src/auto_quantize_model/modelopt_quant_overrides.py`) vs extending `src/auto_quantize_model/modelopt_configs.py`.
- [ ] Job-001-103-002 Implement `apply_quant_cfg_overrides(base_cfg: dict, overrides: dict) -> dict` (exact signature TBD) using a deep copy and a narrow, explicit merge strategy.
- [ ] Job-001-103-003 Implement axis/block_sizes mutual exclusion logic:
  - If override includes `axis`, ensure `block_sizes` is removed from the resulting quantizer attributes.
  - If override includes `block_sizes`, ensure `axis` is removed.
- [ ] Job-001-103-004 Implement `block_sizes` key normalization:
  - Accept `block_sizes` mapping keys as ints or strings and normalize to expected types.
  - Ensure nested `scale_block_sizes` is handled similarly if present.
- [ ] Job-001-103-005 Decide and implement `default` propagation behavior:
  - If a config uses `quant_cfg["default"]`, decide whether overlays should also update it to match weight/input overrides.
- [ ] Job-001-103-006 Add clear validation errors for bad overlays (unknown top-level keys, incompatible values, unsupported structures).

## Notes

- Keep the helper conservative and explicit; this is a core reproducibility piece and should not “guess” too much.
- Favor deterministic behavior: given the same base config and override dict, output should be byte-for-byte stable after JSON serialization.

