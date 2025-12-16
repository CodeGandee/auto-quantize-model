# Subtask 1.4: Implement Hydra runner for LM-only sensitivity

## Scope

Implement a Hydra-based CLI entry point for Qwen3‑VL **LM-only** per-layer sensitivity runs that:

- Composes the Hydra config tree from Subtask 1.2
- Uses shared implementation code from Subtask 1.3
- Supports both single runs and multirun sweeps
- Supports `report_only` regeneration from an existing manifest
- Writes artifacts to a configurable layout (Hydra run dir vs published `models/.../layer-analysis/`)

## Planned outputs

- Hydra runner script:
  - `scripts/qwen/qwen3_lm_sensitivity.py`
- Runner-level config schema assumptions documented in-code (and validated at runtime).
- A minimal set of user-facing commands (captured in docs in Subtask 1.6).

## TODOs

- [x] Job-001-104-001 Create the Hydra runner skeleton
  - Add `scripts/qwen/qwen3_lm_sensitivity.py` using `@hydra.main(config_path=..., config_name=...)`.
  - Ensure it runs via Pixi: `pixi run -e <env> python scripts/qwen/qwen3_lm_sensitivity.py ...`.

- [x] Job-001-104-002 Define runner config contract and validation
  - Define expected config fields (model path/dtype, dataset captions path, quant_pair selection, autoquant settings, output layout).
  - Validate required fields early and emit actionable error messages.

- [x] Job-001-104-003 Wire the runner to shared library code
  - Call the shared “run LM AutoQuant sensitivity” function (Subtask 1.3).
  - Convert Hydra config values into plain function arguments.

- [x] Job-001-104-004 Implement output directory logic
  - Decide how the runner chooses the final output directory:
    - Use Hydra run dir by default (for iteration), OR
    - Allow an explicit publish dir that bypasses Hydra run dir
  - Ensure multiruns produce non-colliding paths (include dataset size + quant_pair in output name).

- [x] Job-001-104-005 Add `report_only` mode
  - Mirror current Qwen3 script behavior:
    - Read `*_quant_manifest.json` from output dir
    - Regenerate `per-layer-sensitivity.md` and `per-layer-sensitivity.json`

- [x] Job-001-104-006 Make multirun sweeps ergonomic
  - Ensure `-m quant_pair=... dataset.size=...` works without modifying code.
  - Optionally provide a `hydra.sweeper.params` preset config for common sweeps.

## Notes

- Keep Hydra-specific behavior in the runner only; shared library functions should remain framework-agnostic.
