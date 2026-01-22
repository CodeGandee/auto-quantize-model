---
description: "Refactor plan: consolidate Qwen3-VL tutorial-pack runners into a shared Python command utility"
---

# Refactor Plan: Qwen3-VL Tutorial Pack Runner → `src/auto_quantize_model` Command Utility

## What to Refactor

The tutorial packs under `docs/tutorial/howto/` are nearly identical and currently duplicate:

- Orchestration logic (asset checks, dataset preset resolution, workspace layout, scenario loop, snapshot/verify).
  - `docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/run_demo.sh`
  - `docs/tutorial/howto/tut-qwen3-vl-4b-layer-sensitivity/run_demo.sh`
- Python “glue” scripts (summarization + artifact sanitization), duplicated per pack.
  - `docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/scripts/summarize_manifest.py`
  - `docs/tutorial/howto/tut-qwen3-vl-4b-layer-sensitivity/scripts/summarize_manifest.py`
  - `docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/scripts/sanitize_artifacts.py`
  - `docs/tutorial/howto/tut-qwen3-vl-4b-layer-sensitivity/scripts/sanitize_artifacts.py`

Refactor target: introduce a shared, tested Python command utility under `src/auto_quantize_model/qwen/` that:

- Defines data models for “Qwen3 series sensitivity scenarios” (model spec, dataset preset, scenario spec, paths).
- Runs all selected scenarios (`all_layers` + `lm_only`) with consistent output layout.
- Produces schema-locked summaries via the existing `src/auto_quantize_model/qwen/tutorial_pack_summary.py`.
- Performs snapshot/verify against `expected_report/` (diffs summaries only, enforces non-degeneracy).

## Why Refactor

- **Avoid drift**: today, the two packs have the same contract but differ only by a few model-specific lines. Any future fixes (CLI flags, validation, summary schema rules) must be applied twice.
- **Single “source of truth”**: the orchestration logic is the tutorial pack’s public interface. Keeping it duplicated makes regressions likely.
- **Testability**: bash logic (snapshot cleanup, scenario enumeration, diffing) is hard to unit test. Moving it to Python allows deterministic tests (no GPU required).
- **Scalability**: adding a new Qwen3-VL model (e.g. another size/variant) currently implies copying a whole tutorial pack. A shared runner makes it “configure, don’t copy”.

## How to Refactor (Step-by-Step)

### Step 0: Inventory and freeze current external contract

1) Treat the current `run_demo.sh` contract as stable:
   - `--snapshot-report`, `--device`, `--dataset-size`, `--modes`, `--quant-pairs`
2) Keep the stable summary schema contract:
   - `specs/002-revise-qwen3-vl-tutorial/contracts/summary.schema.json`

### Step 1: Add Qwen3 tutorial-runner data models in `src/auto_quantize_model/qwen/`

Create a small set of explicit data models (prefer `@dataclass(frozen=True)` for “data-only”):

- `Qwen3ModelSpec`
  - `model_id` (e.g. `qwen3_vl_4b_instruct`)
  - `checkpoint_link` (repo path)
  - `hydra_name`, `hydra_variant` (for LM-only runner compatibility)
- `DatasetPreset`
  - `size`, `max_calib_samples`, `num_calib_batches`, `batch_size`, `calib_seq_len`, `score_size`
  - resolved asset paths: COCO root, VLM DB, captions file
- `ScenarioSpec`
  - `mode` (`all_layers` | `lm_only`)
  - `quant_pair` (`wint4_afp16`, `wint4_aint8`, …)
  - `dataset_preset`, `device`
  - derived `scenario_id` (e.g. `all_layers/wint4_afp16`)

### Step 2: Implement a shared runner + snapshot/verify logic in Python

Add a new module, e.g.:

- `src/auto_quantize_model/qwen/tutorial_pack_runner.py`

Responsibilities:

1) **Asset gating**
   - Validate checkpoint link exists (optionally create symlink from `$HF_SNAPSHOTS_ROOT` like current bash).
   - Validate dataset assets exist for the chosen preset.
2) **Workspace layout**
   - Create `tmp/tutorial_workspace_<pack>_<timestamp>/{outputs,summaries}/<mode>/<quant_pair>/`.
3) **Scenario execution**
   - `all_layers`: invoke existing runner
     - `models/qwen3_vl_4b_instruct/helpers/qwen3_vl_4b_autoquant_all_layers/run_qwen3_vl_4b_autoquant_all_layers.py`
   - `lm_only`: invoke existing runner (Hydra script) or call library function directly
     - `scripts/qwen/qwen3_lm_sensitivity.py` (short-term)
     - `auto_quantize_model.qwen.autoquant_sensitivity.run_qwen3_vl_lm_autoquant_sensitivity` (long-term)
4) **Summarization**
   - Replace pack-local wrappers by calling:
     - `TutorialPackScenarioSummary.from_manifest(...)`
     - `write_summary_json(...)`, `write_summary_md(...)`
5) **Snapshot/verify**
   - Snapshot: refresh `expected_report/<mode>/<quant_pair>/` and delete stale modes/pairs not selected.
   - Verify: `diff` (or Python file compare) of `summary.json` + `summary.md` only.
   - Enforce non-degeneracy: fail if `has_nonzero_sensitivity=false`.

### Step 3: Provide a stable CLI entrypoint for the runner

Add a small CLI module, e.g.:

- `src/auto_quantize_model/qwen/cli_tutorial_pack_runner.py`

CLI should:

- Preserve the existing flags used by both tutorial packs.
- Add required “injection points” to make it reusable:
  - `--model-id` (or `--model`), so one CLI supports all Qwen3 series models.
  - `--expected-report-dir` and `--pack-name` (or derive from tutorial dir).

### Step 4: Thin the tutorial packs to wrappers (no duplicated logic)

Keep each tutorial pack’s `run_demo.sh`, but turn it into a minimal wrapper:

- Determine `REPO_ROOT` and `SCRIPT_DIR`
- Call the shared Python command with the model id + expected_report dir
- Forward user flags unchanged

Optionally remove the duplicated pack-local Python scripts by:

- deleting `docs/.../scripts/summarize_manifest.py` and `sanitize_artifacts.py`, or
- keeping tiny wrappers that only import and call shared functions (for backward compatibility).

### Step 5: Add unit tests for runner logic (CPU-only, deterministic)

Add tests that do not require GPU:

- Dataset preset resolution (paths + numeric budgets).
- Scenario enumeration from `--modes` + `--quant-pairs`.
- Snapshot cleanup rules (stale mode/pair deletion).
- Verify diff behavior limited to `summary.*`.
- Non-degeneracy gate logic (synthetic summary JSON fixtures).

### Step 6: Manual validation + migration workflow

1) For each tutorial pack:
   - `--snapshot-report` (regenerate expected snapshots)
   - default verify run (no diffs)
2) Confirm both packs still:
   - produce the same `expected_report/<mode>/<quant_pair>` layout,
   - keep summary schema stable,
   - fail loudly on missing assets / degeneracy.

## Impact Analysis

### Functional impact

- **Expected behavior should not change**: the user-facing tutorial command (`run_demo.sh`) and its CLI flags remain stable.
- **Internals change**: orchestration moves from duplicated bash + pack-local scripts to a shared Python runner.

### Risks

- **Behavior drift between bash and Python**: subtle differences in quoting, path resolution, or diff semantics.
  - Mitigation: keep the same CLI contract, add unit tests for the runner’s pure logic, and perform snapshot/verify manual runs for both packs.
- **Hydra integration pitfalls** (if still shelling out):
  - Mitigation: short-term keep the existing invocation; long-term extract a non-Hydra callable API and let Hydra be “one of many frontends”.
- **Snapshot churn**:
  - Mitigation: continue diffing only schema-locked summaries; keep sanitizer rules stable.

## Expected Outcome

- One tested Python utility under `src/auto_quantize_model/qwen/` becomes the single place to maintain:
  - dataset preset resolution,
  - scenario identity + output layout,
  - snapshot/verify rules,
  - summary generation + non-degeneracy checks.
- Tutorial packs become thin documentation wrappers (model-specific defaults only).
- Adding a new Qwen3-VL model tutorial becomes:
  - “add model spec + README”, not “copy/paste a whole runner”.

## TODO Checklist

- [ ] Create `context/tasks/refactor` directory (if missing) and track this plan file.
- [ ] Add Qwen3 tutorial-pack data models under `src/auto_quantize_model/qwen/` (model spec, dataset preset, scenario spec).
- [ ] Implement `TutorialPackRunner` in `src/auto_quantize_model/qwen/tutorial_pack_runner.py`.
- [ ] Implement CLI entrypoint `src/auto_quantize_model/qwen/cli_tutorial_pack_runner.py`.
- [ ] Update `docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/run_demo.sh` to call the shared CLI.
- [ ] Update `docs/tutorial/howto/tut-qwen3-vl-4b-layer-sensitivity/run_demo.sh` to call the shared CLI.
- [ ] Replace pack-local summarizer/sanitizer with shared implementations (delete or keep tiny wrappers).
- [ ] Add unit tests for runner logic (scenario enumeration, snapshot cleanup, verify diff scope, non-degeneracy).
- [ ] Regenerate `expected_report/` snapshots for both packs using `--snapshot-report`.
- [ ] Run verify-mode for both packs and record any migration notes in the READMEs.

## Example Refactor Snippets (Before → After)

### Before: duplicated bash runner (model-specific lines only)

```bash
# docs/.../tut-qwen3-vl-8b-.../run_demo.sh
MODEL_DIR="$REPO_ROOT/models/qwen3_vl_8b_instruct/checkpoints/Qwen3-VL-8B-Instruct"
pixi run python scripts/qwen/qwen3_lm_sensitivity.py model.name=qwen3_vl_8b_instruct model.variant=3-vl-8b-instruct ...
WORKSPACE_DIR="$REPO_ROOT/tmp/tutorial_workspace_qwen3_vl_8b_layer_sensitivity_$(date +%Y%m%d_%H%M%S)"
```

```bash
# docs/.../tut-qwen3-vl-4b-.../run_demo.sh
MODEL_DIR="$REPO_ROOT/models/qwen3_vl_4b_instruct/checkpoints/Qwen3-VL-4B-Instruct"
pixi run python scripts/qwen/qwen3_lm_sensitivity.py model.name=qwen3_vl_4b_instruct model.variant=3-vl-4b-instruct ...
WORKSPACE_DIR="$REPO_ROOT/tmp/tutorial_workspace_qwen3_vl_4b_layer_sensitivity_$(date +%Y%m%d_%H%M%S)"
```

### After: tutorial pack wrapper delegates to shared Python CLI

```bash
# docs/.../run_demo.sh (thin wrapper)
exec pixi run python -m auto_quantize_model.qwen.cli_tutorial_pack_runner \
  --model-id qwen3_vl_8b_instruct \
  --expected-report-dir "$SCRIPT_DIR/expected_report" \
  "$@"
```

And the shared runner owns the contract:

```python
# src/auto_quantize_model/qwen/tutorial_pack_runner.py
@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    mode: Mode
    quant_pair: str
    dataset_size: DatasetSize
    device: str

class TutorialPackRunner:
    def __init__(self) -> None:
        self.m_repo_root: Path | None = None
        self.m_expected_report_dir: Path | None = None

    @classmethod
    def from_pack(cls, *, repo_root: Path, expected_report_dir: Path) -> "TutorialPackRunner":
        runner = cls()
        runner.m_repo_root = repo_root
        runner.m_expected_report_dir = expected_report_dir
        return runner
```

## References

- Similar tutorial packs (duplication source):
  - `docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/`
  - `docs/tutorial/howto/tut-qwen3-vl-4b-layer-sensitivity/`
- Existing summary builder to reuse:
  - `src/auto_quantize_model/qwen/tutorial_pack_summary.py`
- Existing LM-only runner + callable API:
  - `scripts/qwen/qwen3_lm_sensitivity.py`
  - `src/auto_quantize_model/qwen/autoquant_sensitivity.py`
- Existing all-layers runner (shared across models via `--model-dir`):
  - `models/qwen3_vl_4b_instruct/helpers/qwen3_vl_4b_autoquant_all_layers/run_qwen3_vl_4b_autoquant_all_layers.py`
- Summary schema contract:
  - `specs/002-revise-qwen3-vl-tutorial/contracts/summary.schema.json`
- Context7 library IDs (3rd-party references):
  - PyTorch: `/pytorch/pytorch`
  - Hydra: `/facebookresearch/hydra`
  - NVIDIA Model Optimizer: `/websites/nvidia_github_io_model-optimizer`
