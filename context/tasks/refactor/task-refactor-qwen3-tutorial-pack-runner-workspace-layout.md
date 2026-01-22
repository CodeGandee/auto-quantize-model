---
description: "Refactor plan: unify Qwen3 tutorial runner workspace layout (merge summaries into outputs; keep only one layer-sensitivity-report.md)"
---

# Refactor Plan: Qwen3 Tutorial Pack Runner Workspace Layout

## What to Refactor

Adjust the shared runner output layout and post-processing rules implemented in:

- `src/auto_quantize_model/qwen/tutorial_pack_runner.py`
- `src/auto_quantize_model/qwen/cli_tutorial_pack_runner.py`

Specifically:

1) **Merge `summaries/` into `outputs/`** so per-scenario `summary.json` is written under:
   - `tmp/<workspace>/outputs/<mode>/<quant_pair>/summary.json`

2) **Merge the old `summary.md` content into `layer-sensitivity-report.md`** so the human-readable markdown output is a single file that contains:
   - the stable “scenario summary” table (previously `summary.md`), and
   - the per-layer sensitivity table (previously `layer-sensitivity-report.md`)

3) **Limit Markdown “layer sensitivity report” output** so that reports are generated for the `all_layers` mode only (and removed/suppressed for `lm_only`):

- `tmp/<workspace>/outputs/all_layers/<quant_pair>/layer-sensitivity-report.md`

All other markdown reports should not be generated (or should be removed after the run), while keeping machine-readable artifacts intact:

- `summary.json` (required for snapshot/verify)
- `layer-sensitivity-report.json` (kept for programmatic inspection/debugging)

## Why Refactor

- **One folder per scenario**: today the workspace splits stable summaries (`summaries/`) and raw artifacts (`outputs/`). Users naturally look in `outputs/`; having two trees makes browsing/debugging harder.
- **One markdown artifact**: users want a single human-readable file that includes both the scenario summary and the per-layer table, instead of `summary.md` plus `layer-sensitivity-report.md`.
- **Reduce noise and disk usage**: generating multiple large markdown reports (per mode × quant-pair) is redundant; keeping a single canonical `layer-sensitivity-report.md` improves UX and avoids clutter.
- **Preserve verification contract**: snapshot/verify should continue to operate on stable machine-readable artifacts; with this refactor, the stable contract should be `summary.json` only (markdown becomes optional and non-gating).
  - Note: `expected_report/` should still contain a sanitized `outputs/` tree for documentation/debugging (including the canonical `layer-sensitivity-report.md`), but verification diffs only `summary.json`.

## How to Refactor (Step-by-Step)

### Step 1: Change workspace layout helpers (remove `summaries/`)

- Update `create_workspace_dir()` to create only `outputs/` (no `summaries/`).
- Replace `resolve_summary_dir()` with summary path helpers returning:
  - `summary_json = output_dir / "summary.json"`

### Step 2: Write only `summary.json` into `outputs/<mode>/<quant_pair>/`

- Update summary generation to write `summary.json` into the scenario’s `output_dir`.
- Update snapshot/verify code paths to read/copy/diff `summary.json` from `output_dir`.
- Update expected snapshot semantics to write sanitized artifacts under:
  - `expected_report/outputs/<mode>/<quant_pair>/summary.json`
  - and (optionally) `expected_report/outputs/all_layers/<quant_pair>/layer-sensitivity-report.md`

### Step 3: Merge the old `summary.md` content into `layer-sensitivity-report.md`

- Implement a helper that renders the “scenario summary table” (currently produced by `write_summary_md`) as markdown text (string).
- When the scenario produces `layer-sensitivity-report.md`, rewrite that file as:
  1) the scenario summary header/table, then
  2) a separator, then
  3) the original layer-sensitivity markdown body.

This keeps the report file human-friendly while maintaining the stable verification contract in `summary.json`.

### Step 4: Enforce “only all_layers layer-sensitivity-report.md”

Add a small, explicit policy function, for example:

- Keep Markdown report only when `mode == "all_layers"` (all quant pairs).
- For all other scenarios:
  - if the underlying runner produces `layer-sensitivity-report.md`, delete it after the scenario completes, or
  - (preferable if supported) pass a flag/env var to suppress generating that markdown file upstream.

Also ensure the runner never deletes `layer-sensitivity-report.json` or `summary.json`.

### Step 5: Update tests

- Unit tests:
  - Update `test_workspace_and_layout_helpers` to assert the new paths (no `summaries/` directory).
  - Update any test helpers that create/read `summary.md` (it should no longer exist).
  - Add a unit test for the markdown merge function (summary table appears in the merged report).
- Integration tests:
  - Ensure snapshot cleanup works under `expected_report/outputs/` and verification diffs only `summary.json`.
  - Add an assertion that after a (mocked) run, non-canonical scenarios do not contain `layer-sensitivity-report.md` (if the integration test simulates filesystem).

### Step 6: Update docs (if needed)

- Ensure tutorial READMEs reference the new summary location (under `outputs/...`) if they mention `summaries/`.
- Ensure any troubleshooting instructions point users to:
  - `outputs/<mode>/<quant_pair>/summary.json`
  - `outputs/<mode>/<quant_pair>/layer-sensitivity-report.json`
  - and only one Markdown report path.

## Impact Analysis

### Behavior changes

- Workspace layout changes from:
  - `tmp/<workspace>/summaries/<mode>/<pair>/summary.*`
  - to `tmp/<workspace>/outputs/<mode>/<pair>/summary.json`
- Markdown reports become “single-canonical”:
  - Only `outputs/all_layers/<quant_pair>/layer-sensitivity-report.md` remains (all_layers only; no lm_only markdown).
- The canonical markdown report becomes “merged”:
  - It includes the per-scenario summary table at the top (previously `summary.md`).

### Risks

- **Scripts/tools expecting `summaries/`**: ad-hoc user scripts may break.
  - Mitigation: keep a short compatibility shim for one release (optional), or clearly document the new location.
- **Upstream runner always writes the markdown report**: deleting it post-run is safe but slightly wasteful.
  - Mitigation: if upstream supports disabling markdown generation, prefer that; otherwise post-delete is acceptable.
- **Verify/snapshot correctness**: changes can accidentally make verification diff the wrong files or rely on markdown.
  - Mitigation: keep verify/snapshot strictly pinned to `summary.json` only and add/adjust tests.

## Expected Outcome

- A single, predictable scenario directory layout under `tmp/<workspace>/outputs/...` that contains:
  - raw run logs/artifacts,
  - `summary.json` next to the artifacts,
  - `layer-sensitivity-report.json` for machine processing,
  - and markdown reports for `all_layers` only: `outputs/all_layers/<quant_pair>/layer-sensitivity-report.md`.
- Snapshot/verify continues to be summary-only and deterministic (diffing only `summary.json`), while `expected_report/outputs/` retains a sanitized copy of key artifacts for reference.

## TODO

- [ ] Update workspace creation to remove `summaries/` directory
- [ ] Move summary writing to `outputs/<mode>/<quant_pair>/summary.json` (stop writing `summary.md`)
- [ ] Update snapshot/verify code to read/copy/diff `summary.json` from `outputs/` (no markdown diffs)
- [ ] Update expected snapshots to include `expected_report/outputs/` (sanitized artifacts + canonical markdown report)
- [ ] Implement a markdown merge step: prepend summary table into canonical `layer-sensitivity-report.md`
- [ ] Implement markdown report retention policy (keep only `all_layers/<quant_pair>`)
- [ ] Update unit tests for new paths and behavior
- [ ] Update integration tests for snapshot/verify behavior under new layout
- [ ] Update tutorial docs that mention `summaries/` (if any)

## Example Refactor Snippets (Before → After)

### 1) Summary path resolution

Before:

```python
summary_dir = resolve_summary_dir(workspace_dir, scenario)
build_and_write_summary(manifest_json, summary_dir, scenario=scenario, dataset_size=dataset_size)
verify_scenario(expected_report_dir, scenario, summary_dir)
```

After:

```python
output_dir = resolve_output_dir(workspace_dir, scenario)
build_and_write_summary(manifest_json, output_dir, scenario=scenario, dataset_size=dataset_size)
verify_scenario(expected_report_dir, scenario, output_dir)
```

### 2) Keep only one markdown report

Before:

```python
# Upstream runners generate layer-sensitivity-report.md for every scenario.
run_scenario(...)
```

After:

```python
run_scenario(...)
if scenario.mode != "all_layers":
    (output_dir / "layer-sensitivity-report.md").unlink(missing_ok=True)
```

### 3) Merge `summary.md` content into the canonical report

Before:

```python
write_summary_md(summary_dir / "summary.md", summary)
# layer-sensitivity-report.md is produced by the underlying runner.
```

After:

```python
summary_md = render_summary_markdown(summary)
report_path = output_dir / "layer-sensitivity-report.md"
if report_path.is_file() and scenario.mode == "all_layers":
    report_path.write_text(summary_md + "\n\n---\n\n" + report_path.read_text(encoding="utf-8"), encoding="utf-8")
```

## References

- Current runner implementation: `src/auto_quantize_model/qwen/tutorial_pack_runner.py`
- CLI entrypoint: `src/auto_quantize_model/qwen/cli_tutorial_pack_runner.py`
- Tutorial workspaces (examples):
  - `tmp/tutorial_workspace_qwen3_vl_4b_layer_sensitivity_20260122_042840`
  - `tmp/tutorial_workspace_qwen3_vl_4b_layer_sensitivity_1768982576`
- Related prior refactor plan: `context/tasks/refactor/task-refactor-qwen3-tutorial-pack-runner.md`
