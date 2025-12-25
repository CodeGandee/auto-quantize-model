<!--
Sync Impact Report

- Version change: N/A (template) → 1.0.0
- Principles defined:
  - Template principle 1 → I. Python-First, Pixi-First
  - Template principle 2 → II. Typed, Linted, Maintainable
  - Template principle 3 → III. Reproducible Experiments and Artifacts
  - Template principle 4 → IV. Testing Discipline (manual + unit + integration)
  - Template principle 5 → V. Documentation and Research Traceability
- Sections defined:
  - Template section 2 → Engineering Standards (Python, environment, docs, tests)
  - Template section 3 → Workflow and Quality Gates
- Removed sections: None
- Templates requiring updates:
  - ✅ `.specify/templates/plan-template.md`
  - ✅ `.specify/templates/spec-template.md` (no change needed)
  - ✅ `.specify/templates/tasks-template.md`
  - ⚠ `.specify/templates/commands/*.md` (directory missing in this repo; removed stale reference from plan template)
- Follow-up TODOs: None
-->

# auto-quantize-model Constitution

## Core Principles

### I. Python-First, Pixi-First

- All development commands MUST run through Pixi (e.g. `pixi run pytest`, `pixi run ruff check .`, `pixi run mypy .`).
- New Python APIs MUST be type-annotated and documented for Python users (NumPy-style docstrings where applicable).
- Code MUST favor readability and composability over cleverness; optimize only after measuring and documenting trade-offs.

Rationale: This repo is primarily a Python research/engineering workspace; consistent environments and readable APIs reduce friction and improve reproducibility.

### II. Typed, Linted, Maintainable

- All new/modified Python code MUST pass `ruff` and `mypy` in the Pixi environment.
- Imports SHOULD be absolute and grouped by standard library, third-party, and local modules.
- Functional classes MUST follow clear OOP conventions: member variables are prefixed `m_`, initialized in `__init__`, read via `@property`, and mutated via explicit `set_*()` methods with validation; prefer `cls.from_*()` factory constructors over large `__init__` signatures.
- Data model classes MUST use framework-native naming (no `m_` prefix) and SHOULD prefer `attrs` (`@define`, `kw_only=True`) unless `pydantic` is needed for schema validation (e.g., web I/O).

Rationale: Static checks and explicit conventions prevent “research code drift” and make it safe to iterate quickly.

### III. Reproducible Experiments and Artifacts

- Every quantization/training run MUST record enough metadata to reproduce it: code entrypoint, resolved config, dataset provenance (paths + counts + split), and key hyperparameters.
- Temporary outputs MUST go under `tmp/` and MUST NOT be committed; curated reports MUST go under `models/*/reports/<run-id>/` and include a human-readable `summary.md`.
- Any exported model artifact (e.g., ONNX) MUST document its IO contract and evaluation settings (imgsz, conf/iou thresholds, max_det, providers).

Rationale: Comparisons are only meaningful when experiments are reproducible and artifacts are traceable.

### IV. Testing Discipline (manual + unit + integration)

- New behavior MUST be accompanied by the right-sized test coverage:
  - Unit tests in `tests/unit/` for deterministic logic and data transforms.
  - Integration tests in `tests/integration/` for end-to-end flows.
  - Manual tests in `tests/manual/` for heavy/interactive workflows (training, GPU inference, long-running experiments).
- Tests MUST be runnable via Pixi (`pixi run pytest`) and SHOULD be deterministic; heavy artifacts MUST be written under `tmp/` (never committed).

Rationale: Quantization workflows are easy to regress silently; tests and manual scripts provide fast feedback without requiring full training runs in CI.

### V. Documentation and Research Traceability

- Documentation MUST be readable and structured; avoid hard line breaks in prose; prefer numbered sections where it improves scanability.
- When implementing research-backed methods (e.g., QAT stabilization), docs MUST include the methodology and the “why” (math where needed), plus links to primary references and the exact commands/configs used.
- Reports and design docs SHOULD include minimal, concrete code snippets over prose-only descriptions when documenting Python interfaces.

Rationale: This repo is used to compare cutting-edge methods; reviewers must be able to audit what was done and why.

## Engineering Standards (Python, environment, docs, tests)

### Python style and quality

- All parameters and return values MUST be type-annotated.
- Python code MUST pass `ruff` and `mypy` before being considered complete.

### Runtime environment

- Prefer Pixi-managed environments; avoid relying on system Python.
- Any docs or scripts that assume an environment MUST state that assumption (e.g., “Run with `pixi run ...`”).

### Documentation standards

- Prefer well-structured Markdown with clear headings; use section numbering where helpful.
- Do not hard-wrap paragraphs in generated Markdown/prose docs.

### Testing standards

- Use `pytest`.
- Place tests under `tests/unit/`, `tests/integration/`, and `tests/manual/` following existing repo conventions.

## Workflow and Quality Gates

### Local quality gates (must be green before review)

- Lint: `pixi run ruff check .`
- Type-check: `pixi run mypy .`
- Tests: `pixi run pytest`

### Review expectations

- PR descriptions MUST include how changes were validated (commands run, relevant artifacts, or why validation is not applicable).
- For experimental runs, PRs MUST link to the report directory and include a concise `summary.md` with key results and caveats.

### Artifacts policy

- Do not commit anything under `tmp/`.
- Prefer keeping large binary artifacts out of Git; commit only the minimal set needed for review (configs, summaries, plots, small databases).

## Governance

- This constitution supersedes other development practices in this repo.
- Amendments MUST be made via PR and MUST include:
  - a clear description of the change and rationale,
  - updates to `.specify/templates/*` if required for alignment,
  - an updated Sync Impact Report (top comment in this file),
  - a version bump following semantic versioning.
- Versioning policy:
  - MAJOR: breaking governance changes or principle removals/redefinitions.
  - MINOR: new principle/section added or materially expanded guidance.
  - PATCH: clarifications, wording, typo fixes, non-semantic refinements.
- Reviewers MUST check changes against this constitution (environment, lint/type/test gates, artifact policy, and documentation requirements).

**Version**: 1.0.0 | **Ratified**: 2025-12-25 | **Last Amended**: 2025-12-25
