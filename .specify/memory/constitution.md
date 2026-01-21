<!--
Sync Impact Report
- Version change: template → 1.0.0
- Modified principles:
  - Principle 1 slot → I. Pixi-First, Reproducible Environments
  - Principle 2 slot → II. Library-First + Stable Repository Layout
  - Principle 3 slot → III. Quality Gates: Ruff + Mypy + Pytest (NON-NEGOTIABLE)
  - Principle 4 slot → IV. Reproducible Experiments & Artifact Hygiene
  - Principle 5 slot → V. Documentation as Product
- Added sections: none (filled existing placeholders)
- Removed sections: none
- Templates requiring updates:
  - ✅ updated: .specify/templates/plan-template.md
  - ✅ updated: .specify/templates/tasks-template.md
  - ⚠ pending: (none) — `.specify/templates/commands/` not present in this repo
-->

# auto-quantize-model Constitution

## Core Principles

### I. Pixi-First, Reproducible Environments
All development, testing, and scripts MUST run inside the Pixi-managed Python 3.12 environment(s) defined in `pyproject.toml`/`pixi.lock`. Commands in docs/PRs MUST be expressed as `pixi run ...` (and include `-e <env>` when a specific CUDA stack is required). Do not rely on system Python; do not “pip install” outside Pixi. If dependencies change, `pixi.lock` MUST be updated in the same change set.

### II. Library-First + Stable Repository Layout
Reusable logic MUST live under `src/auto_quantize_model/` and be written as importable modules with type hints. Experiment entrypoints MUST live under `scripts/` (or `models/*/helpers/` when model-specific) and delegate to library code rather than duplicating logic. Configuration MUST use the repo’s established patterns (`conf/` for Hydra/OmegaConf when applicable). Tests MUST live under `tests/unit/`, `tests/integration/`, or `tests/manual/` as appropriate.

### III. Quality Gates: Ruff + Mypy + Pytest (NON-NEGOTIABLE)
Before merging, changes MUST pass: `pixi run ruff check .`, `pixi run mypy .`, and `pixi run pytest`. New logic MUST be covered by the right-sized tests: unit tests for deterministic pure logic, integration tests for file/metadata/IO boundaries, and manual tests for heavy GPU training or long-running experiments. If tests are omitted, the spec/plan MUST explicitly justify why (e.g., out-of-scope, non-deterministic, prohibitively expensive) and provide an alternative validation path.

### IV. Reproducible Experiments & Artifact Hygiene
All experiments MUST be reproducible: record resolved configuration, dataset provenance (name/version/path), random seeds, and enough environment/code provenance to re-run (e.g., git commit hash when available). Generated artifacts MUST go under `tmp/` (uncommitted) unless explicitly curated; curated run summaries/reports belong under `models/*/reports/<run-id>/` (or a similarly documented, version-controlled reports directory). Do not commit large model binaries, datasets, or transient checkpoints unless the change is explicitly intended as a tracked asset and reviewed as such.

### V. Documentation as Product
Behavioral changes MUST be documented where users will look: `README.md`, `docs/`, and/or the relevant `specs/<id-...>/` documents (plan/research/data-model/quickstart/contracts). Documentation MUST include concrete, copy-pastable commands using Pixi. Markdown prose SHOULD avoid hard line breaks within paragraphs; use headings and lists for structure. Public Python interfaces SHOULD have docstrings (NumPy style preferred) when non-trivial.

## Python Engineering Standards

- Type annotate new public functions/methods (parameters and return values) and keep interfaces small and explicit.
- Prefer clarity and maintainability over cleverness; keep functions focused and avoid “do-everything” scripts.
- Use absolute imports and keep import groups ordered (stdlib, third-party, local).
- Data/config containers SHOULD be lightweight; keep business logic in service/helper modules rather than “fat” data objects.

## Development Workflow

- Use the `specs/<###-feature-name>/` workflow for significant changes: write `spec.md` → `plan.md` → `tasks.md`, and keep feature documentation alongside implementation.
- Branch names SHOULD follow the numeric prefix convention used in this repo (e.g., `001-yolov10-qat-validation`) for traceability.
- Keep runtime artifacts out of git: use `tmp/` for scratch outputs and commit only curated reports under `models/*/reports/<run-id>/` when needed.
- Prefer deterministic defaults (seeded runs, explicit configs); when nondeterminism is unavoidable, document it and bound it (repeats, tolerances, or acceptance criteria).

## Governance
This constitution governs repository-wide engineering practices and supersedes conflicting guidance in templates and docs.

- **Compliance review**: Every PR MUST demonstrate compliance with the Core Principles (tests + quality gates, Pixi commands, artifact hygiene, and documentation updates as applicable).
- **Amendment process**: Amendments MUST be made via PR editing this file and MUST include a rationale, a version bump, an updated “Sync Impact Report” comment, and any required template/doc updates.
- **Versioning policy**: Use semantic versioning for governance changes: MAJOR for backwards-incompatible principle changes/removals, MINOR for new principles/sections or materially expanded requirements, PATCH for clarifications and non-semantic refinements.
- **Review expectations**: Reviewers MUST check for constitution compliance explicitly (especially environment usage, test coverage category, and artifact locations) before approving.

**Version**: 1.0.0 | **Ratified**: 2025-12-25 | **Last Amended**: 2026-01-21
