# Repository Guidelines

This guide explains how to work effectively in this repository, for both human and AI contributors.

## Project Structure & Modules

- Core library code: `src/auto_quantize_model/`
- Context & AI knowledge base: `context/` (see directory READMEs)
- Model assets and tools: `models/yolo11/`, `models/yolo10/`
  - Bootstrap scripts: `models/*/bootstrap.sh`
  - ONNX/export helpers: `models/*/helpers/`
- Tests: `tests/` (`unit/`, `integration/`, `manual/`)
- Docs sources: `docs/`
- CI/automation: `.github/workflows/`
- Temporary data: `tmp/` (ignored by Git; do not commit anything here)

## Build, Test, and Development Commands

All commands should run through Pixi:

- Install / update env: `pixi install`
- Run tests: `pixi run pytest`
- Lint: `pixi run ruff check .`
- Type check: `pixi run mypy .`
- Example model tools:
  - `pixi run python models/yolo11/helpers/convert_to_onnx.py yolo11n`
  - `pixi run python models/yolo10/helpers/infer_and_annotate.py yolov10s tmp/yolo10-infer/image.jpg`

## Coding Style & Naming

- Python, 4-space indentation, Black-like formatting.
- Prefer explicit, descriptive names (no single-letter vars except trivial loops).
- Use type hints for new functions.
- Keep functions small and focused; avoid large, multi-purpose scripts.
- Follow existing patterns in `src/auto_quantize_model/` and model helper scripts.

## Testing Guidelines

- Use `pytest` with tests under `tests/unit/` and `tests/integration/`.
- Name tests `test_*.py`; keep unit tests fast and deterministic.
- Manual or heavy experiments belong in `tests/manual/` or `tmp/`, not CI.

## Commit & Pull Request Guidelines

- Commit messages: short, imperative, and scoped, e.g.:
  - `Configure pixi env and YOLO11 tooling`
  - `Add YOLOv10 assets and ONNX export`
- PRs should:
  - Describe the change and motivation.
  - Reference relevant issues or context files (e.g., `context/tasks/...`).
  - Note any new commands, config changes, or migration steps.

