# Subtask 1.2: Verify environment and YOLO11 repo wiring

## Scope

Confirm that the **auto-quantize-model** repository’s environment and YOLO11 assets are ready for ModelOpt-based quantization work. This includes verifying that `nvidia-modelopt` is installed and importable via `pixi`, and understanding how `models/yolo11/` is organized (checkpoints, helpers, ONNX outputs, and source).

## Planned outputs

- Confirmation that the `pixi` environment can import `modelopt` without errors.
- A quick map of where YOLO11 checkpoints, ONNX exports, and helper scripts live in the repo.
- Notes about any missing dependencies, bootstrap steps, or quirks that future milestones must account for.

## TODOs

- [ ] Job-001-102-001: Inspect `pyproject.toml` and `pixi.lock` to verify that `nvidia-modelopt` (or equivalent ModelOpt package) is included in the environment.
- [ ] Job-001-102-002: Run `pixi run python -c "import modelopt"` (or equivalent) to confirm that ModelOpt is importable in this project’s environment.
- [ ] Job-001-102-003: Review the `models/yolo11/` directory structure (including `bootstrap.sh`, `checkpoints/`, `helpers/`, `onnx/`, and `src/`) to understand the existing YOLO11 export and tooling layout.
- [ ] Job-001-102-004: Document any required setup steps (e.g., running `models/yolo11/bootstrap.sh`) to obtain YOLO11 checkpoints and ONNX exports before quantization.
- [ ] Job-001-102-005: Record a short summary of the environment and YOLO11 layout in the main task file or a small context note for future reference.

## Notes

- Prefer using `pixi` for all environment-related commands to stay consistent with project guidelines.
- If ModelOpt is missing, note the expected installation instructions rather than modifying the environment directly, unless explicitly asked.
