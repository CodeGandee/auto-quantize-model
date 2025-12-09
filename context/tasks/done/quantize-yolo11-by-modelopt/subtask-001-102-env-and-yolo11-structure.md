# Subtask 1.2: Verify environment and YOLO11 repo wiring

## Scope

Confirm that the **auto-quantize-model** repository’s environment and YOLO11 assets are ready for ModelOpt-based quantization work. This includes verifying that `nvidia-modelopt` is installed and importable via `pixi`, and understanding how `models/yolo11/` is organized (checkpoints, helpers, ONNX outputs, and source).

## Planned outputs

- Confirmation that the `pixi` environment can import `modelopt` without errors.
- A quick map of where YOLO11 checkpoints, ONNX exports, and helper scripts live in the repo.
- Notes about any missing dependencies, bootstrap steps, or quirks that future milestones must account for.

## TODOs

- [x] Job-001-102-001: Inspect `pyproject.toml` and `pixi.lock` to verify that `nvidia-modelopt` (or equivalent ModelOpt package) is included in the environment.
- [x] Job-001-102-002: Run `pixi run python -c "import modelopt"` (or equivalent) to confirm that ModelOpt is importable in this project’s environment.
- [x] Job-001-102-003: Review the `models/yolo11/` directory structure (including `bootstrap.sh`, `checkpoints/`, `helpers/`, `onnx/`, and `src/`) to understand the existing YOLO11 export and tooling layout.
- [x] Job-001-102-004: Document any required setup steps (e.g., running `models/yolo11/bootstrap.sh`) to obtain YOLO11 checkpoints and ONNX exports before quantization.
- [x] Job-001-102-005: Record a short summary of the environment and YOLO11 layout in the main task file or a small context note for future reference.

## Notes

- Prefer using `pixi` for all environment-related commands to stay consistent with project guidelines.
- If ModelOpt is missing, note the expected installation instructions rather than modifying the environment directly, unless explicitly asked.

## Implementation summary

- Environment: `nvidia-modelopt` is included in `pyproject.toml` and locked in `pixi.lock` (version `0.39.0`), and `pixi run python -c "import modelopt; print(modelopt.__version__)"` succeeds, confirming ModelOpt is importable in this project’s environment.
- YOLO11 layout: `models/yolo11/` contains `bootstrap.sh`, a `src/` clone of the Ultralytics repository, downloaded checkpoints under `checkpoints/` (`yolo11n/s/m/l/x.pt`), ONNX exports under `onnx/` (e.g., `yolo11n.onnx`), and helper tooling under `helpers/` (notably `helpers/convert_to_onnx.py`).
- Required setup: to (re)initialize YOLO11 assets, run `models/yolo11/bootstrap.sh` from the repo root, then use `pixi run python models/yolo11/helpers/convert_to_onnx.py yolo11n` (or another variant) to generate ONNX exports into `models/yolo11/onnx/` before running ModelOpt-based quantization workflows.
