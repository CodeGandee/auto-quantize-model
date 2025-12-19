Custom build artifacts
=====================

This directory follows the "external reference collection directory" pattern:
it holds symlinks to large, locally built artifacts (wheels, shared libraries,
build logs) that should not be committed to this repo.

Managed entries:

- `vllm-*.whl` (symlink)
  - Source: local build from `extern/vllm` (via `extern/build-vllm.sh`)
  - Used by Pixi tasks like `postinstall-vllm-rtx5090`
- `onnxruntime_gpu-*.whl` (symlink)
  - Source: local build from `extern/onnxruntime` (via `extern/build-onnxruntime-cuda-12_8.sh`)
  - Typical build outputs live under `tmp/` and/or `/workspace/source-builds/`

The `bootstrap.sh` script will:

- Discover wheels in common build output locations.
- Optionally use explicit wheel paths passed via flags or env vars.
- Create or update symlinks in this directory pointing to the chosen wheels.

## VS Code file watching

This directory may contain large wheel files and related artifacts. If VS Code
shows warnings about being unable to watch for file changes (often due to Linux
inotify limits), consider excluding `custom-build/` from file watching in your
workspace settings (repo root `.vscode/settings.json`):

```json
{
  "files.watcherExclude": {
    "**/custom-build/**": true
  }
}
```

Typical usage (from the repo root):

- `bash custom-build/bootstrap.sh` – link vLLM and ONNXRuntime wheels if found.
- `bash custom-build/bootstrap.sh --artifact vllm` – link only the vLLM wheel.
- `bash custom-build/bootstrap.sh --artifact onnxruntime` – link only the ONNXRuntime wheel.
- `bash custom-build/bootstrap.sh --vllm-path /abs/vllm-*.whl` – link a specific vLLM wheel.
- `bash custom-build/bootstrap.sh --onnxruntime-path /abs/onnxruntime_gpu-*.whl` – link a specific ONNXRuntime wheel.
- `bash custom-build/bootstrap.sh --clean` – remove existing wheel links from this directory.

Environment variables:

- `VLLM_WHEEL_DIR`, `VLLM_WHEEL_PATH` (vLLM discovery)
- `ONNXR_WHEEL_DIR`, `ONNXR_WHEEL_PATH` (ONNXRuntime discovery)
- `SOURCE_BUILDS_ROOT` (defaults to `/workspace/source-builds`)
