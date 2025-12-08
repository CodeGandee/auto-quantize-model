Custom vLLM wheels
===================

This directory is for locally built vLLM wheels that come from the
`extern/vllm` source tree in this repo, typically built via
`extern/build-vllm.sh` (for example, using the Pixi task
`pixi run build-vllm-wheel` which writes wheels into `tmp/vllm-build/`).

The `bootstrap.sh` script will:

- Discover a `vllm-*.whl` built from `extern/vllm` in common locations
  (repo `tmp/vllm-build/`, `extern/vllm/tmp/vllm-build/`, or
  `/workspace/python-pkgs`).
- Optionally use an explicit wheel path passed via `--path`.
- Create or update a symlink in this directory pointing to the chosen wheel.

Typical usage (from the repo root):

- `bash custom-build/bootstrap.sh` – auto-discover the latest wheel and link it here.
- `bash custom-build/bootstrap.sh --path /absolute/path/to/vllm.whl` – link a specific wheel.
- `bash custom-build/bootstrap.sh --clean` – remove existing vLLM wheel links from this directory.
