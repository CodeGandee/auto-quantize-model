Howto: Build ONNX Runtime (CUDA EP) inside the Pixi rtx5090 env (CUDA 12.8)

## HEADER
- Purpose: Document how to build a CUDA-enabled ONNX Runtime wheel from the `extern/onnxruntime` checkout using this repo’s Pixi `rtx5090` environment, with all build outputs written under `tmp/`.
- Status: Working (scripted). Assumes system cuDNN headers/libs are available (common on Ubuntu images); otherwise set `CUDNN_HOME` to your cuDNN install root.
- Date: 2025-12-19
- Owner: AI assistant (Codex CLI)
- References:
  - ONNX Runtime “Build Execution Providers” (CUDA): https://onnxruntime.ai/docs/build/eps.html#cuda
  - ONNX Runtime repo: https://github.com/microsoft/onnxruntime

## 1. Prereqs

1. Ensure the ONNX Runtime checkout exists:

```bash
bash extern/bootstrap.sh
```

2. Ensure the Pixi env exists:

```bash
pixi install
```

3. Verify the `rtx5090` toolchain has CUDA 12.8 `nvcc`:

```bash
pixi run -e rtx5090 nvcc --version
```

4. cuDNN is required for the CUDA Execution Provider build.

This repo’s `rtx5090` Pixi env provides the CUDA toolkit, and often also has
cuDNN headers/libs via the PyPI `nvidia-cudnn-cu12` package (installed as part
of the CUDA/PyTorch stack). If cuDNN is not present in the env, you can fall
back to system cuDNN (common on Ubuntu images).

On Ubuntu/Debian images, you can check:

```bash
ls -1 /usr/include/cudnn.h
ldconfig -p | grep -i cudnn | head
```

If your system uses a different install root, set `CUDNN_HOME` accordingly.

## 2. Build (outputs under tmp/)

Use the helper script:

```bash
pixi run -e rtx5090 bash extern/build-onnxruntime-cuda-12_8.sh -o tmp/onnxruntime
```

Defaults (override via env vars if needed):

- `CUDA_HOME`:
  - Prefers the CUDA toolkit from the active Pixi env (detects via `nvcc` on `PATH` and/or `$CONDA_PREFIX/targets/*`)
  - Fallback (if present): `/usr/local/cuda-12.8.1`
- `CUDNN_HOME`:
  - Prefers cuDNN from the active Pixi env (detects `nvidia.cudnn` under site-packages)
  - Then `/usr` (Debian/Ubuntu)
- `ONNXR_CUDA_ARCHS`: `120` (RTX 5090 / `sm_120`)
- `ONNXR_BUILD_DIR`: `<output-dir>/build/Linux/Release-cuda1281`
- `ONNXR_LOG_FILE`: `<output-dir>/logs/build-<timestamp>.log`

To force explicit paths (recommended if you have a non-standard setup):

```bash
pixi run -e rtx5090 bash -lc '
  export CUDA_HOME="$CONDA_PREFIX/targets/x86_64-linux"
  export CUDNN_HOME="/usr"
  export ONNXR_CUDA_ARCHS="120"
  bash extern/build-onnxruntime-cuda-12_8.sh -o tmp/onnxruntime
'
```

## 3. Where the artifacts land

After a successful build:

- Wheel:
  - `<output-dir>/build/Linux/Release-cuda1281/Release/dist/*.whl`
- Log:
  - `<output-dir>/logs/build-*.log`
- CMake/Ninja outputs, shared libs, and intermediates:
  - Under `<output-dir>/build/Linux/Release-cuda1281/`

## 4. Install the wheel into the Pixi env (optional)

If you want the Pixi `rtx5090` environment to use your custom build:

```bash
pixi run -e rtx5090 python -m pip uninstall -y onnxruntime-gpu || true
pixi run -e rtx5090 python -m pip install --no-deps --force-reinstall \
  tmp/onnxruntime/build/Linux/Release-cuda1281/dist/*.whl
```

(If you built to a different `-o <output-dir>`, replace `tmp/onnxruntime` accordingly.)

## 5. Quick validation

```bash
pixi run -e rtx5090 python - <<'PY'
import onnxruntime as ort
print("onnxruntime:", getattr(ort, "__version__", "<no __version__>"))
print("available providers:", ort.get_available_providers())
PY
```

If CUDA EP is working, you should see `CUDAExecutionProvider` in the provider list.

## 6. Cleanup

Delete the build outputs:

```bash
rm -rf tmp/onnxruntime
```
