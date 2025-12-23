# How to build `onnxoptimizer` in the `rtx5090` Pixi env

This repo uses Python 3.12 in the `rtx5090` environment, so `onnxoptimizer` is
typically built from source (there are often no prebuilt wheels for CPython
3.12).

## Quick path (recommended)

Run the helper script from the repo root:

```bash
pixi install -e rtx5090
pixi run -e rtx5090 build-onnxoptimizer-rtx5090
```

This will:

- Ensure `extern/onnxoptimizer` exists (shallow clone) and its submodules are initialized.
- Build a wheel into `custom-build/` (ignored by Git).
- Install that wheel into the active Pixi environment.

## Why the build needs extra CMake flags

When building `onnxoptimizer` (and its vendored `protobuf`) inside this
container, the default protobuf build can fail with:

- `fatal error: absl/log/absl_log.h: No such file or directory`

The fix used here is to force protobuf to fetch a compatible Abseil at build
time:

- `-Dprotobuf_FORCE_FETCH_DEPENDENCIES=ON`

Some envs also lack `zlib.h` headers even if `libz` is present, so the script
disables zlib support in protobuf:

- `-Dprotobuf_WITH_ZLIB=OFF`

If you later add `zlib` development headers to the environment, you can remove
this flag by setting `ONNXOPT_CMAKE_ARGS` when running the task/script.

## Manual build (if you need to tweak things)

```bash
# 1) Clone + submodules
git clone --depth=1 https://github.com/onnx/optimizer.git extern/onnxoptimizer
git -C extern/onnxoptimizer submodule update --init --recursive

# 2) Build + install (inside the Pixi env)
pixi run -e rtx5090 bash -lc '
  CMAKE_ARGS="-Dprotobuf_FORCE_FETCH_DEPENDENCIES=ON -Dprotobuf_WITH_ZLIB=OFF" \
  MAX_JOBS="$(nproc)" \
  python -m pip wheel --no-deps -w custom-build extern/onnxoptimizer
  python -m pip install --no-deps --force-reinstall custom-build/onnxoptimizer-*.whl
'
```

Sanity check:

```bash
pixi run -e rtx5090 python -c "import onnxoptimizer; print(onnxoptimizer.__version__)"
```
