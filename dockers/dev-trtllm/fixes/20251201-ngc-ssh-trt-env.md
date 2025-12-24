# Fix: Ensure NVIDIA NGC SSH sessions can use TensorRT

- **Date**: 2025-12-01
- **Container**: `dev-stage-2-1` (`auto-quantize:stage-2`)
- **User affected**: `me` (SSH user, password `123456`)
- **Symptom**:
  - Environment over SSH was missing critical NVIDIA variables like `LD_LIBRARY_PATH` and `LIBRARY_PATH`.
  - This can cause issues for TensorRT / CUDA tools in SSH or non-interactive shells, even though `docker exec` works.

## What was changed inside the running container

1. Verified baseline behavior:
   - `docker exec dev-stage-2-1 python3 -c "import tensorrt as trt; print('TRT', trt.__version__)"`
   - `sshpass -p 123456 ssh -p 13444 -o StrictHostKeyChecking=no me@localhost 'python3 -c "import tensorrt as trt; print(\"TRT\", trt.__version__)"'`
   - Confirmed TensorRT works via both `docker exec` and SSH, but SSH env lacked `LD_LIBRARY_PATH`/`LIBRARY_PATH`.

2. Inspected NVIDIA init scripts:
   - Checked `/etc/shinit_v2` and `/etc/bash.bashrc` to confirm the standard NGC init pattern:
     - `/etc/bash.bashrc` sources `/etc/shinit_v2`.
     - `/etc/shinit_v2` only conditionally adjusts `LD_LIBRARY_PATH` for CUDA compat and does not set defaults for SSH sessions.
   - Observed that `docker exec` environment includes:
     - `LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64`
     - `LIBRARY_PATH=/usr/local/cuda/lib64/stubs:`
   - Observed that SSH environment for user `me` initially had:
     - `_CUDA_COMPAT_PATH=/usr/local/cuda/compat`
     - **No** `LD_LIBRARY_PATH` / `LIBRARY_PATH`.

3. Applied the NGC SSH env fix by appending to `/etc/shinit_v2`:

   ```bash
   # Custom: Ensure NVIDIA library paths exist for SSH/non-interactive shells
   export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64}
   export LIBRARY_PATH=${LIBRARY_PATH:-/usr/local/cuda/lib64/stubs:}
   ```

   - This was done via:
     - `docker exec dev-stage-2-1 bash -lc 'cat >> /etc/shinit_v2 << "EOF" ... EOF'`
   - The `${VAR:-...}` pattern ensures we **only** set these when they are missing, so we do not override Dockerfile `ENV` in `docker exec` contexts.

4. Re-validated SSH behavior for user `me`:
   - `sshpass -p 123456 ssh -p 13444 -o StrictHostKeyChecking=no me@localhost 'env | egrep "LD_LIBRARY_PATH|LIBRARY_PATH|CUDA" | sort'`
     - Confirmed:
       - `LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64`
       - `LIBRARY_PATH=/usr/local/cuda/lib64/stubs:`
       - `_CUDA_COMPAT_PATH=/usr/local/cuda/compat`
   - `sshpass -p 123456 ssh -p 13444 -o StrictHostKeyChecking=no me@localhost 'python3 -c "import tensorrt as trt, os; print(\"TRT\", trt.__version__); print(\"LD_LIBRARY_PATH=\", os.environ.get(\"LD_LIBRARY_PATH\"))"'`
     - Confirmed TensorRT still imports correctly and sees the expected `LD_LIBRARY_PATH` in SSH sessions.

## Notes and follow-ups

- This fix is **runtime-only** for the currently running container. It should be ported back into the Docker build (e.g., `stage-2.Dockerfile` or a related install script) by appending the same `export` lines to `/etc/shinit_v2` during image build.
- The paths used here intentionally mirror the existing `LD_LIBRARY_PATH` and `LIBRARY_PATH` observed in `docker exec` environments, rather than assuming Triton/TensorRT-specific locations under `/usr/local/tensorrt`.
- If future NGC images change their default library locations, update these exports accordingly.

