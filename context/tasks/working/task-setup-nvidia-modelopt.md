# Task: Set up NVIDIA ModelOpt for YOLO11 quantization

This task explains how to set up the environment to run **NVIDIA TensorRT Model Optimizer (ModelOpt)** and use it to quantize the **YOLO11** model from this repo. It follows the guidance in `context/hints/about-automatic-mixed-precision-quantization-tools.md`, focusing on ModelOpt’s ONNX/PTQ path (AutoCast / auto-mixed-precision) rather than deep TensorRT internals.

You can choose between:

- **Option A – Docker (recommended on fresh machines)**
- **Option B – Native / non-Docker install (works with this repo’s pixi env)**

Both options assume:

- An NVIDIA GPU with recent drivers.
- CUDA-compatible host (Linux preferred).
- Basic familiarity with this repo’s workflow (`pixi install`, `pixi run`).

---

## 1. Prepare YOLO11 ONNX model

ModelOpt’s ONNX/PTQ path expects an ONNX model. In this repo, the canonical pipeline for YOLO11 is:

1. **Bootstrap the YOLO11 assets** (download checkpoints, clone ultralytics, etc.):

   ```bash
   pixi run bash models/yolo11/bootstrap.sh
   ```

2. **Export YOLO11 to ONNX** using the helper:

   ```bash
   # Example: nano model
   pixi run python models/yolo11/helpers/convert_to_onnx.py yolo11n
   ```

   This should produce something like:

   - `models/yolo11/onnx/yolo11n.onnx`

You will point ModelOpt’s ONNX/PTQ example (or your own script) at this ONNX file.

---

## 2. Option A – Docker-based setup (TensorRT-LLM image with ModelOpt)

NVIDIA provides **TensorRT-LLM** container images that already include **ModelOpt**. This is the most isolated and reproducible way to run ModelOpt.

### 2.1 Prerequisites

- NVIDIA GPU + drivers installed.
- Docker installed.
- **NVIDIA Container Toolkit** configured so `docker run --gpus all ...` works.
  - See: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

Verify GPU visibility from Docker (on the host you should see your GPUs in `nvidia-smi`, and from a test container you should see GPUs as well).

### 2.2 Pick a TensorRT-LLM / ModelOpt image

1. Visit NVIDIA NGC and search for **`tensorrt-llm`**:
   - https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt-llm
2. Choose a **recent tag** compatible with your driver/CUDA (e.g. a `24.xx` `*-py3` tag).
3. Note the full image name, for example:
   - `nvcr.io/nvidia/tensorrt-llm:<TAG>`

These images include TensorRT-LLM and **TensorRT-Model-Optimizer (ModelOpt)** pre-installed, according to the ModelOpt README.

### 2.3 Launch container bound to this repo

From the repo root (`auto-quantize-model`), run:

```bash
export IMAGE=nvcr.io/nvidia/tensorrt-llm:<TAG>   # replace <TAG> with a concrete tag from NGC

docker run --gpus all --rm -it \
  -v "$PWD":/workspace/auto-quantize-model \
  -w /workspace/auto-quantize-model \
  $IMAGE \
  bash
```

Inside the container:

1. Confirm Python and ModelOpt are available:

   ```bash
   python -c "import modelopt; print(modelopt.__version__)"
   ```

2. (Optional but recommended) Install any repo-specific Python deps if the image’s Python env is not yet aligned with this project:

   ```bash
   # If pixi is not available in the image, use pip/conda to install dependencies
   # or create a venv. At minimum, you need packages required by YOLO11 export
   # (ultralytics, etc.) and by ModelOpt ONNX/PTQ.
   ```

3. Ensure the YOLO11 ONNX file exists (from Section 1) at:

   - `/workspace/auto-quantize-model/models/yolo11/onnx/yolo11n.onnx`

At this point, you can follow ModelOpt’s ONNX/PTQ examples (e.g., adapting `examples/onnx_ptq` from the ModelOpt repo) to point at the YOLO11 ONNX file and run auto/mixed-precision quantization.

---

## 3. Option B – Native / non-Docker install (with pixi)

If you prefer to stay on the host without containers, you can install ModelOpt into the Python environment used for this repo.

### 3.1 Prerequisites

- NVIDIA GPU + drivers installed.
- CUDA toolkit / CUDA runtime compatible with your GPU.
- This repo’s **pixi** environment installed:

  ```bash
  pixi install
  ```

### 3.2 Install NVIDIA ModelOpt (PyPI)

ModelOpt is published on NVIDIA’s PyPI index as `nvidia-modelopt`. For ONNX/PTQ workflows (suitable for YOLO11), install with the ONNX extra:

```bash
pixi run pip install "nvidia-modelopt[onnx]~=0.15.0" \
  --extra-index-url https://pypi.nvidia.com
```

Notes:

- You can substitute `[onnx]` with `[torch]` or `[all]` depending on which parts of ModelOpt you plan to use (e.g., PyTorch PTQ/QAT vs ONNX PTQ).
- The version constraint `~=0.15.0` is an example; you can relax or bump this to match the latest ModelOpt documentation, as long as it’s compatible with your CUDA/toolchain.

Verify the install:

```bash
pixi run python -c "import modelopt; print(modelopt.__version__)"
```

### 3.3 Confirm YOLO11 + ONNX + ModelOpt can coexist

1. Check YOLO11 ONNX file exists:

   ```bash
   ls models/yolo11/onnx
   # expect e.g. yolo11n.onnx
   ```

2. Run a tiny ModelOpt smoke test (in the pixi env):

   ```bash
   pixi run python - << 'PY'
   import modelopt
   from modelopt.onnx import quantization as onnx_quant

   print("ModelOpt version:", modelopt.__version__)
   # We only import the ONNX quantization namespace here; the actual
   # YOLO11 quantization script will live elsewhere.
   print("ONNX quantization module:", onnx_quant)
   PY
   ```

If the imports succeed, the environment is ready to run YOLO11-specific quantization scripts.

---

## 4. Where to plug in YOLO11-specific quantization

Once ModelOpt is available (Docker or native), the next steps are:

1. **Write a small quantization driver script** (in `models/yolo11/helpers/` or under `src/auto_quantize_model/`) that:
   - Loads `models/yolo11/onnx/<model-name>.onnx`.
   - Builds a calibration/eval dataloader for your YOLO11 dataset.
   - Calls ModelOpt’s ONNX/PTQ or AutoCast APIs to:
     - Run **automatic mixed-precision search** (per-layer format selection).
     - Export a quantized ONNX model or TensorRT-ready artifacts.
2. **Keep this task file focused on environment setup**; put details of calibration datasets, metrics, and deployment into separate context/task files.

For API details and up-to-date examples, refer to:

- NVIDIA TensorRT-Model-Optimizer (ModelOpt) GitHub:
  - https://github.com/NVIDIA/TensorRT-Model-Optimizer
- The ONNX/PTQ examples in that repo (e.g. `examples/onnx_ptq`) and the official docs linked from the README.

With this environment in place, you can now implement and iterate on YOLO11 quantization experiments using ModelOpt’s automatic mixed-precision capabilities.
