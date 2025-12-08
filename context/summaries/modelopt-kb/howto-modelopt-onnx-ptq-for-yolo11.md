Howto: ModelOpt ONNX PTQ for YOLO11 (INT8 / mixed precision)

## HEADER
- **Purpose**: Provide a practical recipe for using NVIDIA ModelOpt’s ONNX PTQ tooling to quantize YOLO11 (or similar CNN detectors) to INT8 and prepare mixed-precision TensorRT deployment.
- **Status**: Draft, based on subtask 001-101 findings
- **Date**: 2025-12-01
- **Owner**: AI assistant (Codex CLI)
- **Source**: See `context/tasks/working/quantize-yolo11-by-modelopt/subtask-001-101-modelopt-docs-and-apis.md` for the underlying analysis.

## 1. Prerequisites and model/export

- You have this repo checked out and the pixi environment installed (`pixi install`).
- YOLO11 ONNX export exists, e.g.:

```bash
pixi run python models/yolo11/helpers/convert_to_onnx.py yolo11n
ls models/yolo11/onnx
# expect: yolo11n.onnx
```

- The ModelOpt submodule is present at `extern/TensorRT-Model-Optimizer` and `nvidia-modelopt[onnx]` is installed in your Python env (inside or outside pixi as per `context/tasks/done/task-setup-nvidia-modelopt.md`).
- You have an NVIDIA GPU and TensorRT installed for engine building (`trtexec` on PATH or TensorRT Python API available).

## 2. Prepare calibration data for YOLO11

ModelOpt ONNX PTQ expects calibration data in `.npy` or `.npz` format:

- For a single input tensor:
  - `.npy` file with shape `[N, C, H, W]` matching ONNX input.
- For multi-input models:
  - `.npz` file with a dict mapping input names → numpy arrays.

For YOLO11, you should:

- Use a few hundred images from the target detection dataset (e.g., subset of COCO or your deployment data).
- Apply the same preprocessing pipeline as runtime inference (resize/letterbox, normalization, etc.).

Example (pseudo-code) for building a `.npy` calibration file:

```python
import numpy as np
from pathlib import Path

def preprocess(img_path):
    # TODO: reuse YOLO11 preprocessing from your pipeline
    # returns CHW float32 tensor-like
    ...

root = Path("/path/to/calib/images")
paths = sorted(root.glob("*.jpg"))[:500]

calib = []
for p in paths:
    calib.append(preprocess(p))

calib_arr = np.stack(calib, axis=0).astype("float32")
np.save("calib_yolo11n.npy", calib_arr)
```

You can also follow the structure in `extern/TensorRT-Model-Optimizer/examples/onnx_ptq/image_prep.py` as a reference.

## 3. Quantize YOLO11 ONNX with ModelOpt (INT8 Q/DQ)

The key ONNX PTQ entrypoints are:

- CLI: `python -m modelopt.onnx.quantization`
- Python: `from modelopt.onnx.quantization import quantize`

The core parameters you care about:

- `onnx_path`: path to your YOLO11 ONNX export (e.g., `models/yolo11/onnx/yolo11n.onnx`).
- `output_path`: where to save the quantized ONNX (e.g., `models/yolo11/onnx/yolo11n-int8-qdq.onnx`).
- `quantize_mode`: `int8` (FP8 and INT4 are available but not required initially for YOLO11 mixed FP16/INT8).
- `calibration_data`: `.npy`/`.npz` calibration file.
- `calibration_method`: `max` or `entropy` for INT8/FP8; `awq_clip` or `rtn_dq` for INT4.
- `calibration_eps`: execution providers for ONNX Runtime during calibration (`cpu`, `cuda:0`, `trt`).
- Optional: `op_types_to_exclude`, `nodes_to_exclude` to keep certain ops/layers in FP16/FP32.
- Optional: `calibrate_per_node` for memory-constrained calibration on large models.

### 3.1 Basic CLI example

```bash
python -m modelopt.onnx.quantization \
  --onnx_path=models/yolo11/onnx/yolo11n.onnx \
  --quantize_mode=int8 \
  --calibration_data=calib_yolo11n.npy \
  --calibration_method=max \
  --output_path=models/yolo11/onnx/yolo11n-int8-qdq.onnx
```

This:

- Loads the YOLO11 ONNX model.
- Runs calibration over `calib_yolo11n.npy`.
- Inserts Q/DQ nodes following TensorRT-friendly rules.
- Saves `yolo11n-int8-qdq.onnx`.

### 3.2 CLI example with selective quantization and TensorRT EP

If you have TensorRT plugins (e.g., YOLO NMS), or want to leave certain ops in higher precision:

```bash
python -m modelopt.onnx.quantization \
  --onnx_path=models/yolo11/onnx/yolo11n.onnx \
  --quantize_mode=int8 \
  --calibration_data=calib_yolo11n.npy \
  --calibration_method=max \
  --output_path=models/yolo11/onnx/yolo11n-int8-qdq.onnx \
  --calibration_eps trt cuda:0 cpu \
  --trt_plugins=/path/to/yolo_plugins.so \
  --op_types_to_exclude=Resize,Concat \
  --calibrate_per_node
```

Typical patterns:

- Keep detection heads / NMS / output reshapes in FP16/FP32 via `op_types_to_exclude` or `nodes_to_exclude`.
- Quantize backbone and neck convs/matrix-mults for most of the performance gains.

### 3.3 Python API example

```python
from modelopt.onnx.quantization import quantize

quantize(
    onnx_path="models/yolo11/onnx/yolo11n.onnx",
    quantize_mode="int8",
    calibration_data="calib_yolo11n.npy",
    calibration_method="max",
    output_path="models/yolo11/onnx/yolo11n-int8-qdq.onnx",
    calibration_eps=["cuda:0", "cpu"],
    op_types_to_exclude=None,
    nodes_to_exclude=None,
)
```

You can parameterize this in a helper script (e.g., `models/yolo11/helpers/quantize_with_modelopt.py`) and wire it into `pixi run` later.

### 3.4 Caveats: Default Layer Exclusions and “Extreme INT8”

**Important**: By default, ModelOpt **excludes** several operation types from INT8 quantization to preserve accuracy. This includes:

- `Softmax`, `Sigmoid`, `Tanh` (activations often sensitive to quantization)
- `Concat`, `Slice`, `Reshape`, `Transpose` (structural ops)
- `Add`, `Sub`, `Mul`, `Div` (element-wise arithmetic, unless fused)

This means a "default" INT8 quantization run will result in a mixed-precision graph where these ops remain in FP32/FP16. For example, in YOLO11:
- The attention mechanism (Softmax in PSA blocks) will remain in FP32/FP16.
- The detection head's final arithmetic (decoding boxes) will remain in FP32/FP16.

If you require **full** INT8 coverage (e.g., for specific NPU constraints or maximum throughput benchmarking), you must explicitly force these ops to be quantized using the `op_types_to_quantize` parameter. However, be aware that forcing sensitive ops like Softmax or Sigmoid to INT8 often degrades accuracy significantly.

In this repository we experimented with an “extreme INT8” configuration that:

- Included a very broad set of op types in `op_types_to_quantize` (Conv, MatMul, MaxPool, Mul, Add, Concat, Sigmoid, Softmax, Split, Transpose, Reshape, Slice, Resize, Sub, Div, etc.).
- Used the same COCO-based calibration tensor as the “normal” YOLO11n PTQ runs.

Empirical observations from that extreme configuration:

- At the ONNX Runtime level (QDQ ONNX evaluated directly), COCO2017 mAP dropped from ≈0.42 (FP32) to ≈0.12 (extreme INT8) on a 500-image slice — a **very large** accuracy collapse.
- The resulting “extreme” QDQ ONNX model also triggered internal errors when attempting to build a TensorRT engine in explicit quantization mode (assertion failures inside TensorRT’s builder on some heavily quantized paths).

Implications for future users:

- Treat “extreme INT8 everywhere” as a **diagnostic or stress test only**, not a realistic deployment configuration.
- For detectors like YOLO11, safer practice is:
  - Let ModelOpt’s defaults exclude sensitive ops (Softmax, some elementwise ops, structural ops).
  - Optionally, **selectively** expand `op_types_to_quantize` after measuring per-layer sensitivity, rather than enabling “quantize everything” in one go.
  - Always verify both ONNX-level accuracy and TensorRT engine build stability before committing to a more aggressive INT8 scheme.

## 4. Build TensorRT engines for mixed FP16/INT8

ModelOpt ONNX quantization creates explicit Q/DQ ONNX graphs that are compatible with TensorRT’s **explicit quantization** path.

- Recommended TensorRT flags:
  - Enable FP16 and INT8 so TensorRT can choose kernels: `--fp16 --int8` (and optionally `--best`).
  - Use `--saveEngine` to persist the engine for later runs.

Example:

```bash
trtexec \
  --onnx=models/yolo11/onnx/yolo11n-int8-qdq.onnx \
  --saveEngine=models/yolo11/trt/yolo11n-int8-mixed.plan \
  --fp16 --int8 --best
```

You should also build a pure FP16 engine from the original ONNX for baseline comparison:

```bash
trtexec \
  --onnx=models/yolo11/onnx/yolo11n.onnx \
  --saveEngine=models/yolo11/trt/yolo11n-fp16.plan \
  --fp16
```

With both engines, run your YOLO11 evaluation script to compute:

- Latency per image / throughput.
- mAP / precision / recall on your evaluation set.

## 5. Practical tips and TensorRT best practices

- Use **explicit quantization** (Q/DQ ONNX) rather than legacy implicit INT8; implicit INT8 is deprecated in TensorRT, while ModelOpt ONNX PTQ is designed for explicit Q/DQ graphs.
- Mixed precision in TensorRT comes from:
  - Allowing multiple precisions in the builder (FP16 + INT8 flags).
  - Providing an ONNX graph with Q/DQ that indicates where INT8 can be used.
- Do not aggressively quantize everything:
  - TensorRT docs (“Working with Quantized Types”) and ModelOpt guides both recommend leaving some output-adjacent or numerically delicate layers in higher precision.
  - For YOLO11, this likely means keeping detection heads, NMS, and some final reshapes / postprocessing ops in FP16/FP32.
- Ensure calibration data:
  - Matches runtime preprocessing exactly (resize/letterbox, normalization).
  - Is drawn from the same distribution as deployment (diverse object sizes, aspect ratios, lighting).
  - Uses ≥500 samples for CNN/ViT-like models, per ModelOpt ONNX PTQ README, unless you intentionally trade some accuracy for speed in calibration.

## 6. Where to look in this repo and upstream

- Local ModelOpt docs and examples:
  - `extern/TensorRT-Model-Optimizer/README.md`
  - `extern/TensorRT-Model-Optimizer/docs/source/guides/_onnx_quantization.rst`
  - `extern/TensorRT-Model-Optimizer/docs/source/guides/_basic_quantization.rst`
  - `extern/TensorRT-Model-Optimizer/docs/source/guides/_choosing_quant_methods.rst`
  - `extern/TensorRT-Model-Optimizer/examples/onnx_ptq/README.md`
  - `extern/TensorRT-Model-Optimizer/examples/onnx_ptq/image_prep.py`
  - `extern/TensorRT-Model-Optimizer/examples/onnx_ptq/evaluate.py`
- TensorRT official docs for explicit quantization and best practices:
  - https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html
- This repo’s task context:
  - `context/tasks/working/quantize-yolo11-by-modelopt/subtask-001-101-modelopt-docs-and-apis.md` (detailed notes and findings)
  - `context/tasks/working/task-quantize-yolo11-by-modelopt.md` (overall plan and follow-up subtasks)
