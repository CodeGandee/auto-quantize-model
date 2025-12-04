Howto: Build mixed-precision TensorRT engines from QDQ ONNX models

## HEADER
- **Purpose**: Show how to take a QDQ (QuantizeLinear/DequantizeLinear) ONNX model (for example produced by ModelOpt ONNX PTQ or QAT) and build a mixed-precision (INT8 + FP16/FP32) TensorRT engine.
- **Status**: Draft, based on TensorRT docs and ModelOpt ONNX PTQ behavior
- **Date**: 2025-12-04
- **Owner**: AI assistant (Codex CLI)
- **Scope**: QDQ ONNX → TensorRT explicit quantization path; focuses on INT8 + FP16 mixed precision but also applies conceptually to other low-precision formats when supported.
- **Sources**:
  - TensorRT “Working with Quantized Types”: https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html
  - TensorRT overview (explicit vs implicit quantization, ONNX): https://docs.nvidia.com/deeplearning/tensorrt/latest/architecture/architecture-overview.html
  - ModelOpt ONNX PTQ guide: `extern/TensorRT-Model-Optimizer/docs/source/guides/_onnx_quantization.rst`

## 1. What “QDQ ONNX → mixed-precision TensorRT” actually means

- A **QDQ ONNX model** is an ONNX graph that uses `QuantizeLinear` (Q) and `DequantizeLinear` (DQ) operators to represent quantization, typically exported from:
  - Quantization-aware training (QAT) in PyTorch or TensorFlow, or
  - Post-training quantization (PTQ) like ModelOpt ONNX PTQ.
- When TensorRT parses ONNX:
  - `QuantizeLinear` becomes an `IQuantizeLayer`.
  - `DequantizeLinear` becomes an `IDequantizeLayer`.
  - TensorRT treats those as **explicit quantization** cues and propagates Q/DQ to maximize low-precision coverage while preserving arithmetic semantics.
- A **mixed-precision engine** here means:
  - Layers inside Q/DQ “islands” run in INT8 (or another quantized format supported by TensorRT).
  - Other layers without Q/DQ (or explicitly excluded from quantization) run in FP16 or FP32.
  - You enable multiple precisions in the builder (INT8 + FP16), and TensorRT chooses the best kernels per layer.

In this project, ModelOpt ONNX PTQ typically produces `*-int8-qdq.onnx` that are ready to feed into TensorRT’s explicit quantization pipeline.

## 2. Quick path with `trtexec` (CLI only)

Assume you already have a QDQ ONNX file:

- Example from ModelOpt:
  - `models/yolo11/onnx/yolo11n-int8-qdq.onnx`

You can directly build a mixed-precision engine with TensorRT’s `trtexec`:

```bash
trtexec \
  --onnx=models/yolo11/onnx/yolo11n-int8-qdq.onnx \
  --saveEngine=models/yolo11/trt/yolo11n-int8-mixed.plan \
  --int8 \
  --fp16 \
  --best
```

Key points:

- `--int8` enables INT8 kernels and tells TensorRT to use the Q/DQ information in the ONNX graph (no calibrator needed when Q/DQ comes from QAT/PTQ).
- `--fp16` allows non-quantized layers (or layers that TensorRT chooses not to run in INT8) to run in FP16 for additional speedup.
- `--best` lets TensorRT spend more time searching for optimal tactics; you can drop it for faster builds if needed.

You usually also want a **pure FP16 baseline** for comparison:

```bash
trtexec \
  --onnx=models/yolo11/onnx/yolo11n.onnx \
  --saveEngine=models/yolo11/trt/yolo11n-fp16.plan \
  --fp16
```

Then run your evaluation script to compare:

- Latency / throughput: FP16 vs INT8-mixed.
- Accuracy (mAP or task-specific metric): FP16 vs INT8-mixed.

Useful `trtexec` debug flags:

- `--dumpLayerInfo` to inspect which layers run in which precision.
- `--separateProfileRun` when using multiple optimization profiles.

## 3. Python API path (fine-grained control)

If you need more control (e.g., custom tactic sources, per-layer constraints, or integrating into an application), you can replicate the above in Python using the TensorRT API.

Minimal example:

```python
import tensorrt as trt

logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(network_flags)
parser = trt.OnnxParser(network, logger)

qdq_onnx_path = "models/yolo11/onnx/yolo11n-int8-qdq.onnx"
with open(qdq_onnx_path, "rb") as f:
    model_bytes = f.read()
if not parser.parse(model_bytes):
    for i in range(parser.num_errors):
        print(parser.get_error(i))
    raise RuntimeError("Failed to parse QDQ ONNX")

config = builder.create_builder_config()
config.max_workspace_size = 4 << 30  # 4 GiB, adjust to your GPU

# Enable mixed precision: explicit INT8 + FP16
config.set_flag(trt.BuilderFlag.INT8)
config.set_flag(trt.BuilderFlag.FP16)

# For QDQ/QAT/PTQ models, no calibrator is needed:
config.int8_calibrator = None

engine = builder.build_engine(network, config)
with open("models/yolo11/trt/yolo11n-int8-mixed.plan", "wb") as f:
    f.write(engine.serialize())
```

Notes:

- `NetworkDefinitionCreationFlag.EXPLICIT_BATCH` should be used for ONNX models with explicit batch dimensions (most modern exports).
- `BuilderFlag.INT8` + Q/DQ ONNX puts TensorRT into **explicit quantization** mode; calibration is unnecessary because quantization parameters are already baked into Q/DQ.
- `BuilderFlag.FP16` lets TensorRT choose FP16 kernels for layers not governed by INT8 Q/DQ, giving you mixed-precision behavior.

Once the engine is built, you can run inference either via:

- Plain TensorRT runtime bindings, or
- Higher-level frameworks (e.g., Triton Inference Server) that load the `.plan` engine.

## 4. How Q/DQ controls which layers are INT8 vs higher precision

From TensorRT’s “Working with Quantized Types” documentation:

- ONNX represents quantization explicitly: `Q` → quantization, `DQ` → dequantization.
- During optimization, TensorRT **propagates Q/DQ**:
  - Moves Q nodes backward so quantization happens as early as possible.
  - Moves DQ nodes forward so dequantization happens as late as possible.
  - Fuses and reorders Q/DQ with commuting operators to maximize low-precision coverage.
- Result:
  - Layers surrounded by Q/DQ run in INT8.
  - Layers outside Q/DQ islands stay in FP16/FP32 (depending on enabled builder flags).

If you want explicit control over which layers remain in higher precision:

- At the ONNX / ModelOpt stage:
  - Use ModelOpt’s `nodes_to_exclude` / `op_types_to_exclude` when generating the QDQ ONNX so that sensitive layers are left unquantized.
  - This is usually the easiest and most portable way to enforce “FP16-only” or “FP32-only” regions.
- At the TensorRT stage:
  - Advanced users can use per-layer precision constraints via the TensorRT C++/Python API (for example, setting `layer.precision` and/or enabling precision constraints in the builder config). This is more involved and generally not necessary if Q/DQ and ModelOpt are configured correctly.

## 5. Verifying Q/DQ Fusion

TensorRT automatically fuses Q/DQ nodes into compute layers (like Convolution or GEMM) during the engine build process. You do not need a special flag to "remove" them; this is the fundamental way TensorRT processes Q/DQ networks.

To verify that this fusion is happening and that Q/DQ nodes are not being executed as standalone kernels (which would be slow), you should inspect the engine's layer information.

**Using `trtexec`:**

Add the `--exportLayerInfo` and `--profilingVerbosity=detailed` flags to your command:

```bash
trtexec \
  --onnx=models/yolo11/onnx/yolo11n-int8-qdq.onnx \
  --saveEngine=models/yolo11/trt/yolo11n-int8-mixed.plan \
  --int8 \
  --fp16 \
  --best \
  --exportLayerInfo=layer_info.json \
  --profilingVerbosity=detailed
```

**How to interpret the JSON output:**

Open the generated `layer_info.json` file and look for the `Layers` section.

*   **Good Sign (Fusion):** You see compute layers (e.g., `Convolution`, `Gemm`) with `precision: INT8`. This means the Q/DQ nodes were successfully fused, and the operation is running in INT8.
*   **Bad Sign (No Fusion):**
    *   You see many standalone `Scale` or `ElementWise` layers that correspond to your Q/DQ nodes.
    *   You see compute layers running in `FP32` or `FP16` surrounded by reformatting nodes (e.g., `Reformat`).

**Common reasons for fusion failure:**
1.  **Unsupported Patterns:** The graph structure (e.g., specific activations between Conv and Q/DQ) prevents fusion.
2.  **Mismatched Scales:** The scales of the Q and DQ nodes do not match or are not constant, preventing TensorRT from collapsing them.

## 6. Practical checklist for this repo

For QDQ ONNX models produced by ModelOpt (e.g., YOLO11):

1. **Generate QDQ ONNX with ModelOpt ONNX PTQ**
   - Use the recipe in `context/summaries/modelopt-kb/howto-modelopt-onnx-ptq-for-yolo11.md` to get `*-int8-qdq.onnx`.
   - Ensure you exclude numerically sensitive ops/layers as needed (e.g., Resize/Concat or detection heads) so they stay in FP16/FP32.
2. **Build mixed-precision TensorRT engine**
   - Using `trtexec`:
     - `trtexec --onnx=<qdq.onnx> --int8 --fp16 --saveEngine=<engine.plan> [--best]`.
   - Or via Python as shown above.
3. **Validate**
   - Compare FP16 and INT8-mixed engines on:
     - Latency / throughput at your target batch size and input resolution.
     - Task accuracy (mAP for object detection, etc.).
   - Inspect layer precisions with `--dumpLayerInfo` (CLI) or TensorRT’s layer inspector API if you suspect unexpected FP32/FP16 fallbacks.

## 7. References and further reading

- TensorRT explicit quantization and Q/DQ behavior:
  - https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html
- TensorRT architecture overview, including ONNX parsing:
  - https://docs.nvidia.com/deeplearning/tensorrt/latest/architecture/architecture-overview.html
- ModelOpt ONNX PTQ guide and examples in this repo:
  - `extern/TensorRT-Model-Optimizer/docs/source/guides/_onnx_quantization.rst`
  - `extern/TensorRT-Model-Optimizer/examples/onnx_ptq/README.md`
  - `context/summaries/modelopt-kb/howto-modelopt-onnx-ptq-for-yolo11.md`

