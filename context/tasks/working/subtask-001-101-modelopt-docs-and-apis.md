# Subtask 1.1: Study ModelOpt docs and ONNX/CNN usage

## Scope

Understand how NVIDIA ModelOpt (TensorRT Model Optimizer) supports ONNX and CNN-style models, with an emphasis on the quantization workflows and options that will be relevant for applying mixed FP16/INT8 quantization to YOLO11 in the **auto-quantize-model** project.

## Planned outputs

- A short written summary of the recommended ONNX quantization flow using ModelOpt (CLI and/or Python APIs).
- Notes on any special considerations for CNNs or object detection models (e.g., calibration data requirements, Q/DQ insertion, supported ops).
- A list of key CLI flags, config options, or example scripts that we expect to reuse for YOLO11 quantization.
- Links or references to the most relevant documents and example files in `extern/TensorRT-Model-Optimizer`.

## TODOs

- [x] Job-001-101-001: Read the main `extern/TensorRT-Model-Optimizer/README.md` to understand the overall scope and techniques offered by ModelOpt.
- [x] Job-001-101-002: Review ModelOpt documentation for ONNX quantization and CNN/object-detection-relevant examples (e.g., `examples/onnx_ptq` and associated docs).
- [x] Job-001-101-003: Identify and record the primary CLI and/or Python entrypoints we will likely use for ONNX quantization (including any calibration-related flags).
- [x] Job-001-101-004: Capture any constraints or recommendations from TensorRT “Best Practices” (especially around ONNX Q/DQ models and mixed precision) that affect how we quantize YOLO11.
- [x] Job-001-101-005: Summarize the findings in a brief note and link it from the main task file or a dedicated context doc so later milestones can reference it.

## Findings: ModelOpt ONNX/CNN usage for YOLO11

### Recommended ONNX PTQ flow (ModelOpt + TensorRT)

- Start from an FP32/FP16 ONNX export of YOLO11 (see `models/yolo11/helpers/convert_to_onnx.py`), with opset ≥ 13 for INT8 and ≥ 21 if we later explore FP8/INT4.
- Prepare calibration data as a `.npy` or `.npz` file matching the ONNX input signature. For CNN/ViT-style vision models, ModelOpt’s ONNX PTQ example recommends ≈500 images; for YOLO11 we should use a representative subset of the target detection dataset, preprocessed with the same resize/letterbox pipeline as inference.
- Run ModelOpt ONNX PTQ either via CLI or Python:

  - CLI (prototype shape, YOLO11 path to be filled in later):

    ```bash
    python -m modelopt.onnx.quantization \
      --onnx_path=<path-to-yolo11.onnx> \
      --quantize_mode=int8 \
      --calibration_data=calib.npy \
      --calibration_method=max \
      --output_path=<path-to-yolo11-int8-qdq.onnx>
    ```

  - Python API:

    ```python
    from modelopt.onnx.quantization import quantize

    quantize(
        onnx_path="path/to/yolo11.onnx",
        quantize_mode="int8",
        calibration_data="calib.npy",
        calibration_method="max",
        output_path="path/to/yolo11-int8-qdq.onnx",
    )
    ```

- Deploy by compiling the quantized ONNX into a TensorRT engine with mixed precision enabled (for example, via `trtexec --onnx=<quant.onnx> --saveEngine=<engine> --best` or an equivalent TensorRT Python builder flow).

### CNN / object-detection specific considerations

- Calibration data:
  - ModelOpt’s ONNX PTQ docs (`examples/onnx_ptq/README.md`) explicitly recommend ≥500 calibration samples for CNN/ViT models; for YOLO11 we should aim for a few hundred diverse images with realistic object scales, aspect ratios, and lighting, drawn from the same distribution as deployment.
  - Calibration arrays must match ONNX input names and shapes, using `.npy` (single tensor) or `.npz` (dict of input-name → array) as described in `docs/source/guides/_onnx_quantization.rst`.

- Q/DQ insertion and ops:
  - ModelOpt ONNX quantization generates explicit Q/DQ nodes in the ONNX graph following TensorRT’s Q/DQ rules. TensorRT then consumes these Q/DQ nodes to run quantized kernels where supported.
  - For YOLO-style detectors, we may want to keep some layers in higher precision (for example, IO tensors and certain detection-head / NMS-related ops) and quantize the bulk of the convolutional backbone and neck. ModelOpt supports this via `op_types_to_exclude`, `nodes_to_exclude`, and related options so we can selectively leave sensitive nodes in FP16/FP32.
  - If the exported YOLO11 ONNX includes TensorRT plugins (for example, NMS plugins), ONNX PTQ docs recommend using TensorRT Execution Provider during calibration (`--calibration_eps trt cuda:0 cpu`) and passing the plugin library via `--trt_plugins`/`trt_plugins` so custom ops run correctly during calibration and inference.

- Supported precisions:
  - ONNX PTQ currently supports FP8, INT8, INT4 (plus NVFP4/MXFP8 via the PyTorch→ONNX path). For YOLO11 mixed FP16/INT8, the primary path is `quantize_mode="int8"` and then enabling FP16+INT8 execution in TensorRT.

### Primary CLI and Python entrypoints for YOLO11 ONNX quantization

- CLI:

  - Module: `python -m modelopt.onnx.quantization`
  - Core arguments we expect to use for YOLO11:
    - `--onnx_path`: path to YOLO11 ONNX (e.g., `models/yolo11/onnx/yolo11n.onnx`).
    - `--output_path`: path for the quantized ONNX (e.g., `.../yolo11n-int8-qdq.onnx`).
    - `--quantize_mode=<int8|fp8|int4>`: precision mode; we will start with `int8`.
    - `--calibration_data=<calib.npy|calib.npz>`: calibration dataset in numpy form.
    - `--calibration_method=<max|entropy|awq_clip|rtn_dq>`: calibration algorithm; `max`/`entropy` are the basic options for INT8/FP8, `awq_clip`/`rtn_dq` are relevant for INT4.
    - `--calibration_eps=<cpu|cuda:0|trt ...>`: execution providers for calibration; for models with TensorRT plugins we should include `trt` and `cuda:0` with CPU fallback.
    - `--calibration_shapes` / `--override_shapes` / `--input_shapes_profile`: to fix static shapes or provide shape profiles when YOLO11 exports with dynamic dims.
    - `--calibrate_per_node`: optional; enables per-node calibration to reduce memory usage on large models.
    - `--op_types_to_quantize`, `--op_types_to_exclude`, `--nodes_to_exclude`, `--op_types_to_exclude_fp16`: control which ops stay quantized vs. FP16/FP32 (useful for keeping detection heads / output layers in higher precision).
    - `--trt_plugins`, `--trt_plugins_precision`: to register TensorRT plugins (e.g., YOLO NMS) during calibration.

- Python:

  - Function: `modelopt.onnx.quantization.quantize(...)` with signature documented in the online reference and surfaced via the ONNX PTQ README.
  - Key parameters for YOLO11:
    - `onnx_path`, `output_path`, `quantize_mode`, `calibration_data`, `calibration_method`.
    - `calibration_eps`, `calibration_shapes` / `override_shapes`, `input_shapes_profile` for dynamic shapes and mixed backends.
    - `op_types_to_quantize`, `op_types_to_exclude`, `nodes_to_exclude` for selectively quantizing backbone/neck vs. detector head and plugins.
    - `trt_plugins`, `trt_plugins_precision` to integrate TensorRT plugin libraries.

### TensorRT best-practice constraints relevant to YOLO11

- Explicit Q/DQ and mixed precision:
  - TensorRT documentation (“Working with Quantized Types”) recommends explicit quantization using Q/DQ nodes; implicit INT8 quantization is deprecated. ModelOpt’s ONNX PTQ path aligns with this by generating Q/DQ explicitly in ONNX.
  - Mixed precision in TensorRT is achieved by allowing multiple precisions (FP16 + INT8) during engine build. With explicit Q/DQ ONNX from ModelOpt, TensorRT uses Q/DQ placement plus builder flags (for example, `--best` or enabling both FP16 and INT8 in the builder) to choose per-layer kernels; we should rely on that rather than forcing all layers to INT8.

- Q/DQ placement and sensitive layers:
  - TensorRT’s Q/DQ best-practice guidance emphasizes that aggressive quantization of all layers (including those close to outputs) can hurt accuracy. For YOLO11, we should be prepared to:
    - Keep some late-stage detection layers, postprocessing, and NMS in FP16/FP32.
    - Quantize the main convolutional backbone/neck where most of the compute and memory bandwidth live.
  - ModelOpt’s `op_types_to_exclude` / `nodes_to_exclude` knobs give us a clean way to implement this selective strategy.

- Calibration:
  - TensorRT best practices highlight the importance of representative calibration data that matches deployment distributions and shapes. This is consistent with ModelOpt’s ONNX PTQ guidance; for YOLO11 we should avoid synthetic or overly small calibration sets except for smoke tests.

### Key local docs and examples to reuse

- ModelOpt overview:
  - `extern/TensorRT-Model-Optimizer/README.md`

- ONNX/PTQ guides:
  - `extern/TensorRT-Model-Optimizer/docs/source/guides/_onnx_quantization.rst`
  - `extern/TensorRT-Model-Optimizer/docs/source/guides/_basic_quantization.rst`
  - `extern/TensorRT-Model-Optimizer/docs/source/guides/_choosing_quant_methods.rst`

- ONNX PTQ examples:
  - `extern/TensorRT-Model-Optimizer/examples/onnx_ptq/README.md`
  - `extern/TensorRT-Model-Optimizer/examples/onnx_ptq/download_example_onnx.py`
  - `extern/TensorRT-Model-Optimizer/examples/onnx_ptq/image_prep.py`
  - `extern/TensorRT-Model-Optimizer/examples/onnx_ptq/torch_quant_to_onnx.py`
  - `extern/TensorRT-Model-Optimizer/examples/onnx_ptq/evaluate.py`

- External references:
  - NVIDIA docs: “Working with Quantized Types” and TensorRT best-practices pages (for Q/DQ recommendations and mixed-precision guidance).

These findings will be used by the main task `context/tasks/working/task-quantize-yolo11-by-modelopt.md` and subsequent subtasks to design the concrete YOLO11 ONNX quantization and TensorRT engine-building workflow.

## Notes

- Prioritize up-to-date docs (ModelOpt’s official site and the current `extern/TensorRT-Model-Optimizer` checkout) to avoid stale guidance.
- Focus on ONNX/PTQ material; LLM-specific content can be skimmed unless it reveals general quantization patterns that also apply to CNNs.
