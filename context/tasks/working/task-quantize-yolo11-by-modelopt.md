# Task: Quantize YOLO11 with NVIDIA ModelOpt

## What to do

- Use NVIDIA ModelOpt’s ONNX quantization tools together with TensorRT to generate, benchmark, and compare mixed FP16/INT8 quantized YOLO11 models from this repo’s YOLO11 assets, then select and document the best-performing configuration (latency vs. accuracy) for practical deployment.

## 1. Understand tooling, docs, and repo setup

Short description: Confirm how ModelOpt ONNX quantization works, how YOLO11 is wired in this repo, and what constraints we have (hardware, datasets, target metrics).

### Scope

Understand the ModelOpt ONNX quantization workflow (with a focus on CNN/ONNX usage relevant to YOLO11), verify that the local environment and `pixi` configuration are ready to run it, and clarify which YOLO11 variants and success criteria will guide the rest of the quantization work.

### Planned outputs

- A brief summary of how to use ModelOpt for ONNX/CNN quantization, based on docs and the `extern/TensorRT-Model-Optimizer` checkout.
- Confirmation that `nvidia-modelopt` is available in the `pixi` environment and can be imported.
- A concise overview of the `models/yolo11/` layout (checkpoints, helpers, ONNX exports, and source).
- A decision on which YOLO11 variant(s) to focus on (e.g., `yolo11n`, `yolo11s`).
- A documented set of success criteria: target hardware, acceptable mAP drop, and latency/throughput targets.

### Milestones (subtasks)

#### 1.1 Study ModelOpt docs and ONNX/CNN usage

Goal: Read the ModelOpt docs and examples (especially ONNX and CNN-related material) plus TensorRT best practices so we know the recommended way to run ONNX quantization for models like YOLO11, including any constraints or flags we should care about.

- Subtask spec and findings: context/tasks/working/subtask-001-101-modelopt-docs-and-apis.md

#### 1.2 Verify environment and YOLO11 repo wiring

Goal: Confirm that the `pixi` environment has `nvidia-modelopt` installed and usable, and that the `models/yolo11/` directory (checkpoints, helpers, ONNX, src) is correctly set up for later quantization and engine-building steps.

- Subtask spec: context/tasks/working/subtask-001-102-env-and-yolo11-structure.md

#### 1.3 Choose YOLO11 variants and define success criteria

Goal: Decide which YOLO11 checkpoint(s) we will use for experiments and write down concrete success criteria (hardware, metrics, acceptable accuracy drop) so later milestones have a clear optimization target.

- Subtask spec: context/tasks/working/subtask-001-103-variants-and-success-criteria.md

### TODOs

- [x] Job-001-101: Complete subtask 1.1: Study ModelOpt ONNX/CNN docs and TensorRT best practices, focusing on how to apply them to YOLO11 (see `context/tasks/working/subtask-001-101-modelopt-docs-and-apis.md` for the summary).
- [ ] Job-001-102: Complete subtask 1.2: Verify `pixi`/ModelOpt environment and review the `models/yolo11/` structure.
- [ ] Job-001-103: Complete subtask 1.3: Select YOLO11 variant(s) and document success criteria for the quantization work.

## 2. Prepare YOLO11 ONNX export and calibration/evaluation data

Short description: Ensure we have reproducible ONNX exports and representative data for both calibration (INT8) and evaluation (accuracy/latency).

- [ ] Job-002-001: Run or document `pixi run python models/yolo11/helpers/convert_to_onnx.py <model-name>` to generate `models/yolo11/onnx/<model-name>.onnx` for at least one YOLO11 checkpoint.
- [ ] Job-002-002: Verify the exported ONNX graph loads cleanly with `onnx` or `onnxruntime` (shape inference, no unsupported ops for TensorRT/ModelOpt).
- [ ] Job-002-003: Assemble a small but representative calibration dataset (e.g., subset of COCO or the project’s target dataset) and store its location/format in this repo’s context (without checking in large data).
- [ ] Job-002-004: Implement or identify an existing YOLO11 inference script (PyTorch or ONNXRuntime/TensorRT-based) to compute baseline accuracy metrics (mAP, precision/recall) on the evaluation subset.
- [ ] Job-002-005: Record baseline FP32/FP16 latency and accuracy for the chosen YOLO11 variant(s) using the evaluation script and the target hardware.

## 3. Integrate ModelOpt ONNX quantization for INT8 and mixed-precision

Short description: Use ModelOpt ONNX tools to create INT8-quantized (Q/DQ) YOLO11 ONNX models configured for mixed FP16/INT8 execution in TensorRT.

- [ ] Job-003-001: Prototype a simple CLI invocation using `python -m modelopt.onnx.quantization` on a YOLO11 ONNX model (initially with random calibration data) to validate that the workflow functions end-to-end.
- [ ] Job-003-002: Replace random data with a real calibration dataloader (using the calibration dataset from Milestone 2), wiring it either via ModelOpt’s CLI flags or a small Python driver script as recommended by the docs.
- [ ] Job-003-003: Configure quantization for INT8 activations/weights while keeping model IO in FP16/FP32 as appropriate for TensorRT-friendly mixed precision (per ModelOpt ONNX recommendations).
- [ ] Job-003-004: Explore ModelOpt ONNX options relevant to mixed precision (e.g., per-node calibration, shape overrides, handling of ops that should remain FP16) and capture chosen settings in a config file or documented CLI command.
- [ ] Job-003-005: Generate one or more quantized YOLO11 ONNX artifacts (e.g., `*-int8-qdq.onnx`) and store them under `models/yolo11/onnx/` with clear naming that encodes quantization settings.

## 4. Build TensorRT engines and benchmark FP16/INT8 mixed-precision

Short description: Convert the (quantized) ONNX models into TensorRT engines that enable both FP16 and INT8, then measure latency and accuracy against the baseline.

- [ ] Job-004-001: Decide whether to use `trtexec`, Python TensorRT APIs, or an existing helper in this repo for engine building, keeping the workflow reproducible via `pixi run` commands.
- [ ] Job-004-002: Build a pure FP16 TensorRT engine from the baseline YOLO11 ONNX model for comparison (FP16 kernels enabled, INT8 disabled).
- [ ] Job-004-003: Build TensorRT engines from the ModelOpt-quantized ONNX models with both FP16 and INT8 precisions enabled so TensorRT can choose the fastest kernel per layer (mixed precision).
- [ ] Job-004-004: Extend the evaluation script to run inference with the TensorRT engines (FP16-only vs. FP16+INT8 mixed) and compute accuracy metrics on the evaluation dataset.
- [ ] Job-004-005: Collect detailed performance data (latency per image, throughput, GPU memory usage) for each engine on the target hardware and store results in a structured format (e.g., JSON or CSV).

## 5. Search for the best mixed FP16/INT8 configuration and document results

Short description: Systematically explore quantization and engine-building variants to find the best trade-off, then document the recommended configuration and how to reproduce it.

- [ ] Job-005-001: Identify key “knobs” to vary: calibration dataset size, per-layer quantization toggles (leave some layers FP16), TensorRT builder flags, and any ModelOpt-specific ONNX options that impact accuracy/latency.
- [ ] Job-005-002: Design a small experiment grid (or script) that sweeps over a manageable set of configurations (e.g., different calibration sizes and selective de-quantization of sensitive layers).
- [ ] Job-005-003: Automate running the experiment grid end-to-end (quantize → build TensorRT engine → benchmark → log metrics) using a single entrypoint command via `pixi run`.
- [ ] Job-005-004: Analyze results to identify the configuration(s) that meet the pre-defined success criteria (accuracy drop, latency, resource usage) and select a “best” mixed FP16/INT8 setup.
- [ ] Job-005-005: Write a brief deployment and reproduction guide (linking from this task file to a doc or README section) that explains how to regenerate the chosen quantized YOLO11 model and TensorRT engine on a fresh machine.
