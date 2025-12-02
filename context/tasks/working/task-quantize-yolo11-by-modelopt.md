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
- [x] Job-001-102: Complete subtask 1.2: Verify `pixi`/ModelOpt environment and review the `models/yolo11/` structure.
- [x] Job-001-103: Complete subtask 1.3: Select YOLO11 variant(s) and document success criteria for the quantization work.

### Status summary for Task 1

- ModelOpt docs and APIs: high-level findings and ONNX/CNN usage notes are summarized in `context/tasks/working/subtask-001-101-modelopt-docs-and-apis.md`, which now points to the detailed how-to and conceptual notes under `context/summaries/modelopt-kb/`.
- Environment and YOLO11 wiring: `nvidia-modelopt` is present and importable in the `pixi` environment, and the `models/yolo11/` layout (bootstrap, checkpoints, helpers, ONNX, src) is documented in `context/tasks/working/subtask-001-102-env-and-yolo11-structure.md`.
- Variants and success criteria: `yolo11n` (primary) and `yolo11s` (secondary) are selected as target checkpoints, with hardware assumptions (RTX 3090, 640×640, batch size 1) and quantitative accuracy/latency/memory criteria captured in `context/tasks/working/subtask-001-103-variants-and-success-criteria.md`.

## 2. Prepare YOLO11 ONNX export and calibration/evaluation data

Short description: Ensure we have reproducible ONNX exports and representative data for both calibration (INT8) and evaluation (accuracy/latency).

### Scope

Prepare YOLO11 ONNX exports for the chosen variants, validate that they load cleanly, assemble a calibration dataset for INT8 quantization, and implement a baseline evaluation path to measure FP32/FP16 accuracy and latency on the target hardware.

### Planned outputs

- Validated YOLO11 ONNX models for the selected variants (at least `yolo11n`, optionally `yolo11s`).
- Documented calibration dataset location/format suitable for ModelOpt ONNX PTQ.
- A baseline inference script (PyTorch, ONNXRuntime, or TensorRT) for computing mAP and related metrics.
- Baseline FP32/FP16 accuracy and latency measurements on the RTX 3090 for the chosen variants.

### Milestones (subtasks)

#### 2.1 Export and validate YOLO11 ONNX

Goal: Export YOLO11 checkpoints to ONNX for the chosen variants and verify that the resulting graphs load cleanly and are usable for TensorRT/ModelOpt workflows.

- Subtask spec: context/tasks/working/subtask-002-101-export-and-validate-yolo11-onnx.md

#### 2.2 Prepare calibration dataset for YOLO11

Goal: Assemble a representative calibration dataset (paths, preprocessing recipe, and storage format) that can be consumed by ModelOpt ONNX PTQ for INT8 calibration.

- Subtask spec: context/tasks/working/subtask-002-102-calibration-data-for-yolo11.md

#### 2.3 Implement baseline evaluation and metrics

Goal: Implement or adapt an inference script and use it to record baseline FP32/FP16 accuracy and latency metrics for the selected YOLO11 variants on the RTX 3090.

- Subtask spec: context/tasks/working/subtask-002-103-baseline-eval-and-metrics.md

### TODOs

- [ ] Job-002-101: Complete subtask 2.1: Export and validate YOLO11 ONNX for the chosen variants.
- [ ] Job-002-102: Complete subtask 2.2: Prepare calibration dataset and document its location/format.
- [ ] Job-002-103: Complete subtask 2.3: Implement baseline evaluation and record FP32/FP16 metrics.

## 3. Integrate ModelOpt ONNX quantization for INT8 and mixed-precision

Short description: Use ModelOpt ONNX tools to create INT8-quantized (Q/DQ) YOLO11 ONNX models configured for mixed FP16/INT8 execution in TensorRT.

### Scope

Prototype and refine the ModelOpt ONNX PTQ workflow for YOLO11, starting from a basic CLI invocation and progressing to a configuration that uses real calibration data and produces TensorRT-ready INT8/QDQ ONNX models with appropriate mixed-precision settings.

### Planned outputs

- A minimal, working ModelOpt ONNX PTQ invocation for YOLO11 ONNX.
- Integration of the calibration dataset from section 2 into the ONNX PTQ flow (CLI or Python driver).
- One or more quantized YOLO11 ONNX models (`*-int8-qdq.onnx`) with documented quantization settings suitable for TensorRT mixed FP16/INT8 execution.

### Milestones (subtasks)

#### 3.1 Prototype ONNX PTQ pipeline for YOLO11

Goal: Stand up a basic ModelOpt ONNX PTQ flow for YOLO11 using a simple CLI/Python invocation and placeholder/random calibration data to validate end-to-end behavior.

- Subtask spec: context/tasks/working/subtask-003-101-prototype-onnx-ptq-pipeline.md

#### 3.2 Wire real calibration data into ONNX PTQ

Goal: Replace placeholder data with the real calibration dataset from section 2 and ensure the PTQ process runs reliably with the expected shapes and preprocessing.

- Subtask spec: context/tasks/working/subtask-003-102-wire-calibration-into-onnx-ptq.md

#### 3.3 Generate quantized YOLO11 ONNX artifacts

Goal: Configure INT8/mixed-precision options and produce one or more quantized YOLO11 ONNX models, capturing the chosen settings for later TensorRT engine building.

- Subtask spec: context/tasks/working/subtask-003-103-generate-quantized-yolo11-onnx-artifacts.md

### TODOs

- [ ] Job-003-101: Complete subtask 3.1: Prototype the YOLO11 ONNX PTQ pipeline with ModelOpt.
- [ ] Job-003-102: Complete subtask 3.2: Integrate the real calibration dataset into ONNX PTQ.
- [ ] Job-003-103: Complete subtask 3.3: Generate and document quantized YOLO11 ONNX artifacts.

## 4. Build TensorRT engines and benchmark FP16/INT8 mixed-precision

Short description: Convert the (quantized) ONNX models into TensorRT engines that enable both FP16 and INT8, then measure latency and accuracy against the baseline.

### Scope

Establish a reproducible TensorRT engine-building path for YOLO11, create FP16 and mixed FP16/INT8 engines from the ONNX models, and benchmark them for latency, throughput, and accuracy using the evaluation setup from section 2.

### Planned outputs

- A chosen and documented TensorRT build approach (`trtexec` and/or Python APIs) wrapped in `pixi` commands.
- FP16-only and mixed FP16/INT8 TensorRT engines for the selected YOLO11 variants.
- Benchmark results (latency, throughput, memory usage, and accuracy) for each engine on the RTX 3090.

### Milestones (subtasks)

#### 4.1 Build FP16 baseline TensorRT engine(s)

Goal: Decide on the engine-building interface and build FP16-only TensorRT engines from the baseline YOLO11 ONNX models for use as comparison points.

- Subtask spec: context/tasks/working/subtask-004-101-build-fp16-baseline-engines.md

#### 4.2 Build mixed FP16/INT8 engines and integrate with evaluation

Goal: Build mixed FP16/INT8 TensorRT engines from the quantized ONNX models and extend the evaluation script to run these engines alongside the FP16 baseline.

- Subtask spec: context/tasks/working/subtask-004-102-build-mixed-engines-and-eval.md

#### 4.3 Collect and store TensorRT benchmark metrics

Goal: Run benchmarks for all relevant engines and store the resulting latency, throughput, memory, and accuracy metrics in a structured, reproducible format.

- Subtask spec: context/tasks/working/subtask-004-103-collect-trt-benchmark-metrics.md

### TODOs

- [ ] Job-004-101: Complete subtask 4.1: Build FP16 baseline TensorRT engines for YOLO11.
- [ ] Job-004-102: Complete subtask 4.2: Build mixed FP16/INT8 engines and integrate them into the evaluation script.
- [ ] Job-004-103: Complete subtask 4.3: Collect and store detailed TensorRT benchmark metrics.

## 5. Search for the best mixed FP16/INT8 configuration and document results

Short description: Systematically explore quantization and engine-building variants to find the best trade-off, then document the recommended configuration and how to reproduce it.

### Scope

Define and explore a manageable search space of quantization and engine-building configurations for YOLO11, then select the best-performing mixed FP16/INT8 setup that meets the success criteria and document how to reproduce it.

### Planned outputs

- A documented list of configurable knobs for ModelOpt/TensorRT in this project.
- An automated experiment script or pipeline that sweeps over a small configuration grid.
- A selected “best” configuration with recorded metrics and a concise reproduction/deployment guide.

### Milestones (subtasks)

#### 5.1 Design configuration search space

Goal: Identify the most impactful quantization and engine-building knobs and design a small, tractable experiment grid for YOLO11.

- Subtask spec: context/tasks/working/subtask-005-101-design-config-search-space.md

#### 5.2 Automate experiment grid execution

Goal: Implement an automated pipeline (e.g., a single `pixi run` entrypoint) that runs the quantize → build → benchmark loop across the configuration grid.

- Subtask spec: context/tasks/working/subtask-005-102-automate-experiment-grid.md

#### 5.3 Analyze results and write deployment guide

Goal: Analyze experiment results to select the best configuration and write a short guide describing how to regenerate the chosen quantized model and TensorRT engine.

- Subtask spec: context/tasks/working/subtask-005-103-analyze-results-and-write-guide.md

### TODOs

- [ ] Job-005-101: Complete subtask 5.1: Design the configuration search space and experiment grid.
- [ ] Job-005-102: Complete subtask 5.2: Automate grid execution and logging.
- [ ] Job-005-103: Complete subtask 5.3: Analyze results and document the recommended configuration and reproduction steps.
