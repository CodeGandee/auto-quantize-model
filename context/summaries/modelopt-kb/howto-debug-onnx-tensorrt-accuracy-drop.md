Howto: Debug ONNX / TensorRT accuracy drops for ModelOpt QDQ / INT8 models

## HEADER
- **Purpose**: Provide a practical checklist and tooling hints for diagnosing large accuracy drops when moving from a framework model to ONNX and TensorRT, especially for ModelOpt-generated QDQ / INT8 models.
- **Status**: Draft, based on common NVIDIA / ONNX practices and TensorRT docs
- **Date**: 2025-12-04
- **Owner**: AI assistant (Codex CLI)
- **Scope**: FP32 → ONNX → QDQ / INT8 ONNX (ModelOpt) → TensorRT engine; applicable to classification, detection, and transformer models.
- **Sources**:
  - TensorRT Best Practices: https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html
  - TensorRT Q/DQ docs (“Working with Quantized Types”): https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html
  - ONNX Runtime quantization and debugging (`qdq_loss_debug`): https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
  - NVIDIA / community threads on ONNX–TensorRT INT8 mismatches (GitHub issues, NVIDIA forums)

## 1. Establish the comparison pipeline

When accuracy drops, you want to locate the first stage where predictions diverge. A common pipeline is:

1. Framework FP32 (PyTorch / TensorFlow baseline).
2. ONNX FP32 (exported model, no QDQ).
3. ONNX QDQ / INT8 (ModelOpt output, evaluated with ONNX Runtime).
4. TensorRT FP16/FP32 engine (from FP32 ONNX).
5. TensorRT INT8 / mixed engine (from QDQ ONNX).

The goal is to measure accuracy and output similarity at each stage so you can say, for example:

- “FP32 ONNX matches PyTorch, but QDQ ONNX already collapses” → issue is in quantization.
- “QDQ ONNX is fine, only TensorRT INT8 is bad” → issue is in TRT parsing/fusion or TRT-specific behavior.

For detection models like YOLO, always compare pre-NMS logits or box/class scores (same thresholding) and not just post-NMS metrics, to avoid masking issues.

## 2. Sanity-check preprocessing and evaluation

Before blaming quantization or TensorRT, confirm basic consistency:

- **Inputs**:
  - Same resize / letterbox strategy (e.g., 640×640 with symmetric padding).
  - Same color order (BGR vs RGB), normalization ([0,1] vs mean/std) and datatype.
  - Same batch size and dynamic shape settings.
- **Outputs / decoding**:
  - Same output tensor interpretation (e.g., YOLO `[..., x, y, w, h, cls_scores...]`).
  - Same confidence thresholds and NMS parameters.
  - Same label mapping / class ordering.
- **Evaluation**:
  - Same dataset split (e.g., COCO2017 val).
  - Same IoU thresholds and mAP definition (COCO metrics vs custom).

Quick check: run a small subset (e.g., 50–100 images) through the framework, ONNX (ONNX Runtime), and TensorRT baseline FP16 engines using the same preprocessing and evaluation code. If FP16 TensorRT is already off, fix preprocessing / decoding before looking at INT8.

## 3. Compare outputs numerically at each stage

Once basic consistency is established, compare raw outputs for a small set of inputs (calibration samples or a small validation subset).

### 3.1 Framework vs FP32 ONNX

Export ONNX, then for a few images:

- Run PyTorch model → `y_ref`.
- Run ONNX via ONNX Runtime → `y_onnx`.
- Compute differences: MSE, max absolute error, cosine similarity.

Example (classification-like output):

```python
import numpy as np

def compare_logits(y_ref: np.ndarray, y_other: np.ndarray) -> None:
    y_ref = y_ref.reshape(-1)
    y_other = y_other.reshape(-1)
    diff = y_ref - y_other
    mse = float(np.mean(diff**2))
    max_abs = float(np.max(np.abs(diff)))
    num = float(np.dot(y_ref, y_other))
    den = float(np.linalg.norm(y_ref) * np.linalg.norm(y_other) + 1e-12)
    cos = num / den
    print(f"MSE={mse:.3e}, max|Δ|={max_abs:.3e}, cos={cos:.6f}")
```

If FP32 ONNX already diverges, fix export (dynamic axes, opset, custom ops) before touching quantization or TensorRT.

### 3.2 FP32 ONNX vs QDQ / INT8 ONNX

Use ONNX Runtime for both models:

- Run both models on the same batch of inputs.
- Compare raw outputs numerically (as above).
- For detection, compare the per-anchor logits before NMS, not only final boxes.

ONNX Runtime’s `qdq_loss_debug` utilities (documented in the ORT quantization guide) can automate per-node difference analysis: they run the FP32 and QDQ graphs side by side and compute per-layer errors to pinpoint sensitive nodes.

### 3.3 QDQ / INT8 ONNX vs TensorRT INT8 engine

Once QDQ ONNX accuracy looks acceptable, compare its outputs vs TensorRT:

- Drive both ONNX Runtime and TensorRT from the same preprocessed tensors.
- For each sample, compare the raw output tensor (same shape, before NMS).

If ONNX QDQ and TensorRT INT8 outputs match closely but mAP differs, the bug is likely in downstream postprocessing (NMS/decoding) or evaluation code. If they differ significantly, focus on TensorRT parsing and layer behaviors.

## 4. Use layer-wise / subgraph tools (Polygraphy, ONNX Runtime debug, ModelOpt)

For deeper analysis, you generally want per-layer or subgraph comparisons using the ecosystem tools rather than rolling everything by hand.

### 4.1 Polygraphy (TensorRT + ONNX, PyPI: `polygraphy`)

Polygraphy can run inference across multiple backends (TensorRT, ONNX Runtime, etc.) and compare outputs:

- Full-model comparison example (TRT vs ONNX Runtime):

```bash
polygraphy run model-int8-qdq.onnx \
  --onnxrt \
  --trt \
  --trt-logger-severity=warning \
  --save-inputs inputs.json \
  --save-outputs outputs.json \
  --compare-outputs
```

This:

- Uses ONNX Runtime and TensorRT on the same ONNX model.
- Saves inputs and outputs.
- Reports discrepancies between backends.

If full-model outputs diverge, you can then use Polygraphy’s debug tools:

- `polygraphy debug accuracy`:
  - Compares layerwise outputs between a “golden” backend (e.g., ONNX Runtime) and TensorRT.
  - Helps find the first layer where TensorRT deviates significantly.
- `polygraphy surgeon`:
  - Extracts subgraphs around problematic layers so you can test them independently.
- `polygraphy debug reduce`:
  - Automatically reduces a failing ONNX model to a minimal reproducer by iteratively removing nodes and checking whether the failure persists.

Typical pattern:

1. Use `polygraphy run` to confirm ONNX vs TensorRT accuracy mismatch.
2. Use `polygraphy debug accuracy` to locate the first bad layer.
3. Use `polygraphy surgeon` / `debug reduce` to isolate and minimize the failing region for closer inspection.

### 4.2 ONNX Runtime QDQ debug (PyPI: `onnxruntime`)

ONNX Runtime’s quantization tools include `qdq_loss_debug`, which can:

- Run FP32 and QDQ/INT8 ONNX models side by side.
- Compute per-node loss metrics (MSE, cosine similarity, etc.).

High-level usage (conceptual):

- Provide:
  - The FP32 ONNX model.
  - The QDQ/INT8 ONNX model.
  - A calibration dataset (e.g., the same `.npy` used for quantization or a list of images).
- Run the debug script to:
  - Generate a report of per-node errors.
  - Identify nodes whose outputs differ the most between FP32 and QDQ models.

This is particularly useful before involving TensorRT at all: if QDQ ONNX is already wrong, you can fix quantization (layer exclusions, calibration, etc.) without worrying about TRT-specific issues.

### 4.3 ModelOpt introspection

ModelOpt emits useful information during quantization:

- Which ops are quantized vs skipped.
- How many nodes fall into each precision category (e.g., `int8` vs `fp32_fp16`).
- Any non-quantizable partitions it detected.

If you serialize or summarize these (as done in `datasets/quantize-calib/yolo11n-int8-scheme/precision-scheme.md` in this repo), you can:

- Quickly see if critical layers (detection heads, layer norms, etc.) have been quantized when they should not be.
- Check whether certain ops (e.g., Resize, Concat) were excluded as intended.

Combined, Polygraphy + ONNX Runtime QDQ debug + ModelOpt logs let you:

- Confirm where outputs start to diverge (framework vs QDQ ONNX vs TRT).
- See which layers and ops are involved in the problematic region.
- Adjust quantization schemes accordingly.

## 5. TensorRT-specific debugging practices

When ONNX QDQ outputs look reasonable but TensorRT INT8 output is bad, focus on TensorRT:

### 5.1 Inspect engine structure and layer precisions

Use `trtexec` with detailed reporting:

```bash
trtexec \
  --onnx=model-int8-qdq.onnx \
  --saveEngine=model-int8.plan \
  --int8 --fp16 \
  --best \
  --exportLayerInfo=layer_info.json \
  --profilingVerbosity=detailed
```

In `layer_info.json` (or the console logs), check:

- Are compute-heavy layers (Conv, MatMul) running in INT8 as expected?
- Are there unexpected FP32 fallbacks or extra reformat layers around sensitive ops?
- Are Q/DQ nodes fused into compute layers, or do you see standalone “scale”/“quantize” layers consuming time?

### 5.2 Check for unsupported or partially supported ops

In TensorRT logs:

- Look for warnings about unsupported ops or fallback paths (e.g., to slower kernels or higher precision).
- For custom ops / plugins, ensure:
  - Plugins support INT8 (if required).
  - Plugin libraries are loaded and compatible with the TensorRT version.

Sometimes, accuracy issues arise because some part of the network is effectively running in FP32 or with different semantics than the framework (e.g., different padding rules, different reduction orders).

### 5.3 Enforce or relax type constraints

Depending on your experiment, you may:

- Use `--int8 --fp16` to allow mixed precision and let TensorRT choose.
- Use more restrictive flags (like `--stronglyTyped` in some workflows) to force TensorRT to follow the QDQ types exactly, helpful for analyzing quantization behavior at the cost of flexibility.

Experimenting with these flags, plus inspecting `layer_info.json`, can reveal if TensorRT is making unexpected precision choices that hurt accuracy.

## 6. Common root causes of large accuracy drops

In practice, big drops usually come from a handful of issues:

1. **Pre/post-processing mismatch**:
   - Image scaling, normalization, color channel order, or letterbox math differs between calibration/inference and the original training setup.
   - NMS thresholds or decode logic differ between baseline and quantized/inference code.

2. **Over-quantization of sensitive layers**:
   - Detection heads, classifier heads, attention/qkv projections, or layer norms quantized to INT8 when they should remain FP16/FP32.
   - Residual/Sigma-like paths where quantization noise accumulates heavily.

3. **Poor calibration or range selection**:
   - Calibration set too small or not representative.
   - Calibration method (e.g., simple max) not robust to outliers for the given model.
   - Activation ranges too tight or too wide, causing saturation or loss of small-but-important values.

4. **Mismatched Q/DQ semantics between tools**:
   - Framework expects fake-quant behavior but ONNX/TensorRT interprets Q/DQ differently.
   - Dynamic ranges or zero-points differ between framework export and ModelOpt/ONNX quantization.

5. **TensorRT-specific limitations or bugs**:
   - Specific ops or patterns suboptimally handled in certain TensorRT versions.
   - Differences between ONNX Runtime and TensorRT in interpreting ONNX graphs (broadcasting, padding, reductions).

## 7. ModelOpt-focused practices for improving INT8 accuracy

When using ModelOpt ONNX PTQ (`modelopt.onnx.quantization.quantize`):

- **Control which ops are quantized**:
  - Use `op_types_to_exclude` / `nodes_to_exclude` to keep known-sensitive layers (e.g., detection heads, some attention blocks) in FP16/FP32.
  - Start with backbone-only quantization and incrementally add more layers to INT8.
- **Tune calibration**:
  - Use a representative calibration set (same distribution as deployment).
  - Consider trying different `calibration_method`s (e.g., `entropy` vs `max`) for some models.
  - Increase the number of calibration samples if ranges look noisy.
- **Experiment with `dq_only`**:
  - With `dq_only=True`, ModelOpt may simplify some paths while still conveying quantization semantics; this can sometimes make TensorRT’s job easier, at the cost of losing explicit Q nodes.
- **Leverage comparison scripts**:
  - Build small utilities that run baseline ONNX and quantized ONNX side by side on the calibration set and record metric summaries (MSE, SQNR, cosine similarity).
  - Use these to spot problematic layers before going to TensorRT.

In a repository like this one, it often helps to:

- Reuse existing ONNX eval scripts (e.g., `scripts/yolo11/eval_yolo11_onnx_coco.py`) as the ground truth for mAP.
- Add parallel TensorRT evaluation scripts (as we did for YOLO11n) to ensure ONNX vs TRT differences can be isolated cleanly.

## 8. Recommended step-by-step workflow

If you see a large accuracy drop moving to TensorRT or INT8, follow this sequence:

1. **Verify FP16 TensorRT vs FP32 ONNX / framework**:
   - Same preprocessing, decoding, and evaluation.
   - If FP16 is bad, fix that first.
2. **Quantize with ModelOpt and evaluate QDQ ONNX with ONNX Runtime**:
   - If QDQ ONNX mAP is already collapsed, iterate on quantization config (exclusions, calibration, methods).
3. **Compare QDQ ONNX vs TensorRT INT8 outputs**:
   - If they differ numerically, inspect TensorRT logs, layer info, and unsupported ops.
4. **Use per-layer / subgraph comparison tools**:
   - Polygraphy, ORT `qdq_loss_debug`, or custom scripts to identify the first misbehaving layer.
5. **Adjust quantization scheme and rebuild**:
   - Exclude sensitive layers from INT8.
   - Tune calibration or quantization parameters.
   - Rebuild TensorRT engines and re-run evaluation.
6. **Document the final working configuration**:
   - Record which layers stay FP16/FP32, what calibration settings work, and which TensorRT flags are used.

This process is iterative but, once done, typically yields a stable mixed-precision scheme that maintains most of the FP16 accuracy while benefiting from INT8 speedups.
