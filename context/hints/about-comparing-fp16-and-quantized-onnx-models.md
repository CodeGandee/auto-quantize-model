# About Comparing FP16 and Quantized ONNX Models

This guide covers metrics, methodologies, and tools for evaluating quantization error when comparing original FP16 ONNX models with quantized (Q/DQ) models, with a focus on NVIDIA ModelOpt workflows.

## Overview

Quantization converts high-precision models (FP32/FP16) to lower-precision formats (INT8/FP8/INT4) to reduce model size, memory bandwidth, and increase inference throughput. However, this introduces quantization error that must be carefully evaluated to ensure acceptable accuracy degradation.

## Key Evaluation Metrics

### 1. Task-Specific Accuracy Metrics

These are the most important metrics as they reflect real-world model performance:

**Object Detection Models:**
- **mAP (mean Average Precision)**: Primary metric averaging precision-recall across all classes
  - mAP@0.50: IoU threshold of 50% (more lenient)
  - mAP@0.75: IoU threshold of 75% (stricter)
  - mAP@[0.50:0.95]: COCO-style average across multiple IoU thresholds (0.50 to 0.95 in 0.05 steps)
- **Per-class AP**: Identify which classes are most affected by quantization
- **Precision/Recall**: Trade-off between false positives and false negatives
- **F1 Score**: Harmonic mean of precision and recall

**Classification Models:**
- **Top-1 Accuracy**: Percentage of correct predictions
- **Top-5 Accuracy**: Whether correct class is in top 5 predictions

**Segmentation Models:**
- **IoU (Intersection over Union)**: Overlap between predicted and ground truth masks
- **Dice Coefficient**: Similarity measure for segmentation quality

### 2. Layer-Level and Activation Metrics

These metrics help identify which layers are most affected by quantization:

**SQNR (Signal-to-Quantization-Noise Ratio)**
- Measures ratio of signal power to quantization noise power
- Expressed in dB: `SQNR_dB = 10 * log10(P_signal / P_noise)`
- Higher SQNR indicates better quantization quality
- Can be calculated per-layer to identify problematic layers
- Formula: `SQNR = E[x²] / E[(x - x_quantized)²]`

**QDQ Error**
- Measures difference between original model and quantize-dequantize (Q/DQ) nodes running in FP32
- Simulates quantization effect without full INT8 execution
- Useful for identifying layers that don't tolerate quantization well

**Cosine Similarity**
- Measures angular similarity between original and quantized activations
- Range: -1 to 1 (1 = perfect similarity, 0 = orthogonal, -1 = opposite)
- Formula: `cos_sim = (A · B) / (||A|| * ||B||)`
- Less sensitive to magnitude differences than MSE

**MSE (Mean Squared Error)**
- Average squared difference between original and quantized outputs
- Formula: `MSE = (1/n) * Σ(y_orig - y_quant)²`
- Very sensitive to outliers and magnitude differences

### 3. Output-Level Comparison Metrics

**PSNR (Peak Signal-to-Noise Ratio)**
- Commonly used for image quality assessment
- Formula: `PSNR = 10 * log10(MAX² / MSE)` where MAX is maximum pixel value
- Higher PSNR indicates better quality (typically > 30 dB is good)
- Normalized against maximum signal value, making it comparable across different images

**SNR (Signal-to-Noise Ratio)**
- Ratio of signal power to noise power
- Formula: `SNR = 10 * log10(P_signal / P_noise)`
- Depends on input signal magnitude, less comparable across different inputs than PSNR

**Per-Tensor Statistics**
- Min/Max/Mean/Std differences between FP16 and quantized outputs
- Useful for understanding distribution shifts

## Common Evaluation Workflow

### 1. Baseline FP16 Model Evaluation

```python
# Evaluate original FP16 model on validation dataset
fp16_metrics = evaluate_model(fp16_model, validation_data)
print(f"FP16 mAP@0.50:0.95: {fp16_metrics['map']}")
print(f"FP16 mAP@0.50: {fp16_metrics['map50']}")
```

### 2. Quantization with NVIDIA ModelOpt

```python
import modelopt.onnx.quantization as moq
import numpy as np

# Prepare calibration data (typically 100-1000 samples)
calibration_data = np.load("calibration_data.npy")

# Quantize model (PTQ - Post-Training Quantization)
moq.quantize(
    onnx_path="model_fp16.onnx",
    calibration_data=calibration_data,
    output_path="model_int8_qdq.onnx",
    quantize_mode="int8",  # or "fp8", "int4"
    calibration_method="entropy",  # or "max" for int8/fp8
)
```

### 3. Quantized Model Evaluation

```python
# Evaluate quantized model on same validation dataset
int8_metrics = evaluate_model(int8_model, validation_data)
print(f"INT8 mAP@0.50:0.95: {int8_metrics['map']}")
print(f"INT8 mAP@0.50: {int8_metrics['map50']}")

# Calculate degradation
degradation = fp16_metrics['map'] - int8_metrics['map']
relative_degradation = (degradation / fp16_metrics['map']) * 100
print(f"Accuracy drop: {degradation:.4f} ({relative_degradation:.2f}%)")
```

### 4. Layer-wise Analysis (for debugging)

```python
# Extract activations from both models
fp16_activations = collect_activations(fp16_model, calibration_data)
int8_activations = collect_activations(int8_model, calibration_data)

# Calculate SQNR per layer
for layer_name in fp16_activations.keys():
    fp16_act = fp16_activations[layer_name]
    int8_act = int8_activations[layer_name]
    
    signal_power = np.mean(fp16_act ** 2)
    noise_power = np.mean((fp16_act - int8_act) ** 2)
    sqnr_db = 10 * np.log10(signal_power / noise_power)
    
    print(f"{layer_name}: SQNR = {sqnr_db:.2f} dB")
```

## Acceptable Accuracy Thresholds

### Industry Standards

- **High-precision applications** (medical, autonomous driving): < 1% accuracy drop, mAP > 0.90
- **General object detection**: < 2-3% accuracy drop acceptable, mAP > 0.50 good, > 0.70 excellent
- **Real-time applications**: May accept 3-5% accuracy drop for speed requirements

### ModelOpt Best Practices

According to NVIDIA documentation and benchmarks:
- **PTQ (Post-Training Quantization)** typically achieves < 1-2% accuracy loss with proper calibration
- **QAT (Quantization Aware Training)** can recover accuracy to within 0.1-0.5% of FP16 baseline
- Use QAT when PTQ shows > 2% accuracy degradation

## Tools for Evaluation

### 1. NVIDIA ModelOpt Built-in Tools

```python
# ModelOpt provides calibration utilities
from modelopt.onnx.quantization import calib_utils

# Create calibrator
calibrator = calib_utils.Calibrator(
    model_path="model.onnx",
    calibration_data=calib_data,
    method="entropy"  # or "max"
)
```

### 2. ONNX Runtime Quantization Debugging

```python
from onnxruntime.quantization.qdq_loss_debug import (
    collect_activations,
    create_activation_matching,
    compute_activation_error
)

# Collect activations from both models
fp16_acts = collect_activations(fp16_model, data)
int8_acts = collect_activations(int8_model, data)

# Match corresponding layers
matched = create_activation_matching(fp16_acts, int8_acts)

# Compute errors
for name, (fp16_act, int8_act) in matched.items():
    error = compute_activation_error(fp16_act, int8_act)
    print(f"{name}: Error = {error}")
```

### 3. TensorRT Engine Analysis

```bash
# Build TensorRT engines for comparison
trtexec --onnx=model_fp16.onnx --fp16 --saveEngine=model_fp16.trt
trtexec --onnx=model_int8_qdq.onnx --int8 --saveEngine=model_int8.trt

# Benchmark performance
trtexec --loadEngine=model_fp16.trt --shapes=input:1x3x640x640
trtexec --loadEngine=model_int8.trt --shapes=input:1x3x640x640
```

### 4. TRT Engine Explorer (trex)

```python
# Visualize and analyze TensorRT engine structure
import trex
plan_fp16 = trex.EnginePlan("model_fp16.trt")
plan_int8 = trex.EnginePlan("model_int8.trt")

# Compare layer fusion and precision
plan_fp16.df  # DataFrame with layer information
plan_int8.df
```

## Common Issues and Solutions

### Issue 1: Large Accuracy Drop (> 5%)

**Diagnosis:**
- Check calibration data quality and diversity
- Analyze per-layer SQNR to find problematic layers
- Verify dynamic range is appropriate

**Solutions:**
- Increase calibration dataset size (100-1000 samples recommended)
- Try different calibration methods ("entropy" vs "max")
- Use selective quantization: keep sensitive layers in FP16
- Consider QAT (Quantization Aware Training) instead of PTQ

### Issue 2: INT8 Model Slower Than FP16

**Diagnosis:**
- Check if model is actually using INT8 compute (inspect with trex)
- Verify hardware has INT8 support (Turing, Ampere, Ada Lovelace, Hopper)
- Look for excessive Q/DQ node insertions

**Solutions:**
- Ensure TensorRT fuses Q/DQ nodes properly
- Remove unnecessary Q/DQ pairs
- Check that operators support INT8 execution

### Issue 3: Inconsistent Results Between Runs

**Diagnosis:**
- Check if calibration is deterministic
- Verify random seed is set
- Look for numerical instability in specific layers

**Solutions:**
```python
# Set deterministic behavior
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
```

## Calibration Methods Comparison

NVIDIA ModelOpt supports multiple calibration methods:

### For INT8/FP8:
- **entropy** (default): Uses KL-divergence to find optimal quantization range, generally better accuracy
- **max**: Uses min/max values from activations, simpler but may be affected by outliers

### For INT4:
- **awq_clip** (default): Activation-aware Weight Quantization with clipping
- **awq_lite**: Lighter version of AWQ
- **awq_full**: Full AWQ optimization
- **rtn_dq**: Round-to-nearest with dequantization

## Selective Quantization Strategy

When full quantization causes too much accuracy loss:

1. **Identify sensitive layers** using SQNR or QDQ error metrics
2. **Keep sensitive layers in FP16** while quantizing others
3. **Iteratively test** different layer combinations

```python
# Example: Exclude specific layers from quantization
moq.quantize(
    onnx_path="model.onnx",
    calibration_data=calib_data,
    output_path="model_mixed.onnx",
    nodes_to_exclude=["sensitive_layer1", "attention_head"],
    op_types_to_exclude_fp16=["LayerNorm", "Softmax"]
)
```

## Complete Evaluation Checklist

- [ ] Establish FP16 baseline metrics on full validation set
- [ ] Prepare calibration dataset (100-1000 representative samples)
- [ ] Quantize model with appropriate method (PTQ or QAT)
- [ ] Evaluate quantized model on same validation set
- [ ] Calculate accuracy degradation (absolute and relative)
- [ ] If degradation > 2%, perform layer-wise SQNR analysis
- [ ] Identify problematic layers and consider selective quantization
- [ ] Build TensorRT engines and verify performance gains
- [ ] Test edge cases and challenging samples
- [ ] Document final metrics and deployment configuration

## References

- [NVIDIA ModelOpt Documentation](https://nvidia.github.io/TensorRT-Model-Optimizer/)
- [ONNX Quantization Guide](https://nvidia.github.io/TensorRT-Model-Optimizer/guides/_onnx_quantization.html)
- [TensorRT Explicit Quantization](https://developer.nvidia.com/blog/achieving-fp32-accuracy-for-int8-inference-using-quantization-aware-training-with-tensorrt/)
- [ONNX Runtime Quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
- [Selective Quantization Paper](https://arxiv.org/html/2507.12196v1)
- [COCO Detection Metrics](https://cocodataset.org/#detection-eval)
