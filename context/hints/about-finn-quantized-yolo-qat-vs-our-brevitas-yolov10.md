# About: QAT in `finn-quantized-yolo` vs our Brevitas+Ultralytics YOLOv10 QAT

This note explains (1) where quantization-aware training (QAT) lives for `extern/finn-quantized-yolo`, (2) what that repo actually does in-code, and (3) how it differs from our YOLOv10m Brevitas pipeline.

## What `finn-quantized-yolo` contains (and what it doesn’t)

`extern/finn-quantized-yolo` is primarily an evaluation + deployment repo for LPYOLO on FPGA (PYNQ-Z2 + FINN). Its README explicitly points training elsewhere:

- Training happens in: https://github.com/sefaburakokcu/quantized-yolov5 (linked from the `finn-quantized-yolo` README)
- This repo consumes pretrained artifacts: ONNX files and deploy.zip bundles containing `finn-accel.bit`, `finn-accel.hwh`, and `scale.npy`

In other words: the “QAT” claim is true for the project overall, but the QAT training loop is not implemented inside `finn-quantized-yolo` itself.

Source: https://github.com/sefaburakokcu/finn-quantized-yolo

## How quantization shows up in `finn-quantized-yolo` code

The deployment/inference path is very explicit about integer I/O and an external output scaling factor:

- `src/deploy/driver/driver.py` defines the accelerator IO contract:
  - Input dtype: `UINT8`
  - Output dtype: `INT8`
  - Input layout: NHWC `(1, 416, 416, 3)`
  - Output layout: NHWC `(1, 13, 13, 18)`
- `src/deploy/save_inference_results.py` prepares input as `uint8`, calls the FPGA overlay, then rescales the output using `scale.npy`:

```python
img = img.astype(np.uint8)
output = driver.execute(driver_in)   # output is INT8
output = scale * output              # dequant/rescale outside the accelerator
output = output.transpose(0, 3, 1, 2)  # NHWC -> NCHW
```

The FINN/PYNQ runtime uses `FINNExampleOverlay(Overlay)` from `src/deploy/driver/driver_base.py`, which is a thin wrapper around PYNQ overlay loading + DMA buffer allocation.

PYNQ background (Overlay + `allocate`): https://github.com/xilinx/pynq/blob/master/docs/source/overlay_design_methodology/python_overlay_api.rst

## Where QAT training is done for LPYOLO

The training repo is `quantized-yolov5`, which is a fork of Ultralytics YOLOv5 extended with Brevitas QAT.

Two notable points visible from its README:

- It "utilizes Brevitas … for quantization-aware training (QAT)".
- It calls out an export-time activation change specifically for FINN:
  - "Sigmoid … when training whereas HardTanh is used when exporting the model for FINN."

Source: https://github.com/sefaburakokcu/quantized-yolov5

### How `quantized-yolov5` implements QAT

The quantized-yolov5 approach is a **YAML-driven layer replacement** strategy integrated into the Ultralytics YOLOv5 framework. Here's how it works:

#### 1. Model Configuration (YAML)

Models are defined in YAML files with explicit bit-width parameters:

```yaml
# models/hub/yolov1-tiny-quant.yaml
nc: 1  # Face detection (1 class)

# Bit-width parameters used throughout the model
bit_width: [8, 4, 4, 8]  # [in_weight, weight, act, out_weight]
use_hardtanh: False  # Training uses Sigmoid, export switches to HardTanh

backbone:
  # Input layer: 8-bit weights, 4-bit activations
  - [-1, 1, QuantConv, [16, 3, 1, None, 1, True, True, in_weight_bit_width, act_bit_width]]
  - [-1, 1, nn.MaxPool2d, [2, 2, 0]]

  # Hidden layers: 4-bit weights, 4-bit activations
  - [-1, 1, QuantConv, [32, 3, 1, None, 1, True, True, weight_bit_width, act_bit_width]]
  # ... more layers ...

head:
  # Detection head layers
  - [-1, 1, QuantConv, [1024, 3, 1, None, 1, True, True, weight_bit_width, act_bit_width]]

  # Final layer: 8-bit output weights, conditional HardTanh
  - [-1, 1, QuantSimpleConv, [(nc+5)*3, 3, 1, None, 1, out_weight_bit_width, act_bit_width, use_hardtanh]]

  # Detection decode
  - [[14], 1, Detect, [nc, anchors, False, use_hardtanh]]
```

#### 2. Custom Brevitas Layer Wrappers

The repo implements custom quantized layers in `models/common.py`:

```python
class QuantConv(nn.Module):
    """Quantized Conv + BN + ReLU block with configurable bit-widths"""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1,
                 use_act=True, use_bn=True,
                 weight_bit_width=4, act_bit_width=2):
        super().__init__()

        # Select weight quantizer (binary vs multi-bit)
        if weight_bit_width == 1:
            weight_quant = CommonWeightQuant  # Binary weights
        else:
            weight_quant = CommonIntWeightPerChannelQuant  # Multi-bit per-channel

        # Select activation quantizer (binary vs unsigned INT)
        if act_bit_width == 1:
            act_quant = CommonActQuant  # Binary activations
        else:
            act_quant = CommonUintActQuant  # Unsigned INT activations (for ReLU)

        # Quantized convolution
        self.conv = QuantConv2d(
            in_channels=c1, out_channels=c2,
            kernel_size=k, stride=s, padding=autopad(k, p),
            groups=g, bias=False,
            weight_quant=weight_quant,
            weight_bit_width=weight_bit_width
        )

        # Batch normalization (fused into conv after training)
        self.bn = nn.BatchNorm2d(c2) if use_bn else nn.Identity()

        # Quantized activation
        self.act = QuantReLU(
            act_quant=act_quant,
            bit_width=act_bit_width,
            return_quant_tensor=True  # Pass quant metadata to next layer
        ) if use_act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class QuantSimpleConv(nn.Module):
    """Final detection layer with HardTanh option for FINN export"""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1,
                 weight_bit_width=4, act_bit_width=2,
                 use_hardtanh=False):
        super().__init__()
        self.use_hardtanh = use_hardtanh

        self.conv = QuantConv2d(
            in_channels=c1, out_channels=c2,
            kernel_size=k, stride=s, padding=autopad(k, p),
            groups=g, bias=False,
            weight_quant=CommonIntWeightPerChannelQuant,
            weight_bit_width=weight_bit_width
        )

        # HardTanh for FINN export (hardware-friendly, no transcendental functions)
        if self.use_hardtanh:
            self.hard_quant = QuantHardTanh(
                max_val=1.0, min_val=-1.0,
                bit_width=8,
                return_quant_tensor=False  # Output raw tensor
            )

    def forward(self, x):
        x = self.conv(x)
        if self.use_hardtanh:
            x = self.hard_quant(x / 2)  # Scale to [-1, 1] range
        return x


class QuantBottleneck(nn.Module):
    """Quantized residual block with identity path quantization"""
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5,
                 weight_bit_width=4, act_bit_width=2):
        super().__init__()
        c_ = int(c2 * e)  # Hidden channels
        self.cv1 = QuantConv(c1, c_, 1, 1, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width)
        self.cv2 = QuantConv(c_, c2, 3, 1, g=g, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width)
        self.add = shortcut and c1 == c2

        # Quantize residual path for proper bit-width management
        if self.add:
            self.quant_identity = QuantIdentity(
                bit_width=weight_bit_width,
                return_quant_tensor=True
            )

    def forward(self, x):
        if self.add:
            return self.quant_identity(x) + self.cv2(self.cv1(x))
        else:
            return self.cv2(self.cv1(x))
```

#### 3. Brevitas Quantizer Primitives

The repo defines reusable quantizer configurations in `models/quant_common.py`:

```python
from brevitas.quant import Int8WeightPerTensorFloat, Uint8ActPerTensorFloatMaxInit

class CommonIntWeightPerChannelQuant(Int8WeightPerTensorFloat):
    """Per-channel weight quantizer with configurable bit-width"""
    scaling_min_val = 2e-16  # Prevent division by zero
    bit_width = None  # Must be specified per layer (e.g., 4, 8)
    scaling_per_output_channel = True  # Per-channel quantization
    restrict_scaling_type = RestrictValueType.LOG_FP  # Log-domain scaling

class CommonUintActQuant(Uint8ActPerTensorFloatMaxInit):
    """Unsigned activation quantizer for ReLU outputs"""
    scaling_min_val = 2e-16
    bit_width = None  # Must be specified per layer
    max_val = 6.0  # ReLU6-like clipping
    restrict_scaling_type = RestrictValueType.LOG_FP
```

#### 4. Training Process

Training uses the standard Ultralytics YOLOv5 training loop with quantized layers:

```bash
# train.py
python train.py \
    --data widerface.yaml \
    --cfg models/hub/yolov1-tiny-quant.yaml \
    --weights '' \  # Train from scratch, or provide FP32 checkpoint
    --batch-size 128 \
    --epochs 300 \
    --device 0

# Quantizers learn scales during full training (no separate calibration step)
# BN statistics + quantizer range parameters are updated via gradient descent
```

**Key difference from our approach**: They train from scratch or FP32 checkpoint with quantizers active from the start, while we do PTQ initialization followed by optional short QAT fine-tuning.

#### 5. FINN-ONNX Export with Activation Switching

The export process (in `export.py`) switches activations from Sigmoid to HardTanh:

```python
import brevitas.onnx as bo

def export_finn_onnx(model, im, file):
    """Export for FINN compiler (HardTanh mode)"""
    # Switch to HardTanh activation (done at export time)
    prepare_cfg_for_export(model.yaml)  # Sets use_hardtanh=True

    # Reload model with HardTanh
    model_hardtanh = attempt_load(weights, map_location=device, fuse=False)

    # Export FINN-ONNX
    input_shape = list(im.numpy().shape)
    bo.export_finn_onnx(
        module=model_hardtanh,
        input_shape=input_shape,
        export_path=file.with_suffix('.finn.onnx'),
    )

    # Save output scale factor separately (for dequantization at inference)
    scale = model_hardtanh.model[-1].hard_quant.quant_output_scale().detach().cpu().numpy()
    np.save(file.with_suffix('.npy'), scale)
```

**Why HardTanh?** FPGA implementations avoid expensive transcendental functions (exp/log for Sigmoid). HardTanh is a simple clamp operation: `max(-1, min(1, x))`.

The detection head logic changes accordingly:

```python
# models/yolo.py - Detect.forward()
if not self.training:
    if self.use_hardtanh:
        # HardTanh range: [-1, 1], no sigmoid
        y = y  # No activation
        # Box decode: different math for [-1, 1] range
        y[..., 0:2] = (y[..., 0:2] + self.grid[i]) * self.stride[i]  # xy
    else:
        # Standard Sigmoid range: [0, 1]
        y = y.sigmoid()
        y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
```

## Our QAT implementation (YOLOv10m + Brevitas)

Our QAT is implemented end-to-end inside this repo:

- Entry point: `scripts/cv-models/quantize_yolov10m_brevitas_w4.py` (`qat` subcommand)
- Key mechanics:
  - Start from a YOLOv10 checkpoint.
  - Inject Brevitas quantizers (weights always quantized; activations optional depending on `w4a16` vs `w4a8`).
  - Optionally calibrate activation quantizers for `w4a8`.
  - Run a short fine-tune using Ultralytics’ `YOLOv10DetectionTrainer`.
  - Export QCDQ ONNX via Brevitas (QuantizeLinear/DequantizeLinear nodes) and optionally run a conservative ONNX optimizer pass that preserves Q/DQ.

Representative code shape (simplified):

```python
model = load_yolov10_detection_model(...)
model = quantize_model_brevitas_ptq(model, weight_bit_width=4, act_bit_width=8 or None)
if act_bit_width is not None:
    calibrate_activation_quantizers(model, ...)
trainer = YOLOv10DetectionTrainer(overrides=...)
trainer.model = model
trainer.train()
export_brevitas_qcdq_onnx(Yolov10HeadOutput(trained_model, head="one2many"), out_path=...)
```

## Key differences (why the workflows look very different)

### 1. Repo scope and "where QAT lives"
- **`finn-quantized-yolo`**: deployment + eval; QAT training is delegated to `quantized-yolov5`.
- **`quantized-yolov5`**: training framework with YAML-configured quantized layers; full QAT from scratch.
- **Ours**: training/fine-tune (QAT) + export are implemented in-repo; PTQ-first with optional short QAT.

### 2. Quantization initialization strategy
- **`quantized-yolov5`**: QAT from scratch (or FP32 checkpoint) with quantizers active from the start; trains for hundreds of epochs with full dataset (WiderFace).
- **Ours**: PTQ first (no training, graph transform with `layerwise_quantize()`), then optional short QAT fine-tuning (1-3 epochs, ~100 images).

**Impact**: Their approach optimizes for final accuracy (invest training time upfront); our approach optimizes for research speed (explore many bit-widths quickly).

### 3. Layer replacement strategy
- **`quantized-yolov5`**: YAML-driven declarative configuration; each layer explicitly specifies bit-widths in the model definition file. Easy to understand layer-by-layer quantization, but requires manual YAML editing.
  ```yaml
  - [-1, 1, QuantConv, [32, 3, 1, None, 1, True, True, weight_bit_width, act_bit_width]]
  ```
- **Ours**: Programmatic graph transform via Brevitas API; flexible for selective quantization and automated exploration.
  ```python
  quantized = layerwise_quantize(model, compute_layer_map={nn.Conv2d: (qnn.QuantConv2d, {...})})
  ```

### 4. Target runtime and ONNX format
- **FINN/PYNQ flow** (quantized-yolov5 → finn-quantized-yolo): Integer I/O (`UINT8` input, `INT8` output), FINN-ONNX format with custom ops, external `scale.npy` for output dequantization, NHWC layout.
- **Our flow**: Floating-point I/O (FP16/FP32), standard QCDQ ONNX with QuantizeLinear/DequantizeLinear nodes in-graph, NCHW layout, targets ONNX Runtime CUDA EP.

**Impact**: Their export targets FPGA deployment (FINN compiler → bitstream); ours targets GPU inference (ONNX Runtime).

### 5. Model family and detection head
- **LPYOLO**: YOLOv3-tiny variant, face detection (1 class), output shape `(1, 13, 13, 18)` NHWC, Python-side decode + NMS.
- **Ours**: YOLOv10m, COCO detection (80 classes), output shape `(1, 84, 8400)` NCHW, exports detection head output (one2many).

### 6. Export-time activation constraints
- **`quantized-yolov5`**: Switches activations at export time (Sigmoid during training → HardTanh for FINN export) to avoid expensive transcendental functions on FPGA hardware. Detection decode logic changes accordingly (different math for [-1, 1] vs [0, 1] range).
- **Ours**: Preserves original activations (Sigmoid, SiLU, etc.) in exported ONNX; focuses on Q/DQ semantics and maintaining Ultralytics/YOLOv10 head behavior.

### 7. Calibration approach
- **`quantized-yolov5`**: Implicit calibration via batch normalization statistics and quantizer range learning during full training (no explicit calibration step).
- **Ours**: Explicit calibration step using `calibration_mode()` with ~100 COCO images for activation quantizers (PTQ modes with A8).

### 8. Compute type
- **FINN deployment**: True integer compute throughout (no dequantization inside accelerator); ultra-low bit-widths possible (2W4A, 4W2A, etc.).
- **Ours**: Fake quantization (Q/DQ nodes dequantize to FP16/FP32 before Conv); no speedup on current GPU stacks, but fast iteration and inspectable graphs.

## Practical implications for "compare with our implementation"

If you are comparing accuracy/latency between the two projects, align these first:

- **Task/dataset**: LPYOLO (face detection, WiderFace) vs YOLOv10m (COCO, 80 classes)
- **I/O contract**: integer I/O + external scale (`scale.npy`) vs float I/O + Q/DQ inside ONNX
- **Postprocessing**: Python decode+NMS vs whatever your runtime uses
- **Quantization budget**: LPYOLO models mention 8-bit input/output and varying W/A bits internally (e.g., 2W4A, 4W4A); our pipeline explicitly explores `w4a16`, `w4a8`, `w8a16`, `w8a8`
- **Training investment**: Full training (hundreds of epochs, full dataset) vs PTQ + short QAT (1-3 epochs, ~100 images)
- **Target hardware**: FPGA (true INT compute) vs GPU (fake quantization)

## Quick Reference: When to Use Each Approach

### Use `quantized-yolov5` (Training Framework) If:
- You want to **train FINN-targeted models from scratch**
- You have **time and compute for full training** (hundreds of epochs)
- You need **fine-grained control over layer-wise bit-widths** via YAML configuration
- You want to **customize YOLO architecture for FPGA**
- You plan to **compile with FINN** for FPGA deployment
- You target **face detection on WiderFace** or similar single-task detection

### Use `finn-quantized-yolo` (Deployment) If:
- You have **pretrained FINN-ONNX models** already (from quantized-yolov5)
- You target **PYNQ-Z2 FPGA for deployment**
- You want **ultra-low-bit inference** (2W4A, 4W2A, etc.)
- You need **true integer compute** (no FP operations)
- You accept **single-task models** (e.g., face detection)

### Use Our Brevitas YOLOv10m Approach If:
- You want to **quickly explore different bit-widths** (W4/W8, A8/A16)
- You have **limited training time/data** (PTQ + short QAT)
- You target **NVIDIA GPUs for inference**
- You want **standard ONNX output** (portable, inspectable)
- You need **COCO detection** (80 classes)
- You want to **research quantization impact** on YOLOv10 architecture

## Summary Comparison Table

| Feature | Our Brevitas YOLOv10m | quantized-yolov5 (Training) | finn-quantized-yolo (Deployment) |
|---------|----------------------|----------------------------|----------------------------------|
| **Scope** | End-to-end (train+export+eval) | Training+export | Deployment+inference |
| **Model** | YOLOv10m | YOLOv5/v3/v1 | YOLOv3-tiny variant |
| **Task** | COCO (80 classes) | WiderFace (1 class) | Face detection (1 class) |
| **Quantization** | PTQ + optional short QAT | Full QAT from scratch | Consumes pretrained (QAT done elsewhere) |
| **Training Duration** | 1-3 epochs (fine-tuning) | 300+ epochs (full training) | N/A (deployment only) |
| **Dataset Size** | ~100 images for QAT | Full dataset (thousands) | N/A |
| **Bit-widths** | W4A16, W4A8, W8A16, W8A8 | Configurable via YAML (1-8 bit) | 2W4A, 3W5A, 4W2A, 4W4A, 6W4A, 8W3A |
| **Layer Config** | Programmatic (`layerwise_quantize()`) | YAML declarative | N/A (consumes artifacts) |
| **Calibration** | Explicit (`calibration_mode()`) | Implicit (BN stats + training) | N/A |
| **ONNX Format** | QCDQ (standard Q/DQ nodes) | FINN-ONNX + scale.npy | N/A (uses bitstream) |
| **I/O Dtype** | FP16/FP32 (in-graph Q/DQ) | UINT8 input, INT8 output | UINT8 input, INT8 output |
| **I/O Layout** | NCHW | NCHW (converts to NHWC at export) | NHWC |
| **Export Activation** | Preserves original (Sigmoid, SiLU) | Sigmoid → HardTanh switch | HardTanh (from training repo) |
| **Target Hardware** | NVIDIA GPU (fake quant) | FPGA via FINN | PYNQ-Z2 FPGA |
| **Runtime** | ONNX Runtime CUDA EP | FINN compiler output | PYNQ Overlay (bitstream) |
| **Compute** | FP16/FP32 after dequant | True INT compute (FPGA) | True INT compute (FPGA) |
| **Brevitas Version** | 0.12.1 (2024) | ≥0.7.1 (2021-2022) | N/A |
| **PyTorch Version** | 2.9.0 (needs compat shim) | ≥1.7.0 | N/A (ARM PyTorch for PYNQ) |
| **Key Strength** | Fast iteration, flexible exploration | Full control, proven accuracy | Extreme efficiency, true INT compute |
| **Key Trade-off** | No speedup (fake quant) | Requires full training | Deployment only, FINN toolchain |

## Sources

**External Repositories:**
- `finn-quantized-yolo` (deployment repo): https://github.com/sefaburakokcu/finn-quantized-yolo
- `quantized-yolov5` (training repo): https://github.com/sefaburakokcu/quantized-yolov5
- LPYOLO paper: https://arxiv.org/abs/2207.10482
- PYNQ Overlay API: https://github.com/xilinx/pynq/blob/master/docs/source/overlay_design_methodology/python_overlay_api.rst
- Deployment walkthrough (Medium): https://medium.com/@bestamigunay1/end-to-end-deployment-of-lpyolo-low-precision-yolo-for-face-detection-on-fpga-13c3284ed14b

**Brevitas and FINN:**
- Brevitas documentation: https://xilinx.github.io/brevitas/
- Brevitas GitHub: https://github.com/Xilinx/brevitas
- Brevitas paper: Pappalardo et al., "Brevitas: An Open-Source Library for Neural Network Quantization", https://arxiv.org/abs/2106.11948
- FINN documentation: https://finn.readthedocs.io/
- FINN GitHub: https://github.com/Xilinx/finn
- FINN paper: Blott et al., "FINN-R: An End-to-End Deep-Learning Framework for Fast Exploration of Quantized Neural Networks", https://arxiv.org/abs/1809.04570

**Our Implementation:**
- Plan: `context/plans/plan-quantize-yolov10m-w4a8-w4a16-brevitas.md`
- Main script: `scripts/cv-models/quantize_yolov10m_brevitas_w4.py`
- PTQ implementation: `src/auto_quantize_model/cv_models/yolov10_brevitas.py`
- QAT implementation: `src/auto_quantize_model/cv_models/yolov10_lightning_qat.py`
- Export compatibility: `src/auto_quantize_model/brevitas_onnx_export_compat.py`
- Related hint: `context/hints/about-brevitas-yolo-w4a8-w4a16-onnx-nvidia-gpu.md`
