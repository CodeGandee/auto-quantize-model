# About Edge NPU-Friendly Object Detection and Segmentation Models

This guide surveys modern object detection and segmentation models that are well-suited for deployment on edge devices with NPUs (phone SoCs, Jetson-class modules, edge TPUs, NPUs in embedded boards) where you want end-to-end latency typically under 100 ms per frame. The focus is on models that:
- Have small, efficient variants (often “nano” / “tiny” / “lite”),
- Quantize cleanly to INT8 / mixed precision,
- Are already used or benchmarked on edge devices, and
- Have good tooling support (ONNX, TFLite, TensorRT, OpenVINO, CoreML, NNAPI, etc.).

Use this as a model-selection cheat sheet when deciding what to target before running automatic mixed-precision or sensitivity-aware quantization.

## Quick Recommendations

- If you want **one default detector** for mid-range NPUs (≈4–30 TOPS): use **Ultralytics YOLO11n / YOLOv8n** (detection) for new projects.
- On **high-end edge modules** (e.g., NVIDIA Jetson Orin, D-Robotics RDK S100 at 80+ TOPS BPU): you can typically afford **YOLO11s/m or YOLOv8s/m** for higher accuracy while staying well under 100 ms, or even **RT-DETR-R50**-class backbones.
- If you need **fast instance segmentation**: use **YOLO11n-seg / YOLOv8n-seg** (Ultralytics) or **MobileSAM** for prompt-based segmentation.
- If you are locked into **TFLite / EdgeTPU**: use **EfficientDet-Lite (Lite0–Lite2)** or **SSD MobileNet V1/V2**.
- If you want **transformer-based real-time detection** on strong edge NPUs (Jetson Orin, high-end phones): use **RT-DETR / RT-DETRv2 small backbones** or **RTMDet-n/s**.
- For **semantic segmentation** (dense pixel labels) on NPUs: consider **BiSeNetV2**, **DDRNet-23-slim**, **SeaFormer**, or newer hardware-aware models such as **HARD-Edge**.

Below, “sub-100 ms” means per-frame latency that is reasonably achievable on typical 4–30 TOPS NPUs after INT8 or mixed-precision optimization and with moderate input resolutions (≈320–640 px); on **high-end devices like Jetson Orin or RDK S100 (80+ TOPS)**, the same models generally run far below this threshold, allowing you to use larger backbones at similar latency.

## Object Detection Models for Edge NPUs

### 1. Ultralytics YOLO11 / YOLOv8 (Nano/Small)

**Why it’s strong for edge NPUs**
- Modern YOLO family with comprehensive tooling (Python API, CLI, Ultralytics HUB).
- Offers multiple sizes (`n`, `s`, `m`, `l`, `x`) and tasks (detect, segment, pose); `n` and `s` are the sweet spot for NPUs.
- Widely benchmarked and adopted, with many guides for ONNX, TensorRT, OpenVINO, TFLite, CoreML, and mobile deployments.
- Architecturally simple for quantizers (conv + BN + SiLU/LeakyReLU; no exotic layers in the small variants).

**Speed evidence**
- Ultralytics MobileSAM docs compare YOLOv8n-seg and YOLO11n-seg against SAM/MobileSAM and report CPU latency of **24.5 ms (YOLOv8n-seg)** and **30.1 ms (YOLO11n-seg)** per image on a 2025 Apple M4 Pro CPU with `torch==2.6.0` and `ultralytics==8.3.90`. With INT8 on an NPU, these models typically run much faster, so a **100 ms budget is very lax** for `n`/`s` variants on 320–640 px inputs.
- For pure detection, the `n` models are even lighter than their `*-seg` counterparts.
- On **NVIDIA Jetson Orin NX**, an optimized YOLOv8n pipeline reaches **≈52 FPS in FP16** and **≈65 FPS with INT8 quantization** using TensorRT and careful preprocessing, corresponding to ≈19 ms and ≈15 ms per frame respectively.  
  Source: SimaLabs “Achieving 60 FPS YOLOv8 Object Detection on NVIDIA Jetson Orin NX with INT8 Quantization”.  
  https://www.simalabs.ai/resources/60-fps-yolov8-jetson-orin-nx-int8-quantization-simabit
- MLPerf v4.0 benchmarks show **Jetson AGX Orin 64GB reaching ≈383 FPS with YOLOv8n** in INT8 mode, i.e., ≈2.6 ms per frame under optimized conditions, highlighting how much headroom high-end edge devices have for larger models and multiple concurrent streams.  
  Source summary: Lowtouch.ai “NVIDIA Jetson AGX Orin and RTX 4090 Comparison”.  
  https://www.lowtouch.ai/nvidia-jetson-agx-orin-and-rtx-4090-in-ai-applications/
- Reference: MobileSAM vs YOLO comparison table in Ultralytics docs.  
  Source: https://docs.ultralytics.com/models/mobile-sam/

**When to use**
- Default choice when you control the model family and want strong accuracy–latency trade-offs on many hardware targets.
- Good baseline for automatic mixed-precision quantization tools (TensorRT ModelOpt, OpenVINO NNCF, Intel INC, AIMET).

**Code snippet: basic YOLO11n detection + export**
```python
from ultralytics import YOLO

# 1. Load a small, edge-friendly detector
model = YOLO("yolo11n.pt")  # or "yolov8n.pt"

# 2. Run quick inference
results = model("image.jpg")  # or a video / webcam source

# 3. Export for deployment (ONNX, TensorRT, etc.)
model.export(format="onnx", opset=13, dynamic=True)     # generic ONNX
model.export(format="engine", half=True, int8=True)     # TensorRT FP16/INT8 (requires TensorRT)
```

**Links**
- YOLO11 models and docs (Ultralytics): https://docs.ultralytics.com/models/yolo11/
- YOLOv8 models and docs (Ultralytics): https://docs.ultralytics.com/models/yolov8/
- Ultralytics GitHub: https://github.com/ultralytics/ultralytics

### 2. RT-DETR / RT-DETRv2 (Real-Time Detection Transformer)

**Why it’s strong for edge NPUs**
- End-to-end transformer-based detector **without NMS**, which simplifies deployment pipelines.
- Designed explicitly for real-time performance while maintaining high AP.
- Small backbones (e.g., ResNet-18/34, MobileNet-like variants from community repos) can be competitive on strong edge NPUs.

**Speed evidence**
- The original RT-DETR paper “DETRs Beat YOLOs on Real-time Object Detection” reports **RT-DETR-R50 at 108 FPS** and **RT-DETR-R101 at 74 FPS** on a T4 GPU with TensorRT FP16 at 800×1333 input resolution (≈9.3 ms and 13.5 ms per frame, respectively), while outperforming YOLOs in accuracy.  
  Source (paper homepage): https://zhao-yian.github.io/RTDETR/  
  Source (arXiv): https://arxiv.org/abs/2304.08069  
- RT-DETRv2 improves the baseline with “bag-of-freebies” enhancements.  
  Source (official repo): https://github.com/lyuwenyu/RT-DETR

**When to use**
- When you need a **transformer-based detector** with end-to-end decoding, but still want real-time performance on relatively capable NPUs (Jetson Orin, Apple M-series, Snapdragon with strong Hexagon NPU).
- Good candidate for **latency-sensitive multi-object tracking** where NMS costs become noticeable.

**Code snippet: RT-DETR via Ultralytics**
```python
from ultralytics import YOLO

# RT-DETR pretrained COCO model (Ultralytics packaging)
model = YOLO("rtdetr-l.pt")  # or smaller variant if available
results = model("image.jpg")

# Export to ONNX for NPU compilers (TensorRT, OpenVINO, etc.)
model.export(format="onnx", opset=13)
```

### 3. RTMDet (OpenMMLab)

**Why it’s strong for edge NPUs**
- Designed specifically as a **real-time detector** with high throughput and strong COCO AP.
- Lightweight variants (`rtmdet-tiny`, `rtmdet-nano`) offer excellent speed–accuracy trade-offs.
- Implemented in MMDetection, which exports cleanly to ONNX and serves as input to many hardware compilers.

**Speed evidence**
- Roboflow’s “Best Object Detection Models” and OpenMMLab benchmarks highlight RTMDet’s performance, with `rtmdet-l` achieving **~300+ FPS on an NVIDIA RTX 3090** at strong AP. That translates to per-frame latencies well below 10 ms on high-end GPUs, leaving plenty of headroom for edge-sized variants on NPUs.  
  Source: https://blog.roboflow.com/best-object-detection-models/  
  OpenMMLab RTMDet docs: https://github.com/open-mmlab/mmdetection/tree/main/configs/rtmdet

**When to use**
- Useful if you already use MMDetection or OpenMMLab for training and want a consistent export path to edge compilers.
- Often a strong alternative to YOLO models when you want different design trade-offs (e.g., anchor-free, decoupled heads).

### 4. EfficientDet-Lite (Lite0–Lite2) and SSD MobileNet

**Why it’s strong for edge NPUs**
- Officially supported by **TensorFlow Lite** and **Google Coral EdgeTPU**, with pre-trained and quantization-ready models.
- Architectures are explicitly tuned for mobile deployment (depthwise separable convolutions, low parameter count).
- Widely used in real-world embedded applications (cameras, Raspberry Pi + TPU, IoT devices).

**Speed evidence**
- The 2024 paper “Benchmarking Deep Learning Models for Object Detection on Edge Computing Devices” evaluates YOLOv8 (Nano/Small/Medium), EfficientDet-Lite (Lite0–Lite2), and SSD (SSD MobileNet V1, SSDLite MobileDet) on Raspberry Pi 3/4/5, Coral TPU, and Jetson Orin Nano. It finds that **SSD MobileNet V1 is the fastest and most energy-efficient** (but lowest accuracy), while YOLOv8 Medium is more accurate but slower. Jetson Orin Nano is the fastest device overall.  
  Source: https://arxiv.org/abs/2409.16808
- Coral EdgeTPU official docs provide quantized detection models based on MobileNet SSD and EfficientDet-Lite.  
  Source: https://coral.ai/models/object-detection/

**When to use**
- When you must stay within the TensorFlow/TFLite ecosystem or deploy on Google Coral / some Android NPUs that integrate tightly with TFLite/NNAPI.
- Great fit for strict power / memory budgets where small models like `EfficientDet-Lite0` or `SSD MobileNet` can hit **tens of FPS** at relatively low resolutions.

**Code snippet: TFLite INT8 conversion for EfficientDet-Lite**
```python
import tensorflow as tf

# Load a SavedModel version of EfficientDet-Lite
saved_model_dir = "efficientdet-lite0-savedmodel"
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

# Enable full integer quantization for NPU/EdgeTPU
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen  # yields calibration batches
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()
open("efficientdet-lite0-int8.tflite", "wb").write(tflite_model)
```

## Segmentation Models for Edge NPUs

### 1. YOLO11n-seg / YOLOv8n-seg (Instance Segmentation)

**Why it’s strong for edge NPUs**
- Brings YOLO’s speed to instance segmentation by adding mask heads on top of the detection backbone.
- Small variants (`n`, `s`) are extremely light and quantize cleanly to INT8.
- Unified ecosystem with detection, segmentation, pose, tracking, and export support makes it convenient for edge deployment pipelines.

**Speed evidence**
- Ultralytics MobileSAM docs compare segmentation models and show:  
  - **YOLOv8n-seg:** 6.7 MB, 3.4M parameters, **24.5 ms CPU latency** per image.  
  - **YOLO11n-seg:** 5.9 MB, 2.9M parameters, **30.1 ms CPU latency** per image.  
  - MobileSAM and SAM variants are orders of magnitude slower on CPU.  
  Source: https://docs.ultralytics.com/models/mobile-sam/
- These values are already far below 100 ms on a general-purpose CPU; NPUs with INT8 acceleration will typically reduce latency further.

**When to use**
- Default choice for **real-time instance segmentation** on NPUs when you can control both training and deployment.
- Works very well with automatic mixed-precision quantization tools that support ONNX/TensorRT/OpenVINO.

**Code snippet: YOLO11n-seg inference**
```python
from ultralytics import YOLO

model = YOLO("yolo11n-seg.pt")  # or "yolov8n-seg.pt"
results = model("image.jpg")    # masks + boxes + classes

# Export segmentation model for NPU runtimes
model.export(format="onnx", opset=13)
```

### 2. MobileSAM (Mobile Segment Anything)

**Why it’s strong for edge NPUs**
- A lightweight variant of Meta’s Segment Anything Model (SAM) designed specifically for mobile and edge devices, using a Tiny-ViT encoder.
- Provides **prompt-based segmentation** (points / boxes), which is powerful for interactive labeling and building segmentation datasets from detections.
- Can be combined with a fast detector (e.g., YOLO11) to auto-annotate segmentation masks on devices or servers.

**Speed evidence**
- Ultralytics docs report that MobileSAM is about **5× smaller and 7× faster** than the original SAM, operating at roughly **12 ms per image** (8 ms encoder + 4 ms mask decoder) on a single GPU.  
  Source (MobileSAM docs): https://docs.ultralytics.com/models/mobile-sam/  
- CPU speed comparison table in the same docs shows MobileSAM at 25,381 ms/im vs YOLO segmentation models at tens of milliseconds on CPU and tiny model sizes, highlighting that MobileSAM is far more tractable than SAM but YOLO-seg remains the fastest choice for strict real-time needs.

**When to use**
- When you need **flexible / promptable segmentation** (segment anything-like behavior) rather than fixed-category instance segmentation.
- For building segmentation datasets on-device from bounding boxes or sparse prompts, especially when integrated with Ultralytics’ `auto_annotate`.

**Code snippet: MobileSAM with point prompts**
```python
from ultralytics import SAM

model = SAM("mobile_sam.pt")
results = model.predict(
    "image.jpg",
    points=[900, 370],    # or [[x1, y1], [x2, y2], ...]
    labels=[1],           # 1 = positive point, 0 = negative
)
```

### 3. Real-Time Semantic Segmentation: BiSeNetV2, DDRNet, SeaFormer, HARD-Edge

**Why they’re strong for edge NPUs**
- These models are designed with lightweight encoder–decoder architectures and extensive use of depthwise separable convolutions or efficient multi-branch designs to reach **hundreds of FPS** on desktop GPUs at Cityscapes/ADE20K resolutions, implying ample margin for sub-100 ms inference on edge NPUs.
- Typically have parameter counts in the range of 0.1M–20M and FLOPs in the low single-digit to tens of GFLOPs.

**Representative references**
- **BiSeNetV2**: An improved bilateral segmentation network with lightweight design, used widely as a real-time baseline in many papers; supports 70+ FPS on Cityscapes in original work.  
  Source (original paper): “BiSeNet V2: Bilateral Network with Guided Aggregation for Real-time Semantic Segmentation” (2021).
- **DDRNet-23-slim**: Dual-Resolution Network focusing on maintaining high-resolution features with low overhead, often used as a strong real-time baseline.  
  Source repo: https://github.com/ydhongHIT/DDRNet
- **SeaFormer** and related transformer-lite models: E.g., SeaFormer-S / SeaFormer-B with low FLOPs and 100–200+ FPS on Cityscapes depending on resolution and configuration.  
  Source: SeaFormer real-time segmentation papers (see CVPR/ACCV resources).
- **HARD-Edge / HARD-XXS**: “HARD: Hardware-Aware lightweight Real-time semantic segmentation model Deployable from MCUs to GPUs” (ACCV 2024) proposes tiny segmentation models such as HARD-XXS (≈0.11M params, 0.93–1.49 GFLOPs) reaching hundreds of FPS (e.g., >500 FPS at 360×640/480×480).  
  Source: https://openaccess.thecvf.com/content/ACCV2024/papers/Kwon_HARD__Hardware-Aware_lightweight_Real-time_semantic_segmentation_model_Deployable_from_ACCV_2024_paper.pdf

**When to use**
- When you need **dense semantic segmentation** (class per pixel) for tasks such as lane detection, drivable area segmentation, or medical imaging masks on-device.
- If you target very low-power NPUs or even MCUs (HARD-Edge models), but still require real-time inference.

**Implementation notes**
- Many of these models are available as PyTorch repos that export cleanly to ONNX; after that, your usual quantization + NPU compiler pipeline applies (TensorRT, OpenVINO, TFLite, CoreML).
- For extremely constrained NPUs, start with the smallest variants (e.g., BiSeNetV2-Lite, HARD-XXS) and scale up as latency budget allows.

## Practical Selection Rules for <100 ms on NPUs

- **Mid-range NPUs (≈4–30 TOPS)**  
  - Start with `n`/`nano` or `tiny` variants (YOLO11n/YOLOv8n, RTMDet-n) at 320–512 px input; most devices will run these well under 100 ms after INT8 quantization.  
  - Favor very lightweight segmentation heads (YOLO11n-seg/YOLOv8n-seg, BiSeNetV2-Lite, HARD-XXS) to keep latency in the **50–80 ms** range for segmentation.

- **High-end edge NPUs (Jetson Orin, RDK S100, ≥40–80+ TOPS)**  
  - You can usually step up to **YOLO11s/m or YOLOv8s/m**, **RT-DETR with R50 backbones**, or **RTMDet-s** at 640 px while remaining well below 100 ms per stream, even with multiple concurrent camera inputs.  
  - For segmentation, **YOLO11s-seg/YOLOv8s-seg**, **BiSeNetV2**, or **DDRNet-23-slim** are typically safe under 100 ms when quantized and optimized with TensorRT or equivalent runtimes.  
  - Exploit parallelism: high-end modules can run several small models (e.g., detector + segmenter + tracker) simultaneously within a 100 ms end-to-end budget by pinning them to different compute engines (GPU, NPU/BPU, DLA).

- **General rules**  
  - Use INT8 or mixed precision: Leverage automatic mixed-precision tools (ModelOpt, OpenVINO NNCF, Intel INC, AIMET) to keep critical layers in FP16/BF16 while pushing most of the network to INT8; this helps preserve accuracy while maximizing throughput.  
  - Measure on target hardware: Latency depends heavily on NPU implementation (memory bandwidth, compiler quality). Treat published FPS from GPUs as *upper bounds* and run on your own device with a calibration workload.  
  - Prefer “standard ops” architectures: Models using mostly convs, depthwise convs, pointwise convs, standard activations, and simple pooling are easier for NPU compilers to optimize than those with complex custom layers.  
  - Align model family with tooling: If your deployment stack is Ultralytics + TensorRT, YOLO/RT-DETR is ideal. If it is TFLite/EdgeTPU, EfficientDet-Lite and SSD MobileNet make more sense. For OpenVINO, YOLO/RTMDet/RT-DETR can be excellent after ONNX conversion.

In practice, for most edge NPUs you can achieve **<30 ms per frame** with quantized YOLO11n/YOLOv8n detectors and, on high-end devices like Jetson Orin or RDK S100, still stay comfortably under **100 ms** even when moving up to `s`/`m` detectors or adding segmentation heads, as long as you apply proper quantization and runtime optimization.
