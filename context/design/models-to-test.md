# Models to Test for Quantization and Layer Sensitivity

This document lists the primary models we plan to use for post-training quantization experiments, mixed FP16/INT8 deployment, and layer/region sensitivity analysis.

## Object Detection Models

### YOLO11 (Ultralytics)

- **Task**: Real-time object detection (treated as detector only here, even though the family supports additional tasks).
- **Summary**: YOLO11 is the latest generation of the Ultralytics You Only Look Once (YOLO) family, designed to provide higher accuracy and speed than earlier YOLO versions while retaining a simple, production-friendly API. It supports multiple model scales (e.g., nano, small, medium) and predicts bounding boxes, confidence scores, and class labels in a single pass.
- **Why we test it**: Ultralytics provides official PyTorch checkpoints, ONNX export support, and detailed docs, making YOLO11 a strong baseline for evaluating post-training quantization, mixed FP16/INT8 deployment, and per-layer sensitivity on modern NVIDIA GPUs.
- **Baseline performance (COCO, YOLO11 Detect)**:

  | Variant   | Input | mAPval 50-95 | Speed CPU ONNX (ms) | Speed T4 TensorRT10 (ms) | Params (M) | FLOPs (B) |
  |----------|-------|--------------|----------------------|---------------------------|------------|-----------|
  | YOLO11n  | 640   | 39.5         | 56.1 ± 0.8           | 1.5 ± 0.0                 | 2.6        | 6.5       |
  | YOLO11s  | 640   | 47.0         | 90.0 ± 1.2           | 2.5 ± 0.0                 | 9.4        | 21.5      |
  | YOLO11m  | 640   | 51.5         | 183.2 ± 2.0          | 4.7 ± 0.1                 | 20.1       | 68.0      |
  | YOLO11l  | 640   | 53.4         | 238.6 ± 1.4          | 6.2 ± 0.1                 | 25.3       | 86.9      |
  | YOLO11x  | 640   | 54.7         | 462.8 ± 6.7          | 11.3 ± 0.2                | 56.9       | 194.9     |

- **References**:
  - Docs: https://docs.ultralytics.com/models/yolo11/
  - Blog overview: https://www.ultralytics.com/blog/how-to-use-ultralytics-yolo11-for-object-detection
  - GitHub: https://github.com/ultralytics/ultralytics
  - Hugging Face collection: https://huggingface.co/Ultralytics/YOLO11

### YOLO11-seg (Ultralytics instance segmentation)

- **Task**: Instance segmentation (grouped under detectors for this project).
- **Summary**: YOLO11-seg extends the YOLO11 detection head with mask prediction so each detected object has a pixel-level segmentation mask. Ultralytics ships segmentation-specific checkpoints (e.g., `yolo11n-seg.pt`, `yolo11s-seg.pt`) and supports direct export to ONNX and other deployment formats.
- **Why we test it**: The segmentation branches are often more numerically sensitive than pure detection heads, making YOLO11-seg a useful stress test for quantization and mixed-precision strategies built around the YOLO11 backbone.
- **Baseline performance (COCO, YOLO11 Segment)**:

  | Variant      | Input | mAPbox 50-95 | mAPmask 50-95 | Speed CPU ONNX (ms) | Speed T4 TensorRT10 (ms) | Params (M) | FLOPs (B) |
  |-------------|-------|--------------|---------------|----------------------|---------------------------|------------|-----------|
  | YOLO11n-seg | 640   | 38.9         | 32.0          | 65.9 ± 1.1           | 1.8 ± 0.0                 | 2.9        | 9.7       |
  | YOLO11s-seg | 640   | 46.6         | 37.8          | 117.6 ± 4.9          | 2.9 ± 0.0                 | 10.1       | 33.0      |
  | YOLO11m-seg | 640   | 51.5         | 41.5          | 281.6 ± 1.2          | 6.3 ± 0.1                 | 22.4       | 113.2     |
  | YOLO11l-seg | 640   | 53.4         | 42.9          | 344.2 ± 3.2          | 7.8 ± 0.2                 | 27.6       | 132.2     |
  | YOLO11x-seg | 640   | 54.7         | 43.8          | 664.5 ± 3.2          | 15.8 ± 0.7                | 62.1       | 296.4     |

- **References**:
  - Docs: https://docs.ultralytics.com/tasks/segment/
  - GitHub: https://github.com/ultralytics/ultralytics
  - Hugging Face collection: https://huggingface.co/Ultralytics/YOLO11

## Segmentation Models

### SegFormer

- **Task**: Semantic segmentation.
- **Summary**: SegFormer is a transformer-based segmentation model that combines a hierarchical Transformer encoder with a lightweight MLP decoder to achieve strong accuracy, speed, and robustness across diverse segmentation tasks. It avoids heavy decoders, instead leveraging multi-scale transformer features for rich representations.
- **Why we test it**: As a relatively lightweight transformer-style segmentation backbone with strong performance, SegFormer is a natural candidate for studying how quantization and per-layer precision selection affect transformer-based vision models (vs. convolutional backbones like YOLO11).
- **Baseline performance (SegFormer-B0, ADE20K)**:

  > Exact throughput and mIoU numbers vary by implementation and hardware; NVIDIA’s paper reports strong ADE20K/Cityscapes performance with a good accuracy–speed trade-off. For concrete reference, see their official benchmarks and the `nvidia/segformer-b0-finetuned-ade-512-512` model card on Hugging Face.

- **References**:
  - Intro/overview: https://roboflow.com/model/segformer
  - Blog/tutorial: https://www.labellerr.com/blog/segformer/
  - Original implementation (NVIDIA): https://github.com/NVlabs/SegFormer
  - PyTorch implementation: https://github.com/FrancescoSaverioZuppichini/SegFormer
  - Hugging Face docs: https://huggingface.co/docs/transformers/en/model_doc/segformer
  - Hugging Face model card: https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512

### MobileSAM (Mobile Segment Anything)

- **Task**: Promptable segmentation (Segment Anything–style).
- **Summary**: MobileSAM is a lightweight variant of Meta’s Segment Anything Model (SAM) that replaces the heavy ViT-H image encoder with a much smaller encoder trained via distillation, while keeping a compatible mask decoder. This yields a model that is dramatically smaller yet still offers competitive mask quality and zero-shot segmentation capabilities.
- **Why we test it**: MobileSAM explicitly targets mobile and edge scenarios where precision and bandwidth are constrained, making it a strong candidate for evaluating how aggressive quantization and mixed-precision schemes interact with prompt-based segmentation and distilled encoders.
- **Baseline performance (MobileSAM vs FastSAM, whole pipeline)**:

  | Metric      | FastSAM | MobileSAM |
  |------------|---------|-----------|
  | Parameters | 68M     | 9.66M     |
  | Speed      | 64 ms   | 12 ms     |

  > Table from the official MobileSAM repository, comparing end-to-end encoder+decoder size and latency (hardware and exact setup detailed in their README).

- **References**:
  - Ultralytics docs: https://docs.ultralytics.com/models/mobile-sam/
  - Kornia docs: https://kornia.readthedocs.io/en/latest/models/mobile_sam.html
  - Official GitHub: https://github.com/ChaoningZhang/MobileSAM

## Vision-Language Models (VLM)

### MobileVLM V2

- **Task**: Vision-language understanding and generation (multimodal VLM).
- **Summary**: MobileVLM V2 is a family of improved vision-language models built on the original MobileVLM, combining a lightweight vision encoder with a compact language model and refined training recipes. It is designed to deliver competitive multimodal performance while remaining deployable on mobile and edge devices.
- **Why we test it**: MobileVLM V2 serves as a representative VLM to see how the quantization and mixed-precision techniques we use for CNNs and transformers extend to multimodal encoders, cross-attention blocks, and text decoders, especially when targeting FP16/INT8 mixed precision end-to-end.
- **Baseline performance (MobileVLM V2-1.7B, high level)**:

  > The MobileVLM V2 paper and model card report that MobileVLM_V2-1.7B achieves on-par or better performance than some 3B-scale VLMs on standard benchmarks, while the 3B variant surpasses many 7B+ models. Exact benchmark tables are available in the paper (`MobileVLM V2: Faster and Stronger Baseline for Vision Language Model`) and the Hugging Face model card.

- **References**:
  - GitHub (official): https://github.com/Meituan-AutoML/MobileVLM
  - Hugging Face model card: https://huggingface.co/mtgv/MobileVLM_V2-1.7B
  - Paper summary: https://huggingface.co/papers/2402.03766
