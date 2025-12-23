Here’s what I’m seeing as the *actual* state of practice for **YOLO (conv-heavy) → INT4 weights** as of late 2025 — and where you can find real code / models.

## 1) Reality check: INT4 “just works” much less often for YOLO than for LLMs

Most “YOLO INT4” attempts run into a simple blocker: **mainstream inference stacks don’t broadly accelerate INT4 convolutions**.

* **TensorRT**: INT4 support is primarily **weight-only quantization (WoQ) for GEMM/MatMul**, not general CNN conv pipelines; WoQ is described as available only for GEMM layers. ([NVIDIA Docs][1])
* **ONNX Runtime**: “Quantize to Int4/UInt4” is **block-wise weight-only** and (in the doc) is limited to ops like **MatMul** (const B) and **Gather** (const data) — i.e., not your typical YOLO Conv2d graph. ([ONNX Runtime][2])
* **Ultralytics YOLO (YOLOv8)**: upstream explicitly says **no INT4 quant/export support** (closed “not planned”). ([GitHub][3])
* NVIDIA community guidance reflects this reality: **INT4/FP4 is usually discussed for LLMs**, while **CNNs typically use INT8** to preserve accuracy and because tooling/kernel support is better. ([NVIDIA Developer Forums][4])

**Implication:** If your target is “standard YOLOv5/7/8/11 ONNX → TensorRT/ORT GPU”, then **INT8 is the commodity path**; **true INT4 for conv layers usually means custom kernels/plugins or non-GPU targets (FPGA/NPU) designed for low-bit conv).**

---

## 2) Successful INT4(-ish) YOLO examples that *do* exist

### A) FPGA/QNN toolchains (this is where you see real W4A? models + ONNX artifacts)

The most concrete “it works + you can download it” example I found is:

* **LPYOLO / FINN + Brevitas**: `sefaburakokcu/finn-quantized-yolo`

  * Uses **Brevitas QAT**, exports to **ONNX**, then deploys via **Xilinx FINN** on PYNQ-Z2. ([GitHub][5])
  * **Pretrained ONNX models are provided**, including **4W4A** (4-bit weights, 4-bit activations) and other mixes. The repo lists direct downloads for `4w4a.onnx` etc. ([GitHub][5])
  * The accompanying LPYOLO paper highlights **4W4A** as a practical accuracy/latency tradeoff on that platform. ([arXiv][6])

This is, today, the clearest public example of **“INT4 weights YOLO-style detector + downloadable ONNX.”**

Related training code:

* `sefaburakokcu/quantized-yolov5` provides the **training/QAT** side (Brevitas-based) that feeds into the FINN deployment repo. ([GitHub][7])

### B) Research PTQ for low-bit YOLO exists — but deployment often falls back to INT8

A key paper is **Q-YOLO (PTQ)**, which explicitly studies **4-bit quantization** and shows naive 4-bit baselines can collapse (e.g., a 4-bit YOLOv5s percentile baseline doing terribly). ([arXiv][8])
But it also states a major practical constraint: **inference frameworks “only support symmetric 8-bit quantization”**, so their real deployment speed tests use **8-bit** in TensorRT/OpenVINO. ([arXiv][8])

There is an *open repo* that implements the method on top of YOLOv5:

* `jie311/Q-YOLO-My` (PTQ pipeline + export hooks) ([GitHub][9])

### C) “It’s hard” signals from real users

* Neuralmagic SparseML has an issue where someone tried **YOLOv8 INT4 training** and reported **large mAP drops (~-20 mAP@0.50)**, which matches the general experience that YOLO conv stacks are fragile at 4-bit without careful design + target-kernel support. ([GitHub][10])

---

## 3) Pre-quantized ONNX INT4 YOLO models (publicly downloadable)

If what you want is *“give me INT4(-weight) YOLO ONNX models I can download today”*, the best match I found is:

* **`sefaburakokcu/finn-quantized-yolo`** provides **pretrained ONNX** models with bitwidth combos including:

  * **4W4A**, **4W2A**, **2W4A**, **6W4A**, **8W3A**, etc. ([GitHub][5])

Outside of this FPGA/QNN ecosystem, I did **not** find widely-used, broadly-compatible **INT4 YOLO ONNX** model zoos for TensorRT/ORT GPU flows — and the framework limitations above largely explain why.

---

## 4) Practical guidance if your goal is “YOLO with INT4 weights”

* If you need **drop-in deployment on GPU (TensorRT/ORT)**: plan on **INT8** (PTQ calibration or QAT). The INT4 path is usually blocked by op/kernel coverage. ([NVIDIA Docs][1])
* If you truly need **W4A? conv**: you’ll likely be in **FPGA / custom NPU** territory. The **Brevitas → (Q)ONNX → FINN** workflow is the most “real” open path I saw, with actual downloadable models. ([GitHub][5])
* If you’re okay with “INT4 research PTQ” but not necessarily accelerated deployment: look at **Q-YOLO** and its open implementation(s); just be aware many deployments still end up using **8-bit** because of tool support. ([arXiv][8])

If you tell me your intended deployment target (TensorRT GPU? CPU? FPGA? specific NPU like RK/VitisAI/etc.), I can narrow this to the *exact* toolchain that will actually run end-to-end for YOLO at 4-bit.

[1]: https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html "Working with Quantized Types — NVIDIA TensorRT Documentation"
[2]: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html "Quantize ONNX models | onnxruntime"
[3]: https://github.com/ultralytics/ultralytics/issues/8337?utm_source=chatgpt.com "4-bit quantisation or support for .gguf · Issue #8337"
[4]: https://forums.developer.nvidia.com/t/performing-int4-fp4-quantization-on-thor-for-yolov7/345343 "Performing int4/fp4 quantization on Thor for Yolov7 - Jetson Thor - NVIDIA Developer Forums"
[5]: https://github.com/sefaburakokcu/finn-quantized-yolo "GitHub - sefaburakokcu/finn-quantized-yolo: Low-Precision YOLO on PYNQ with FINN"
[6]: https://arxiv.org/pdf/2207.10482?utm_source=chatgpt.com "LPYOLO: Low Precision YOLO for Face Detection on FPGA"
[7]: https://github.com/sefaburakokcu/quantized-yolov5 "GitHub - sefaburakokcu/quantized-yolov5: Low Precision(quantized) Yolov5"
[8]: https://arxiv.org/html/2307.04816v1 "Q-YOLO: Efficient Inference for Real-time Object Detection † Equal contribution. ⋆ Corresponding author."
[9]: https://github.com/jie311/Q-YOLO-My "GitHub - jie311/Q-YOLO-My: A Pytorch Quantization for YOLO series: High accuracy, Low latency and Easy to use ~"
[10]: https://github.com/neuralmagic/sparseml/issues/2346?utm_source=chatgpt.com "YOLOv8 - INT4 Training · Issue #2346 · neuralmagic ..."
