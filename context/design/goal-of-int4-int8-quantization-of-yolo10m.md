# Goal: INT4 quantization experiments for YOLOv10m (ONNX-first)

## HEADER

- **Purpose**: State the research goal and define pragmatic implementation paths to generate YOLOv10m ONNX artifacts representing INT4 (plus float compute) for downstream inspection/simulation, with INT8 serving as the “easy baseline” comparison.
- **Status**: Draft
- **Date**: 2025-12-23
- **Dependencies**:
  - `models/cv-models/yolov10m/checkpoints/yolov10m.onnx`
  - `scripts/cv-models/quantize_yolov10m_int8_onnx.sh`
  - `scripts/cv-models/eval_yolov10m_onnx_coco.py`
  - `tests/manual/yolo10_layer_sensitivity_sweep/scripts/run_layer_sensitivity_sweep.py`
  - `scripts/cv-models/make_yolov10m_candidate_schemes.py`
- **Target**: Quantization researchers working on low-bit representation and downstream simulation workflows.

---

## 1. Goal and scope

We want to experiment with **INT4 quantization of YOLOv10m** and compare it against an **INT8 baseline**.

This is research-oriented: accuracy is not the priority at first. The first priority is to generate artifacts that are:

1. **Inspectably INT4 in ONNX**: weights (at least for selected Conv layers) are stored as INT4/UINT4 or as packed 4-bit payloads.
2. **Runnable for sanity**: the model can be executed end-to-end somewhere (CPU is acceptable) to confirm the graph is valid and produces outputs.
3. **Controllable**: we can pick which layers are “INT4” vs “kept float/high precision”, ideally driven by sensitivity analysis.

INT8 is expected to be “easy mode” because it is widely supported by ModelOpt/ORT and already exists in our workflow; the hard part is INT4 for a Conv-heavy detector.

Non-goals (initial phase):

- High accuracy, optimal calibration, or speedups.
- TensorRT engine generation and benchmarking (can be a later phase).
- “True INT4 compute” for Conv layers. If the backend dequantizes to float and computes in FP16/FP32, that is acceptable for this phase.

## 2. What we mean by “INT4 YOLOv10m ONNX”

YOLOv10m is Conv-heavy. Current “INT4 inference” support in mainstream stacks is centered on **MatMul/Gemm weight-only** (LLM-style), not Conv-dominated detectors. Therefore, for YOLOv10m we should treat “INT4 ONNX” as:

- **INT4 weight storage + dequantize to float**: weights are stored as INT4/UINT4 (or as packed 4-bit payloads), then the graph dequantizes them back to FP16/FP32 before `Conv` executes.
- This is still valuable for research because:
  - it is **inspectable** (you can see INT4 weights in the graph),
  - it supports **downstream simulation** (you can model quantization noise and scaling),
  - it is runnable on CPU/GPU because compute happens in float after dequantization.

This is distinct from “backend executes Conv in INT4,” which depends on kernels/importer support and is not required here.

## 3. Existing Community Status

This section documents why “YOLO (Conv-heavy) INT4 quantization” is not a first-class, off-the-shelf workflow in common toolchains today. The intent is to justify investing in a custom ONNX rewrite, even if it grows into non-trivial engineering work.

### 3.1 ONNX (spec-level): INT4 storage is supported; INT4 Conv compute is not standardized

There are three distinct questions that often get conflated:

1. **Can ONNX represent INT4 values in a model file?** Yes.
2. **Can ONNX represent a convolution that *computes* in INT4 (i.e., an “INT4 Conv kernel”)?** Not in the portable, standardized operator set (today).
3. **Do the mainstream quantization tools generate an INT4 Conv representation for YOLO?** Generally no (see ModelOpt/ORT sections below).

What ONNX can represent (relevant to our “inspectable + runnable” goal):

- ONNX added `INT4`/`UINT4` tensor element types for low-bit compression:

> “As a result, two new types were introduced in `onnx==1.17.0` supporting a limited set of operators to enable compression using 4 bit data-types:
> `UINT4`: 4 bit unsigned integer, values in range [0, 15]
> `INT4`: 4 bit signed integer, using two’s complement representation. Values in range [-8, 7].”
>
> Source: ONNX “4 bit integer types” (`tmp/int4_yolo_refs/2025-12-23_07-37-55/onnx_docs/int4.html`) / upstream https://onnx.ai/onnx/technical/int4.html

- ONNX `DequantizeLinear` accepts low-bit inputs (including `tensor(int4)` / `tensor(uint4)`) and produces float/float16 outputs, which enables **INT4 weight storage + float compute**:

> “`T1` in ( … `tensor(int4)` … `tensor(uint4)` … ): The type of the inputs ‘x_zero_point’ and ‘x’.”
>
> Source: ONNX `DequantizeLinear` type constraints (`tmp/int4_yolo_refs/2025-12-23_07-37-55/onnx_docs/dequantizelinear.html`) / upstream https://onnx.ai/onnx/operators/onnx__DequantizeLinear.html

What ONNX does *not* provide as a portable standard today (this is the spec-level blocker for “true INT4 Conv compute”):

- The **standard** quantized convolution operators are constrained to **8-bit** inputs/weights:

> “`T1` in ( `tensor(int8)`, `tensor(uint8)` ): Constrain input type to 8-bit integer tensor.”
>
> “`T2` in ( `tensor(int8)`, `tensor(uint8)` ): Constrain filter type to 8-bit integer tensor.”
>
> Source: ONNX `QLinearConv` type constraints (`tmp/int4_yolo_refs/2025-12-23_07-37-55/onnx_docs/qlinearconv.html`) / upstream https://onnx.ai/onnx/operators/onnx__QLinearConv.html

> “`T1` in ( `tensor(int8)`, `tensor(uint8)` ): Constrain input x and its zero point data type to 8-bit integer tensor.”
>
> “`T2` in ( `tensor(int8)`, `tensor(uint8)` ): Constrain input w and its zero point data type to 8-bit integer tensor.”
>
> Source: ONNX `ConvInteger` type constraints (`tmp/int4_yolo_refs/2025-12-23_07-37-55/onnx_docs/convinteger.html`) / upstream https://onnx.ai/onnx/operators/onnx__ConvInteger.html

This means:

- You *can* build an “INT4 YOLOv10m ONNX” that stores selected Conv weights as `INT4` and inserts `DequantizeLinear` to float before `Conv` (runnable on CPU/GPU because Conv compute stays float).
- You *cannot* express “Conv computes in INT4” as a **portable, standard** ONNX graph via `QLinearConv` / `ConvInteger`, because those ops do not accept `INT4/UINT4`.
- This is a **representation/spec gap**, not a “quantization algorithm is impossible” gap: we can implement our own quantizer + graph rewrite to store weights in INT4 and dequantize to float, but “true INT4 Conv compute” would require non-standard/custom ops and kernels.

### 3.2 NVIDIA ModelOpt (ONNX PTQ): INT4 mode targets Gemm/MatMul, not Conv

- In the ModelOpt ONNX quantization entrypoint, the default “op types quantized” list is explicitly documented per mode; INT4 mode is listed as:
  - INT4 mode: **Gemm, MatMul**
  - Source (repo checkout in this workspace): `extern/TensorRT-Model-Optimizer/modelopt/onnx/quantization/quantize.py`
  - Upstream reference: https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/main/modelopt/onnx/quantization/quantize.py
- This aligns with the broader ecosystem focus: INT4 is most commonly implemented for GEMM/MatMul weight-only use cases, and Conv-heavy detection models do not benefit from those kernels unless a Conv INT4 path exists (which is not the common case).

Proof (from the repo source we vendor in `extern/`):

> “The default operation types that this ONNX post-training quantization (PTQ) tool quantizes in different quantization
> modes are as follows:
> - INT8 mode: … Conv … MatMul …
> - INT4 mode: Gemm, MatMul
> - FP8 mode: Conv, Gemm, MatMul”
>
> Source: `extern/TensorRT-Model-Optimizer/modelopt/onnx/quantization/quantize.py`

### 3.3 ONNX Runtime (ORT): INT4 tooling is MatMul/Gather weight-only

- ORT’s quantization docs describe INT4/UINT4 as a block-wise **weight-only** flow and describe supported operator families as MatMul/Gather-centric.
  - https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
- ORT’s quantizer implementation defaults reinforce this focus:
  - Default operator types: `{"MatMul"}`
  - Default quant axes mapping includes `{"MatMul": 0, "Gather": 1}`
  - Source (repo checkout in this workspace): `extern/onnxruntime/onnxruntime/python/tools/quantization/matmul_nbits_quantizer.py`
  - Upstream reference: https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/matmul_nbits_quantizer.py
- For a Conv-heavy model like YOLOv10m, this means “apply ORT INT4 tooling” is unlikely to produce an INT4 representation for the bulk of the model, regardless of accuracy concerns.

Proof (from ORT’s contrib operator docs + quantizer defaults we vendor in `extern/`):

> “MatMulNBits performs a matrix multiplication where the right-hand-side matrix (weights) is quantized to N bits.”
>
> Source: `extern/onnxruntime/docs/ContribOperators.md` (section `com.microsoft.MatMulNBits`)

> `self.op_types_to_quantize = set(op_types_to_quantize) if op_types_to_quantize else {"MatMul"}`
>
> “Default {MatMul: 0, Gather: 1}”
>
> Source: `extern/onnxruntime/onnxruntime/python/tools/quantization/matmul_nbits_quantizer.py`

### 3.4 Intel Neural Compressor (INC) / ONNX Neural Compressor: also MatMul-centric

- Community INC-style “INT4 ONNX” artifacts commonly rely on ORT’s MatMul-focused int4 operators. For example, Intel’s published INT4 Whisper ONNX model notes an ORT requirement to support a MatMul INT4 operator (`MatMulFpQ4`), which is a transformer/MatMul pathway rather than Conv.
  - https://huggingface.co/Intel/whisper-medium-onnx-int4-inc
- The ONNX-focused neural compressor project emphasizes weight-only quantization for LLMs via MatMul N-bits quantizers (again highlighting MatMul rather than Conv as the practical target for INT4).
  - https://github.com/onnx/neural-compressor

### 3.5 Ultralytics YOLO tooling: no “INT4 quant/export” workflow (as of late 2025)

- Ultralytics’ own issue tracker includes user requests for 4-bit quantization support that were closed as “not planned,” reinforcing that “YOLO INT4 export” is not a common supported path in mainstream YOLO tooling.
  - https://github.com/ultralytics/ultralytics/issues/8337
  - Snapshot (includes `stateReason:"NOT_PLANNED"` in embedded JSON): `tmp/int4_yolo_refs/2025-12-23_07-37-55/ultralytics_issue_8337.html`

Proof (from the downloaded snapshot JSON payload):

> `"state":"CLOSED","stateReason":"NOT_PLANNED"`
>
> Source: `tmp/int4_yolo_refs/2025-12-23_07-37-55/ultralytics_issue_8337.html` (embedded `react-app.embeddedData` JSON)

### 3.6 Community signals: “INT4 for LLMs; INT8 for CNNs” and real-world pain

These are “status-of-practice” datapoints that show what practitioners run into when trying YOLO INT4 today.

- NVIDIA Developer Forums (Jetson Thor / YOLOv7):

> “Usually, INT4/FP4 quantization is applied to the LLM model.
>
> For CNN, it’s more common to use INT8 to preserve precision.”
>
> Source: `tmp/int4_yolo_refs/2025-12-23_07-37-55/nvidia_forum_yolov7_int4_fp4.html` (embedded JSON-LD) / upstream https://forums.developer.nvidia.com/t/performing-int4-fp4-quantization-on-thor-for-yolov7/345343

- NeuralMagic SparseML issue (YOLOv8 INT4 training report):

> “However, the performance is quite inferior (-20mAP@0.50)?”
>
> Source: `tmp/int4_yolo_refs/2025-12-23_07-37-55/sparseml_issue_2346.html` (embedded JSON-LD) / upstream https://github.com/neuralmagic/sparseml/issues/2346

### 3.7 (Context) TensorRT: INT4 is documented as GEMM weight-only

Even though TensorRT is not the priority for this phase, the reason many toolchains focus INT4 on GEMM/MatMul becomes clear from TensorRT’s own documentation:

- TensorRT states INT4 quantization support as **GEMM weight-only quantization** (not general Conv INT4 quantization).
  - https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/troubleshooting.html
- TensorRT “working with quantized types” further describes weight-only INT4 as an optimization for GEMM layers, with high-precision compute and low-bit weight storage + dequantization.
  - https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html

## 4. What the community has tried (with extracted tables)

Downloaded references for this section live under:

- `tmp/int4_yolo_refs/2025-12-23_07-37-55/`

### 4.1 PTQ evidence: Q-YOLO shows “naive 4-bit” collapse and partial recovery

Source:

- Paper: `tmp/int4_yolo_refs/2025-12-23_07-37-55/Q-YOLO_2307.04816.pdf`
- Extracted text: `tmp/int4_yolo_refs/2025-12-23_07-37-55/Q-YOLO_2307.04816_text.txt`
- arXiv: https://arxiv.org/abs/2307.04816
- Public repo (YOLOv5-based implementation): https://github.com/jie311/Q-YOLO-My
  - Snapshot: `tmp/int4_yolo_refs/2025-12-23_07-37-55/q-yolo-my_README.md`

The paper reports severe collapse for a naive 4-bit baseline (Percentile PTQ) and a partial recovery with their method (Q-YOLO). Below is a condensed extract from the paper’s Table 1 (COCO val2017, AP metrics):

| Model | Bits (W-A) | Method | AP | AP50 | AP75 |
| --- | --- | --- | ---: | ---: | ---: |
| YOLOv5s | 32-32 | Real-valued | 37.4 | 57.1 | 40.1 |
| YOLOv5s | 4-4 | Percentile | 7.0 | 14.2 | 6.3 |
| YOLOv5s | 4-4 | Q-YOLO | 14.0 | 26.2 | 13.5 |
| YOLOv5m | 32-32 | Real-valued | 45.1 | 64.1 | 49.0 |
| YOLOv5m | 4-4 | Percentile | 19.4 | 35.6 | 19.1 |
| YOLOv5m | 4-4 | Q-YOLO | 28.8 | 46.0 | 30.5 |
| YOLOv7 | 32-32 | Real-valued | 50.8 | 69.6 | 54.9 |
| YOLOv7 | 4-4 | Percentile | 16.7 | 26.9 | 17.8 |
| YOLOv7 | 4-4 | Q-YOLO | 37.3 | 55.0 | 40.9 |

Interpretation for our YOLOv10m experiments:

- “4-bit everywhere” can catastrophically fail without careful activation-range handling.
- There are PTQ methods that partially recover accuracy, but they introduce extra algorithmic complexity (activation histogram search, per-layer range rules, etc.).
- For our current goal (inspectable/run-anywhere INT4 ONNX), this motivates starting with a naive representation-first approach, then optionally layering on research PTQ methods.

Deployment reality note (from the same paper):

> “As most current inference frameworks only support symmetric quantization and 8-bit quantization, we had to choose a symmetric 8-bit quantization scheme…”
>
> Source: `tmp/int4_yolo_refs/2025-12-23_07-37-55/Q-YOLO_2307.04816_text.txt` (extracted from `Q-YOLO_2307.04816.pdf`)

### 4.2 QAT evidence: LPYOLO shows 4W4A is practical in a specialized deployment stack

Source:

- Paper: `tmp/int4_yolo_refs/2025-12-23_07-37-55/LPYOLO_2207.10482.pdf`
- Extracted text: `tmp/int4_yolo_refs/2025-12-23_07-37-55/LPYOLO_2207.10482_text.txt`
- arXiv: https://arxiv.org/abs/2207.10482

The paper reports quantization-aware training (Brevitas) for a TinyYOLOv3-derived model deployed via FINN on FPGA. Extracted tables (WiderFace validation, mAP by difficulty; and latency breakdown):

Accuracy results (Table 2, WiderFace mAP):

| Quantization | Easy | Medium | Hard |
| --- | ---: | ---: | ---: |
| Non-Quantized | 0.765 | 0.631 | 0.294 |
| 2W4A | 0.671 | 0.469 | 0.199 |
| 3W5A | 0.733 | 0.563 | 0.247 |
| 4W2A | 0.705 | 0.521 | 0.224 |
| 4W4A | 0.757 | 0.590 | 0.261 |
| 6W4A | 0.764 | 0.608 | 0.274 |
| 8W3A | 0.740 | 0.571 | 0.252 |

Latency results (Table 3, ms per image on PYNQ-Z2; PS-only vs PS+PL):

| Quantization | Preprocessing | CNN | Postprocessing |
| --- | ---: | ---: | ---: |
| Non-Quantized | 18.5 | 5017.9 | 7.5 |
| 2W4A | 35.0 | 43.5 | 10.0 |
| 3W5A | 35.0 | 81.5 | 10.0 |
| 4W2A | 35.0 | 37.3 | 10.0 |
| 4W4A | 35.0 | 52.3 | 10.0 |
| 6W4A | 35.0 | 63.5 | 10.0 |
| 8W3A | 35.0 | 49.1 | 10.0 |

Interpretation for our YOLOv10m experiments:

- 4W4A can work when the whole toolchain is built around it (QAT + target hardware/compiler).
- This is not directly transferable to YOLOv10m ONNX + ORT, but it is evidence that “YOLO-ish + 4-bit” is feasible in a purpose-built stack.

### 4.3 Public artifacts: downloadable low-bit YOLO(-ish) ONNX models exist (FPGA/QONNX ecosystem)

Source:

- Repo README snapshot: `tmp/int4_yolo_refs/2025-12-23_07-37-55/finn-quantized-yolo_README.md`
- Repo: https://github.com/sefaburakokcu/finn-quantized-yolo

The repo provides direct download links for ONNX models including `4w4a.onnx` and other bitwidth mixes (see their README table). This is useful as a reference for “what a real low-bit YOLO export looks like” in the Brevitas/QONNX/FINN world.

## 5. Recommended ways forward for YOLOv10m INT4

This section groups approaches by expected engineering complexity. INT8 is baseline and already supported; the focus here is “how do we get an INT4 representation for a Conv-heavy YOLOv10m ONNX model”.

### 5.1 Naive approach (recommended first): INT4 weight storage + DequantizeLinear (float compute)

#### Artifact shape (ONNX)

- **Select Conv nodes to quantize**
  - From sensitivity top‑K mapped to ONNX node names, and/or explicit include/exclude lists.
- **Quantize each selected Conv’s weight initializer**
  - Store weights as `INT4/UINT4` using a simple deterministic scheme (e.g., symmetric per‑output‑channel absmax).
- **Insert `DequantizeLinear` so Conv compute stays float**
  - Replace the Conv weight input with `DequantizeLinear(int4_weight, scale[, zero_point])` so `Conv` still consumes FP16/FP32.
- **Leave unselected Convs unchanged**
  - Keep original float weights and `Conv` nodes intact.

#### Why this is the right first step

- It achieves “INT4 in ONNX” without requiring any INT4 Conv kernels.
- It gives us precise control over which layers are affected (sensitivity-driven mixed precision).
- It is the minimum engineering needed to start downstream simulation and inspection work.

#### CPU sanity checks

- Prefer dequantizing to FP32 and running the model in FP32 on CPU for maximum compatibility.

#### If we develop our own quantization methods (beyond “use ModelOpt as-is”)

- **Target A (recommended first): INT4 representation + float Conv compute**
  - Implement a quantization algorithm (however naive or research‑grade) that produces per‑tensor/per‑channel scales + INT4 weights.
  - Implement an ONNX graph rewrite that stores weights as `INT4/UINT4` and inserts `DequantizeLinear` so `Conv` still runs in FP16/FP32.
  - No custom operator kernels are required; this is the only path that is both **inspectable** and **runnable everywhere** today.

- **Target B: “True INT4 Conv compute” (requires non-standard ops and kernels)**
  - Not expressible with standard ONNX quantized Conv ops (`QLinearConv` / `ConvInteger` are 8-bit-only), so it requires a custom operator and runtime support.
  - Implementation options:
    - **ORT custom op library**: define a domain like `com.yourorg::ConvNBits`, implement CPU and/or CUDA kernels, and load it via `SessionOptions.register_custom_ops_library(...)`.
    - **ORT fork / contrib op**: add a contrib op + kernels inside `extern/onnxruntime` (similar to `com.microsoft::MatMulNBits`) and rebuild ORT wheels.
    - **TensorRT plugin** (later): if the goal is TensorRT execution, implement a TRT plugin layer and convert ONNX to TRT with plugin registration.
  - Weight representation will usually be either:
    - true `tensor(int4)` / `tensor(uint4)` initializers, or
    - packed 4‑bit payloads in `tensor(uint8)` (portable storage; 2× 4‑bit values per byte), plus scale/zero‑point metadata.
  - For CPU-only “sanity” of true INT4 compute, the simplest correct kernel can unpack INT4 → int8/float internally (slow is fine for research), but it still requires the custom op plumbing.

### 5.2 Research methods (optional second phase): activation-aware PTQ (Q-YOLO style)

If we later want “less broken” 4-bit behavior, Q-YOLO’s main contribution is activation-range handling (histogram-based truncation selection). This can be layered on top of the naive approach by improving how activation quantization parameters are chosen (but it adds meaningful complexity: calibration data readers, per-layer distribution analysis, and range search).

### 5.3 Research methods (optional second phase): QAT (Brevitas/QONNX-style)

If we need genuinely robust 4-bit behavior, QAT is the path with the strongest evidence for “YOLO-ish models at 4-bit” (LPYOLO-style). However, it is a different workflow:

- requires access to training code/checkpoints and fine-tuning,
- likely produces a different ONNX topology than the existing YOLOv10m ONNX checkpoint,
- often targets specialized deployment stacks (FINN, FPGA/NPU).

## 6. Success criteria (for this phase)

For each INT4 candidate (and the INT8 baseline):

- The ONNX passes `onnx.checker.check_model`.
- We can programmatically confirm:
  - which Conv nodes are “INT4-weight” vs float,
  - at least some weights are stored as INT4/UINT4 (or packed 4-bit payloads if we fall back).
- We can run a CPU smoke inference (random input) and produce a reproducible summary under `tmp/`.

## 7. Next steps (implementation-oriented)

1. Implement the naive INT4 ONNX rewrite (Section 5.1) as a script under `scripts/cv-models/`.
2. Add a CPU smoke runner for the INT4 artifact (random input), and optionally a tiny COCO slice run if feasible.
3. Keep INT8 via ModelOpt as baseline comparison using the existing wrapper.
