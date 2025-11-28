# 量化敏感层分析方法及工具-调研V1.0

## 一. 什么是“量化敏感层分析”？

量化敏感层分析（Quantization Sensitivity Analysis）的核心目的：
找出模型中对量化最敏感的层（quantization-critical layers），从而有针对性地：
*   决定哪些层保留 FP16 / INT8
*   哪些层可以安全量化到 INT4
*   提高整网量化后的精度、稳定性、吞吐率

适用于 Conv 模型（ResNet/YOLO）、Transformer（LLM/VLM）、NPU 部署。

## 二. 方法分类

### 2.1 方法一：梯度/损失敏感度（FIT / Fisher Information）
*   **代表：** FIT (Fisher Information Trace)、SQNR-based、gradient norm、loss jump
*   **原理：** 衡量量化噪声对 Loss 的影响。常见实现：对每层计算分数，分数越高层越敏感。
*   **优点：** 快、成本低。适合 CNN / Transformer。NVIDIA PTQ 中有类似度量。
*   **缺点：** 一阶近似，不如二阶方法稳定。

### 2.2 方法二：二阶 Hessian 方法（HAWQ、HAWQ-V2、Hessian Trace）
*   **代表：** HAWQ、HAWQ-V2、Hessian-based PTQ
*   **原理：** 对量化噪声 $\Delta W$ 的敏感度近似为：
    (此处原文有公式图片，通常涉及 Hessian 矩阵)
    用 Hutchinson + Power Iteration 得到层的 Hessian 最大特征值 $\lambda_{max}$ 作为敏感度排序。
*   **优点：** 准确度最高。能稳定支持 INT4、INT3、INT2。
*   **缺点：** 计算量大（需 HVP）。

### 2.3 方法三：校准集模拟量化（Simulated Quantization, SQ）
*   **代表：** TensorRT PTQ、QAT、AutoQ、BRECQ 前传模拟
*   **原理：** 用校准数据对每层插入 FakeQuant，观测输出误差（MSE/KL/ACT 距离）。
*   **常用指标：**
    *   Activation Cosine Distance
    *   MSE (per-tensor / per-channel)
    *   KL Divergence
*   **优点：** 直接作用输出，工程上最常用。NVIDIA TensorRT 官方支持。
*   **缺点：** 需要完整前向推理。

### 2.4 方法四：感知权值重要性（AWQ / GPTQ / QDrop）
*   **适用于：** LLM/CV-Transformer
*   **原理：** 量化误差放在“不重要”的权值上。
*   **工具：**
    *   AWQ（Activation-aware weight quantization）
    *   GPTQ（第二阶块wise拟合）
*   **优点：** 是目前 LLM 量化最强方法（INT4 可保持极高精度）。
*   **缺点：** 对 CNN 通用性一般。

## 三. 工具

### 🟩 NVIDIA 官方工具链（最推荐）

#### 1. NVIDIA ModelOpt（2024–2025 主推量化框架）
*   **适用于：** CNN、Transformer、LLM
*   **特点：**
    *   自动敏感度分析
    *   支持 INT8 / INT4 / FP8 / 混合精度
    *   输出可直接用于 TensorRT / NIM / Jetson / NPU
*   **功能：**
    *   `modelopt.quantization.auto_quantize()`
    *   `modelopt.quantization.sensitivity_analyzer`
*   **可生成：**
    *   per-layer MSE
    *   per-layer Hessian metric
    *   per-layer mixed-precision suggestions

#### 2. TensorRT PTQ (Post-Training Quantization Tools)
*   **方法：**
    *   per-layer MSE
    *   per-layer KL
    *   per-layer cosine similarity
    *   calibration cache
*   **输出：** INT8 engine
*   **工具：** `trtexec --sparsity=enable --quantize=INT8 --layer-info`

#### 3. NVIDIA Tao Toolkit
*   自动 PTQ + 自带敏感度评估
*   适合企业 GPU 量产部署

### 🟧 PyTorch / Open-source 工具

#### 4. Intel Neural Compressor (formerly LPOT)
*   **原名：** Intel Low Precision Optimization Tool (LPOT)
*   **特点：** 
    *   支持数十种敏感度分析指标
    *   兼容 PyTorch/ONNX/TensorFlow
    *   提供 per-layer sensitivity 报告
    *   **Accuracy-Aware Tuning:** 自动混合精度搜索

#### 5. Microsoft NN-Tool / Olive
*   自动量化搜索
*   有 layer-wise sensitivity & MSE

#### 6. HAWQ / HAWQ-V2 官方实现
*   CNN + Transformer 通用
*   支持 INT8–INT2

#### 7. BRECQ（最强 PTQ）
*   计算 layer-wise reconstruction error
*   比 HAWQ 更精确

#### 8. GPTQ / AWQ
*   Transformer / LLM 量化专用

## 四. 结论与落地建议（短期 vs 中期）

*   **短期（工程可复现，低风险）：** 使用 NVIDIA ModelOpt 的 `auto_quantize` + SmoothQuant/AWQ 流程做 PTQ，再用 TensorRT-LLM 部署；FIT 作为快速零样本敏感度筛选器。
*   **中期（追求极致 4-bit）：** 结合 SVDQuant 或社区高性能实现（Nunchaku 等），并在支持 NVFP4 的 GPU 上测试性能/精度折中。若 PTQ 不够好，再做 QAT。

## 参考论文

*   **TensorRT Model Optimizer (ModelOpt) — API & auto_quantize.**
    *   [nvidia.github.io](https://nvidia.github.io/TensorRT-Model-Optimizer/reference/generated/modelopt.torch.quantization.model_quant.html)
*   **NVIDIA 技术博客：Optimizing LLMs & NVFP4 相关文章。**
    *   [NVIDIA Developer](https://developer.nvidia.com/blog/optimizing-llms-for-performance-and-accuracy-with-post-training-quantization/)
*   **AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration。**
    *   [arXiv:2306.00978](https://arxiv.org/abs/2306.00978)
*   **SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models。**
    *   [arXiv:2411.05007](https://arxiv.org/abs/2411.05007)
*   **HAWQ：Hessian-Aware Quantization（原始论文）。**
    *   [arXiv:1905.03696](https://arxiv.org/abs/1905.03696)
*   **FIT: A Metric for Model Sensitivity。**
    *   [arXiv:2210.08502](https://arxiv.org/abs/2210.08502)
*   **SmoothQuant（activation-aware smoothing for PTQ）。**
    *   [arXiv:2211.10438](https://arxiv.org/pdf/2211.10438)
