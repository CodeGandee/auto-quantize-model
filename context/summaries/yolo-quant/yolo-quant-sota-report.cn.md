# You Only Look Once (YOLO) low-bit quantization 现状（Post-Training Quantization (PTQ) + Quantization-Aware Training (QAT)）

### 0) 元信息（Metadata）
- **Title**: YOLO low-bit quantization 现状（PTQ + QAT）
- **Date**: 2026-02-02
- **Scope**: Microsoft COCO (Common Objects in Context, COCO) 上 YOLOv5 + YOLOv7 系列结果（基于下列 3 篇公开论文）
- **Package contents**: 本报告 markdown + `figures/` 目录（提取的关键图）
- **Papers reviewed (public)**:
  - **Reducing the Side-Effects of Oscillations in Training of Quantized YOLO Networks (gupta2024-oscillations)**
    - Venue: IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2024.
    - IEEE citation: K. Gupta and A. Asthana, “Reducing the Side-Effects of Oscillations in Training of Quantized YOLO Networks,” WACV, 2024. doi: N/A.
    - arXiv: `https://arxiv.org/abs/2311.05109`
    - Open access PDF (Computer Vision Foundation (CVF)): `https://openaccess.thecvf.com/content/WACV2024/papers/Gupta_Reducing_the_Side-Effects_of_Oscillations_in_Training_of_Quantized_YOLO_WACV_2024_paper.pdf`
    - Local TeX source (workspace-only): 见文末 **Local references**
  - **Overcoming Oscillations in Quantization-Aware Training (nagel2022-oscillations)**
    - Venue: International Conference on Machine Learning (ICML), 2022.
    - IEEE citation: M. Nagel, M. Fournarakis, Y. Bondarenko, and T. Blankevoort, “Overcoming Oscillations in Quantization-Aware Training,” ICML, 2022. doi: N/A.
    - arXiv: `https://arxiv.org/abs/2203.11086`
    - Proceedings PDF (Proceedings of Machine Learning Research (PMLR)): `https://proceedings.mlr.press/v162/nagel22a/nagel22a.pdf`
    - Local TeX source (workspace-only): 见文末 **Local references**
  - **Q-YOLO: Efficient Inference for Real-time Object Detection (qyolo-2023)**
    - IEEE citation: M. Wang, H. Sun, J. Shi, X. Liu, B. Zhang, and X. Cao, “Q-YOLO: Efficient Inference for Real-time Object Detection,” 2023. doi: N/A.
    - arXiv: `https://arxiv.org/abs/2307.04816`
    - Code (GitHub): `https://github.com/Meize0729/Q-YOLO`
    - Local TeX source (workspace-only): 见文末 **Local references**
- **Primary evaluation settings assumed in this report**:
  - Dataset(s): COCO 2017（论文中声明使用 `train2017` 做 calibration/training；`val2017` 做 evaluation）
  - Metric(s): COCO Average Precision (AP)（也常写作 mean Average Precision (mAP)@0.5:0.95，除非论文另有说明）
  - Image size / preprocessing: 640×640（`qyolo-2023` 明确给出；`gupta2024-oscillations` 使用 YOLO defaults）
  - Baseline 32-bit floating point (FP32) reference: 各论文表格中报告的 FP32 数值（不同论文之间不保证完全一致）

### 1) 执行摘要（Executive summary，最多 1 页）
- **Best reported Post-Training Quantization (PTQ) result (headline)**: Q-YOLO PTQ @ W4A4 on COCO val2017 — YOLOv7x: AP 37.6（Δ −14.9 vs FP32 52.5）。
- **Best reported Quantization-Aware Training (QAT) result (headline)**: Gupta et al. QAT（Exponential Moving Average (EMA) + post-hoc Quantization Correction (QC)）@ “4-bit” on COCO — YOLOv7: AP 48.9（Δ −2.3 vs FP32 51.2）。
- **Key takeaways (3–7 bullets)**:
  - 在被审阅的 PTQ pipeline 中，8-bit integer (INT8) PTQ 对 YOLOv5/v7 可以接近无损，但 4-bit PTQ 相比 FP32 仍会出现大幅 accuracy collapse。
  - 对 YOLO detection 来说，同一 bit-width 下，QAT（甚至 3–4 bits）往往能显著优于 PTQ。
  - 由 Straight-Through Estimator (STE) 驱动的 oscillations（weights 以及 quantization scale/step parameters）是 low-bit QAT 的核心稳定性/精度瓶颈之一。
  - “Layer exceptions”（例如 first/last layers 维持更高精度）仍是维持 accuracy 的关键手段。
  - 部署现实：主流 inference stack 大多标准化在 INT8（且常要求 symmetric），因此除非具备自定义 kernels/hardware，否则 4-bit integer (INT4) 的落地空间有限。
- **What’s still unclear / not comparable across papers (3–7 bullets)**:
  - 各论文的 FP32 baseline 存在小幅差异（可能来自 codebase/training/eval 设置差别），导致跨论文 Δ 对比噪声较大。
  - “4-bit” 并不总是同一个定义：有的工作对 first/last layers 做高精度保留；有的未完整声明所有 layer exceptions。
  - PTQ calibration 细节（sample selection、augmentations、observer settings 等）在论文间不一致或描述不完整。
  - speed/latency 并未在同一部署目标下统一报告（INT4 vs INT8、GPU vs CPU、kernel availability）。

### 2) 定义与可比性说明（Definitions and comparability notes）
- **Bit-width notation**:
  - `Wb-Ab` 表示 weight bit-width 为 `b`、activation bit-width 为 `b`（例如 `W4A4`）。
  - 如果 first/last layers 存在 exceptions，本报告会在 Notes 列标注（论文中也常见该做法）。
- **Quantization scope**:
  - **Post-Training Quantization (PTQ)**（Q-YOLO）: quantizes backbone/neck/head；同时保留 input/output layer accuracy（意味着这些层通常 retained higher precision）。
  - **Quantization-Aware Training (QAT)**（Gupta et al.）: quantizes weights + activations；常见做法是 first/last layers 保持 8-bit，同时也报告了“fully 4-bit” variant。
- **Calibration protocol (PTQ)**:
  - Q-YOLO: 1500 张 COCO `train2017` 图像做 calibration；activation histograms 使用 2048 bins；通过 histogram search 选择 clipping/range。
- **Training protocol (QAT)**:
  - Gupta et al.: 从 pretrained FP model 开始做 100-epoch QAT；Learned Step Size Quantization (LSQ)-style learned step size；对 latent weights/scale factors 做 Exponential Moving Average (EMA)；外加 1-epoch 的 post-hoc Quantization Correction (QC)。
- **Hardware / kernel availability assumptions**:
  - Q-YOLO 明确讨论 TensorRT/OpenVINO 部署；并报告 INT8 speed（受 framework constraint 影响）。
- **Important comparability caveats**:
  - Detection AP 对 preprocessing、Non-Maximum Suppression (NMS)/postprocess 以及具体 YOLO implementation 很敏感；跨论文对比应主要视为方向性参考。
  - 论文中报告的 “4-bit” 常包含 layer precision exceptions；除非论文明确说明，否则应把 “W4A4” 视为 “mostly W4A4”。

### 3) PTQ 现状（results + interpretation）

#### 3.1 Snapshot table（best/representative PTQ results）
| Paper | Model | Reported accuracy | FP32 accuracy | Δ vs FP32 | W bits | A bits | Image size |
|---|---|---:|---:|---:|---:|---:|---:|
| qyolo-2023 | YOLOv5s | 14.0 | 37.4 | −23.4 | 4 | 4 | 640 |
| qyolo-2023 | YOLOv5m | 28.8 | 45.1 | −16.3 | 4 | 4 | 640 |
| qyolo-2023 | YOLOv7 | 37.3 | 50.8 | −13.5 | 4 | 4 | 640 |
| qyolo-2023 | YOLOv7x | 37.6 | 52.5 | −14.9 | 4 | 4 | 640 |

- Dataset: COCO val2017
- Metric: AP（COCO AP / mAP@0.5:0.95）
- Quant scheme: weights (W) symmetric per-channel；activations (A) asymmetric per-layer；UH activation quantization (UH) histogram search for activation range
- Calibration: 1500 COCO train2017 images
- Notes: input/output layers 通常会 retained（非端到端 fully quantized）

#### 3.2 解读（Interpretation）
##### Method overview（方法概述）
Q-YOLO（qyolo-2023）是一个面向 YOLO 的 end-to-end PTQ pipeline，核心针对 YOLO 中常见的 activation distribution imbalance。其关键点是 UH activation quantization (UH)：固定一个经验性的 activation lower bound（与 Sigmoid Linear Unit (SiLU) 的分布形态相关），再用 histogram-based search 选择 activation upper bound，使得 quantization error（用 Mean Squared Error (MSE) proxy）最小。

```text
# Q-YOLO PTQ pipeline（简化版；核心 = 用 UH 选 activation range）
calib_data = sample(COCO_train2017, n=1500)                 # calibration set

for layer in model.layers:
    if layer in {input_layer, output_layer}:                # sensitive layers
        keep_fp_or_higher_precision(layer)                  # 论文中说明这些层通常 retained
        continue

    # Weights: symmetric per-channel quantization（MinMax range）
    w_min, w_max = per_channel_minmax(layer.W)              # per-channel 处理 outliers
    layer.W_q = quantize_symmetric(layer.W, w_min, w_max, bits=W_bits)

    # Activations: asymmetric per-layer quantization（UH range selection）
    acts = run_fp32_forward_and_collect(layer, calib_data)  # 用 calibration data 收集 activation samples
    hist = histogram(acts, bins=2048)                       # 2048-bin histogram
    a_min = -0.2785                                         # fixed lower bound（SiLU empirical）
    a_max = uh_search_max(hist, a_min, bits=A_bits)          # 选 upper bound 以最小化 MSE
    layer.A_qparams = (a_min, a_max)                         # per-layer（asymmetric）
```

```text
# UH search（简化版）：在 histogram 的 candidate cutoff 上搜索 a_max，使 MSE 最小
def uh_search_max(hist, a_min, bits):
    best_max, best_mse = None, +inf
    for i in range(128, 2048):                               # candidate cutoffs（histogram bins）
        a_max = bin_center(hist, i)                          # 候选 upper bound
        fp = hist_centers(hist, 0, i)                        # 用 bin centers 近似 FP32 values
        qdq = dequantize(quantize_asymmetric(fp, a_min, a_max, bits))
        mse = mean_squared_error(fp, qdq)                    # Mean Squared Error (MSE) proxy
        if mse < best_mse:
            best_max, best_mse = a_max, mse
    return best_max
```

- Evidence: Table `exp_main` in qyolo-2023（见 **Metadata** 中的 arXiv 链接，以及论文中对 UH algorithm 的描述）。
- Practical takeaway: 在 ≤4 bits 时，activation range selection 往往是 YOLO PTQ 的主导因素；naive MinMax/Percentile 可能直接 collapse。

##### Limitations（局限与可比性问题）
- **4-bit PTQ 会导致显著 accuracy loss**：论文明确指出在 4-bit 下会出现显著的 accuracy drop（与 4-bit integer 表达能力相关）。
  > “When quantizing models to 4 bits, the accuracy experiences a significant loss due to the reduced expressiveness of 4-bit integer representation.”  
  > — qyolo-2023, Main results
- **input/output layers 更敏感，通常会 retained**：论文说明 input/output layers 对 accuracy 更敏感，通常保留这些 layers 的原始 accuracy。
  > “The input and output layers… are more sensitive to the loss of accuracy… the original accuracy of these layers is usually retained.”  
  > — qyolo-2023, Implementation Details
- **weights 比 activations 更敏感**：论文指出 quantizing weights 相比 quantizing activations 会带来更大的 performance degradation。
  > “Compared to quantizing the activation values, quantizing the weights consistently induces larger performance degradation.”  
  > — qyolo-2023, Quantization types discussion
- **部署工具链约束倾向于 INT8/symmetric**：论文指出多数 inference frameworks 仅支持 symmetric quantization 与 8-bit quantization，因此在部署速度测试中选择 symmetric 8-bit quantization scheme。
  > “As most current inference frameworks only support symmetric quantization and 8-bit quantization, we had to choose a symmetric 8-bit quantization scheme…”  
  > — qyolo-2023, Inference speed

#### Appendix (PTQ): Q-YOLO main table（AP only）
以下数值来自 Table `exp_main` in qyolo-2023（Bits 为 W-A）。

| Model | FP32 (32-32) | MinMax (8-8) | Percentile (8-8) | Q-YOLO (8-8) | Percentile (4-4) | Q-YOLO (4-4) |
|---|---:|---:|---:|---:|---:|---:|
| YOLOv5s | 37.4 | 37.2 | 36.9 | 37.4 | 7.0 | 14.0 |
| YOLOv5m | 45.1 | 44.9 | 44.6 | 45.1 | 19.4 | 28.8 |
| YOLOv7  | 50.8 | 50.6 | 50.5 | 50.7 | 16.7 | 37.3 |
| YOLOv7x | 52.5 | 52.3 | 52.0 | 52.4 | 36.8 | 37.6 |

### 4) QAT 现状（results + interpretation）

#### 4.1 Snapshot table（best/representative QAT results）
| Paper | Model | Reported accuracy | FP32 accuracy | Δ vs FP32 | W bits | A bits | Image size |
|---|---|---:|---:|---:|---:|---:|---:|
| gupta2024-oscillations | YOLOv5s | 34.0 | 37.4 | −3.4 | 4 | 4 | 640 |
| gupta2024-oscillations | YOLOv7 | 48.9 | 51.2 | −2.3 | 4 | 4 | 640 |
| gupta2024-oscillations | YOLOv5s | 30.2 | 37.4 | −7.2 | 3 | 3 | 640 |
| gupta2024-oscillations | YOLOv7 | 46.8 | 51.2 | −4.4 | 3 | 3 | 640 |

- Dataset: COCO
- Metric: mAP (AP)
- Quant scheme: per-tensor（LSQ-style）weight+activation QAT with Exponential Moving Average + Quantization Correction (EMA+QC)（Ours “EMA+QC”）
- QAT recipe highlights: 100 epochs；Adam learning rate (lr)=1e-4；EMA decay 0.9999；+1 epoch correction（Batch Normalization (BN) stats fixed）
- Notes: QAT 期间 first/last layers 以 8-bit quantization 处理（detection 常见做法）

#### 4.2 解读（Interpretation）
##### Method overview（方法概述）
Gupta et al.（gupta2024-oscillations）聚焦于 YOLO 在 STE-based low-bit QAT 中的 oscillations 问题，并提出两个互补机制：

1) **Exponential Moving Average (EMA) model**：在 QAT 过程中对 latent weights 以及 learned quantization step sizes（weights/activations）做 exponential moving average（EMA），并用 EMA-smoothed parameters 做最终 inference/evaluation。

2) **Quantization Correction (QC)**：在标准 QAT 结束后，追加一个低成本的 post-hoc correction phase（1 epoch），学习每层的 affine scale/shift（可 fold 到 Batch Normalization (BN)），用于补偿由于 weights/step oscillations 在 quantization thresholds 附近跳变而累积的 quantization error。

下面 pseudo code 的记号约定：`W` = latent weights；`sW` = weight quantization step size；`sA` = activation quantization step size。

```text
# QAT with EMA（简化版；LSQ-style step-size learning，per-tensor quantization）
init_from_pretrained_fp32(model)
set_precision(first_layer, last_layer, bits=8)                 # YOLO 常见 exception
set_precision(other_layers, bits=3_or_4)                       # 目标 low-bit

W_ema, sW_ema, sA_ema = copy(W, sW, sA)                        # EMA buffers（latent weights + step sizes）
alpha = 0.9999                                                  # EMA decay（论文设置）

for epoch in range(100):                                       # QAT epochs
    for batch in train_loader:
        # Forward with fake-quant + STE（weights + activations）
        y = model_forward_fake_quant(batch.x, W, sW, sA)        # forward 中模拟 int quantization
        loss = detection_loss(y, batch.targets)
        loss.backward(); optimizer.step(); optimizer.zero_grad()

        # EMA 用于平滑 oscillating latent state（weights + learned step sizes）
        W_ema  = alpha * W_ema  + (1 - alpha) * W
        sW_ema = alpha * sW_ema + (1 - alpha) * sW
        sA_ema = alpha * sA_ema + (1 - alpha) * sA

eval_model = (W_ema, sW_ema, sA_ema)                            # 最终用 EMA 参数评测
```

```text
# QC post-hoc correction（简化版；1 epoch；fold 后 inference 无额外开销）
freeze(W, sW, sA)                                               # 不再更新 QAT 训练得到的参数
freeze(BN_running_stats=True)                                   # BN statistics 固定（论文设置）
init(gamma=1, beta=0)                                           # per-layer（per-output-channel）affine 参数

for epoch in range(1):                                          # single correction epoch
    for batch in calib_or_train_subset:
        h = model_pre_bn_preactivations(batch.x, W, sW, sA)      # pre-activation（BN 之前）
        h_corr = gamma * h + beta                                # affine correction
        y = model_forward_from_corrected(h_corr)
        loss = detection_loss(y, batch.targets)
        loss.backward(); optimizer_qc.step(); optimizer_qc.zero_grad()

fold_into_BN_or_scales(gamma, beta)                              # absorb 到 BN params / quant scales
```

- Evidence: Table `tab:yolo-qat-ours` 及相关 ablations in gupta2024-oscillations（见 **Metadata** 中的 arXiv/CVF 链接）。
- Practical takeaway: 对 YOLO low-bit QAT，显式处理 oscillations（EMA、dampening/freezing、post-hoc correction）通常能显著降低 4-bit 的 AP 损失。

##### Limitations（局限与训练/部署代价）
- **efficient YOLO 的 extremely low precision（4-bit and lower）仍然困难**：论文明确指出即便使用 state-of-the-art (SOTA) QAT methods，在 4-bit 及更低 precision 上仍然很难。
  > “It is difficult to achieve extremely low precision (4-bit and lower) for efficient YOLO models even with SOTA QAT methods…”  
  > — gupta2024-oscillations, Abstract
- **oscillation 不仅影响 latent weights，也影响 scale factors（weights + activations）**：论文指出 oscillation 会同时影响 latent weights 与（weights/activations 的）scale factors。
  > “The oscillation issue does not only affect the latent weights but also affects the scale factors of both weights and activations.”  
  > — gupta2024-oscillations, Sec. “Side-effects of Oscillations in YOLO”
- **first/last layers 在 QAT 中保持 8-bit**：论文在 QAT 期间将 first/last layers 以 8-bit quantization 处理。
  > “During QAT… we quantize the first and last layer with 8-bit.”  
  > — gupta2024-oscillations, Experimental Setup
- **per-channel quantization 在 depth-wise convolutions 上可能 inferior**：论文指出在 depth-wise convolutions 场景下，per-channel quantization 有时会比 per-tensor 更差。
  > “Per-channel quantization with depth-wise convolutions can sometimes be inferior to per-tensor quantization.”  
  > — gupta2024-oscillations, Comparison against per-channel quantization

#### Appendix (QAT): Gupta et al. “Ours (Exponential Moving Average + Quantization Correction (EMA+QC))” across YOLO variants
以下数值来自 Table `tab:yolo-qat-ours` in gupta2024-oscillations。Δ 为相对同表 FP32 列的差值。

| Model | FP32 | “4-bit” AP | Δ | “3-bit” AP | Δ | “4-bit*” AP | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLOv5n | 28.0 | 23.8 | −4.2 | 18.2 | −9.8 | 20.4 | −7.6 |
| YOLOv5s | 37.4 | 34.0 | −3.4 | 30.2 | −7.2 | 32.0 | −5.4 |
| YOLOv5m | 45.2 | 42.8 | −2.4 | 40.0 | −5.2 | 40.1 | −5.1 |
| YOLOv5l | 49.0 | 46.6 | −2.4 | 44.0 | −5.0 | 43.6 | −5.4 |
| YOLOv5x | 50.7 | 47.9 | −2.8 | 46.8 | −3.9 | 45.2 | −5.5 |
| YOLOv7-tiny | 37.5 | 35.2 | −2.3 | 31.0 | −6.5 | 34.3 | −3.2 |
| YOLOv7 | 51.2 | 48.9 | −2.3 | 46.8 | −4.4 | 47.6 | −3.6 |

#### Appendix (QAT): Baseline comparison (Gupta et al. re-implementations on YOLO)
以下数值来自 Table `tab:compare-baselines` in gupta2024-oscillations。

| Bits | Method | YOLOv5n | YOLOv5s | YOLOv7-tiny |
|---|---|---:|---:|---:|
| 4-bit | LSQ | 20.6 | 32.4 | 32.9 |
| 4-bit | Osc. dampening | 21.5 | 32.9 | 33.5 |
| 4-bit | Ours (EMA) | 22.1 | 33.1 | 34.6 |
| 4-bit | Ours (EMA+QC) | 23.8 | 34.0 | 35.2 |
| 3-bit | LSQ | 15.2 | 27.2 | 28.4 |
| 3-bit | Osc. dampening | 16.4 | 27.5 | 29.2 |
| 3-bit | Ours (EMA) | 16.4 | 28.5 | 30.3 |
| 3-bit | Ours (EMA+QC) | 18.2 | 30.2 | 31.0 |

### 5) Challenges & open problems（含证据引用）
- **Activation range/outlier management（PTQ bottleneck at ≤4 bits）**：YOLO activations 可能高度不均衡（SiLU），导致 MinMax/naive clipping 把 quantization levels 浪费在低概率的 extreme values 上；即便做 truncation 也可能显著伤害 AP。
  - Evidence:
    > “To address the issue of activation value imbalance, we propose… UH activation quantization.”  
    > — qyolo-2023, UH Activation Quantization
  - Key figures:
    - `![Activation histogram imbalance (from qyolo-2023, Fig. 1)](figures/qyolo_fig1_hist.png)`
- **STE oscillations（low-bit QAT instability）**：weights 会在 quantization thresholds 周围来回跳变，把额外的 optimization noise 注入训练过程；在 4 bits 及更低时尤其致命。
  - Evidence:
    > “Weights seemingly randomly oscillate around decision thresholds, leading to detrimental noise during the optimization process.”  
    > — nagel2022-oscillations, Introduction
  - Key figures:
    - `![Training oscillation example (from nagel2022-oscillations)](figures/nagel_mnv2_training_oscillation.png)`
- **Scale/step parameter instability（not just weights）**：learned quantization scale factors 可能在训练后期仍不稳定，导致最终 quantized state 处于 sub-optimal。
  - Evidence:
    > “Quantization scale factors remain unstable even until the end of quantization-aware training.”  
    > — gupta2024-oscillations, Sec. “Oscillation Issue in YOLO networks”
  - Key figures:
    - `![Oscillation in YOLO latent weights (from gupta2024-oscillations, Fig. 2a)](figures/gupta_fig2a_latent_weight_dist.png)`
- **Sensitive layers 仍需要 exceptions（input/output, first/last, parts of head）**：无论 PTQ 还是 QAT，常见做法都是对这些 layers 保留更高 precision，以避免 AP 大幅下降。
  - Evidence:
    > “The input and output layers… are more sensitive to the loss of accuracy.”  
    > — qyolo-2023, Implementation Details
- **Per-channel vs per-tensor trade-offs（efficient backbones）**：per-channel 可能提升表达能力，但在 depthwise-heavy、参数/算力受限的模型上可能更不稳定，需要更谨慎的 recipe/regularization。
  - Evidence:
    > “Per-channel quantization tends to be more unstable… for efficient networks with depth-wise convolutions…”  
    > — gupta2024-oscillations, Comparison against per-channel quantization
- **Deployment gap for INT4**：即便研究上能做 INT4，生产 inference stack 也常只支持 INT8（有时还要求 symmetric），限制了 ≤4-bit 方案的可落地性。
  - Evidence:
    > “As most current inference frameworks only support… 8-bit quantization… we had to choose a symmetric 8-bit quantization scheme…”  
    > — qyolo-2023, Inference speed

### 6) 实践建议（Practical recommendations）
- **For PTQ experiments**:
  - 先把 INT8 PTQ 跑通并达到稳定 parity（验证 pipeline end-to-end），再尝试 INT4。
  - activation-aware 的 range setting（histogram/MSE/Kullback–Leibler (KL)-style）优先；在看到 heavy imbalance/outliers 时避免对 activations 用 MinMax。
  - 对 sensitive layers 保持 higher precision（input/output、first/last conv、以及任何 fragile head components），并明确记录哪些 layers 被排除/保留。
  - 记录 per-layer activation histograms 与 percentile stats（quantization 前/后），用于尽早发现 imbalance。
- **For QAT experiments**:
  - 把 “oscillation management” 当作一等调参项：EMA of weights/steps、显式 dampening/freezing、以及 small post-hoc correction 都可能带来显著提升。
  - first/last layers 保持 8-bit 作为强默认；在收敛稳定后再尝试 “fully 4-bit”。
  - 明确 BN handling（必要时 freeze stats；显式说明 BN folding），并在 ≤4 bits 使用更保守的 optimizer/learning rate。
- **Minimum reporting checklist (for future runs)**:
  - 精确的 YOLO codebase/version + checkpoint + dataset split
  - Metric 定义（AP vs AP50（AP at Intersection over Union (IoU)=0.5）等）、image size、NMS/postprocess settings
  - bit-widths + granularity + symmetric/asymmetric + per-layer exceptions
  - PTQ: calibration set size/selection 与 range-setting method
  - QAT: epochs/learning rate/optimizer、BN treatment、distillation/EMA/correction steps
  - latency 与 deployment target + kernels/framework constraints（INT8 vs INT4 reality）

### 7) References
- Reducing the Side-Effects of Oscillations in Training of Quantized YOLO Networks, WACV 2024. arXiv: `https://arxiv.org/abs/2311.05109`（CVF PDF: `https://openaccess.thecvf.com/content/WACV2024/papers/Gupta_Reducing_the_Side-Effects_of_Oscillations_in_Training_of_Quantized_YOLO_WACV_2024_paper.pdf`）
- Overcoming Oscillations in Quantization-Aware Training, ICML 2022. arXiv: `https://arxiv.org/abs/2203.11086`（PMLR PDF: `https://proceedings.mlr.press/v162/nagel22a/nagel22a.pdf`）
- Q-YOLO: Efficient Inference for Real-time Object Detection, 2023. arXiv: `https://arxiv.org/abs/2307.04816`（Code: `https://github.com/Meize0729/Q-YOLO`）

### 8) Local references（workspace-only）
本节仅对可以访问生成本报告的 workspace/repo 的读者有用。

- Report sources:
  - EN: `context/summaries/yolo-quant/yolo-quant-sota-report.md`
  - CN: `context/summaries/yolo-quant/yolo-quant-sota-report.cn.md`
  - Figures dir: `context/summaries/yolo-quant/figures/`
- Extracted figures used in this report:
  - qyolo-2023 Fig. 1 histogram: `context/summaries/yolo-quant/figures/qyolo_fig1_hist.png`
  - gupta2024-oscillations Fig. 2a latent weight distribution: `context/summaries/yolo-quant/figures/gupta_fig2a_latent_weight_dist.png`
  - nagel2022-oscillations training oscillation example: `context/summaries/yolo-quant/figures/nagel_mnv2_training_oscillation.png`
- Local paper sources (vendored LaTeX):
  - gupta2024-oscillations: main `paper-source/gupta2024-oscillations/tex/od-qat.tex`, bib `paper-source/gupta2024-oscillations/tex/od-qat.bib`
  - nagel2022-oscillations: main `paper-source/nagel2022-oscillations/tex/main.tex`, bib `paper-source/nagel2022-oscillations/tex/dirty.bib`
  - qyolo-2023: main `paper-source/qyolo-2023/tex/paper.tex`, bib `paper-source/qyolo-2023/tex/references.bib`
- Local anchors for items referenced in the main text:
  - qyolo-2023 Table `exp_main`（PTQ results）: `paper-source/qyolo-2023/tex/paper.tex`
  - qyolo-2023 Fig. 1（source PDF）: `paper-source/qyolo-2023/tex/fig1.pdf`
  - gupta2024-oscillations Table `tab:yolo-qat-ours`: `paper-source/gupta2024-oscillations/tex/tables/yolo-qat-ours2.tex`
  - gupta2024-oscillations Table `tab:compare-baselines`: `paper-source/gupta2024-oscillations/tex/tables/baselines-compare.tex`
  - gupta2024-oscillations Fig. 2a（source PDF）: `paper-source/gupta2024-oscillations/tex/figures/osc_yolo/4.pdf`
  - nagel2022-oscillations oscillation figure（source PNG）: `paper-source/nagel2022-oscillations/tex/figures/mnv2_training_oscillation.png`
