# Status quo of You Only Look Once (YOLO) low-bit quantization (Post-Training Quantization (PTQ) + Quantization-Aware Training (QAT))

### 0) Metadata
- **Title**: Status quo of YOLO low-bit quantization (PTQ + QAT)
- **Date**: 2026-02-02
- **Scope**: YOLOv5 + YOLOv7 family results on Microsoft COCO (Common Objects in Context, COCO) (based on the 3 public papers listed below)
- **Package contents**: this markdown + `figures/` directory (extracted figures)
- **Papers reviewed (public)**:
  - **Reducing the Side-Effects of Oscillations in Training of Quantized YOLO Networks (gupta2024-oscillations)**
    - Venue: IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2024.
    - IEEE citation: K. Gupta and A. Asthana, “Reducing the Side-Effects of Oscillations in Training of Quantized YOLO Networks,” WACV, 2024. doi: N/A.
    - arXiv: `https://arxiv.org/abs/2311.05109`
    - Open access PDF (Computer Vision Foundation (CVF)): `https://openaccess.thecvf.com/content/WACV2024/papers/Gupta_Reducing_the_Side-Effects_of_Oscillations_in_Training_of_Quantized_YOLO_WACV_2024_paper.pdf`
    - Local TeX source (workspace-only): see **Local references** at the end
  - **Overcoming Oscillations in Quantization-Aware Training (nagel2022-oscillations)**
    - Venue: International Conference on Machine Learning (ICML), 2022.
    - IEEE citation: M. Nagel, M. Fournarakis, Y. Bondarenko, and T. Blankevoort, “Overcoming Oscillations in Quantization-Aware Training,” ICML, 2022. doi: N/A.
    - arXiv: `https://arxiv.org/abs/2203.11086`
    - Proceedings PDF (Proceedings of Machine Learning Research (PMLR)): `https://proceedings.mlr.press/v162/nagel22a/nagel22a.pdf`
    - Local TeX source (workspace-only): see **Local references** at the end
  - **Q-YOLO: Efficient Inference for Real-time Object Detection (qyolo-2023)**
    - IEEE citation: M. Wang, H. Sun, J. Shi, X. Liu, B. Zhang, and X. Cao, “Q-YOLO: Efficient Inference for Real-time Object Detection,” 2023. doi: N/A.
    - arXiv: `https://arxiv.org/abs/2307.04816`
    - Code (GitHub): `https://github.com/Meize0729/Q-YOLO`
    - Local TeX source (workspace-only): see **Local references** at the end
- **Primary evaluation settings assumed in this report**:
  - Dataset(s): COCO 2017 (`train2017` calibration/training; `val2017` evaluation as stated in papers)
  - Metric(s): COCO Average Precision (AP) (a.k.a. mean Average Precision (mAP)@0.5:0.95) unless otherwise stated
  - Image size / preprocessing: 640×640 (as stated in `qyolo-2023`; YOLO defaults used in `gupta2024-oscillations`)
  - Baseline 32-bit floating point (FP32) reference: the FP32 numbers reported in each paper’s tables (not guaranteed identical across papers)

### 1) Executive summary (1 page max)
- **Best reported Post-Training Quantization (PTQ) result (headline)**: Q-YOLO PTQ @ W4A4 on COCO val2017 — YOLOv7x: AP 37.6 (Δ −14.9 vs FP32 52.5).
- **Best reported Quantization-Aware Training (QAT) result (headline)**: Gupta et al. QAT (Exponential Moving Average (EMA) + post-hoc Quantization Correction (QC)) @ “4-bit” on COCO — YOLOv7: AP 48.9 (Δ −2.3 vs FP32 51.2).
- **Key takeaways (3–7 bullets)**:
  - 8-bit integer (INT8) PTQ can be near-lossless for YOLOv5/v7 in the reviewed PTQ pipeline, but 4-bit PTQ still collapses a lot vs FP32.
  - For YOLO detection, QAT (even at 3–4 bits) can be dramatically better than PTQ at the same bit-width.
  - Straight-Through Estimator (STE)-driven oscillations (weights and scale/step parameters) are a central stability/accuracy bottleneck for low-bit QAT.
    > The Straight-Through Estimator (STE) is a surrogate-gradient trick for non-differentiable quantization: use quantize/dequantize in the forward pass, but in backprop replace `d/dx quantize(x)` with an identity (or clipped-identity) gradient so gradients can flow.
  - “Layer exceptions” (e.g., keeping first/last layers higher precision) remain a key ingredient for keeping accuracy.
  - Deployment reality: common inference stacks largely standardize on INT8 (and often symmetric), limiting practical 4-bit integer (INT4) unless you have custom kernels/hardware.
- **What’s still unclear / not comparable across papers (3–7 bullets)**:
  - FP32 baselines differ slightly across papers (likely codebase/training/eval differences), so cross-paper Δ comparisons are noisy.
  - “4-bit” is not always the same: some works keep first/last layers at higher precision; others don’t fully specify all exceptions.
  - PTQ calibration details (sample selection, augmentations, observer settings) differ or are partially unspecified.
  - Speed/latency is not consistently reported for the same targets (INT4 vs INT8, GPU vs CPU, kernel availability).

### 2) Definitions and comparability notes
- **Bit-width notation**:
  - `Wb-Ab` means weight bit-width `b` and activation bit-width `b` (e.g., `W4A4`).
  - If first/last layers are exceptions, this report annotates it in the Notes column (because papers often do this).
- **Quantization scope**:
  - **Post-Training Quantization (PTQ)** (Q-YOLO): quantizes backbone/neck/head; retains input/output layer accuracy (implies higher precision retained).
  - **Quantization-Aware Training (QAT)** (Gupta et al.): quantizes weights + activations; commonly keeps first/last layers at 8-bit, and also reports a “fully 4-bit” variant.
- **Calibration protocol (PTQ)**:
  - Q-YOLO: 1500 COCO `train2017` images for calibration; activation histograms with 2048 bins; selects clipping/range via histogram search.
- **Training protocol (QAT)**:
  - Gupta et al.: 100-epoch QAT from pretrained FP model; Learned Step Size Quantization (LSQ)-style learned step size; Exponential Moving Average (EMA) of latent weights/scale factors; plus a 1-epoch post-hoc Quantization Correction (QC).
- **Hardware / kernel availability assumptions**:
  - Q-YOLO explicitly notes TensorRT/OpenVINO deployment; reports speed for INT8 (framework constraint).
- **Important comparability caveats**:
  - Detection AP is sensitive to preprocessing, Non-Maximum Suppression (NMS)/postprocess, and exact YOLO implementation; treat cross-paper comparisons as directional only.
  - Reported “4-bit” results often include layer precision exceptions; treat “W4A4” as “mostly W4A4” unless the paper explicitly states otherwise.

### 3) PTQ status quo (results + interpretation)

#### 3.1 Snapshot table (best/representative PTQ results)
| Paper | Model | Reported accuracy | FP32 accuracy | Δ vs FP32 | W bits | A bits | Image size |
|---|---|---:|---:|---:|---:|---:|---:|
| qyolo-2023 | YOLOv5s | 14.0 | 37.4 | −23.4 | 4 | 4 | 640 |
| qyolo-2023 | YOLOv5m | 28.8 | 45.1 | −16.3 | 4 | 4 | 640 |
| qyolo-2023 | YOLOv7 | 37.3 | 50.8 | −13.5 | 4 | 4 | 640 |
| qyolo-2023 | YOLOv7x | 37.6 | 52.5 | −14.9 | 4 | 4 | 640 |

Note: all low-bit entries in tables refer to low-bit integer (INT) quantization (not low-bit floating point).
> “Finally, the fully-quantized network is deployed either on integer arithmetic hardware or simulated on GPUs…”  
> — Q-YOLO: Efficient Inference for Real-time Object Detection (qyolo-2023), PTQ process overview

- Dataset: COCO val2017
- Metric: AP (COCO AP / mAP@0.5:0.95)
- Quant scheme: weights (W) symmetric per-channel; activations (A) asymmetric per-layer; UH activation quantization (UH) histogram search for activation range
- Calibration: 1500 COCO train2017 images
- Notes: input/output layers are typically retained (not fully quantized end-to-end)

#### 3.2 Interpretation
##### Method overview
Q-YOLO (qyolo-2023) is an end-to-end PTQ pipeline tailored to YOLO’s activation statistics. The central idea is to fix an empirically observed activation lower bound (linked to Sigmoid Linear Unit (SiLU)) and then choose the activation upper bound via a histogram-based search to minimize quantization error (UH).

```text
# Q-YOLO PTQ pipeline (simplified; key idea = UH for activation range)
calib_data = sample(COCO_train2017, n=1500)                 # calibration set

for layer in model.layers:
    if layer in {input_layer, output_layer}:                # sensitive layers
        keep_fp_or_higher_precision(layer)                  # "retained" in paper
        continue

    # Weights: symmetric per-channel quantization (MinMax range)
    w_min, w_max = per_channel_minmax(layer.W)              # handle outliers per-channel
    layer.W_q = quantize_symmetric(layer.W, w_min, w_max, bits=W_bits)

    # Activations: asymmetric per-layer quantization (UH range selection)
    acts = run_fp32_forward_and_collect(layer, calib_data)  # collect activation samples
    hist = histogram(acts, bins=2048)                       # 2048-bin histogram
    a_min = -0.2785                                         # fixed lower bound (SiLU empirical)
    a_max = uh_search_max(hist, a_min, bits=A_bits)          # choose upper bound to minimize MSE
    layer.A_qparams = (a_min, a_max)                         # per-layer (asymmetric)
```

```text
# UH search (very simplified): pick activation max that minimizes quantization MSE
def uh_search_max(hist, a_min, bits):
    best_max, best_mse = None, +inf
    for i in range(128, 2048):                               # candidate cutoffs in histogram bins
        a_max = bin_center(hist, i)                          # candidate upper bound
        fp = hist_centers(hist, 0, i)                        # representative FP32 values (bin centers)
        qdq = dequantize(quantize_asymmetric(fp, a_min, a_max, bits))
        mse = mean_squared_error(fp, qdq)                    # Mean Squared Error (MSE) proxy
        if mse < best_mse:
            best_max, best_mse = a_max, mse
    return best_max
```

- Evidence: Table `exp_main` in qyolo-2023 (see arXiv link in **Metadata**, plus the paper’s UH algorithm description).
- Practical takeaway: at ≤4 bits, *activation range selection dominates* PTQ outcome for YOLO; naive MinMax/Percentile can collapse.

##### Limitations
- **4-bit PTQ incurs significant accuracy loss**: the paper explicitly states a large accuracy drop at 4-bit due to limited integer expressiveness.
  > “When quantizing models to 4 bits, the accuracy experiences a significant loss due to the reduced expressiveness of 4-bit integer representation.”  
  > — qyolo-2023, Main results
- **Input/output layers are sensitive and are typically retained**: the paper notes input/output layers are more accuracy-sensitive and their original accuracy is usually retained.
  > “The input and output layers… are more sensitive to the loss of accuracy… the original accuracy of these layers is usually retained.”  
  > — qyolo-2023, Implementation Details
- **Weights are more sensitive than activations under quantization**: the paper reports that quantizing weights causes larger degradation than quantizing activations.
  > “Compared to quantizing the activation values, quantizing the weights consistently induces larger performance degradation.”  
  > — qyolo-2023, Quantization types discussion
- **Deployment tooling constraints bias toward INT8/symmetric**: the paper notes common inference frameworks only support symmetric quantization and 8-bit quantization, so they use a symmetric 8-bit quantization scheme for deployment speed tests.
  > “As most current inference frameworks only support symmetric quantization and 8-bit quantization, we had to choose a symmetric 8-bit quantization scheme…”  
  > — qyolo-2023, Inference speed

#### Appendix (PTQ): Q-YOLO main table (AP only)
Values copied from Table `exp_main` in qyolo-2023 (Bits are W-A).

| Model | FP32 (32-32) | MinMax (8-8) | Percentile (8-8) | Q-YOLO (8-8) | Percentile (4-4) | Q-YOLO (4-4) |
|---|---:|---:|---:|---:|---:|---:|
| YOLOv5s | 37.4 | 37.2 | 36.9 | 37.4 | 7.0 | 14.0 |
| YOLOv5m | 45.1 | 44.9 | 44.6 | 45.1 | 19.4 | 28.8 |
| YOLOv7  | 50.8 | 50.6 | 50.5 | 50.7 | 16.7 | 37.3 |
| YOLOv7x | 52.5 | 52.3 | 52.0 | 52.4 | 36.8 | 37.6 |

Note: all low-bit entries in tables refer to low-bit integer (INT) quantization (not low-bit floating point).
> “Finally, the fully-quantized network is deployed either on integer arithmetic hardware or simulated on GPUs…”  
> — Q-YOLO: Efficient Inference for Real-time Object Detection (qyolo-2023), PTQ process overview

### 4) QAT status quo (results + interpretation)

#### 4.1 Snapshot table (best/representative QAT results)
| Paper | Model | Reported accuracy | FP32 accuracy | Δ vs FP32 | W bits | A bits | Image size |
|---|---|---:|---:|---:|---:|---:|---:|
| gupta2024-oscillations | YOLOv5s | 34.0 | 37.4 | −3.4 | 4 | 4 | 640 |
| gupta2024-oscillations | YOLOv7 | 48.9 | 51.2 | −2.3 | 4 | 4 | 640 |
| gupta2024-oscillations | YOLOv5s | 30.2 | 37.4 | −7.2 | 3 | 3 | 640 |
| gupta2024-oscillations | YOLOv7 | 46.8 | 51.2 | −4.4 | 3 | 3 | 640 |

Note: all low-bit entries in tables refer to low-bit integer (INT) quantization (not low-bit floating point).
> “\u2026$q(\\vec{w}; s, u, v) = s \\cdot \\mathrm{clip}(\\mathrm{round}(\\vec{w}/s), u, v)$\u2026 where \u2026 $\u2018\\mathrm{round}\u2019$ is the round-to-nearest operator\u2026”  
> — Reducing the Side-Effects of Oscillations in Training of Quantized YOLO Networks (gupta2024-oscillations), Preliminaries (quantization function)

- Dataset: COCO
- Metric: mAP (AP)
- Quant scheme: per-tensor (LSQ-style) weight+activation QAT with Exponential Moving Average + Quantization Correction (EMA+QC) (Ours “EMA+QC”)
- QAT recipe highlights: 100 epochs; Adam learning rate (lr)=1e-4; EMA decay 0.9999; +1 epoch correction (Batch Normalization (BN) stats fixed)
- Notes: first/last layers are quantized at 8-bit during QAT (common practice in detection)

#### 4.2 Interpretation
##### Method overview
Gupta et al. (gupta2024-oscillations) target the oscillation phenomenon in STE-based low-bit QAT for YOLO. They propose two complementary mechanisms:

1) **Exponential Moving Average (EMA) model**: maintain an exponential moving average of latent weights *and* learned quantization step sizes (for both weights and activations) during QAT, and use the EMA-smoothed parameters for final inference/evaluation.

2) **Quantization Correction (QC)**: after standard QAT, run a cheap post-hoc correction phase (1 epoch) that learns per-layer affine scale/shift (foldable into Batch Normalization (BN)), intended to compensate accumulated quantization error caused by oscillations near quantization thresholds.

Notation in the pseudo code below: `W` = latent weights; `sW` = weight quantization step size; `sA` = activation quantization step size.

```text
# QAT with EMA (simplified; LSQ-style step-size learning, per-tensor quantization)
init_from_pretrained_fp32(model)
set_precision(first_layer, last_layer, bits=8)                 # common YOLO exception
set_precision(other_layers, bits=3_or_4)                       # target low-bit regime

W_ema, sW_ema, sA_ema = copy(W, sW, sA)                        # EMA buffers (latent weights + step sizes)
alpha = 0.9999                                                  # EMA decay (paper setting)

for epoch in range(100):                                       # QAT epochs
    for batch in train_loader:
        # Forward with fake-quant + STE (weights + activations)
        y = model_forward_fake_quant(batch.x, W, sW, sA)        # simulates int quantization in forward
        loss = detection_loss(y, batch.targets)
        loss.backward(); optimizer.step(); optimizer.zero_grad()

        # EMA smooths oscillating latent states (weights + learned step sizes)
        W_ema  = alpha * W_ema  + (1 - alpha) * W
        sW_ema = alpha * sW_ema + (1 - alpha) * sW
        sA_ema = alpha * sA_ema + (1 - alpha) * sA

eval_model = (W_ema, sW_ema, sA_ema)                            # use EMA parameters for final eval
```

```text
# QC post-hoc correction (simplified; 1 epoch; no extra inference cost after folding)
freeze(W, sW, sA)                                               # do not change QAT-trained parameters
freeze(BN_running_stats=True)                                   # keep BN statistics fixed (paper setting)
init(gamma=1, beta=0)                                           # per-layer (per-output-channel) affine params

for epoch in range(1):                                          # single correction epoch
    for batch in calib_or_train_subset:
        h = model_pre_bn_preactivations(batch.x, W, sW, sA)      # pre-activation (before BN)
        h_corr = gamma * h + beta                                # affine correction
        y = model_forward_from_corrected(h_corr)
        loss = detection_loss(y, batch.targets)
        loss.backward(); optimizer_qc.step(); optimizer_qc.zero_grad()

fold_into_BN_or_scales(gamma, beta)                              # absorb into BN params / quant scales
```

- Evidence: Table `tab:yolo-qat-ours` and related ablations in gupta2024-oscillations (see arXiv/CVF links in **Metadata**).
- Practical takeaway: for YOLO low-bit QAT, explicitly addressing oscillations (EMA, dampening/freezing, post-hoc correction) can materially reduce AP loss at 4-bit.

##### Limitations
- **Extremely low precision remains hard for efficient YOLO**: the paper explicitly states that 4-bit (and lower) is difficult even with state-of-the-art (SOTA) QAT methods.
  > “It is difficult to achieve extremely low precision (4-bit and lower) for efficient YOLO models even with SOTA QAT methods…”  
  > — gupta2024-oscillations, Abstract
- **Oscillation affects both weights and scale factors (weights + activations)**: the paper reports oscillations impact not only latent weights but also learned scale factors.
  > “The oscillation issue does not only affect the latent weights but also affects the scale factors of both weights and activations.”  
  > — gupta2024-oscillations, Sec. “Side-effects of Oscillations in YOLO”
- **First/last layers are kept at higher precision during QAT**: the paper quantizes first/last layers at 8-bit.
  > “During QAT… we quantize the first and last layer with 8-bit.”  
  > — gupta2024-oscillations, Experimental Setup
- **Per-channel quantization can be unstable/inferior for depth-wise convolutions**: the paper notes per-channel quantization can be inferior to per-tensor quantization on depth-wise convolutions.
  > “Per-channel quantization with depth-wise convolutions can sometimes be inferior to per-tensor quantization.”  
  > — gupta2024-oscillations, Comparison against per-channel quantization

#### Appendix (QAT): Gupta et al. “Ours (Exponential Moving Average + Quantization Correction (EMA+QC))” across YOLO variants
Values copied from Table `tab:yolo-qat-ours` in gupta2024-oscillations. Deltas (Δ) are computed vs the FP32 column in the same table.

| Model | FP32 | “4-bit” AP | Δ | “3-bit” AP | Δ | “4-bit*” AP | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLOv5n | 28.0 | 23.8 | −4.2 | 18.2 | −9.8 | 20.4 | −7.6 |
| YOLOv5s | 37.4 | 34.0 | −3.4 | 30.2 | −7.2 | 32.0 | −5.4 |
| YOLOv5m | 45.2 | 42.8 | −2.4 | 40.0 | −5.2 | 40.1 | −5.1 |
| YOLOv5l | 49.0 | 46.6 | −2.4 | 44.0 | −5.0 | 43.6 | −5.4 |
| YOLOv5x | 50.7 | 47.9 | −2.8 | 46.8 | −3.9 | 45.2 | −5.5 |
| YOLOv7-tiny | 37.5 | 35.2 | −2.3 | 31.0 | −6.5 | 34.3 | −3.2 |
| YOLOv7 | 51.2 | 48.9 | −2.3 | 46.8 | −4.4 | 47.6 | −3.6 |

Note: all low-bit entries in tables refer to low-bit integer (INT) quantization (not low-bit floating point).
> “\u2026$q(\\vec{w}; s, u, v) = s \\cdot \\mathrm{clip}(\\mathrm{round}(\\vec{w}/s), u, v)$\u2026 where \u2026 $\u2018\\mathrm{round}\u2019$ is the round-to-nearest operator\u2026”  
> — Reducing the Side-Effects of Oscillations in Training of Quantized YOLO Networks (gupta2024-oscillations), Preliminaries (quantization function)

#### Appendix (QAT): Baseline comparison (Gupta et al. re-implementations on YOLO)
Values copied from Table `tab:compare-baselines` in gupta2024-oscillations.

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

Note: all low-bit entries in tables refer to low-bit integer (INT) quantization (not low-bit floating point).
> “\u2026$q(\\vec{w}; s, u, v) = s \\cdot \\mathrm{clip}(\\mathrm{round}(\\vec{w}/s), u, v)$\u2026 where \u2026 $\u2018\\mathrm{round}\u2019$ is the round-to-nearest operator\u2026”  
> — Reducing the Side-Effects of Oscillations in Training of Quantized YOLO Networks (gupta2024-oscillations), Preliminaries (quantization function)

### 5) Challenges & open problems (with cited evidence)
- **Activation range/outlier management is the PTQ bottleneck at ≤4 bits**: YOLO activations can be highly imbalanced (SiLU), making MinMax/naive clipping waste quantization levels on rare values; aggressive truncation can still hurt AP.
  - Evidence:
    > “To address the issue of activation value imbalance, we propose… UH activation quantization.”  
    > — qyolo-2023, UH Activation Quantization
  - Key figures:
    - `![Activation histogram imbalance (from qyolo-2023, Fig. 1)](figures/qyolo_fig1_hist.png)`
- **STE oscillations drive low-bit QAT instability**: weights can bounce around quantization thresholds, injecting optimization noise and harming convergence; this becomes critical at 4 bits and below.
  - Evidence:
    > “Weights seemingly randomly oscillate around decision thresholds, leading to detrimental noise during the optimization process.”  
    > — nagel2022-oscillations, Introduction
  - Key figures:
    - `![Training oscillation example (from nagel2022-oscillations)](figures/nagel_mnv2_training_oscillation.png)`
- **Scale/step parameter instability (not just weights) matters for YOLO**: learned quantization scale factors can remain unstable late into training, leaving a sub-optimal final quantized state.
  - Evidence:
    > “Quantization scale factors remain unstable even until the end of quantization-aware training.”  
    > — gupta2024-oscillations, Sec. “Oscillation Issue in YOLO networks”
  - Key figures:
    - `![Oscillation in YOLO latent weights (from gupta2024-oscillations, Fig. 2a)](figures/gupta_fig2a_latent_weight_dist.png)`
- **Sensitive layers (input/output, first/last, parts of head) still need exceptions**: both PTQ and QAT commonly preserve higher precision for these layers to avoid large AP drops.
  - Evidence:
    > “The input and output layers… are more sensitive to the loss of accuracy.”  
    > — qyolo-2023, Implementation Details
- **Per-channel vs per-tensor trade-offs on efficient backbones**: per-channel can improve representational fidelity but may destabilize depthwise-heavy, efficient models; careful handling is required.
  - Evidence:
    > “Per-channel quantization tends to be more unstable… for efficient networks with depth-wise convolutions…”  
    > — gupta2024-oscillations, Comparison against per-channel quantization
- **Deployment gap for INT4**: even when INT4 methods exist, production inference stacks often only support INT8 (and sometimes symmetric only), constraining real-world usage of ≤4-bit schemes.
  - Evidence:
    > “As most current inference frameworks only support… 8-bit quantization… we had to choose a symmetric 8-bit quantization scheme…”  
    > — qyolo-2023, Inference speed

### 6) Practical recommendations (actionable)
- **For PTQ experiments**:
  - Start with INT8 PTQ first (validate the pipeline end-to-end), then attempt INT4 only after you have stable INT8 parity.
  - Use activation-aware range setting (histogram/MSE/Kullback–Leibler (KL)-style); avoid MinMax for activations when you see heavy imbalance/outliers.
  - Keep sensitive layers higher precision (input/output, first/last conv, and any fragile head components) and record exactly which layers are excluded.
  - Log per-layer activation histograms and percentile stats (before/after quant) to catch imbalance early.
- **For QAT experiments**:
  - Treat “oscillation management” as a first-class tuning knob: EMA of weights/steps, explicit dampening/freezing, and/or small post-hoc correction can matter.
  - Keep first/last layers at 8-bit as a strong default; only “fully 4-bit” once you have stable convergence.
  - Stabilize BN handling (freeze stats where appropriate; be explicit about BN folding) and keep optimizer/learning rate conservative at ≤4 bits.
- **Minimum reporting checklist (for future runs)**:
  - Exact YOLO codebase/version + checkpoint + dataset split
  - Metric definition (AP vs AP50 (AP at Intersection over Union (IoU)=0.5), etc.), image size, NMS/postprocess settings
  - Bit-widths + granularity + symmetric/asymmetric + per-layer exceptions
  - PTQ: calibration set size/selection and range-setting method
  - QAT: epochs/learning rate/optimizer, BN treatment, distillation/EMA/correction steps
  - Latency and deployment target + kernels/framework constraints (INT8 vs INT4 reality)

### 7) References
- Reducing the Side-Effects of Oscillations in Training of Quantized YOLO Networks, WACV 2024. arXiv: `https://arxiv.org/abs/2311.05109` (CVF PDF: `https://openaccess.thecvf.com/content/WACV2024/papers/Gupta_Reducing_the_Side-Effects_of_Oscillations_in_Training_of_Quantized_YOLO_WACV_2024_paper.pdf`)
- Overcoming Oscillations in Quantization-Aware Training, ICML 2022. arXiv: `https://arxiv.org/abs/2203.11086` (PMLR PDF: `https://proceedings.mlr.press/v162/nagel22a/nagel22a.pdf`)
- Q-YOLO: Efficient Inference for Real-time Object Detection, 2023. arXiv: `https://arxiv.org/abs/2307.04816` (Code: `https://github.com/Meize0729/Q-YOLO`)

### 8) Local references (workspace-only)
This section is only useful for readers who have access to the original workspace/repo that produced this report.

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
  - qyolo-2023 Table `exp_main` (PTQ results): `paper-source/qyolo-2023/tex/paper.tex`
  - qyolo-2023 Fig. 1 (source PDF): `paper-source/qyolo-2023/tex/fig1.pdf`
  - gupta2024-oscillations Table `tab:yolo-qat-ours`: `paper-source/gupta2024-oscillations/tex/tables/yolo-qat-ours2.tex`
  - gupta2024-oscillations Table `tab:compare-baselines`: `paper-source/gupta2024-oscillations/tex/tables/baselines-compare.tex`
  - gupta2024-oscillations Fig. 2a (source PDF): `paper-source/gupta2024-oscillations/tex/figures/osc_yolo/4.pdf`
  - nagel2022-oscillations oscillation figure (source PNG): `paper-source/nagel2022-oscillations/tex/figures/mnv2_training_oscillation.png`
