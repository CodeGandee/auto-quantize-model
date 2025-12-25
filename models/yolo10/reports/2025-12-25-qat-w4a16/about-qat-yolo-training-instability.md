# About: QAT instability in low-bit YOLO training (community notes + mitigation ideas)

This note explains a known issue pattern in ultra-low-bit quantization-aware training (QAT) for YOLO-family detectors (especially ≤4-bit), why it happens, and what the community/research literature suggests doing about it. It is written to contextualize the strange curves in this report (for example `train-vs-val-loss.svg`, `metrics__mAP50*.svg`).

Local copies of the referenced papers (PDF + `pdftotext -layout`) are stored under `tmp/papers/qat-yolo-oscillations/` to keep the report reproducible if links move.

## At a glance

- Symptom: early metric peaks followed by collapse, plus odd train-vs-val loss behavior, is a known failure mode in ≤4-bit QAT.
- Common root cause: STE-based “fake quant” training can produce oscillating weights (and sometimes oscillating quantization scale factors), which adds training noise and can break BN running stats.
- Literature-backed mitigations: BN re-estimation, oscillation dampening / iterative freezing (ICML’22), EMA + post-hoc Quantization Correction (QC) (WACV’24 YOLO), staged/progressive quantization, and conservative QAT hyperparameters (LR/AMP/BN handling).

## What we observed in this run

- The QAT run in `models/yolo10/reports/2025-12-25-qat-w4a16/` shows unstable validation behavior: early mAP peaks followed by collapse, plus an unusual train-vs-val loss relationship in `models/yolo10/reports/2025-12-25-qat-w4a16/train-logs/figures/qat-w4a16__ultralytics__yolov10m-scratch-qat-w4a16__root/train-vs-val-loss.svg`.
- This symptom cluster is commonly reported in “very low bit” QAT (≤4-bit): training can look stable while validation becomes erratic, collapses, or oscillates.

## Why YOLO + ultra-low-bit QAT is hard (intuition)

- Detectors combine classification and localization regressions; quantization noise can disproportionately hurt localization/box quality (and downstream NMS), making validation metrics fragile.
- QAT at ≤4-bit is more sensitive to STE-induced oscillations; per-tensor quantization can amplify the problem for layers with outliers or broad distributions.
- BN/statistics drift can become severe under oscillations; even if training “looks stable”, inference-time stats can be wrong, producing a train/val mismatch.

## Community echo: this is a known problem

Two directly relevant references for the underlying phenomenon and YOLO-specific mitigation:

- ICML 2022 (Nagel et al.): https://proceedings.mlr.press/v162/nagel22a/nagel22a.pdf
- WACV 2024 (Gupta & Asthana): https://openaccess.thecvf.com/content/WACV2024/papers/Gupta_Reducing_the_Side-Effects_of_Oscillations_in_Training_of_Quantized_YOLO_WACV_2024_paper.pdf (also on arXiv: https://arxiv.org/abs/2311.05109)

Related implementation repo (from ICML’22):

- https://github.com/Qualcomm-AI-research/oscillations-qat

Other related YOLO quantization literature that emphasizes staged/progressive stabilization:

- MPQ-YOLO (Neurocomputing 2024, preview): https://www.sciencedirect.com/science/article/abs/pii/S0925231223013334

## Paper summaries (with quotes + key results)

### ICML 2022: Overcoming Oscillations in Quantization-Aware Training (Nagel et al.)

> “When training neural networks with simulated quantization, we observe that quantized weights can, rather unexpectedly, oscillate between two grid-points.”  
> “... it can lead to a significant accuracy degradation due to wrongly estimated batch-normalization statistics during inference and increased noise during training.”  
> “Finally, we propose two novel QAT algorithms to overcome oscillations during training: oscillation dampening and iterative weight freezing.”

Major argument: oscillations are not just a curiosity; they are a root cause for (1) noisy optimization and (2) BN running-statistics mismatch, and both effects are particularly visible at ≤4 bits for efficient architectures (depth-wise separable layers).

Proposed methods (high-level):

- BN re-estimation: after QAT, run a small pass over data in training mode (no gradients) to refresh BN running stats; this addresses a major validation/inference failure mode but does not remove oscillations’ optimization noise.
- Oscillation dampening: add a regularizer that discourages weights from residing near quantization decision boundaries and anneal its strength (more flexibility early, more stability late).
- Iterative weight freezing: detect weights with high oscillation frequency and freeze them (in the integer/quantized domain), typically using a schedule to avoid freezing too early.

Key results (selected excerpts from tables in the paper):

- BN re-estimation can recover large amounts of accuracy at low bits, especially for MobileNetV2 at 3–4 bits (paper Table 2; weights quantized, 20 epochs, average of 3 seeds): MobileNetV2 4-bit pre-BN 68.99±0.44 → post-BN 71.01±0.05, and 3-bit pre-BN 64.97±1.23 → post-BN 69.50±0.04.
- Their oscillation-prevention methods improve low-bit W/A accuracy consistently versus LSQ baselines on efficient nets (paper Tables 6–8). Excerpt (validation accuracy %, ImageNet):

| Network | W/A | Full-precision | LSQ baseline | +Dampen (ours) | +Freeze (ours) |
| --- | --- | --- | --- | --- | --- |
| MobileNetV2 | 4/4 | 71.7 | 69.5 (-2.3) | 70.5 (-1.2) | 70.6 (-1.1) |
| MobileNetV2 | 3/3 | 71.7 | 65.3 (-6.5) | 67.8 (-3.9) | 67.6 (-4.1) |
| MobileNetV3-Small | 4/4 | 65.1 | 61.0 | 63.7 | 63.6 |
| MobileNetV3-Small | 3/3 | 65.1 | 52.0 | 59.0 | 58.9 |
| EfficientNet-lite | 4/4 | 75.4 | 72.3 | 73.5 | 73.5 |
| EfficientNet-lite | 3/3 | 75.4 | 69.7 | 71.1 | 71.0 |

Local copies: `tmp/papers/qat-yolo-oscillations/nagel22_overcoming_oscillations_qat.pdf` and `tmp/papers/qat-yolo-oscillations/nagel22_overcoming_oscillations_qat.txt`.

### WACV 2024: Reducing the Side-Effects of Oscillations in Training of Quantized YOLO Networks (Gupta & Asthana)

> “... we show that it is difficult to achieve extremely low precision (4-bit and lower) for efficient YOLO models even with SOTA QAT methods due to oscillation issue ...”  
> “To mitigate the effect of oscillation, we first propose Exponentially Moving Average (EMA) based update to the QAT model.”  
> “Further, we propose a simple QAT correction method, namely QC, that takes only a single epoch of training after standard QAT procedure ...”

Major argument: for YOLO-style detectors, oscillations are prevalent and impact not only latent weights but also learnable quantization scale factors; they propose practical techniques that improve downstream detection/segmentation metrics at 3–4 bits.

Their training recipe (important difference vs our scratch W4A16 run): they start from a pretrained FP model, apply LSQ weight+activation quantization, run QAT for 100 epochs with Adam (lr=1e-4), maintain an EMA model (decay 0.9999), and then train QC correction factors for 1 epoch with BN statistics fixed.

Proposed methods (high-level):

- EMA in QAT: use an exponential moving average of the quantized model parameters to reduce oscillation side-effects during evaluation/export (smoother effective weights than raw oscillating weights).
- QC (Quantization Correction): after QAT, learn lightweight per-layer correction factors (scale and shift) that correct errors induced by oscillating weights/activations; they keep BN stats fixed and train QC for a single epoch.

Key results on COCO detection mAP (paper Table 2; excerpt):

| Model | FP mAP | EMA 4-bit | EMA+QC 4-bit | EMA 3-bit | EMA+QC 3-bit |
| --- | --- | --- | --- | --- | --- |
| YOLO5-n | 28.0 | 22.1 | 23.8 | 16.3 | 18.2 |
| YOLO5-s | 37.4 | 33.1 | 34.0 | 28.5 | 30.2 |
| YOLO7-tiny | 37.5 | 34.6 | 35.2 | 30.3 | 31.0 |

They also compare against a re-implemented “oscillation dampening” baseline for detection and report consistent improvements (paper Table 3; 4-bit shown below):

| Method (4-bit) | YOLO5-n mAP | YOLO5-s mAP | YOLO7-tiny mAP |
| --- | --- | --- | --- |
| LSQ | 20.6 | 32.4 | 32.9 |
| Oscillation dampening | 21.5 | 32.9 | 33.5 |
| Ours (EMA) | 22.1 | 33.1 | 34.6 |
| Ours (EMA+QC) | 23.8 | 34.0 | 35.2 |

Local copies: `tmp/papers/qat-yolo-oscillations/gupta2024_oscillations_quantized_yolo_wacv.pdf` and `tmp/papers/qat-yolo-oscillations/gupta2024_oscillations_quantized_yolo_wacv.txt`.

## What tends to work (actionable mitigation checklist)

The most common working pattern reported in papers and by practitioners is to reduce “shock” to the optimizer/model and reduce oscillation side-effects.

1) Prefer QAT fine-tuning over scratch QAT  
- Train FP16/FP32 to a reasonable baseline, then enable QAT and fine-tune (often fewer epochs, smaller LR).  
- Scratch + W4 is often substantially harder than “pretrain then quantize”, especially for detection.

2) Use progressive / staged quantization  
- Start at higher precision (e.g., W8) then step down to W4; or keep first/last layers at higher precision (mixed precision).  
- If you must do W4, avoid switching everything at once.

3) Lower the learning rate for QAT (often drastically)  
- A common heuristic is to start QAT with LR 5–10× lower than the FP baseline and use longer warmup.  
- If training uses AMP, consider disabling AMP for QAT (quant noise + AMP can destabilize gradients).

4) Address BN/statistics explicitly  
- BN re-estimation and/or BN freezing are common tools; the ICML’22 paper highlights BN stat corruption as a major failure mode.

5) Oscillation-specific techniques (from the literature)  
- EMA-based QAT updates and/or post-QAT correction (QC) as in the WACV’24 YOLO paper.  
- Oscillation dampening / iterative freezing as in the ICML’22 paper (see `oscillations-qat`).

## Minimal code snippets (conceptual)

EMA-smoothed parameter update (conceptual):

```python
ema_decay = 0.999
for p, p_ema in zip(model.parameters(), ema_model.parameters()):
    p_ema.data.mul_(ema_decay).add_(p.data, alpha=1.0 - ema_decay)
```

BN re-estimation (conceptual):

```python
model.train()
with torch.no_grad():
    for images, _ in calib_loader:
        _ = model(images)
model.eval()
```

Progressive quantization (conceptual):

```python
for w_bits in [8, 6, 4]:
    apply_weight_quant(model, bit_width=w_bits)
    train_for_some_epochs(lr=lr_for_bits[w_bits])
```

## How this connects to this repo’s run artifacts

- Training config summary: `models/yolo10/reports/2025-12-25-qat-w4a16/training-config.yaml`
- Stats DB (TensorBoard scalars + results.csv): `models/yolo10/reports/2025-12-25-qat-w4a16/train-logs/training-stats.db`
- Rendered plots: `models/yolo10/reports/2025-12-25-qat-w4a16/train-logs/figures/`
