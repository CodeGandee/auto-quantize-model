# Research Notes: YOLOv10 W4A16 QAT Stabilization (EMA + QC)

**Branch**: `001-yolov10-qat-validation`  
**Created**: 2025-12-25  
**Purpose**: Consolidate paper findings and make concrete design decisions for validating EMA + QC on YOLOv10 using `yolo10n` and `yolo10s` first.

## Primary References (local copies)

- WACV 2024 (Gupta & Asthana): `tmp/papers/qat-yolo-oscillations/gupta2024_oscillations_quantized_yolo_wacv.pdf` and `tmp/papers/qat-yolo-oscillations/gupta2024_oscillations_quantized_yolo_wacv.txt`
- ICML 2022 (Nagel et al.): `tmp/papers/qat-yolo-oscillations/nagel22_overcoming_oscillations_qat.pdf` and `tmp/papers/qat-yolo-oscillations/nagel22_overcoming_oscillations_qat.txt`
- Repo note on observed instability + curated summary: `models/yolo10/reports/2025-12-25-qat-w4a16/about-qat-yolo-training-instability.md`

## Key Findings (what the papers actually claim)

### WACV 2024: EMA + post-hoc Quantization Correction (QC) for quantized YOLO

- EMA is proposed to reduce the side-effects of oscillating latent weights and (when present) oscillating quantization scale factors during QAT.
- Their EMA update (paper Eq. 6–8) is a standard exponential moving average over parameters and scale factors:
  - \(W'(t) = \alpha W'(t-1) + (1-\alpha) W(t)\)
  - \(s'_W(t) = \alpha s'_W(t-1) + (1-\alpha) s_W(t)\)
  - \(s'_a(t) = \alpha s'_a(t-1) + (1-\alpha) s_a(t)\)
- QC is a 1-epoch post-hoc stage that learns only lightweight correction parameters to compensate for error accumulated from oscillations:
  - \( \hat{h}_l = \gamma_l \cdot h_l + \beta_l \)
  - \( h_l = W_l^q a_{l-1} + b_l \)
  - Correction parameters are initialized as identity and trained on a calibration subset \(D_c\) after QAT.
- Their experimental recipe uses pretrained FP starting points, long QAT (e.g., ~100 epochs), and a single QC epoch with BN statistics fixed.

### ICML 2022: Oscillations in QAT and BN failure mode

- Oscillations can cause (1) noisy optimization and (2) corrupted BN running statistics, which can produce a large train/val mismatch at inference.
- The paper proposes oscillation dampening and iterative freezing, plus emphasizes BN re-estimation as a practical mitigation.

## Decisions (what we will do in this repo)

### Decision: Use Pixi `cu128` as the development and run environment

**Decision**: Use the Pixi environment `cu128` for development and for all validation runs in this feature.  
**Rationale**: `cu128` provides the intended CUDA/PyTorch stack and includes Brevitas, which is required for W4A16 fake-quant in this repo.  
**Alternatives considered**: `cu124` (rejected for this feature: may not include the same pinned toolchain/deps; keep the experiment surface consistent).

### Decision: Validate on `yolo10n` then `yolo10s` (gate `yolo10m`)

**Decision**: Implement and validate the method on `yolo10n` and `yolo10s` first, and only attempt `yolo10m` after the success criteria are met for both smaller variants.  
**Rationale**: This provides the fastest signal on algorithm correctness and stability while minimizing compute waste.  
**Alternatives considered**: Starting directly on `yolo10m` (rejected: too expensive and slow to debug).

### Decision: Use pretrained fine-tuning (not scratch QAT) for validation

**Decision**: Use the provided pretrained checkpoints (`models/yolo10/checkpoints/yolov10n.pt`, `models/yolo10/checkpoints/yolov10s.pt`) as the starting point for QAT.  
**Rationale**: WACV’24 explicitly uses pretrained starting points; scratch W4 training is a known high-risk path and is not required to validate the stabilization algorithm.  
**Alternatives considered**: Scratch training with W4 enabled from epoch 0 (rejected for validation: conflates baseline training difficulty with stabilization correctness).

### Decision: Quantization target is W4A16 (weight-only int4)

**Decision**: Target W4A16 (4-bit weights, floating activations) for this feature.  
**Rationale**: This matches the feature spec and aligns with the instability pattern we are trying to fix; it is also the simplest ultra-low-bit regime to validate before adding activation quantization.  
**Alternatives considered**: W4A8 or W8A8 (rejected: different failure surface; can be added later after EMA+QC correctness is validated).

### Decision: Use Brevitas layerwise quantization for W4A16

**Decision**: Use the existing Brevitas quantization path (`auto_quantize_model.cv_models.yolov10_brevitas.quantize_model_brevitas_ptq`) to insert weight-only fake-quant modules.  
**Rationale**: This is already integrated with YOLOv10 in this repo and supports inspectable QCDQ exports.  
**Alternatives considered**: Re-implement LSQ exactly as in WACV’24 (rejected for now: correctness of EMA+QC can be validated without matching LSQ implementation details).

### Decision: EMA implementation uses Ultralytics `ModelEMA` (with method-controlled enable/disable)

**Decision**: Use Ultralytics `ModelEMA` as the EMA mechanism and make EMA explicitly controllable per method variant (disabled for baseline, enabled for EMA and EMA+QC).  
**Rationale**: Ultralytics already maintains EMA over `state_dict()` values (parameters + buffers), which matches the paper’s intent; the default decay is 0.9999 with an early ramp, close to WACV’24.  
**Alternatives considered**: Custom EMA loop (rejected: higher maintenance; would duplicate existing EMA logic and checkpoint handling).

### Decision: QC is implemented as per-layer affine correction trained for 1 epoch on a calibration subset

**Decision**: Implement QC by inserting learnable per-channel correction parameters \(\gamma_l\) and \(\beta_l\) at selected layer boundaries, initialize as identity, freeze the base QAT model, and train only QC parameters for 1 epoch on a calibration subset \(D_c\).  
**Rationale**: This is the core mechanism of WACV’24 and is the primary algorithm-under-test beyond EMA.  
**Alternatives considered**: BN re-estimation only (rejected: not QC; would not validate the WACV claim).

### Decision: BN handling during QC is “BN stats fixed”

**Decision**: During QC training, keep BN running statistics fixed so QC is not chasing moving inference-time normalization.  
**Rationale**: This matches WACV’24’s stated QC setup and reduces confounding between BN drift and QC correction.  
**Alternatives considered**: Allow BN to update during QC (rejected: violates paper assumptions and risks masking QC effects).

### Decision: Primary stability metric is `metrics/mAP50-95(B)` from Ultralytics `results.csv`

**Decision**: Use `metrics/mAP50-95(B)` as the single “validation quality” metric used for collapse detection and gating.  
**Rationale**: This is a standard COCO detection metric and is already emitted by Ultralytics YOLOv10 training logs.  
**Alternatives considered**: `metrics/mAP50(B)` (rejected: less sensitive to localization quality changes; keep as secondary).

### Decision: Collapse detection and stage gate follow the feature spec

**Decision**: Mark a run as “collapsed” when `final_metric < 0.5 * best_metric` for the primary metric within that run; require 2/2 stable EMA+QC runs for `yolo10n` and `yolo10s` before allowing `yolo10m`.  
**Rationale**: This is an explicit, implementation-independent stability definition suitable for early validation.  
**Alternatives considered**: Threshold on absolute mAP values (rejected: depends heavily on dataset slice and hyperparameters).

### Decision: Reproducibility is captured in a run summary schema

**Decision**: Every run writes a machine-readable `run_summary.json` (schema defined in `contracts/`) that captures: method, model variant, seed, resolved config, dataset provenance/selection, artifact paths, and stability outcome.  
**Rationale**: Enables automated comparison and prevents “non-comparable run” errors during review.  
**Alternatives considered**: Only human-written notes (rejected: too easy to lose critical comparison metadata).
