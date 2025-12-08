# Subtask 1.3: Choose YOLO11 variants and define success criteria

## Scope

Select the YOLO11 checkpoint(s) that will be used for ModelOpt quantization experiments in **auto-quantize-model**, and define clear, measurable success criteria for accuracy and performance on the target hardware and dataset(s).

## Planned outputs

- A decision on which YOLO11 variant(s) (e.g., `yolo11n`, `yolo11s`) to use for quantization experiments.
- Documented assumptions about the target deployment hardware (e.g., specific NVIDIA GPU models) and key workload characteristics (batch size, image resolution).
- A concise set of success criteria (e.g., maximum acceptable mAP drop relative to FP32/FP16, target latency/throughput, and any memory limits).

## TODOs

- [x] Job-001-103-001: List available YOLO11 checkpoints under `models/yolo11/checkpoints/` and note their expected accuracy/speed trade-offs (using upstream docs if needed).
- [x] Job-001-103-002: Choose one or two primary YOLO11 variants to focus on for quantization (balancing realism and experiment time).
- [x] Job-001-103-003: Identify the target hardware platform(s) (e.g., GPU model(s), driver/toolkit versions) and typical inference settings (batch size, resolution) relevant to this project.
- [x] Job-001-103-004: Define quantitative success criteria (e.g., max allowable mAP drop, latency budget per image, throughput goals, memory constraints) for comparing FP16 vs. mixed FP16/INT8 models.
- [x] Job-001-103-005: Capture the chosen variants and success criteria in the main task file and/or a linked context document so later milestones can reference them.

## Notes

- If exact deployment hardware is not yet fixed, define criteria for one or two representative GPU classes (e.g., data center vs. prosumer).
- Keep criteria realistic but firm enough to drive decision-making when selecting the “best” mixed-precision configuration later.

## Implementation summary

- Available checkpoints: `models/yolo11/checkpoints/` currently contains `yolo11n.pt`, `yolo11s.pt`, `yolo11m.pt`, `yolo11l.pt`, and `yolo11x.pt`, matching the typical Ultralytics nano→xlarge accuracy/speed trade-off pattern (n = smallest/fastest, x = largest/most accurate).
- Chosen variants: for this project we will focus on:
  - **Primary**: `yolo11n` — used for fast iteration, debugging the end-to-end ModelOpt/TensorRT pipeline, and initial PTQ experiments.
  - **Secondary**: `yolo11s` — used for more realistic accuracy/performance measurements once the pipeline is stable.
- Target hardware and settings: experiments are assumed to run on a single NVIDIA GeForce RTX 3090 (24 GB, driver 560.35.03, CUDA 12.6) with:
  - Input resolution: 640×640 RGB (standard YOLO11 setting unless otherwise noted).
  - Primary inference mode: batch size 1; optional follow-up measurements may use small batches (e.g., 4–8) for throughput characterization.
- Success criteria (measured vs. a TensorRT FP16-only baseline engine for the same variant, resolution, and batch size on the RTX 3090):
  - **Accuracy**: mAP@[0.5:0.95] drop ≤ 1.0 absolute percentage point compared to FP16; if baseline is low, also require relative drop ≤ 3%.
  - **Latency / throughput**: median per-image latency for the mixed FP16/INT8 engine should be at most 75% of the FP16 baseline (≥ 1.33× speedup) for `yolo11n`; for `yolo11s` we accept at most 80% of FP16 latency (≥ 1.25× speedup).
  - **Memory**: peak GPU memory usage for the mixed FP16/INT8 engine should not exceed the FP16 baseline by more than 10%; any larger increase should be justified by a clearly better latency/throughput trade-off.
