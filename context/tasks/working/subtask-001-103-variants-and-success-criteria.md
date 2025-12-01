# Subtask 1.3: Choose YOLO11 variants and define success criteria

## Scope

Select the YOLO11 checkpoint(s) that will be used for ModelOpt quantization experiments in **auto-quantize-model**, and define clear, measurable success criteria for accuracy and performance on the target hardware and dataset(s).

## Planned outputs

- A decision on which YOLO11 variant(s) (e.g., `yolo11n`, `yolo11s`) to use for quantization experiments.
- Documented assumptions about the target deployment hardware (e.g., specific NVIDIA GPU models) and key workload characteristics (batch size, image resolution).
- A concise set of success criteria (e.g., maximum acceptable mAP drop relative to FP32/FP16, target latency/throughput, and any memory limits).

## TODOs

- [ ] Job-001-103-001: List available YOLO11 checkpoints under `models/yolo11/checkpoints/` and note their expected accuracy/speed trade-offs (using upstream docs if needed).
- [ ] Job-001-103-002: Choose one or two primary YOLO11 variants to focus on for quantization (balancing realism and experiment time).
- [ ] Job-001-103-003: Identify the target hardware platform(s) (e.g., GPU model(s), driver/toolkit versions) and typical inference settings (batch size, resolution) relevant to this project.
- [ ] Job-001-103-004: Define quantitative success criteria (e.g., max allowable mAP drop, latency budget per image, throughput goals, memory constraints) for comparing FP16 vs. mixed FP16/INT8 models.
- [ ] Job-001-103-005: Capture the chosen variants and success criteria in the main task file and/or a linked context document so later milestones can reference them.

## Notes

- If exact deployment hardware is not yet fixed, define criteria for one or two representative GPU classes (e.g., data center vs. prosumer).
- Keep criteria realistic but firm enough to drive decision-making when selecting the “best” mixed-precision configuration later.

