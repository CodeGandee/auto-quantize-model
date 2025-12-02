# Subtask 4.3: Collect and store TensorRT benchmark metrics

## Scope

Benchmark FP16 and mixed FP16/INT8 TensorRT engines for the selected YOLO11 variants on the RTX 3090, and store the resulting metrics in a structured form for later analysis.

## Planned outputs

- A set of benchmark runs for FP16 and mixed engines covering relevant batch sizes (starting with batch size 1).
- Recorded metrics for latency, throughput, GPU memory usage, and accuracy for each engine.
- One or more machine-readable results files (e.g., JSON/CSV) stored in a predictable location in the repo.

## TODOs

- [ ] Job-004-103-001: Define the benchmark protocol (number of warmup/measurement iterations, batch sizes, and metrics to record) for YOLO11 TensorRT engines.
- [ ] Job-004-103-002: Run benchmarks for FP16 and mixed engines on the RTX 3090 according to the protocol and collect latency, throughput, memory, and accuracy metrics.
- [ ] Job-004-103-003: Store the benchmark results in a structured format and add a short summary or pointer in this subtask file for later reference.

## Notes

- These metrics will feed into the configuration search and selection work in section 5.

