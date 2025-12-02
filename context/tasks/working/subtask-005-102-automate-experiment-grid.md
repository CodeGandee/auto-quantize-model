# Subtask 5.2: Automate experiment grid execution

## Scope

Implement an automated pipeline that runs the full quantize → build TensorRT engine → benchmark loop for each configuration in the experiment grid using a single `pixi run` entrypoint.

## Planned outputs

- A script or driver (Python or shell) that iterates over the configuration grid and triggers the necessary steps.
- Logged metrics for each configuration, stored in a consistent format and location.
- Minimal manual intervention required to run the full set of experiments.

## TODOs

- [ ] Job-005-102-001: Design the interface (e.g., CLI flags, config files) that the experiment driver will use to specify different configurations.
- [ ] Job-005-102-002: Implement the driver that loops over configurations, invoking quantization, TensorRT engine building, and benchmarking steps for each.
- [ ] Job-005-102-003: Verify that the pipeline can run at least a small subset of configurations end-to-end and produces logs/results as expected.

## Notes

- Reuse as much of the existing tooling from sections 2–4 as possible to avoid duplication.

