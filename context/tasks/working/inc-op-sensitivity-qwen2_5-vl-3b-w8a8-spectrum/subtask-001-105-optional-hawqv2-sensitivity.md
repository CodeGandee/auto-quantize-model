# Subtask 1.5: (Optional) Implement HAWQ_V2 Hessian-based sensitivity for Qwen2.5-VL-3B

## Scope

Optionally extend the sensitivity tooling to use INC’s `hawq_v2` strategy, which leverages Hessian trace-based metrics to rank layer sensitivity. This subtask is more compute-intensive and can be skipped if `mse_v2` is sufficient, but it can provide a richer view of sensitivity for research or fine-grained mixed-precision design.

## Planned outputs

- An extended sensitivity driver that can run in `hawq_v2` mode with a provided loss function.
- A saved artifact (JSON/Markdown) containing:
  - Per-op Hessian trace scores.
  - (Optional) a comparison summary between MSE and Hessian rankings.

## TODOs

- [ ] Job-001-105-001 Define an appropriate loss function for Qwen2.5-VL sensitivity (e.g., token-level cross-entropy on a small text set).
- [ ] Job-001-105-002 Configure `TuningCriterion(strategy="hawq_v2", strategy_kwargs={"hawq_v2_loss": loss_fn})` in the driver.
- [ ] Job-001-105-003 Run HAWQ_V2 on a small calibration subset to obtain per-op Hessian trace scores and save them (e.g., `qwen2_5_vl_3b_hawq_trace.json`).
- [ ] Job-001-105-004 Optionally compare MSE vs Hessian rankings and note any major differences or insights in a short summary.

## Notes

- Because HAWQ_V2 is more expensive than MSE_V2, keep calibration sizes and iterations small, treating this as an analysis tool rather than a frequent operation.

