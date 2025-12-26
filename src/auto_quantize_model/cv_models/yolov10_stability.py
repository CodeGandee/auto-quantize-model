"""Collapse detection and run-status classification for YOLOv10 QAT validation."""

from __future__ import annotations

from auto_quantize_model.cv_models.yolov10_results_csv import MetricPoint


def summarize_series(series: list[MetricPoint]) -> tuple[float, float]:
    """Return (best_value, final_value) from a metric series."""

    if not series:
        raise ValueError("Metric series is empty.")
    best_value = max(point.value for point in series)
    final_value = series[-1].value
    return float(best_value), float(final_value)


def classify_collapse(
    *,
    series: list[MetricPoint],
    threshold_ratio: float = 0.5,
) -> tuple[bool, float]:
    """Return (collapsed, final_over_best_ratio)."""

    best_value, final_value = summarize_series(series)
    final_over_best_ratio = float(final_value / best_value) if best_value > 0 else 0.0
    collapsed = bool(best_value > 0 and final_value < float(threshold_ratio) * best_value)
    return collapsed, final_over_best_ratio


def classify_run_status(*, collapsed: bool, error: str | None = None) -> str:
    """Map a run outcome to the run_summary.schema.json status values."""

    if error:
        return "incomplete"
    return "completed-failed" if collapsed else "completed-success"

