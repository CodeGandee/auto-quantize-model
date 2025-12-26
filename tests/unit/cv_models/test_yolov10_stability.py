from __future__ import annotations

import pytest

from auto_quantize_model.cv_models.yolov10_results_csv import MetricPoint
from auto_quantize_model.cv_models.yolov10_stability import classify_collapse, classify_run_status, summarize_series


def test_summarize_series_best_and_final() -> None:
    series = [MetricPoint(epoch=0, value=0.1), MetricPoint(epoch=1, value=0.3), MetricPoint(epoch=2, value=0.2)]
    best_value, final_value = summarize_series(series)
    assert best_value == pytest.approx(0.3)
    assert final_value == pytest.approx(0.2)


def test_classify_collapse_rule() -> None:
    series = [MetricPoint(epoch=0, value=0.2), MetricPoint(epoch=1, value=0.8), MetricPoint(epoch=2, value=0.3)]
    collapsed, ratio = classify_collapse(series=series, threshold_ratio=0.5)
    assert collapsed is True
    assert ratio == pytest.approx(0.375)
    assert classify_run_status(collapsed=collapsed) == "completed-failed"


def test_classify_collapse_best_zero_is_not_collapsed() -> None:
    series = [MetricPoint(epoch=0, value=0.0)]
    collapsed, ratio = classify_collapse(series=series, threshold_ratio=0.5)
    assert collapsed is False
    assert ratio == pytest.approx(0.0)
    assert classify_run_status(collapsed=collapsed) == "completed-success"


def test_classify_run_status_incomplete_on_error() -> None:
    assert classify_run_status(collapsed=False, error="boom") == "incomplete"

