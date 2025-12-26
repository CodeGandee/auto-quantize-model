from __future__ import annotations

from pathlib import Path

import pytest

from auto_quantize_model.cv_models.yolov10_results_csv import read_metric_series


def test_read_metric_series_dedup_and_sort() -> None:
    results_csv = Path(__file__).parent / "fixtures" / "yolov10_results.csv"
    series = read_metric_series(results_csv=results_csv, metric_name="metrics/mAP50-95(B)")
    assert [p.epoch for p in series] == [0, 1, 2]
    assert series[0].value == pytest.approx(0.10)
    assert series[1].value == pytest.approx(0.31)  # duplicate epoch -> keep last-seen row
    assert series[2].value == pytest.approx(0.25)


def test_read_metric_series_missing_metric_raises() -> None:
    results_csv = Path(__file__).parent / "fixtures" / "yolov10_results.csv"
    with pytest.raises(KeyError):
        _ = read_metric_series(results_csv=results_csv, metric_name="does/not/exist")

