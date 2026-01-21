"""Ultralytics `results.csv` parsing helpers (YOLOv10 training runs)."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MetricPoint:
    epoch: int
    value: float


def read_metric_series(*, results_csv: Path, metric_name: str) -> list[MetricPoint]:
    """Return a per-epoch metric series for a named column.

    Notes
    -----
    - Ultralytics may append duplicate epoch rows (e.g. resume/rewrite); we keep
      the last-seen value per epoch.
    - Header names are normalized by stripping whitespace.
    """

    results_csv = Path(results_csv)
    if not results_csv.is_file():
        raise FileNotFoundError(f"results.csv not found: {results_csv}")

    values_by_epoch: dict[int, float] = {}

    with results_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Empty results.csv: {results_csv}") from exc

        header_norm = [str(name).strip() for name in header]
        if metric_name not in header_norm:
            raise KeyError(
                f"Metric column {metric_name!r} not found in {results_csv}; "
                f"available={header_norm}"
            )

        metric_idx = header_norm.index(metric_name)
        epoch_idx = header_norm.index("epoch") if "epoch" in header_norm else 0

        for row in reader:
            if not row:
                continue
            if all(not str(cell).strip() for cell in row):
                continue
            if len(row) <= max(epoch_idx, metric_idx):
                continue

            epoch_raw = str(row[epoch_idx]).strip()
            metric_raw = str(row[metric_idx]).strip()
            if not epoch_raw or not metric_raw:
                continue

            try:
                epoch = int(epoch_raw)
            except Exception:
                try:
                    epoch = int(float(epoch_raw))
                except Exception:
                    continue

            try:
                value = float(metric_raw)
            except Exception:
                continue

            values_by_epoch[int(epoch)] = float(value)

    return [MetricPoint(epoch=e, value=values_by_epoch[e]) for e in sorted(values_by_epoch.keys())]

