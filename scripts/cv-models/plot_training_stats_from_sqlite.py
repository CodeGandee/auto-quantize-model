#!/usr/bin/env python
"""Plot TensorBoard-like curves from a training SQLite stats DB.

Reads the SQLite DB produced by `scripts/cv-models/collect_training_stats_sqlite.py`
and writes per-tag plots as SVG files, similar to TensorBoard scalars.

Example:
  pixi run -e rtx5090 python scripts/cv-models/plot_training_stats_from_sqlite.py \
    --db models/yolo10/reports/2025-12-25-qat-w4a16/train-logs/training-stats.db \
    --out-dir models/yolo10/reports/2025-12-25-qat-w4a16/train-logs/figures
"""

from __future__ import annotations

import argparse
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class ScalarSeries:
    tag: str
    steps: list[int]
    values: list[float]


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--run-name", type=str, default=None, help="Run name to plot (defaults to */root if present).")
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.6,
        help="EMA smoothing factor in [0, 1). 0 disables smoothing line.",
    )
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args(argv)


def _sanitize_filename(name: str) -> str:
    name = name.strip().replace("/", "__")
    name = re.sub(r"[^A-Za-z0-9_.()\\-]+", "_", name)
    return name


def _ema(values: list[float], smoothing: float) -> list[float]:
    if not values:
        return []
    if smoothing <= 0.0:
        return values[:]
    alpha = float(smoothing)
    out: list[float] = [float(values[0])]
    for v in values[1:]:
        out.append(alpha * float(out[-1]) + (1.0 - alpha) * float(v))
    return out


def _plot_train_vs_val_loss(
    *,
    train_total: ScalarSeries,
    val_total: ScalarSeries,
    out_path: Path,
    smoothing: float,
    dpi: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not train_total.steps and not val_total.steps:
        return

    plt.figure(figsize=(10.5, 4.5))

    train_values = _ema(train_total.values, smoothing=smoothing) if smoothing > 0.0 else train_total.values
    val_values = _ema(val_total.values, smoothing=smoothing) if smoothing > 0.0 else val_total.values

    if train_total.steps:
        plt.plot(train_total.steps, train_values, linewidth=2.0, color="black", label="train (smoothed)")
    if val_total.steps:
        plt.plot(val_total.steps, val_values, linewidth=2.0, color="red", label="val (smoothed)")

    title = "train vs val loss_total (lower is better)"
    if smoothing > 0.0:
        title += f" — ema({smoothing:.2f})"
    plt.title(title)
    plt.xlabel("step")
    plt.ylabel("loss_total")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, format="svg", dpi=int(dpi))
    plt.close()


def _pick_run_name(connection: sqlite3.Connection, *, requested: Optional[str]) -> str:
    rows = connection.execute("SELECT run_name FROM runs ORDER BY run_id").fetchall()
    run_names = [r[0] for r in rows]
    if not run_names:
        raise RuntimeError("No runs found in DB.")

    if requested is not None:
        if requested not in run_names:
            raise ValueError(f"--run-name {requested!r} not found. Available: {run_names}")
        return requested

    root_candidates = [n for n in run_names if n.endswith("/root") or n == "root"]
    return root_candidates[0] if root_candidates else run_names[0]


def _load_scalar_series(connection: sqlite3.Connection, *, run_name: str) -> list[ScalarSeries]:
    run_row = connection.execute("SELECT run_id FROM runs WHERE run_name = ?", (run_name,)).fetchone()
    if run_row is None:
        raise ValueError(f"Run not found: {run_name}")
    run_id = int(run_row[0])

    tags = connection.execute(
        "SELECT tag_id, tag FROM tags WHERE run_id = ? AND kind = 'scalar' ORDER BY tag",
        (run_id,),
    ).fetchall()

    out: list[ScalarSeries] = []
    for tag_id, tag in tags:
        rows = connection.execute(
            "SELECT step, value FROM scalars WHERE run_id = ? AND tag_id = ? ORDER BY step",
            (run_id, int(tag_id)),
        ).fetchall()
        steps = [int(r[0]) for r in rows]
        values = [float(r[1]) for r in rows]
        out.append(ScalarSeries(tag=str(tag), steps=steps, values=values))
    return out


def _derive_sum_series(*, series_list: list[ScalarSeries], prefix: str, out_tag: str) -> Optional[ScalarSeries]:
    matching = [s for s in series_list if s.tag.startswith(prefix)]
    if not matching:
        return None

    step_to_total: dict[int, float] = {}
    for series in matching:
        for step, value in zip(series.steps, series.values):
            step_to_total[step] = step_to_total.get(step, 0.0) + float(value)

    steps = sorted(step_to_total.keys())
    values = [float(step_to_total[s]) for s in steps]
    return ScalarSeries(tag=out_tag, steps=steps, values=values)


def _plot_scalar(
    *,
    series: ScalarSeries,
    run_name: str,
    out_path: Path,
    smoothing: float,
    dpi: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not series.steps:
        return

    plt.figure(figsize=(10.5, 4.5))
    plt.plot(series.steps, series.values, linewidth=1.0, color="#b7c0ce", label="raw")
    if smoothing > 0.0:
        smoothed = _ema(series.values, smoothing=smoothing)
        plt.plot(series.steps, smoothed, linewidth=2.0, color="#1f2d3d", label=f"ema({smoothing:.2f})")

    title = series.tag
    if series.tag.startswith("train/") or series.tag.startswith("val/"):
        title = f"{series.tag} (lower is better)"
    elif series.tag.startswith("lr/pg"):
        # Ultralytics groups optimizer params into 3 groups:
        # - pg0: bias parameters (no decay; special warmup schedule)
        # - pg1: non-Norm weights (with weight decay)
        # - pg2: Norm layer weights (no decay)
        pg = series.tag[len("lr/pg") :]
        if pg == "0":
            title = f"{series.tag} (pg0: bias params, no weight decay)"
        elif pg == "1":
            title = f"{series.tag} (pg1: non-Norm weights, with weight decay)"
        elif pg == "2":
            title = f"{series.tag} (pg2: Norm weights, no weight decay)"
    plt.title(title)
    plt.xlabel("step")
    plt.ylabel("value")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, format="svg", dpi=int(dpi))
    plt.close()


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    db_path = args.db.resolve()
    out_dir = args.out_dir.resolve()

    con = sqlite3.connect(str(db_path))
    try:
        run_name = _pick_run_name(con, requested=args.run_name)
        series_list = _load_scalar_series(con, run_name=run_name)
    finally:
        con.close()

    run_dir = out_dir / _sanitize_filename(run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Add derived "combined loss" curves to match common TensorBoard usage.
    train_total = _derive_sum_series(series_list=series_list, prefix="train/", out_tag="train/loss_total")
    if train_total is not None:
        series_list.append(train_total)
    val_total = _derive_sum_series(series_list=series_list, prefix="val/", out_tag="val/loss_total")
    if val_total is not None:
        series_list.append(val_total)

    for series in series_list:
        out_path = run_dir / f"{_sanitize_filename(series.tag)}.svg"
        _plot_scalar(series=series, run_name=run_name, out_path=out_path, smoothing=float(args.smoothing), dpi=int(args.dpi))

    if train_total is not None and val_total is not None:
        _plot_train_vs_val_loss(
            train_total=train_total,
            val_total=val_total,
            out_path=run_dir / "train-vs-val-loss.svg",
            smoothing=float(args.smoothing),
            dpi=int(args.dpi),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
