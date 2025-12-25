#!/usr/bin/env python
"""Collect Ultralytics/TensorBoard training stats into a SQLite database.

This script parses TensorBoard event files (scalars, histograms, images, tensors,
etc.) and optionally Ultralytics `results.csv`, then writes everything into a
single SQLite DB for analysis and reproducible reporting.

Example:
  pixi run -e rtx5090 python scripts/cv-models/collect_training_stats_sqlite.py \
    --logdir tmp/.../qat-w4a16/ultralytics/yolov10m-scratch-qat-w4a16 \
    --results-csv tmp/.../qat-w4a16/ultralytics/yolov10m-scratch-qat-w4a16/results.csv \
    --out models/yolo10/reports/<report>/train-logs/training-stats.db
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional


@dataclass(frozen=True)
class RunInfo:
    run_name: str
    run_dir: Path


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logdir", type=Path, required=True, help="TensorBoard logdir to scan recursively.")
    parser.add_argument("--results-csv", type=Path, default=None, help="Optional Ultralytics results.csv.")
    parser.add_argument("--out", type=Path, required=True, help="Output SQLite DB path.")
    parser.add_argument("--run-prefix", type=str, default="", help="Optional prefix for run names in the DB.")
    return parser.parse_args(argv)


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(db_path))
    connection.execute("PRAGMA journal_mode=WAL;")
    connection.execute("PRAGMA synchronous=NORMAL;")
    connection.execute("PRAGMA foreign_keys=ON;")
    return connection


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS runs (
          run_id INTEGER PRIMARY KEY AUTOINCREMENT,
          run_name TEXT NOT NULL UNIQUE,
          run_dir TEXT NOT NULL,
          created_at_unix REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS event_files (
          event_file_id INTEGER PRIMARY KEY AUTOINCREMENT,
          run_id INTEGER NOT NULL,
          path TEXT NOT NULL,
          size_bytes INTEGER NOT NULL,
          mtime_unix REAL NOT NULL,
          FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS tags (
          tag_id INTEGER PRIMARY KEY AUTOINCREMENT,
          run_id INTEGER NOT NULL,
          kind TEXT NOT NULL,
          tag TEXT NOT NULL,
          UNIQUE(run_id, kind, tag),
          FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS scalars (
          scalar_id INTEGER PRIMARY KEY AUTOINCREMENT,
          run_id INTEGER NOT NULL,
          tag_id INTEGER NOT NULL,
          step INTEGER NOT NULL,
          wall_time REAL NOT NULL,
          value REAL NOT NULL,
          FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
          FOREIGN KEY(tag_id) REFERENCES tags(tag_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_scalars_run_tag_step ON scalars(run_id, tag_id, step);

        CREATE TABLE IF NOT EXISTS histograms (
          histogram_id INTEGER PRIMARY KEY AUTOINCREMENT,
          run_id INTEGER NOT NULL,
          tag_id INTEGER NOT NULL,
          step INTEGER NOT NULL,
          wall_time REAL NOT NULL,
          min REAL,
          max REAL,
          num REAL,
          sum REAL,
          sum_squares REAL,
          bucket_limits_json TEXT,
          bucket_counts_json TEXT,
          FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
          FOREIGN KEY(tag_id) REFERENCES tags(tag_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_histograms_run_tag_step ON histograms(run_id, tag_id, step);

        CREATE TABLE IF NOT EXISTS images (
          image_id INTEGER PRIMARY KEY AUTOINCREMENT,
          run_id INTEGER NOT NULL,
          tag_id INTEGER NOT NULL,
          step INTEGER NOT NULL,
          wall_time REAL NOT NULL,
          width INTEGER,
          height INTEGER,
          colorspace INTEGER,
          encoded_image BLOB,
          FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
          FOREIGN KEY(tag_id) REFERENCES tags(tag_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_images_run_tag_step ON images(run_id, tag_id, step);

        CREATE TABLE IF NOT EXISTS tensors (
          tensor_id INTEGER PRIMARY KEY AUTOINCREMENT,
          run_id INTEGER NOT NULL,
          tag_id INTEGER NOT NULL,
          step INTEGER NOT NULL,
          wall_time REAL NOT NULL,
          tensor_proto BLOB,
          FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
          FOREIGN KEY(tag_id) REFERENCES tags(tag_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_tensors_run_tag_step ON tensors(run_id, tag_id, step);

        CREATE TABLE IF NOT EXISTS run_metadata (
          run_metadata_id INTEGER PRIMARY KEY AUTOINCREMENT,
          run_id INTEGER NOT NULL,
          tag_id INTEGER NOT NULL,
          step INTEGER NOT NULL,
          wall_time REAL NOT NULL,
          serialized_metadata BLOB,
          FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
          FOREIGN KEY(tag_id) REFERENCES tags(tag_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_run_metadata_run_tag_step ON run_metadata(run_id, tag_id, step);

        CREATE TABLE IF NOT EXISTS csv_kv (
          csv_kv_id INTEGER PRIMARY KEY AUTOINCREMENT,
          run_name TEXT NOT NULL,
          epoch INTEGER NOT NULL,
          key TEXT NOT NULL,
          value_real REAL,
          value_text TEXT,
          UNIQUE(run_name, epoch, key)
        );

        CREATE INDEX IF NOT EXISTS idx_csv_kv_run_epoch ON csv_kv(run_name, epoch);
        """
    )


def _discover_event_run_dirs(logdir: Path) -> list[RunInfo]:
    event_files = sorted(logdir.rglob("events.out.tfevents.*"))
    if not event_files:
        return []

    run_dirs = sorted({p.parent for p in event_files})
    runs: list[RunInfo] = []
    for run_dir in run_dirs:
        run_name = os.path.relpath(run_dir, start=logdir)
        if run_name == ".":
            run_name = "root"
        runs.append(RunInfo(run_name=run_name, run_dir=run_dir))
    return runs


def _insert_run(connection: sqlite3.Connection, *, run_name: str, run_dir: Path) -> int:
    created_at = time.time()
    connection.execute(
        "INSERT OR REPLACE INTO runs(run_name, run_dir, created_at_unix) VALUES(?, ?, ?)",
        (run_name, str(run_dir), float(created_at)),
    )
    row = connection.execute("SELECT run_id FROM runs WHERE run_name = ?", (run_name,)).fetchone()
    if row is None:
        raise RuntimeError(f"Failed to look up run_id for run_name={run_name}")
    return int(row[0])


def _insert_event_files(connection: sqlite3.Connection, *, run_id: int, run_dir: Path) -> None:
    for event_file in sorted(run_dir.glob("events.out.tfevents.*")):
        stat = event_file.stat()
        connection.execute(
            "INSERT INTO event_files(run_id, path, size_bytes, mtime_unix) VALUES(?, ?, ?, ?)",
            (int(run_id), str(event_file), int(stat.st_size), float(stat.st_mtime)),
        )


def _get_or_create_tag_id(connection: sqlite3.Connection, *, run_id: int, kind: str, tag: str) -> int:
    connection.execute(
        "INSERT OR IGNORE INTO tags(run_id, kind, tag) VALUES(?, ?, ?)",
        (int(run_id), str(kind), str(tag)),
    )
    row = connection.execute(
        "SELECT tag_id FROM tags WHERE run_id = ? AND kind = ? AND tag = ?",
        (int(run_id), str(kind), str(tag)),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"Failed to resolve tag_id for run_id={run_id} kind={kind} tag={tag}")
    return int(row[0])


def _iter_csv_rows(csv_path: Path) -> Iterator[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            cleaned = {str(k).strip(): str(v).strip() for k, v in row.items() if k is not None}
            yield cleaned


def _parse_float(value: str) -> Optional[float]:
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _ingest_results_csv(connection: sqlite3.Connection, *, run_name: str, csv_path: Path) -> None:
    if not csv_path.is_file():
        return
    for row in _iter_csv_rows(csv_path):
        epoch_raw = row.get("epoch", "")
        epoch_val = _parse_float(epoch_raw)
        if epoch_val is None:
            continue
        epoch = int(epoch_val)
        for key, value in row.items():
            if key == "epoch":
                continue
            value_real = _parse_float(value)
            connection.execute(
                "INSERT OR REPLACE INTO csv_kv(run_name, epoch, key, value_real, value_text) VALUES(?, ?, ?, ?, ?)",
                (str(run_name), int(epoch), str(key), value_real, None if value_real is not None else str(value)),
            )


def _json_dumps(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _ingest_tensorboard_events(connection: sqlite3.Connection, *, run_id: int, run_dir: Path) -> None:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(
        str(run_dir),
        size_guidance={
            "scalars": 0,
            "images": 0,
            "histograms": 0,
            "tensors": 0,
            "audio": 0,
            "compressedHistograms": 0,
        },
    )
    accumulator.Reload()
    tags = accumulator.Tags()

    scalar_tags = tags.get("scalars", []) or []
    for tag in scalar_tags:
        tag_id = _get_or_create_tag_id(connection, run_id=run_id, kind="scalar", tag=tag)
        for event in accumulator.Scalars(tag):
            connection.execute(
                "INSERT INTO scalars(run_id, tag_id, step, wall_time, value) VALUES(?, ?, ?, ?, ?)",
                (int(run_id), int(tag_id), int(event.step), float(event.wall_time), float(event.value)),
            )

    histogram_tags = tags.get("histograms", []) or []
    for tag in histogram_tags:
        tag_id = _get_or_create_tag_id(connection, run_id=run_id, kind="histogram", tag=tag)
        for event in accumulator.Histograms(tag):
            h = event.histogram_value
            connection.execute(
                """
                INSERT INTO histograms(
                  run_id, tag_id, step, wall_time,
                  min, max, num, sum, sum_squares,
                  bucket_limits_json, bucket_counts_json
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(run_id),
                    int(tag_id),
                    int(event.step),
                    float(event.wall_time),
                    float(h.min),
                    float(h.max),
                    float(h.num),
                    float(h.sum),
                    float(h.sum_squares),
                    _json_dumps(list(h.bucket_limit)),
                    _json_dumps(list(h.bucket)),
                ),
            )

    image_tags = tags.get("images", []) or []
    for tag in image_tags:
        tag_id = _get_or_create_tag_id(connection, run_id=run_id, kind="image", tag=tag)
        for event in accumulator.Images(tag):
            connection.execute(
                """
                INSERT INTO images(run_id, tag_id, step, wall_time, width, height, colorspace, encoded_image)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(run_id),
                    int(tag_id),
                    int(event.step),
                    float(event.wall_time),
                    int(event.width) if event.width is not None else None,
                    int(event.height) if event.height is not None else None,
                    int(event.colorspace) if event.colorspace is not None else None,
                    sqlite3.Binary(event.encoded_image_string),
                ),
            )

    tensor_tags = tags.get("tensors", []) or []
    for tag in tensor_tags:
        tag_id = _get_or_create_tag_id(connection, run_id=run_id, kind="tensor", tag=tag)
        for event in accumulator.Tensors(tag):
            proto_bytes = event.tensor_proto.SerializeToString()
            connection.execute(
                "INSERT INTO tensors(run_id, tag_id, step, wall_time, tensor_proto) VALUES(?, ?, ?, ?, ?)",
                (int(run_id), int(tag_id), int(event.step), float(event.wall_time), sqlite3.Binary(proto_bytes)),
            )

    run_metadata_tags = tags.get("run_metadata", []) or []
    for tag in run_metadata_tags:
        tag_id = _get_or_create_tag_id(connection, run_id=run_id, kind="run_metadata", tag=tag)
        metadata_obj = accumulator.RunMetadata(tag)

        # TensorBoard returns either:
        # - a RunMetadata protobuf (common), or
        # - a wrapper event (rare; depends on tensorboard version/producer).
        if hasattr(metadata_obj, "SerializeToString"):
            # Conventionally Ultralytics uses tags like "step1", "step2", ...
            step = 0
            if tag.startswith("step"):
                try:
                    step = int(tag[len("step") :])
                except ValueError:
                    step = 0
            connection.execute(
                """
                INSERT INTO run_metadata(run_id, tag_id, step, wall_time, serialized_metadata)
                VALUES(?, ?, ?, ?, ?)
                """,
                (int(run_id), int(tag_id), int(step), 0.0, sqlite3.Binary(metadata_obj.SerializeToString())),
            )
            continue

        metadata_events: Iterable[Any]
        if isinstance(metadata_obj, list):
            metadata_events = metadata_obj
        else:
            metadata_events = [metadata_obj]

        for event in metadata_events:
            run_meta = getattr(event, "run_metadata", None)
            if run_meta is None or not hasattr(run_meta, "SerializeToString"):
                continue
            connection.execute(
                """
                INSERT INTO run_metadata(run_id, tag_id, step, wall_time, serialized_metadata)
                VALUES(?, ?, ?, ?, ?)
                """,
                (
                    int(run_id),
                    int(tag_id),
                    int(getattr(event, "step", 0)),
                    float(getattr(event, "wall_time", 0.0)),
                    sqlite3.Binary(run_meta.SerializeToString()),
                ),
            )


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    logdir = args.logdir.resolve()
    out_db = args.out.resolve()
    results_csv = args.results_csv.resolve() if args.results_csv is not None else None

    connection = _connect(out_db)
    try:
        _create_schema(connection)
        connection.execute("INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)", ("logdir", str(logdir)))
        connection.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)",
            ("generated_at_unix", str(time.time())),
        )

        runs = _discover_event_run_dirs(logdir)
        if not runs:
            raise FileNotFoundError(f"No TensorBoard event files found under {logdir}")

        with connection:
            for run in runs:
                run_name = f"{args.run_prefix}{run.run_name}"
                run_id = _insert_run(connection, run_name=run_name, run_dir=run.run_dir)
                _insert_event_files(connection, run_id=run_id, run_dir=run.run_dir)
                _ingest_tensorboard_events(connection, run_id=run_id, run_dir=run.run_dir)

            # results.csv is typically present at the root run dir; still store it if provided.
            if results_csv is not None:
                _ingest_results_csv(connection, run_name=f"{args.run_prefix}root", csv_path=results_csv)
            else:
                default_csv = logdir / "results.csv"
                if default_csv.is_file():
                    _ingest_results_csv(connection, run_name=f"{args.run_prefix}root", csv_path=default_csv)

    finally:
        connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
