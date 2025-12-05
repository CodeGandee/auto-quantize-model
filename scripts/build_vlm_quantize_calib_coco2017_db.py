#!/usr/bin/env python
"""
Build a small VLM calibration subset from COCO 2017 captions into a SQLite DB.

The DB only stores lightweight metadata:
  - Relative image paths (under the COCO 2017 root).
  - Image / caption identifiers.
  - The caption text itself.

No image data is copied; consumers are expected to resolve image paths
relative to the COCO 2017 dataset root (for this repo, typically
`datasets/coco2017/source-data`).

Example usage:

    pixi run python scripts/build_vlm_quantize_calib_coco2017_db.py \\
        --coco-root datasets/coco2017/source-data \\
        --out datasets/vlm-quantize-calib/coco2017_vlm_calib.db \\
        --max-samples 4096
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass
class CalibSample:
    """Single calibration sample reference for VLM PTQ."""

    split: str
    image_relpath: str
    image_id: int
    caption_id: int
    caption: str


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build a COCO2017-based VLM calibration SQLite DB."
    )
    parser.add_argument(
        "--coco-root",
        type=str,
        default="datasets/coco2017/source-data",
        help=(
            "Root directory of the COCO 2017 dataset. "
            "Expected to contain train2017/, val2017/, annotations/."
        ),
    )
    parser.add_argument(
        "--out",
        type=str,
        default="datasets/vlm-quantize-calib/coco2017_vlm_calib.db",
        help="Output path for the SQLite database.",
    )
    parser.add_argument(
        "--captions-text-out",
        type=str,
        default="datasets/vlm-quantize-calib/coco2017_captions.txt",
        help=(
            "Optional path for a newline-delimited text file of captions "
            "corresponding to the selected calibration subset."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=4096,
        help=(
            "Maximum number of (image, caption) samples to include. "
            "If 0 or negative, include all available samples."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when sampling from COCO captions.",
    )
    return parser.parse_args(argv)


def load_coco_captions(
    annotations_path: Path,
    split: str,
    image_subdir: str,
) -> List[CalibSample]:
    """Load caption annotations from a COCO captions JSON file."""
    if not annotations_path.is_file():
        raise FileNotFoundError(f"COCO captions file not found: {annotations_path}")

    with annotations_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    images_index = {img["id"]: img["file_name"] for img in data.get("images", [])}
    samples: List[CalibSample] = []

    for ann in data.get("annotations", []):
        image_id = int(ann["image_id"])
        caption_id = int(ann["id"])
        caption = str(ann["caption"])

        file_name = images_index.get(image_id)
        if file_name is None:
            # Skip annotations without a corresponding image entry.
            continue

        image_relpath = f"{image_subdir}/{file_name}"
        samples.append(
            CalibSample(
                split=split,
                image_relpath=image_relpath,
                image_id=image_id,
                caption_id=caption_id,
                caption=caption,
            )
        )

    return samples


def choose_subset(
    samples: Iterable[CalibSample],
    max_samples: int,
    seed: int,
) -> List[CalibSample]:
    """Choose a random subset of calibration samples."""
    samples_list = list(samples)
    if max_samples <= 0 or max_samples >= len(samples_list):
        return samples_list

    rng = random.Random(seed)
    return rng.sample(samples_list, k=max_samples)


def init_db(connection: sqlite3.Connection) -> None:
    """Create tables for the calibration SQLite database."""
    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS vlm_calib_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            split TEXT NOT NULL,
            image_relpath TEXT NOT NULL,
            image_id INTEGER NOT NULL,
            caption_id INTEGER NOT NULL,
            caption TEXT NOT NULL
        )
        """
    )

    connection.commit()


def write_samples_to_db(
    connection: sqlite3.Connection,
    samples: Iterable[CalibSample],
) -> int:
    """Insert calibration samples into the SQLite DB."""
    cursor = connection.cursor()
    rows = [
        (
            sample.split,
            sample.image_relpath,
            sample.image_id,
            sample.caption_id,
            sample.caption,
        )
        for sample in samples
    ]

    cursor.executemany(
        """
        INSERT INTO vlm_calib_samples (
            split,
            image_relpath,
            image_id,
            caption_id,
            caption
        ) VALUES (?, ?, ?, ?, ?)
        """,
        rows,
    )
    connection.commit()
    return len(rows)


def write_meta(connection: sqlite3.Connection, key: str, value: str) -> None:
    """Insert or replace a single metadata key/value pair."""
    cursor = connection.cursor()
    cursor.execute(
        """
        INSERT INTO meta (key, value)
        VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """,
        (key, value),
    )
    connection.commit()


def main(argv: List[str] | None = None) -> int:
    """Entry point for building the COCO2017 VLM calibration DB."""
    args = parse_args(argv)

    coco_root = Path(args.coco_root)
    if not coco_root.exists():
        raise FileNotFoundError(f"COCO root not found at {coco_root}")

    annotations_dir = coco_root / "annotations"
    train_json = annotations_dir / "captions_train2017.json"
    val_json = annotations_dir / "captions_val2017.json"

    print(f"[INFO] Using COCO root: {coco_root}")
    print(f"[INFO] Loading captions from: {train_json} and {val_json}")

    train_samples = load_coco_captions(
        annotations_path=train_json,
        split="train2017",
        image_subdir="train2017",
    )
    val_samples = load_coco_captions(
        annotations_path=val_json,
        split="val2017",
        image_subdir="val2017",
    )

    all_samples = train_samples + val_samples
    print(
        f"[INFO] Loaded {len(train_samples)} train and {len(val_samples)} val captions "
        f"({len(all_samples)} total samples)."
    )

    chosen_samples = choose_subset(
        samples=all_samples,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    print(f"[INFO] Selected {len(chosen_samples)} samples for calibration subset.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(out_path)
    try:
        init_db(connection)
        written = write_samples_to_db(connection, chosen_samples)
        write_meta(connection, "source_dataset", "coco2017_captions")
        write_meta(connection, "coco_root_relative", "../coco2017/source-data")
        write_meta(connection, "num_samples", str(written))
        write_meta(connection, "random_seed", str(args.seed))
    finally:
        connection.close()

    print(f"[INFO] Wrote {written} samples to SQLite DB at: {out_path}")

    # Optionally export the selected captions into a simple text file for
    # generic text-only calibration loaders (one caption per line).
    captions_txt_path = Path(args.captions_text_out)
    captions_txt_path.parent.mkdir(parents=True, exist_ok=True)
    with captions_txt_path.open("w", encoding="utf-8") as handle:
        for sample in chosen_samples:
            handle.write(sample.caption.replace("\n", " ").strip())
            handle.write("\n")

    print(f"[INFO] Wrote captions text file with {len(chosen_samples)} lines to: {captions_txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
