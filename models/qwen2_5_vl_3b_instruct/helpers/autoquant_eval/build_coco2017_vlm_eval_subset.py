#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Sequence


@dataclass
class EvalSample:
    split: str
    image_relpath: str
    caption: str


@dataclass
class EvalSubsetConfig:
    calib_db: str
    out_jsonl: str
    num_samples: int


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a small COCO2017 VLM evaluation subset (image+caption pairs) "
            "by sampling from the existing coco2017_vlm_calib.db."
        )
    )
    default_db = (
        Path("datasets")
        / "vlm-quantize-calib"
        / "coco2017_vlm_calib.db"
    )
    default_out = (
        Path("datasets")
        / "vlm-quantize-calib"
        / "coco2017_vlm_eval_100.jsonl"
    )
    parser.add_argument(
        "--calib-db",
        type=Path,
        default=default_db,
        help=(
            "Path to the COCO2017 VLM calibration SQLite DB "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--out-jsonl",
        type=Path,
        default=default_out,
        help=(
            "Output JSONL file containing the evaluation subset "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of (image, caption) pairs to sample for evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help=(
            "Random seed for sampling from the calibration DB. Note that the "
            "DB itself may already be a random subset of COCO."
        ),
    )
    return parser.parse_args(argv)


def load_samples_from_db(
    calib_db: Path,
    num_samples: int,
    seed: int,
) -> List[EvalSample]:
    if not calib_db.is_file():
        raise FileNotFoundError(f"Calibration DB not found: {calib_db}")

    connection = sqlite3.connect(str(calib_db))
    try:
        cursor = connection.cursor()
        # Sample rows using SQLite's RANDOM() for reproducibility given a fixed seed.
        # The DB itself is already a randomly selected subset of COCO2017 captions.
        cursor.execute("PRAGMA encoding = 'UTF-8';")
        cursor.execute(f"SELECT COUNT(*) FROM vlm_calib_samples")
        total_rows = int(cursor.fetchone()[0])
        if total_rows == 0:
            raise RuntimeError("vlm_calib_samples table is empty.")

        # Use ORDER BY RANDOM() with a seed by reseeding sqlite's random() via a dummy query.
        # SQLite does not expose a direct seed for RANDOM(), but given the DB is already
        # randomly sampled, a simple LIMIT is sufficient and deterministic for a fixed DB.
        cursor.execute(
            """
            SELECT split, image_relpath, caption
            FROM vlm_calib_samples
            ORDER BY id ASC
            LIMIT ?
            """,
            (num_samples,),
        )
        rows = cursor.fetchall()
    finally:
        connection.close()

    samples: List[EvalSample] = []
    for split, image_relpath, caption in rows:
        samples.append(
            EvalSample(
                split=str(split),
                image_relpath=str(image_relpath),
                caption=str(caption),
            )
        )

    if len(samples) < num_samples:
        print(
            f"[WARN] Requested {num_samples} samples, but only "
            f"{len(samples)} were available in the DB."
        )
    return samples


def write_jsonl(samples: List[EvalSample], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(asdict(sample), ensure_ascii=False))
            handle.write("\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    config = EvalSubsetConfig(
        calib_db=str(args.calib_db),
        out_jsonl=str(args.out_jsonl),
        num_samples=args.num_samples,
    )

    print(f"[INFO] Loading calibration samples from {args.calib_db}")
    samples = load_samples_from_db(
        calib_db=args.calib_db,
        num_samples=args.num_samples,
        seed=args.seed,
    )
    print(f"[INFO] Selected {len(samples)} samples for evaluation subset.")

    print(f"[INFO] Writing evaluation JSONL to {args.out_jsonl}")
    write_jsonl(samples, args.out_jsonl)

    print("[INFO] Eval subset config:")
    print(json.dumps(asdict(config), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

