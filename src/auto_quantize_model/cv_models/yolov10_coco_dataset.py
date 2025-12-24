"""Run-local COCO2017 (YOLO-format) dataset helpers for Ultralytics YOLOv10.

The repository keeps COCO2017 under `datasets/coco2017/source-data/` as the
original MS-COCO layout:

  train2017/ val2017/ annotations/instances_*.json

Ultralytics training expects a YOLO-format dataset layout:

  <root>/images/{train2017,val2017}/...
  <root>/labels/{train2017,val2017}/...

This module prepares a run-local YOLO-format dataset under `tmp/<run>/dataset/`
by converting COCO JSON annotations to YOLO labels (via Ultralytics'
`convert_coco`) and creating directory symlinks for images.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml  # type: ignore[import-untyped]

from auto_quantize_model.cv_models.yolov10_brevitas import ensure_local_yolo10_src_on_path


@dataclass(frozen=True)
class CocoYoloDataset:
    dataset_root: Path
    dataset_yaml: Path
    provenance_json: Path


def _symlink_dir(*, src_dir: Path, dst_dir: Path) -> None:
    dst_dir.parent.mkdir(parents=True, exist_ok=True)
    if dst_dir.exists() or dst_dir.is_symlink():
        if dst_dir.is_symlink() and dst_dir.resolve() == src_dir.resolve():
            return
        raise FileExistsError(f"Refusing to overwrite existing path {dst_dir}")
    rel = os.path.relpath(src_dir, start=dst_dir.parent)
    dst_dir.symlink_to(rel, target_is_directory=True)


def _symlink_files(*, src_dir: Path, dst_dir: Path) -> int:
    """Symlink all files from `src_dir` into `dst_dir` (flat)."""

    dst_dir.mkdir(parents=True, exist_ok=True)
    created = 0
    for src_path in src_dir.iterdir():
        if not src_path.is_file():
            continue
        dst_path = dst_dir / src_path.name
        if dst_path.exists() or dst_path.is_symlink():
            continue
        rel = os.path.relpath(src_path, start=dst_path.parent)
        dst_path.symlink_to(rel)
        created += 1
    return created


def load_coco_names(*, repo_root: Path) -> dict[int, str]:
    coco_yaml_path = repo_root / "models" / "yolo10" / "src" / "ultralytics" / "cfg" / "datasets" / "coco.yaml"
    data = yaml.safe_load(coco_yaml_path.read_text(encoding="utf-8"))
    names = data.get("names", {})
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {int(i): str(name) for i, name in enumerate(names)}
    raise ValueError(f"Unexpected names format in {coco_yaml_path}")


def write_dataset_yaml(*, dataset_root: Path, names: dict[int, str]) -> Path:
    yaml_path = dataset_root / "coco2017-yolo.yaml"
    payload = {
        "path": str(dataset_root.resolve()),
        "train": "images/train2017",
        "val": "images/val2017",
        "names": names,
    }
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return yaml_path


def prepare_coco2017_yolo_dataset(
    *,
    repo_root: Path,
    coco_root: Path,
    out_dir: Path,
) -> CocoYoloDataset:
    """Prepare (or reuse) a run-local COCO2017 YOLO-format dataset under `out_dir`."""

    coco_root = coco_root.resolve()
    annotations_dir = coco_root / "annotations"
    if not annotations_dir.is_dir():
        raise FileNotFoundError(f"Missing COCO annotations dir: {annotations_dir}")
    if not (coco_root / "train2017").is_dir() or not (coco_root / "val2017").is_dir():
        raise FileNotFoundError(f"Missing COCO train2017/val2017 under: {coco_root}")

    out_dir.mkdir(parents=True, exist_ok=True)
    annotations_subset = out_dir / "annotations_instances"
    dataset_root = out_dir / "coco_yolo"
    provenance_json = out_dir / "provenance.json"

    ensure_local_yolo10_src_on_path(repo_root=repo_root)
    from ultralytics.data.converter import convert_coco  # type: ignore[import-not-found]

    instances_train = annotations_dir / "instances_train2017.json"
    instances_val = annotations_dir / "instances_val2017.json"
    if not instances_train.is_file() or not instances_val.is_file():
        raise FileNotFoundError("Missing COCO instances_{train,val}2017.json under annotations/.")

    annotations_subset.mkdir(parents=True, exist_ok=True)
    _symlink_dir(src_dir=annotations_dir, dst_dir=annotations_subset / "_src_annotations_dir")  # provenance only
    for src in (instances_train, instances_val):
        dst = annotations_subset / src.name
        if dst.exists() or dst.is_symlink():
            continue
        rel = os.path.relpath(src, start=dst.parent)
        dst.symlink_to(rel)

    if not dataset_root.exists():
        convert_coco(
            labels_dir=str(annotations_subset),
            save_dir=str(dataset_root),
            use_segments=False,
            use_keypoints=False,
            cls91to80=True,
        )

    train_images = coco_root / "train2017"
    val_images = coco_root / "val2017"

    train_dst = dataset_root / "images" / "train2017"
    val_dst = dataset_root / "images" / "val2017"
    if train_dst.is_symlink():
        train_dst.unlink()
    if val_dst.is_symlink():
        val_dst.unlink()

    train_created = _symlink_files(src_dir=train_images, dst_dir=train_dst)
    val_created = _symlink_files(src_dir=val_images, dst_dir=val_dst)

    names = load_coco_names(repo_root=repo_root)
    dataset_yaml = write_dataset_yaml(dataset_root=dataset_root, names=names)

    provenance: Dict[str, Any] = {
        "coco_root": str(coco_root),
        "dataset_root": str(dataset_root),
        "dataset_yaml": str(dataset_yaml),
        "annotations_dir": str(annotations_dir),
        "annotations_subset": str(annotations_subset),
        "instances_train": str(instances_train),
        "instances_val": str(instances_val),
        "symlinked_images": {
            "train2017_created": int(train_created),
            "val2017_created": int(val_created),
            "note": "existing symlinks are reused; counts are new links created on this run.",
        },
        "converted_by": "ultralytics.data.converter.convert_coco",
        "notes": "images/* are per-file symlinks; labels/* are generated under dataset_root.",
    }
    provenance_json.write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")

    return CocoYoloDataset(dataset_root=dataset_root, dataset_yaml=dataset_yaml, provenance_json=provenance_json)
