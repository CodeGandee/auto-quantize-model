"""Deterministic COCO2017 subset dataset builder for YOLOv10 validation runs."""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from auto_quantize_model.cv_models.yolov10_coco_dataset import load_coco_names, write_dataset_yaml


@dataclass(frozen=True)
class CocoSubsetYoloDataset:
    dataset_root: Path
    dataset_yaml: Path
    provenance_json: Path
    train_images: int
    val_images: int


def _read_coco_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to read COCO JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"COCO JSON must be an object: {path}")
    return payload


def _index_images(coco: dict[str, Any]) -> dict[int, dict[str, Any]]:
    images = coco.get("images")
    if not isinstance(images, list):
        raise TypeError("COCO JSON missing 'images' list.")
    indexed: dict[int, dict[str, Any]] = {}
    for item in images:
        if not isinstance(item, dict):
            continue
        image_id = item.get("id")
        if isinstance(image_id, int):
            indexed[int(image_id)] = item
    return indexed


def _select_train_images(
    *,
    images: Iterable[dict[str, Any]],
    train_list: Path | None,
    train_max_images: int | None,
    seed: int,
    selection: str,
) -> list[dict[str, Any]]:
    images_sorted = sorted(
        [img for img in images if isinstance(img, dict) and isinstance(img.get("id"), int)],
        key=lambda x: int(x["id"]),
    )

    if train_list is not None:
        allowed: set[str] = set()
        for line in Path(train_list).read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            allowed.add(Path(stripped).name)

        selected = [img for img in images_sorted if str(img.get("file_name", "")) in allowed]
        if train_max_images is not None:
            return selected[: int(train_max_images)]
        return selected

    if train_max_images is None:
        return images_sorted

    max_images = int(train_max_images)
    if max_images <= 0:
        return []

    if selection == "first_n":
        return images_sorted[:max_images]
    if selection != "random":
        raise ValueError(f"Unsupported selection strategy: {selection!r}")

    rng = random.Random(int(seed))
    if max_images >= len(images_sorted):
        return images_sorted
    return sorted(rng.sample(images_sorted, k=max_images), key=lambda x: int(x["id"]))


def _select_val_images(*, images: Iterable[dict[str, Any]], val_max_images: int | None) -> list[dict[str, Any]]:
    images_sorted = sorted(
        [img for img in images if isinstance(img, dict) and isinstance(img.get("id"), int)],
        key=lambda x: int(x["id"]),
    )
    if val_max_images is None:
        return images_sorted
    max_images = int(val_max_images)
    if max_images <= 0:
        return []
    return images_sorted[:max_images]


def _filter_annotations(
    *,
    coco: dict[str, Any],
    selected_image_ids: set[int],
) -> dict[str, Any]:
    annotations = coco.get("annotations")
    if not isinstance(annotations, list):
        raise TypeError("COCO JSON missing 'annotations' list.")
    filtered_annotations: list[dict[str, Any]] = []
    for ann in annotations:
        if not isinstance(ann, dict):
            continue
        image_id = ann.get("image_id")
        if isinstance(image_id, int) and int(image_id) in selected_image_ids:
            filtered_annotations.append(ann)

    images_by_id = _index_images(coco)
    filtered_images = [images_by_id[i] for i in sorted(selected_image_ids) if i in images_by_id]

    payload: dict[str, Any] = {}
    for key in ("info", "licenses", "categories"):
        if key in coco:
            payload[key] = coco[key]
    payload["images"] = filtered_images
    payload["annotations"] = filtered_annotations
    return payload


def _symlink_selected_images(*, src_dir: Path, dst_dir: Path, file_names: list[str]) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    created = 0
    for name in file_names:
        src_path = src_dir / name
        if not src_path.is_file():
            raise FileNotFoundError(f"Missing COCO image file: {src_path}")
        dst_path = dst_dir / name
        if dst_path.exists() or dst_path.is_symlink():
            continue
        rel = os.path.relpath(src_path, start=dst_path.parent)
        dst_path.symlink_to(rel)
        created += 1
    return created


def _coco91_to_coco80_class() -> list[int | None]:
    # Copied from Ultralytics (avoids importing cv2-heavy ultralytics modules during tests).
    return [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        None,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        None,
        24,
        25,
        None,
        None,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        None,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        None,
        60,
        None,
        None,
        61,
        None,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        72,
        None,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        None,
    ]


def _write_yolo_labels(
    *,
    coco: dict[str, Any],
    labels_dir: Path,
    cls91to80: bool = True,
) -> None:
    images_by_id = _index_images(coco)

    annotations = coco.get("annotations")
    if not isinstance(annotations, list):
        raise TypeError("COCO JSON missing 'annotations' list.")

    img_to_anns: dict[int, list[dict[str, Any]]] = {}
    for ann in annotations:
        if not isinstance(ann, dict):
            continue
        image_id = ann.get("image_id")
        if isinstance(image_id, int):
            img_to_anns.setdefault(int(image_id), []).append(ann)

    coco80 = _coco91_to_coco80_class()

    labels_dir.mkdir(parents=True, exist_ok=True)
    for image_id, image in images_by_id.items():
        file_name = str(image.get("file_name") or "")
        stem = Path(file_name).stem if file_name else str(image_id)
        label_path = labels_dir / f"{stem}.txt"

        width = float(image.get("width") or 0)
        height = float(image.get("height") or 0)
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image dimensions for image_id={image_id}: width={width}, height={height}")

        lines: list[str] = []
        for ann in img_to_anns.get(int(image_id), []):
            if ann.get("iscrowd"):
                continue
            bbox = ann.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            x, y, w, h = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
            if w <= 0 or h <= 0:
                continue

            category_id = ann.get("category_id")
            if not isinstance(category_id, int):
                continue
            if cls91to80:
                idx = category_id - 1
                if idx < 0 or idx >= len(coco80):
                    continue
                mapped = coco80[idx]
                if mapped is None:
                    continue
                cls = int(mapped)
            else:
                cls = int(category_id - 1)

            xc = (x + w / 2.0) / width
            yc = (y + h / 2.0) / height
            wn = w / width
            hn = h / height

            lines.append(f"{cls} {xc:.8f} {yc:.8f} {wn:.8f} {hn:.8f}")

        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def prepare_coco2017_yolo_subset_dataset(
    *,
    repo_root: Path,
    coco_root: Path,
    out_dir: Path,
    train_list: Path | None,
    train_max_images: int | None,
    val_max_images: int | None,
    seed: int,
    selection: str,
) -> CocoSubsetYoloDataset:
    """Build a deterministic YOLO-format COCO subset dataset for fast QAT validation."""

    coco_root = Path(coco_root).resolve()
    annotations_dir = coco_root / "annotations"
    instances_train = annotations_dir / "instances_train2017.json"
    instances_val = annotations_dir / "instances_val2017.json"
    if not instances_train.is_file() or not instances_val.is_file():
        raise FileNotFoundError("Missing COCO instances_{train,val}2017.json under annotations/.")

    train_images_dir = coco_root / "train2017"
    val_images_dir = coco_root / "val2017"
    if not train_images_dir.is_dir() or not val_images_dir.is_dir():
        raise FileNotFoundError(f"Missing COCO train2017/val2017 under: {coco_root}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    annotations_subset_dir = out_dir / "annotations_instances"
    annotations_subset_dir.mkdir(parents=True, exist_ok=True)

    coco_train = _read_coco_json(instances_train)
    coco_val = _read_coco_json(instances_val)

    train_images_raw = coco_train.get("images")
    train_images: list[dict[str, Any]] = (
        [img for img in train_images_raw if isinstance(img, dict)] if isinstance(train_images_raw, list) else []
    )

    val_images_raw = coco_val.get("images")
    val_images: list[dict[str, Any]] = (
        [img for img in val_images_raw if isinstance(img, dict)] if isinstance(val_images_raw, list) else []
    )

    selected_train = _select_train_images(
        images=train_images,
        train_list=train_list,
        train_max_images=train_max_images,
        seed=seed,
        selection=selection,
    )
    selected_val = _select_val_images(images=val_images, val_max_images=val_max_images)

    selected_train_ids = {int(img["id"]) for img in selected_train if isinstance(img, dict) and isinstance(img.get("id"), int)}
    selected_val_ids = {int(img["id"]) for img in selected_val if isinstance(img, dict) and isinstance(img.get("id"), int)}

    subset_train_json = annotations_subset_dir / "instances_train2017.json"
    subset_val_json = annotations_subset_dir / "instances_val2017.json"

    subset_train_payload = _filter_annotations(coco=coco_train, selected_image_ids=selected_train_ids)
    subset_val_payload = _filter_annotations(coco=coco_val, selected_image_ids=selected_val_ids)

    subset_train_json.write_text(json.dumps(subset_train_payload, indent=2, sort_keys=True), encoding="utf-8")
    subset_val_json.write_text(json.dumps(subset_val_payload, indent=2, sort_keys=True), encoding="utf-8")

    dataset_root = out_dir / "coco_yolo_subset"
    dataset_root.mkdir(parents=True, exist_ok=True)

    labels_train_dir = dataset_root / "labels" / "train2017"
    labels_val_dir = dataset_root / "labels" / "val2017"
    _write_yolo_labels(coco=subset_train_payload, labels_dir=labels_train_dir, cls91to80=True)
    _write_yolo_labels(coco=subset_val_payload, labels_dir=labels_val_dir, cls91to80=True)

    train_dst = dataset_root / "images" / "train2017"
    val_dst = dataset_root / "images" / "val2017"

    train_file_names = [str(img.get("file_name")) for img in selected_train if str(img.get("file_name", "")).strip()]
    val_file_names = [str(img.get("file_name")) for img in selected_val if str(img.get("file_name", "")).strip()]

    train_created = _symlink_selected_images(src_dir=train_images_dir, dst_dir=train_dst, file_names=train_file_names)
    val_created = _symlink_selected_images(src_dir=val_images_dir, dst_dir=val_dst, file_names=val_file_names)

    names = load_coco_names(repo_root=repo_root)
    dataset_yaml = write_dataset_yaml(dataset_root=dataset_root, names=names)

    provenance_json = out_dir / "provenance.json"
    provenance: dict[str, Any] = {
        "coco_root": str(coco_root),
        "dataset_root": str(dataset_root),
        "dataset_yaml": str(dataset_yaml),
        "subset_annotations_dir": str(annotations_subset_dir),
        "instances_train": str(instances_train),
        "instances_val": str(instances_val),
        "subset_instances_train": str(subset_train_json),
        "subset_instances_val": str(subset_val_json),
        "train_selection": {
            "strategy": selection if train_list is None else "file_list",
            "seed": int(seed),
            "train_list": str(train_list) if train_list is not None else None,
            "train_max_images": int(train_max_images) if train_max_images is not None else None,
            "selected_image_ids": sorted(list(selected_train_ids)),
        },
        "val_selection": {
            "strategy": "first_n",
            "val_max_images": int(val_max_images) if val_max_images is not None else None,
            "selected_image_ids": sorted(list(selected_val_ids)),
        },
        "symlinked_images": {
            "train2017_created": int(train_created),
            "val2017_created": int(val_created),
            "note": "existing symlinks are reused; counts are new links created on this run.",
        },
        "converted_by": "auto_quantize_model.cv_models.yolov10_coco_subset_dataset._write_yolo_labels",
        "notes": "images/* are per-file symlinks; labels/* are generated under dataset_root.",
    }
    provenance_json.write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")

    return CocoSubsetYoloDataset(
        dataset_root=dataset_root,
        dataset_yaml=dataset_yaml,
        provenance_json=provenance_json,
        train_images=int(len(train_file_names)),
        val_images=int(len(val_file_names)),
    )
