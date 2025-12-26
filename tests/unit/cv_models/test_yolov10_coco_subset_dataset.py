from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from auto_quantize_model.cv_models.yolov10_coco_subset_dataset import prepare_coco2017_yolo_subset_dataset


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _write_fixture_json(*, fixture: Path, out_path: Path) -> dict[str, Any]:
    payload = json.loads(fixture.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Fixture is not a JSON object: {fixture}")
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def test_prepare_coco_subset_dataset_is_deterministic(tmp_path: Path) -> None:
    fixtures_dir = Path(__file__).parent / "fixtures"
    train_fixture = fixtures_dir / "coco_instances_train.json"
    val_fixture = fixtures_dir / "coco_instances_val.json"

    coco_root = tmp_path / "coco"
    (coco_root / "annotations").mkdir(parents=True, exist_ok=True)
    (coco_root / "train2017").mkdir(parents=True, exist_ok=True)
    (coco_root / "val2017").mkdir(parents=True, exist_ok=True)

    train_payload = _write_fixture_json(fixture=train_fixture, out_path=coco_root / "annotations" / "instances_train2017.json")
    val_payload = _write_fixture_json(fixture=val_fixture, out_path=coco_root / "annotations" / "instances_val2017.json")

    train_images = train_payload.get("images")
    if isinstance(train_images, list):
        for image in train_images:
            if not isinstance(image, dict):
                continue
            name = str(image.get("file_name"))
            (coco_root / "train2017" / name).write_bytes(b"")

    val_images = val_payload.get("images")
    if isinstance(val_images, list):
        for image in val_images:
            if not isinstance(image, dict):
                continue
            name = str(image.get("file_name"))
            (coco_root / "val2017" / name).write_bytes(b"")

    out1 = prepare_coco2017_yolo_subset_dataset(
        repo_root=_repo_root(),
        coco_root=coco_root,
        out_dir=tmp_path / "out1",
        train_list=None,
        train_max_images=2,
        val_max_images=1,
        seed=123,
        selection="random",
    )
    assert out1.dataset_yaml.is_file()
    assert out1.provenance_json.is_file()
    assert out1.train_images == 2
    assert out1.val_images == 1

    prov1 = json.loads(out1.provenance_json.read_text(encoding="utf-8"))
    assert prov1["val_selection"]["selected_image_ids"] == [101]
    train_selected_1 = prov1["train_selection"]["selected_image_ids"]
    assert isinstance(train_selected_1, list)
    assert len(train_selected_1) == 2

    out2 = prepare_coco2017_yolo_subset_dataset(
        repo_root=_repo_root(),
        coco_root=coco_root,
        out_dir=tmp_path / "out2",
        train_list=None,
        train_max_images=2,
        val_max_images=1,
        seed=123,
        selection="random",
    )
    prov2 = json.loads(out2.provenance_json.read_text(encoding="utf-8"))
    assert prov2["train_selection"]["selected_image_ids"] == train_selected_1
