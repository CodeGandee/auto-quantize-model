#!/usr/bin/env python
"""Quantize Ultralytics YOLOv10m (`yolov10m.pt`) with Brevitas and export QCDQ ONNX.

This script implements the Brevitas-based PTQ/QAT workflow described in:
`context/tasks/working/quantize-yolov10m-w4a8-w4a16-brevitas/`.

All artifacts are written under a caller-provided `--run-root` (expected to be
under `tmp/`).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

import yaml  # type: ignore[import-untyped]

from auto_quantize_model.cv_models.yolov10_brevitas import (
    calibrate_activation_quantizers,
    count_qdq_nodes,
    default_repo_root,
    ensure_local_yolo10_src_on_path,
    export_brevitas_qcdq_onnx,
    export_yolov10_head_onnx,
    infer_onnx_io_contract,
    load_yolov10_detection_model,
    optimize_onnx_keep_qdq,
    quantize_model_brevitas_ptq,
    torch_load_weights_only_disabled,
    Yolov10HeadOutput,
    write_json,
)


DEFAULT_CHECKPOINT = Path("models/yolo10/checkpoints/yolov10m.pt")
DEFAULT_CALIB_LIST = Path("datasets/quantize-calib/quant100.txt")


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Brevitas PTQ/QAT export for YOLOv10m.")
    sub = parser.add_subparsers(dest="command", required=True)

    baseline = sub.add_parser("baseline", help="Export baseline ONNX via Ultralytics.")
    baseline.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    baseline.add_argument("--run-root", type=Path, required=True)
    baseline.add_argument("--imgsz", type=int, default=640)
    baseline.add_argument("--opset", type=int, default=13)
    baseline.add_argument("--prefer-fp16", action="store_true", default=True)
    baseline.add_argument("--no-prefer-fp16", dest="prefer_fp16", action="store_false")

    ptq = sub.add_parser("ptq", help="Export PTQ QCDQ ONNX via Brevitas.")
    ptq.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ptq.add_argument("--run-root", type=Path, required=True)
    ptq.add_argument("--mode", type=str, choices=["w4a16", "w4a8", "w8a16", "w8a8"], required=True)
    ptq.add_argument("--imgsz", type=int, default=640)
    ptq.add_argument("--opset", type=int, default=13)
    ptq.add_argument("--export-fp16-input", action="store_true", default=True)
    ptq.add_argument("--no-export-fp16-input", dest="export_fp16_input", action="store_false")
    ptq.add_argument("--calib-list", type=Path, default=DEFAULT_CALIB_LIST)
    ptq.add_argument("--calib-device", type=str, default="cuda:0")
    ptq.add_argument("--calib-batch-size", type=int, default=4)
    ptq.add_argument("--calib-max-images", type=int, default=None)
    ptq.add_argument("--optimize", action="store_true", default=True)
    ptq.add_argument("--no-optimize", dest="optimize", action="store_false")

    qat = sub.add_parser("qat", help="Run a short QAT fine-tune (subset COCO) and export ONNX.")
    qat.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    qat.add_argument("--run-root", type=Path, required=True)
    qat.add_argument("--mode", type=str, choices=["w4a16", "w4a8"], required=True)
    qat.add_argument("--imgsz", type=int, default=640)
    qat.add_argument("--opset", type=int, default=13)
    qat.add_argument("--calib-list", type=Path, default=DEFAULT_CALIB_LIST)
    qat.add_argument("--calib-device", type=str, default="cuda:0")
    qat.add_argument("--calib-batch-size", type=int, default=4)
    qat.add_argument("--calib-max-images", type=int, default=None)

    qat.add_argument(
        "--coco-root",
        type=Path,
        default=Path("datasets/coco2017/source-data"),
        help="COCO root with train2017/, val2017/, annotations/.",
    )
    qat.add_argument(
        "--train-list",
        type=Path,
        default=DEFAULT_CALIB_LIST,
        help="Image list for a small labeled QAT subset (train2017 images).",
    )
    qat.add_argument("--val-max-images", type=int, default=20)
    qat.add_argument("--epochs", type=int, default=1)
    qat.add_argument("--batch", type=int, default=2)
    qat.add_argument("--device", type=str, default="0")
    qat.add_argument("--seed", type=int, default=0)
    qat.add_argument("--workers", type=int, default=0)
    qat.add_argument("--amp", action="store_true", default=False)
    qat.add_argument("--val", action="store_true", default=False)
    qat.add_argument("--lr", type=float, default=1e-4)
    qat.add_argument("--weight-decay", type=float, default=5e-4)
    qat.add_argument("--log-every-n-steps", type=int, default=10)

    qat.add_argument("--optimize", action="store_true", default=True)
    qat.add_argument("--no-optimize", dest="optimize", action="store_false")

    return parser.parse_args(argv)


def json_load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def coco_select_image_filenames_from_list(list_path: Path) -> list[str]:
    filenames: list[str] = []
    with list_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            filenames.append(Path(raw).name)
    if not filenames:
        raise ValueError(f"No filenames found in {list_path}")
    return filenames


def coco_select_first_n_filenames(instances_path: Path, *, max_images: int) -> list[str]:
    data = json_load(instances_path)
    images = sorted(data.get("images", []), key=lambda x: str(x.get("file_name", "")))
    return [str(img["file_name"]) for img in images[: int(max_images)]]


def coco_write_subset_json(
    *,
    instances_path: Path,
    keep_filenames: Sequence[str],
    out_path: Path,
) -> dict[str, int]:
    data = json_load(instances_path)
    keep_set = set(str(x) for x in keep_filenames)

    images = [img for img in data.get("images", []) if str(img.get("file_name", "")) in keep_set]
    keep_ids = {int(img["id"]) for img in images}
    annotations = [ann for ann in data.get("annotations", []) if int(ann.get("image_id", -1)) in keep_ids]

    subset = dict(data)
    subset["images"] = images
    subset["annotations"] = annotations

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(subset), encoding="utf-8")
    return {"images": len(images), "annotations": len(annotations)}


def ensure_symlinks(
    *,
    src_dir: Path,
    filenames: Sequence[str],
    dst_dir: Path,
) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        src = src_dir / filename
        dst = dst_dir / filename
        if dst.exists():
            continue
        if not src.is_file():
            raise FileNotFoundError(f"Missing COCO image {src}")
        rel = os.path.relpath(src, start=dst.parent)
        dst.symlink_to(rel)


def load_coco_names(repo_root: Path) -> dict[int, str]:
    coco_yaml_path = repo_root / "models" / "yolo10" / "src" / "ultralytics" / "cfg" / "datasets" / "coco.yaml"
    data = yaml.safe_load(coco_yaml_path.read_text(encoding="utf-8"))
    names = data.get("names", {})
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {int(i): str(name) for i, name in enumerate(names)}
    raise ValueError(f"Unexpected names format in {coco_yaml_path}")


def write_dataset_yaml(dataset_root: Path, *, names: dict[int, str]) -> Path:
    yaml_path = dataset_root / "coco_yolo_subset.yaml"
    payload = {
        "path": str(dataset_root.resolve()),
        "train": "images/train2017",
        "val": "images/val2017",
        "names": names,
    }
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return yaml_path


def ensure_empty_label_files(labels_dir: Path, *, filenames: Sequence[str]) -> None:
    labels_dir.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        label_path = (labels_dir / filename).with_suffix(".txt")
        if not label_path.exists():
            label_path.write_text("", encoding="utf-8")


def run_baseline(args: argparse.Namespace, *, repo_root: Path) -> Dict[str, Any]:
    out_dir = args.run_root / "onnx"
    fp16_path = out_dir / "yolov10m-baseline-fp16.onnx"
    model = load_yolov10_detection_model(checkpoint_path=args.checkpoint, repo_root=repo_root)
    info = export_yolov10_head_onnx(
        model,
        out_path=fp16_path,
        head="one2many",
        imgsz=int(args.imgsz),
        opset=int(args.opset),
        prefer_fp16=bool(args.prefer_fp16),
        fp16_device="cuda:0",
    )

    out_path = Path(info["out_path"])
    if not bool(info.get("fp16_used", False)):
        fp32_path = out_dir / "yolov10m-baseline-fp32.onnx"
        if out_path.exists():
            out_path.rename(fp32_path)
        info["out_path"] = str(fp32_path)

    onnx_path = Path(info["out_path"])
    info["io_contract"] = infer_onnx_io_contract(onnx_path)
    write_json(args.run_root / "baseline_export.json", info)
    write_json(args.run_root / "onnx" / "baseline_io.json", info["io_contract"])
    return info


def run_ptq(args: argparse.Namespace, *, repo_root: Path) -> Dict[str, Any]:
    mode = str(args.mode)
    if mode == "w4a16":
        out_name = "yolov10m-w4a16-qcdq-ptq.onnx"
        weight_bit_width = 4
        act_bit_width = None
        fp16_input = bool(args.export_fp16_input)
        export_device = "cuda:0" if fp16_input else "cpu"
    elif mode == "w4a8":
        out_name = "yolov10m-w4a8-qcdq-ptq.onnx"
        weight_bit_width = 4
        act_bit_width = 8
        fp16_input = False
        export_device = "cpu"
    elif mode == "w8a16":
        out_name = "yolov10m-w8a16-qcdq-ptq.onnx"
        weight_bit_width = 8
        act_bit_width = None
        fp16_input = bool(args.export_fp16_input)
        export_device = "cuda:0" if fp16_input else "cpu"
    elif mode == "w8a8":
        out_name = "yolov10m-w8a8-qcdq-ptq.onnx"
        weight_bit_width = 8
        act_bit_width = 8
        fp16_input = False
        export_device = "cpu"
    else:
        raise ValueError(f"Unsupported PTQ mode: {mode}")

    out_dir = args.run_root / "onnx"
    onnx_path = out_dir / out_name
    model = load_yolov10_detection_model(checkpoint_path=args.checkpoint, repo_root=repo_root)
    model = quantize_model_brevitas_ptq(model, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width)

    calib_info: Dict[str, Any] | None = None
    if act_bit_width is not None:
        calib_info = calibrate_activation_quantizers(
            model,
            image_list_path=args.calib_list,
            repo_root=repo_root,
            imgsz=int(args.imgsz),
            batch_size=int(args.calib_batch_size),
            device=str(args.calib_device),
            max_images=args.calib_max_images,
        )

    export_model = Yolov10HeadOutput(model, head="one2many")
    export_info = export_brevitas_qcdq_onnx(
        export_model,
        out_path=onnx_path,
        imgsz=int(args.imgsz),
        opset=int(args.opset),
        fp16_input=fp16_input,
        device=export_device,
    )
    export_info["qdq_counts"] = count_qdq_nodes(onnx_path)
    export_info["io_contract"] = infer_onnx_io_contract(onnx_path)
    if calib_info is not None:
        export_info["calibration"] = calib_info

    if args.optimize:
        opt_path = out_dir / onnx_path.name.replace(".onnx", "-opt.onnx")
        opt_info = optimize_onnx_keep_qdq(onnx_path=onnx_path, out_path=opt_path)
        export_info["optimized"] = opt_info
        export_info["optimized_io_contract"] = infer_onnx_io_contract(opt_path)

    write_json(args.run_root / f"ptq_{mode}_export.json", export_info)
    return export_info


def run_qat(args: argparse.Namespace, *, repo_root: Path) -> Dict[str, Any]:
    mode = str(args.mode)
    act_bit_width = 8 if mode == "w4a8" else None
    fp16_input = False

    coco_root = args.coco_root
    instances_train = coco_root / "annotations" / "instances_train2017.json"
    instances_val = coco_root / "annotations" / "instances_val2017.json"
    train_images_dir = coco_root / "train2017"
    val_images_dir = coco_root / "val2017"

    if not instances_train.is_file() or not instances_val.is_file():
        raise FileNotFoundError("COCO instances JSON not found under --coco-root annotations/.")
    if not train_images_dir.is_dir() or not val_images_dir.is_dir():
        raise FileNotFoundError("COCO train2017/val2017 images not found under --coco-root.")

    train_filenames = coco_select_image_filenames_from_list(args.train_list)
    val_filenames = coco_select_first_n_filenames(instances_val, max_images=int(args.val_max_images))

    qat_root = args.run_root / "qat"
    annotations_dir = qat_root / "annotations_subset"
    dataset_root = qat_root / "coco_yolo_subset"

    train_subset_stats = coco_write_subset_json(
        instances_path=instances_train,
        keep_filenames=train_filenames,
        out_path=annotations_dir / "instances_train2017.json",
    )
    val_subset_stats = coco_write_subset_json(
        instances_path=instances_val,
        keep_filenames=val_filenames,
        out_path=annotations_dir / "instances_val2017.json",
    )

    ensure_local_yolo10_src_on_path(repo_root=repo_root)
    from ultralytics.data.converter import convert_coco  # type: ignore[import-not-found]

    if not dataset_root.exists():
        convert_coco(
            labels_dir=str(annotations_dir),
            save_dir=str(dataset_root),
            use_segments=False,
            use_keypoints=False,
            cls91to80=True,
        )

        ensure_symlinks(
            src_dir=train_images_dir,
            filenames=train_filenames,
            dst_dir=dataset_root / "images" / "train2017",
        )
        ensure_symlinks(
            src_dir=val_images_dir,
            filenames=val_filenames,
            dst_dir=dataset_root / "images" / "val2017",
        )
        ensure_empty_label_files(dataset_root / "labels" / "train2017", filenames=train_filenames)
        ensure_empty_label_files(dataset_root / "labels" / "val2017", filenames=val_filenames)

    else:
        # Reuse a previous conversion under the same run root.
        ensure_symlinks(
            src_dir=train_images_dir,
            filenames=train_filenames,
            dst_dir=dataset_root / "images" / "train2017",
        )
        ensure_symlinks(
            src_dir=val_images_dir,
            filenames=val_filenames,
            dst_dir=dataset_root / "images" / "val2017",
        )

    names = load_coco_names(repo_root)
    dataset_yaml = write_dataset_yaml(dataset_root, names=names)

    model = load_yolov10_detection_model(checkpoint_path=args.checkpoint, repo_root=repo_root)
    model = quantize_model_brevitas_ptq(model, weight_bit_width=4, act_bit_width=act_bit_width)
    calib_info: Dict[str, Any] | None = None
    if act_bit_width is not None:
        calib_info = calibrate_activation_quantizers(
            model,
            image_list_path=args.calib_list,
            repo_root=repo_root,
            imgsz=int(args.imgsz),
            batch_size=int(args.calib_batch_size),
            device=str(args.calib_device),
            max_images=args.calib_max_images,
        )

    from auto_quantize_model.cv_models.yolov10_ultralytics_qat import run_ultralytics_qat

    qat_project_dir = qat_root / "ultralytics"
    trained_model, qat_outputs, qat_summary = run_ultralytics_qat(
        model=model,
        checkpoint_path=args.checkpoint,
        dataset_yaml=dataset_yaml,
        out_dir=qat_project_dir,
        run_name=f"yolov10m-brevitas-{mode}",
        imgsz=int(args.imgsz),
        epochs=int(args.epochs),
        batch=int(args.batch),
        device=str(args.device),
        seed=int(args.seed),
        workers=int(args.workers),
        amp=bool(args.amp),
        val=bool(args.val),
        lr0=float(args.lr),
        weight_decay=float(args.weight_decay),
        log_every_n_steps=int(args.log_every_n_steps),
    )

    out_dir = args.run_root / "onnx"
    onnx_path = out_dir / f"yolov10m-{mode}-qcdq-qat.onnx"
    export_model = Yolov10HeadOutput(trained_model, head="one2many")
    export_info = export_brevitas_qcdq_onnx(
        export_model,
        out_path=onnx_path,
        imgsz=int(args.imgsz),
        opset=int(args.opset),
        fp16_input=fp16_input,
        device="cpu",
    )
    export_info["qdq_counts"] = count_qdq_nodes(onnx_path)
    export_info["io_contract"] = infer_onnx_io_contract(onnx_path)
    export_info["qat_dataset"] = {
        "coco_root": str(coco_root),
        "train_list_path": str(args.train_list),
        "val_max_images": int(args.val_max_images),
        "dataset_yaml": str(dataset_yaml),
        "annotations_dir": str(annotations_dir),
        "dataset_root": str(dataset_root),
        "train_images": len(train_filenames),
        "val_images": len(val_filenames),
        "train_subset": train_subset_stats,
        "val_subset": val_subset_stats,
    }
    export_info["qat_training"] = {"framework": "ultralytics", **qat_summary}
    export_info["qat_training_outputs"] = {
        "save_dir": str(qat_outputs.save_dir),
        "tensorboard_log_dir": str(qat_outputs.tensorboard_log_dir),
        "results_csv": str(qat_outputs.results_csv),
        "loss_curve_csv": str(qat_outputs.loss_curve_csv),
        "loss_curve_png": str(qat_outputs.loss_curve_png),
    }
    if calib_info is not None:
        export_info["calibration"] = calib_info

    if args.optimize:
        opt_path = out_dir / onnx_path.name.replace(".onnx", "-opt.onnx")
        opt_info = optimize_onnx_keep_qdq(onnx_path=onnx_path, out_path=opt_path)
        export_info["optimized"] = opt_info
        export_info["optimized_io_contract"] = infer_onnx_io_contract(opt_path)

    write_json(args.run_root / f"qat_{mode}_export.json", export_info)
    return export_info


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = default_repo_root()

    args.run_root.mkdir(parents=True, exist_ok=True)

    with torch_load_weights_only_disabled():
        if args.command == "baseline":
            summary = run_baseline(args, repo_root=repo_root)
        elif args.command == "ptq":
            summary = run_ptq(args, repo_root=repo_root)
        elif args.command == "qat":
            summary = run_qat(args, repo_root=repo_root)
        else:
            raise ValueError(f"Unknown command: {args.command}")

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
