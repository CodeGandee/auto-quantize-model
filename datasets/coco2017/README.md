# COCO 2017 Dataset Integration

This directory wires an existing local COCO 2017 dataset into the repository.
The data itself is expected to live elsewhere on disk.

## Layout

- `bootstrap.yaml` — configuration for discovering the dataset location.
- `bootstrap.sh` — helper script that creates a symlink named
  `source-data` pointing at your local COCO 2017 directory.

Once bootstrapped, all paths in this repository assume the COCO 2017 root
is available at `datasets/coco2017/source-data`.

## Bootstrapping

By default, the bootstrap script looks for the dataset in:

- Environment variable `DATASET_ROOT_COCO2017`, if set.
- Otherwise `/data2/dataset/coco2017`.

If a candidate directory is found, you will be asked to confirm it; you can
also provide a path explicitly or override it interactively.

Examples:

- Use discovered/default location (with confirmation):
  - `datasets/coco2017/bootstrap.sh`
- Provide an explicit dataset path:
  - `datasets/coco2017/bootstrap.sh --path /absolute/path/to/coco2017`
- Clean the repo-local link:
  - `datasets/coco2017/bootstrap.sh --clean`

## Dataset Structure

The linked COCO 2017 directory (the target of `source-data`) is expected to
follow the **Standard MS COCO Layout**. This is the format used by MMDetection, Detectron2, and the evaluation scripts in this repository.

### 1. Standard Layout (Recommended)

This is the structure expected by `scripts/yolo11/eval_yolo11_onnx_coco.py` and other tools in this repo.

```text
coco2017/
├── annotations/
│   ├── instances_train2017.json      # Object detection & segmentation
│   ├── instances_val2017.json        # (Required for evaluation)
│   ├── captions_train2017.json       # Image captioning
│   ├── captions_val2017.json
│   ├── person_keypoints_train2017.json
│   └── person_keypoints_val2017.json
├── train2017/                        # 118k training images
│   ├── 000000000009.jpg
│   └── ...
├── val2017/                          # 5k validation images (Required for evaluation)
│   ├── 000000000139.jpg
│   └── ...
└── test2017/                         # 41k test images (Optional)
```

### 2. YOLO / Ultralytics Layout (Alternative)

Some YOLO training frameworks (like Ultralytics) prefer a structure where images and labels are separated. While this repository's evaluation scripts primarily use the standard JSON annotations, you might encounter this structure if you have used YOLO training tools previously.

```text
coco2017/
├── images/
│   ├── train2017/
│   └── val2017/
├── labels/                           # TXT files generated from JSONs
│   ├── train2017/
│   └── val2017/
├── annotations/                      # Original JSONs are often kept here too
│   └── instances_val2017.json
├── train2017.txt                     # List of image paths
└── val2017.txt
```

**Note:** If your dataset is in this format, ensure that `source-data` points to the root `coco2017/` folder. Our scripts will look for `annotations/instances_val2017.json`. If your images are nested under `images/val2017` instead of just `val2017`, you may need to adjust paths in configuration files or create symlinks (e.g., `ln -s images/val2017 val2017`) to satisfy the standard layout expectation.

Some distributions also keep the original `*_trainval2017.zip` archives alongside
the extracted files; these are not required once the JSONs are unpacked.
