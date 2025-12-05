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

## Dataset structure

The linked COCO 2017 directory (the target of `source-data`) is expected to
follow the standard MS COCO layout:

- `train2017/` — training images (`*.jpg`).
- `val2017/` — validation images.
- `test2017/` — test images (no public labels).
- `annotations/` — JSON annotations, typically including:
  - `instances_train2017.json`, `instances_val2017.json` for detection/segmentation.
  - `captions_train2017.json`, `captions_val2017.json` for the COCO Caption task
    (5 human-written captions per image).
  - `person_keypoints_train2017.json`, `person_keypoints_val2017.json` for keypoints.

Some distributions also keep the original `*_trainval2017.zip` archives alongside
the extracted files; these are not required once the JSONs are unpacked.
