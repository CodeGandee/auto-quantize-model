# COCO 2017 Dataset Integration

This directory wires an existing local COCO 2017 dataset into the repository.
The data itself is expected to live elsewhere on disk.

## Layout

- `bootstrap.yaml` — configuration for discovering the dataset location.
- `bootstrap.sh` — helper script that creates a symlink named
  `source-data` pointing at your local COCO 2017 directory.

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

