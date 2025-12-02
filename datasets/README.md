# Datasets

This directory contains dataset-specific integration glue for the
`auto-quantize-model` project. Each dataset lives in its own subdirectory and
exposes a small, local bootstrap script plus configuration.

Common conventions:

- Each dataset folder has a `bootstrap.sh` script.
- The script reads a local `bootstrap.yaml` (where applicable) to discover
  dataset locations and create repo-local symlinks (for example
  `source-data`).
- The bootstrap script should never download or modify the underlying dataset;
  it only links or performs light local preparation.

To see available datasets and their usage, check the README in each
subdirectory (for example `datasets/coco2017/README.md`).

