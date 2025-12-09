# Subtask 2.2: Prepare calibration dataset for YOLO11

## Scope

Assemble a small but representative calibration dataset for YOLO11 INT8 PTQ, define where it lives on disk, and document how it is preprocessed and serialized for consumption by ModelOpt ONNX PTQ.

## Planned outputs

- A selected source dataset (COCO2017 training set via `datasets/coco2017/source-data`) and a defined subset of 100 images for calibration.
- A persistent text file listing the 100 chosen image paths at `datasets/quantize-calib/quant100.txt`.
- A preprocessing recipe that matches the YOLO11 inference pipeline (resize/letterbox, normalization, etc.).
- One or more calibration artifacts (e.g., `.npy`/`.npz` files or a directory structure) and their documented location.

## TODOs

- [x] Job-002-102-001: Use the COCO2017 training split under `datasets/coco2017/source-data` as the calibration source and randomly sample 100 images from it.
- [x] Job-002-102-002: Save the randomly selected image paths (one per line, relative or absolute) into `datasets/quantize-calib/quant100.txt` and ensure the directory exists and is ignored appropriately by Git.
- [x] Job-002-102-003: Implement or document a preprocessing script that reads `quant100.txt`, converts those images into tensors matching YOLO11’s ONNX input signature, and serializes them to `.npy`/`.npz` files suitable for ModelOpt.
- [x] Job-002-102-004: Record the calibration dataset path(s), file formats, and any required environment variables or symlinks (e.g., `datasets/coco2017/source-data`) in this subtask file or a linked context note.

## Notes

- Calibration subset size is fixed at 100 randomly chosen training images from COCO2017; randomness should be controlled (e.g., fixed seed) if exact reproducibility is important.
- ModelOpt ONNX PTQ consumes calibration **tensors**, not raw image paths: the CLI expects `--calibration_data` to point to a `.npy` (single tensor) or `.npz` (dict of tensors), and the Python API likewise takes in-memory arrays, so packing the sampled images into `.npy`/`.npz` is the most convenient and CLI-compatible format.
- Avoid checking any actual images or large `.npy`/`.npz` calibration blobs into Git; only the `quant100.txt` file and associated scripts should live in the repo.

## Implementation summary

- Sampled 100 images from the COCO2017 training split under `datasets/coco2017/source-data/train2017` using a fixed random seed and wrote their full repository-relative paths to `datasets/quantize-calib/quant100.txt`.
  - Command used:
    - `pixi run python - << 'PY' ...` (enumerates `*.jpg` under `datasets/coco2017/source-data/train2017`, samples 100 with `random.Random(42)`, writes `quant100.txt`).  
- Added a YOLO11-style preprocessing script at `scripts/yolo11/make_yolo11_calib_npy.py` that:
  - Reads an image list (e.g., `datasets/quantize-calib/quant100.txt`).
  - Applies letterbox resizing to 640×640, BGR→RGB conversion, and [0,1] scaling.
  - Produces a float32 tensor with shape `[N, 3, 640, 640]` saved as `.npy`.
- Generated a concrete calibration tensor for 100 images:
  - Command: `pixi run python scripts/yolo11/make_yolo11_calib_npy.py --list datasets/quantize-calib/quant100.txt --out datasets/quantize-calib/calib_yolo11_640.npy`
  - Output: `datasets/quantize-calib/calib_yolo11_640.npy` (shape `(100, 3, 640, 640)`).
- Updated `datasets/.gitignore` so that large calibration arrays are not committed:
  - Added patterns `quantize-calib/*.npy` and `quantize-calib/*.npz` while keeping `quant100.txt` tracked.
