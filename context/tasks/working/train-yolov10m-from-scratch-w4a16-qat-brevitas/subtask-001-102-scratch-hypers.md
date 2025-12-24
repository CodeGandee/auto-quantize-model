# Subtask 1.2: Scratch hyperparameters config

## Scope

- Port the COCO “scratch” recipe from:
  - `extern/quantized-yolov5/data/hyps/hyp.scratch.yaml`
- Write an Ultralytics-friendly hyperparameter config under:
  - `conf/cv-models/yolov10m/hyp.scratch.yaml`
- Document any keys that do not map 1:1 between YOLOv5 and YOLOv10 Ultralytics training.

## Planned outputs

- `conf/cv-models/yolov10m/hyp.scratch.yaml` with explicit:
  - optimizer settings (SGD, lr0, lrf, momentum, weight_decay, warmup_*),
  - augmentation knobs (hsv_*, degrees, translate, scale, shear, perspective, flipud, fliplr, mosaic, mixup, copy_paste).

## TODOs

- [ ] Job-001-102-001 Extract the reference values from `extern/quantized-yolov5/.../hyp.scratch.yaml`.
- [ ] Job-001-102-002 Map keys into Ultralytics trainer args (and omit unsupported keys explicitly).
- [ ] Job-001-102-003 Ensure the training script can load this YAML and pass it as overrides.

## Notes

- Keep the config minimal: only keys we actually pass to Ultralytics.

