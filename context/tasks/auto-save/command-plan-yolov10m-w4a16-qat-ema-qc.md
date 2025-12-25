# Command: Plan a stable YOLOv10m W4A16 QAT pipeline (EMA + QC)

- Read `models/yolo10/reports/2025-12-25-qat-w4a16/about-qat-yolo-training-instability.md`.
- Read the WACV 2024 paper (Gupta & Asthana) on oscillations in quantized YOLO training.
- Create an implementation plan per `magic-context/instructions/planning/make-implementation-plan.md`.
- In the plan, document the maths and methodology (quantization/STE oscillations, EMA, QC, BN handling, etc.).
- If studying open-source reference implementations, clone them under `tmp/<subdir>` for inspection (do not rely on search engines).
- If using PTQ as a QAT starting point, consider NVIDIA ModelOpt (available in the Pixi environment).
- Design the code/config so we can later run QAT method and hyperparameter sweeps via Hydra configs under `conf/`.
