# Subtask 3.2: Wire real calibration data into ONNX PTQ

## Scope

Replace placeholder calibration data in the ONNX PTQ pipeline with the real calibration dataset prepared in section 2, ensuring that ModelOpt’s calibration step runs reliably and uses the correct shapes and preprocessing.

## Planned outputs

- A ModelOpt ONNX PTQ configuration (CLI or Python) that consumes the real calibration dataset.
- Successful PTQ runs using the real calibration data for the chosen YOLO11 variants.
- Documentation of any constraints or performance considerations observed during calibration.

## TODOs

- [ ] Job-003-102-001: Update the ONNX PTQ invocation (CLI/Python) to point to the real calibration artifacts (e.g., `.npy`/`.npz` files) produced in Subtask 2.2.
- [ ] Job-003-102-002: Run PTQ for at least `yolo11n` using the real calibration dataset and confirm that calibration completes without shape or provider errors.
- [ ] Job-003-102-003: Record calibration-related settings (e.g., `calibration_eps`, batch size, number of samples) and any performance or stability observations.

## Notes

- Ensure the calibration preprocessing exactly matches the model’s expected input pipeline to avoid skewed quantization.

