# Summary

| variant | mAP_50_95 | mAP_50 | mean_ms | p90_ms | providers | onnx |
|---|---:|---:|---:|---:|---|---|
| baseline | 0.6022 | 0.7736 | 4.038 | 4.091 | CUDAExecutionProvider CPUExecutionProvider | `/workspace/code/auto-quantize-model/tmp/yolov10m_brevitas_w4a8_w4a16/2025-12-23_16-12-40/onnx/yolov10m-baseline-fp32.onnx` |
| ptq-w8a16 | 0.5983 | 0.7697 | 4.534 | 4.594 | CUDAExecutionProvider CPUExecutionProvider | `tmp/yolov10m_brevitas_w4a8_w4a16/2025-12-23_16-12-40/onnx/yolov10m-w8a16-qcdq-ptq-opt.onnx` |
| ptq-w8a8 | 0.5932 | 0.7696 | 5.279 | 5.348 | CUDAExecutionProvider CPUExecutionProvider | `tmp/yolov10m_brevitas_w4a8_w4a16/2025-12-23_16-12-40/onnx/yolov10m-w8a8-qcdq-ptq-opt.onnx` |
| ptq-w4a16 | 0.1277 | 0.2541 | 4.336 | 4.408 | CUDAExecutionProvider CPUExecutionProvider | `/workspace/code/auto-quantize-model/tmp/yolov10m_brevitas_w4a8_w4a16/2025-12-23_16-12-40/onnx/yolov10m-w4a16-qcdq-ptq-opt.onnx` |
| ptq-w4a8 | 0.1150 | 0.2265 | 5.302 | 5.370 | CUDAExecutionProvider CPUExecutionProvider | `/workspace/code/auto-quantize-model/tmp/yolov10m_brevitas_w4a8_w4a16/2025-12-23_16-12-40/onnx/yolov10m-w4a8-qcdq-ptq-opt.onnx` |
| qat-w4a8-pl | 0.2156 | 0.4150 | 5.264 | 5.336 | CUDAExecutionProvider CPUExecutionProvider | `tmp/yolov10m_brevitas_w4a8_w4a16/2025-12-23_16-12-40/onnx/yolov10m-w4a8-qcdq-qat-pl-opt.onnx` |
| qat-w4a8 | 0.2912 | 0.4949 | 5.286 | 5.352 | CUDAExecutionProvider CPUExecutionProvider | `tmp/yolov10m_brevitas_w4a8_w4a16/2025-12-23_16-12-40/onnx/yolov10m-w4a8-qcdq-qat-opt.onnx` |

## Datasets

### Evaluation

- `data_root`: `/workspace/code/auto-quantize-model/datasets/coco2017/source-data`
- `instances`: `/workspace/code/auto-quantize-model/datasets/coco2017/source-data/annotations/instances_val2017.json`
- `images_dir`: `/workspace/code/auto-quantize-model/datasets/coco2017/source-data/val2017`
- `max_images`: `100`
- `imgsz`: `640`
- `conf`: `0.001`
- `iou`: `0.7`
- `max_det`: `300`
- `pre_nms_topk`: `30000`
- `warmup_runs`: `10`
- `skip_latency`: `10`

### Calibration (A8 variants)

- `ptq-w8a8`: list=`datasets/quantize-calib/quant100.txt`, used=100, batch=4, device=`cuda:0`
- `ptq-w4a8`: list=`datasets/quantize-calib/quant100.txt`, used=100, batch=4, device=`cuda:0`
- `qat-w4a8-pl`: list=`datasets/quantize-calib/quant100.txt`, used=100, batch=4, device=`cuda:0`
- `qat-w4a8`: list=`None`, used=100, batch=4, device=`cuda:0`

### QAT (Lightning)

- `dataset_yaml`: `tmp/yolov10m_brevitas_w4a8_w4a16/2025-12-23_16-12-40/qat/coco_yolo_subset/coco_yolo_subset.yaml`
- `dataset_root`: `tmp/yolov10m_brevitas_w4a8_w4a16/2025-12-23_16-12-40/qat/coco_yolo_subset`
- `train_images`: `100` (list: `datasets/quantize-calib/quant100.txt`)
- `val_images`: `20` (val_max_images: `20`)
- TensorBoard: `tmp/yolov10m_brevitas_w4a8_w4a16/2025-12-23_16-12-40/qat/lightning/yolov10m-brevitas-w4a8/lightning`
- Loss curve: `tmp/yolov10m_brevitas_w4a8_w4a16/2025-12-23_16-12-40/qat/lightning/yolov10m-brevitas-w4a8/loss/loss_curve.png` (csv: `tmp/yolov10m_brevitas_w4a8_w4a16/2025-12-23_16-12-40/qat/lightning/yolov10m-brevitas-w4a8/loss/loss_curve.csv`)

