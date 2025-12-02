
Quantization Scheme Summary
===========================


**Model path:** `models/yolo11/onnx/yolo11n-int8-qdq-proto.onnx`

**Total nodes:** 700
## Precision category counts

|Precision category|Node count|
| :--- | :--- |
|fp32_fp16|120|
|int8|202|

## Sample nodes

|Name|Op type|Precision|
| :--- | :--- | :--- |
|images_cast_to_fp16|Cast|int8|
|/model.0/conv/Conv|Conv|int8|
|/model.0/act/Sigmoid|Sigmoid|fp32_fp16|
|/model.0/act/Mul|Mul|int8|
|/model.1/conv/Conv|Conv|int8|
|/model.1/act/Sigmoid|Sigmoid|fp32_fp16|
|/model.1/act/Mul|Mul|int8|
|/model.2/cv1/conv/Conv|Conv|int8|
|/model.2/cv1/act/Sigmoid|Sigmoid|fp32_fp16|
|/model.2/cv1/act/Mul|Mul|fp32_fp16|
|/model.2/Split|Split|int8|
|/model.2/m.0/cv1/conv/Conv|Conv|int8|
|/model.2/m.0/cv1/act/Sigmoid|Sigmoid|fp32_fp16|
|/model.2/m.0/cv1/act/Mul|Mul|int8|
|/model.2/m.0/cv2/conv/Conv|Conv|int8|
|/model.2/m.0/cv2/act/Sigmoid|Sigmoid|fp32_fp16|
|/model.2/m.0/cv2/act/Mul|Mul|int8|
|/model.2/m.0/Add|Add|int8|
|/model.2/Concat|Concat|int8|
|/model.2/cv2/conv/Conv|Conv|int8|
