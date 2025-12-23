# About: Using Brevitas to get W4A8 / W4A16(-like) YOLO ONNX that runs on NVIDIA GPU

## TL;DR

Yes, **Brevitas can be used for QAT** to produce YOLO-like models with **4-bit weights** and either **8-bit activations (W4A8)** or **float activations (W4A16-like)**, and export an **ONNX graph that runs on NVIDIA GPU via ONNX Runtime CUDA EP**.

However, for Conv-heavy detectors you should assume:

- The exported ONNX is typically **“fake-quantized”** (quantize/dequantize around ops; compute still happens in FP16/FP32), not “true INT4 Conv kernels”.
- “4-bit” is commonly represented as **`int8` tensors with `Clip` to the 4-bit range** (e.g., [-7, 7]) rather than actual ONNX `int4` tensor element types.
- On current stacks, **Conv INT4 acceleration is not expected**; “runs on GPU” ≠ “executes Conv in INT4”.

## What Brevitas exports (and why it can still run on GPU)

Brevitas supports exporting quantized PyTorch models to ONNX using standard ONNX ops in a style called **QCDQ**.

From the Brevitas ONNX export tutorial:

> “QCDQ allows to execute low precision fake-quantization in ONNX Runtime, meaning operations actually happen among floating-point values.”
>
> Source: https://raw.githubusercontent.com/Xilinx/brevitas/master/docs/v0.12.0/tutorials/onnx_export.html

This matters for “runnable on NVIDIA GPU” because:

- The exported graph is mostly standard ONNX ops (`QuantizeLinear`, `Clip`, `DequantizeLinear`, then `Conv`/`Relu`/etc.).
- ORT CUDA EP can execute those standard ops on GPU (even if the Conv itself runs in FP16/FP32 after dequantization).

## What “W4A8” and “W4A16” mean in practice

### W4A8 (weights 4-bit, activations 8-bit)

Brevitas can express this at the PyTorch level (QAT) by using quantized layers and activation quantizers, e.g.:

```python
import torch
import brevitas.nn as qnn

class W4A8Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = qnn.QuantConv2d(3, 16, 3, padding=1, weight_bit_width=4)
        self.act = qnn.QuantReLU(bit_width=8)

    def forward(self, x):
        return self.act(self.conv(x))
```

In ONNX export, Brevitas documents that sub-8-bit uses **`Clip` between `QuantizeLinear` and `DequantizeLinear`**:

> “The addition of the `Clip` function between the `QuantizeLinear` and `DeQuantizeLinear`, allows to quantize a tensor to bit-width < 8.”
>
> Source: https://raw.githubusercontent.com/Xilinx/brevitas/master/docs/v0.12.0/tutorials/onnx_export.html

And for “QOps” style export (QLinearConv/QLinearMatMul), it explicitly calls out that 4-bit weights end up as clipped `int8` payloads:

> “the `Clip` operation over the weights won’t be captured in the exported ONNX graph. Instead, it will be performed at export-time, and the clipped tensor will be exported in the ONNX graph.”
>
> Source: https://raw.githubusercontent.com/Xilinx/brevitas/master/docs/v0.12.0/tutorials/onnx_export.html

### W4A16 (recommended interpretation: W4 + FP16 activations)

Standard ONNX `QuantizeLinear` only outputs 8-bit integer tensors, so “A16 as int16” is not what most ONNX quant workflows mean. In practice, for detector research, **W4A16 usually means weight-only 4-bit with FP16 activations/compute**.

A “W4A16-like” Brevitas block is just weight quantization with no activation quantizer:

```python
import torch
import brevitas.nn as qnn

class W4A16LikeBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = qnn.QuantConv2d(3, 16, 3, padding=1, weight_bit_width=4)

    def forward(self, x):
        return self.conv(x)
```

If you want FP16 compute, use `.half()` and a FP16 representative input during export/inference (backend-dependent).

## Runtime support reality (Conv-heavy YOLO)

Even if the ONNX is valid and runs on CUDA EP, don’t assume you get low-bit speedups for Conv:

> “We did not observe a similar behavior for other operations such as `QuantConvNd`.”
>
> (Context: ORT sometimes optimizes QCDQ `QuantLinear` into int8 QGEMM, but not Conv in their observation.)
>
> Source: https://raw.githubusercontent.com/Xilinx/brevitas/master/docs/v0.12.0/tutorials/onnx_export.html

So for YOLO:

- **Runnable on GPU**: yes (ORT CUDA EP can run QCDQ graphs).
- **True INT4 Conv execution**: generally no (you’d need non-standard ops/kernels or a TRT plugin).

## Practical gotchas in this repo (Torch 2.9 + Brevitas 0.12.1)

In `pixi -e rtx5090`, we have `brevitas==0.12.1` (see `pyproject.toml` rtx5090 dependencies), but PyTorch in that env is very new (`torch==2.9.0+cu128`).

Brevitas’ ONNX export helper currently relies on Torch ONNX internals that moved in newer PyTorch versions:

- Brevitas code: https://github.com/Xilinx/brevitas/blob/master/src/brevitas/export/onnx/__init__.py

If you see errors like:

- `ModuleNotFoundError: No module named 'torch.onnx._globals'`
- `AttributeError: module 'torch.onnx.symbolic_helper' has no attribute '_export_onnx_opset_version'`

One workaround (for TorchScript export) is to patch the opset getter to use the new location:

```python
from torch.onnx._internal.torchscript_exporter._globals import GLOBALS

import brevitas.export.onnx as be_onnx
import brevitas.export.onnx.standard.function as be_std_fn

def onnx_export_opset():
    return GLOBALS.export_onnx_opset_version

be_onnx.onnx_export_opset = onnx_export_opset
be_std_fn.onnx_export_opset = onnx_export_opset
```

Then export with TorchScript mode:

```python
from brevitas.export import export_onnx_qcdq

export_onnx_qcdq(model, args=example_input, export_path="model.onnx", opset_version=13, dynamo=False)
```

If you prefer not to patch, the alternative is to run Brevitas export in a separate environment with a PyTorch version known to work with that Brevitas release.

## Running the exported ONNX on NVIDIA GPU (ORT CUDA EP)

```python
import onnxruntime as ort

sess = ort.InferenceSession(
    "model.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
```

## What this means for “YOLOv10m W4A8/W4A16”

- **Yes for research artifacts**: Brevitas is a viable QAT framework to produce “W4 + (A8 or FP16)” models and export an ONNX graph runnable with ORT CUDA.
- **But it is not a 1-command drop-in** for our existing YOLOv10m ONNX checkpoint: you must start from the PyTorch model, replace/augment layers with Brevitas quant layers (or insert quant proxies), and (for QAT) fine-tune.
- If your primary goal is “inspectable ONNX with real `int4` element types” (not `int8`+`Clip`), Brevitas’ standard ONNX exports are usually not that representation; you likely need a custom ONNX rewrite (see `context/design/goal-of-int4-int8-quantization-of-yolo10m.md`).
