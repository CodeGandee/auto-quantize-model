# extern/: external tool checkouts

This directory holds local Git checkouts of external optimization / quantization
tooling used as references by this project and its notes under `context/hints/`.

The contents are **not** committed to this repo. They are ignored via
`extern/.gitignore` and can be safely deleted and re-cloned at any time.

## VS Code file watching

This directory can contain a very large number of files (vendored repositories).
On Linux, VS Code may hit inotify watch limits and show warnings like “unable to
watch for file changes”.

Recommended: exclude `extern/` from VS Code’s file watcher (and typically also
other large vendor/build directories). Add this to your workspace settings
(`.vscode/settings.json` at the repo root):

```json
{
  "files.watcherExclude": {
    "**/extern/**": true
  }
}
```

## Layout

- `TensorRT-Model-Optimizer/`
  - Shallow clone of NVIDIA’s TensorRT Model Optimizer (ModelOpt) repo.
  - Upstream: <https://github.com/NVIDIA/TensorRT-Model-Optimizer>
  - Used for inspecting ModelOpt PTQ/QAT, AutoQuantize, AWQ, FP8 / NVFP4, etc.

- `neural-compressor/`
  - Shallow clone of Intel Neural Compressor.
  - Upstream: <https://github.com/intel/neural-compressor>
  - Used for reference on Intel’s low-bit LLM quantization, GPTQ/AWQ/SmoothQuant,
    and mixed-precision flows.

- `nncf/`
  - Shallow clone of OpenVINO’s Neural Network Compression Framework (NNCF).
  - Upstream: <https://github.com/openvinotoolkit/nncf>
  - Used to study NNCF-based quantization / sparsity algorithms referenced in
    OpenVINO flows.

- `openvino/`
  - Shallow clone of the OpenVINO toolkit.
  - Upstream: <https://github.com/openvinotoolkit/openvino>
  - Used for cross-referencing runtime / graph passes that interact with NNCF
    or quantized models.

- `onnxruntime/`
  - Shallow clone of Microsoft ONNX Runtime.
  - Upstream: <https://github.com/microsoft/onnxruntime>
  - Used to build a custom ONNX Runtime with CUDA Execution Provider (CUDA
    toolkit 12.8.1) for running ONNX models on GPU, and to inspect/export
    configuration for CUDA EP builds.

- `finn-quantized-yolo/`
  - Shallow clone of an example FINN-based quantized YOLO workflow.
  - Upstream: <https://github.com/sefaburakokcu/finn-quantized-yolo>
  - Used as a reference for FINN / FPGA-oriented quantization pipelines.

- `quantized-yolov5/`
  - Shallow clone of the LPYOLO/FINN companion training repo (Brevitas QAT on YOLOv5).
  - Upstream: <https://github.com/sefaburakokcu/quantized-yolov5>
  - Used to study their Brevitas-based QAT + FINN-oriented export choices.

- `vllm/`
  - Shallow clone of the vLLM inference engine.
  - Upstream: <https://github.com/vllm-project/vllm>
  - Used both as:
    - A reference for vLLM’s quantization / ModelOpt integration, and
    - A build source for wheels or editable installs (see `extern/build-vllm.sh`).

- `build-vllm.sh`
  - Helper script to build vLLM from `extern/vllm` as a wheel or editable
    install inside your current Python environment.

## (Re)creating the checkouts

If you ever remove the directories under `extern/`, you can recreate them with
shallow clones (from the repo root):

```bash
cd extern
git clone --depth=1 https://github.com/NVIDIA/TensorRT-Model-Optimizer.git TensorRT-Model-Optimizer
git clone --depth=1 https://github.com/intel/neural-compressor.git neural-compressor
git clone --depth=1 https://github.com/openvinotoolkit/nncf.git nncf
git clone --depth=1 https://github.com/openvinotoolkit/openvino.git openvino
git clone --depth=1 https://github.com/vllm-project/vllm.git vllm
git clone --depth=1 https://github.com/microsoft/onnxruntime.git onnxruntime
git clone --depth=1 https://github.com/sefaburakokcu/finn-quantized-yolo.git finn-quantized-yolo
git clone --depth=1 https://github.com/sefaburakokcu/quantized-yolov5.git quantized-yolov5
```

For ONNX Runtime, see the upstream docs for detailed CUDA build instructions:

- Build with CUDA EP: <https://onnxruntime.ai/docs/build/eps.html#cuda>
- High-level Linux example (CUDA toolkit 12.8.1):

  ```bash
  cd extern/onnxruntime

  # Ensure CUDA 12.8.1 and cuDNN 9.x are installed and discoverable:
  #   export CUDA_HOME=/usr/local/cuda-12.8.1
  #   export CUDNN_HOME=/usr/local/cuda-12.8.1
  #   export PATH=\"$CUDA_HOME/bin:$PATH\"

  ./build.sh \\
    --config Release \\
    --build_dir build/Linux/Release-cuda1281 \\
    --update --build --parallel \\
    --build_wheel \\
    --use_cuda \\
    --cuda_home \"$CUDA_HOME\" \\
    --cudnn_home \"$CUDNN_HOME\" \\
    --cuda_version 12.8 \\
    --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=\"80;86;89\" \
    --skip_tests

  # Resulting wheel will be under build/Linux/Release-cuda1281/dist/
  ```

