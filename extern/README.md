# extern/: external tool checkouts

This directory holds local Git checkouts of external optimization / quantization
tooling used as references by this project and its notes under `context/hints/`.

The contents are **not** committed to this repo. They are ignored via
`extern/.gitignore` and can be safely deleted and re-cloned at any time.

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
```

