# Status: Open

Issue: Intel Neural Compressor GPU limitations for VLM layer sensitivity (ONNX requirement)
===========================================================================================

Context
-------

- We are using **Intel Neural Compressor (INC)** primarily as a *per-layer sensitivity engine* for large models (LLMs/VLMs), not just as a production quantizer.
- For the **Qwen2.5‑VL‑3B‑Instruct** VLM, we want:
  - Per-op / per-layer sensitivity metrics (MSE_V2, possibly HAWQ-style metrics).
  - **GPU-accelerated** runs (RTX 5090) to keep turnaround times practical.
  - A path that plays well with NVIDIA-deployment stacks (ModelOpt / TensorRT-LLM) via ONNX.
- INC’s GPU story is largely built on top of **ONNX Runtime**:
  - For NVIDIA GPUs, the supported INC backend is ONNX Runtime (CUDA), not PyTorch+CUDA directly.
  - That implies: to use INC on GPU, the model must be exported to ONNX and must run cleanly through ONNX Runtime.


Symptoms / Problem
------------------

- For **full VLM graphs** like Qwen2.5‑VL‑3B:
  - Exporting the entire model (vision + text) to ONNX with `torch.onnx.export` is currently **fragile**:
    - Even with `attn_implementation="eager"`, export fails in the attention stack with:
      - `RuntimeError: _Map_base::at` deep inside functorch/custom attention.
    - This occurs both for:
      - Full VLM wrappers (vision + text).
      - Text-only wrappers (language tower) built on `Qwen2_5_VLForConditionalGeneration`.
  - `optimum-onnx` does not yet provide a dedicated `OnnxConfig` for `qwen2_5_vl`:
    - It ultimately calls `torch.onnx.export` under the hood, so it inherits the same failures.
- For **vision-only exports**, we can produce ONNX models:
  - We successfully exported `model.visual` at fixed 672x672 resolution to:
    - `models/qwen2_5_vl_3b_instruct/onnx/qwen2_5_vl_3b_vision_672_fp32.onnx`
    - `.../qwen2_5_vl_3b_vision_672_fp32.onnx_data` (aggregated external data).
  - This required:
    - Custom patch+grid construction (`build_patches_and_grid`).
    - Enforcing that H/W are multiples of 28 (Qwen2.5‑VL vision constraint).
  - But **full model export still fails**, and the language tower ONNX export is blocked by the same trace-time error.

Net effect:

- INC can run:
  - On **CPU** via the PyTorch FX adaptor (we are already doing this for per-op MSE_V2 sensitivity).
  - On **GPU** only for models we can successfully export to ONNX and run under ONNX Runtime.
- For large VLMs with nontrivial attention/patching stacks (Qwen2.5‑VL, etc.), ONNX export is not yet robust → there is **no clean INC-on-GPU path** for full-graph layer sensitivity today.


Why this matters
----------------

- **Runtime practicality**:
  - For a 3B-parameter VLM, CPU-only INC sensitivity (MSE_V2) is already slow:
    - Even tiny runs (3 calib samples, ~4–8 ops) take many minutes.
    - Full coverage (all ops) would be hours.
  - GPU acceleration via ONNX Runtime would make per-layer sensitivity analysis more feasible, but we cannot reach that path if full ONNX export fails.

- **Tooling expectations vs reality**:
  - INC documentation emphasizes:
    - LLM/VLM support (including INT4 for Qwen-VL-style models via AutoRound).
    - Transformer-like APIs and ONNX Runtime backends.
  - In practice, these GPU-friendly flows are focused on:
    - **Weight-only** or light activation quantization for inference.
    - Models with stable ONNX export paths (LLaMA, Qwen, some VLMs/IP pipelines).
  - Heavy per-layer PTQ/sensitivity on huge VLMs is not well covered by the current stack when ONNX export is brittle.


What works vs what does not
---------------------------

**Works (today):**

- CPU-only INC sensitivity on:
  - PyTorch FX models (Qwen2.5‑VL‑3B) via:
    - `MSE_V2TuneStrategy` + `calculate_op_sensitivity` (patched to force MSE helpers to run).
  - Exported ONNX subgraphs (e.g. Qwen2.5‑VL vision encoder at fixed resolution) via ONNX Runtime (CPU or GPU).
- Full ONNX export + ONNXRuntime GPU **for Qwen2‑VL** (non-2.5), as demonstrated in:
  - `onnx-community/Qwen2-VL-2B-Instruct`:
    - They use patched `Qwen2VLForConditionalGeneration`, export embedding/text/vision ONNX models, and post-process with `optimum.onnx.graph_transformations.check_and_save_model`.

**Does not work reliably (today):**

- Full ONNX export of **Qwen2.5‑VL‑3B‑Instruct** (vision + text) with current PyTorch 2.7.x + Transformers version:
  - Fails during tracing with `RuntimeError: _Map_base::at` in attention.
- Text-only ONNX export using `Qwen2_5_VLForConditionalGeneration`:
  - Even with a simple `(input_ids, attention_mask) -> logits` wrapper, export hits the same error.
- Therefore:
  - There is no current way to run full-graph INC sensitivity on GPU for Qwen2.5‑VL‑3B via ONNXRuntime backend.


Impact on our plans
-------------------

- For the **INC op-sensitivity plan** on Qwen2.5‑VL‑3B:
  - We cannot rely on **INC+ONNXRuntime GPU** for full-model sensitivity.
  - We must treat INC as:
    - A **CPU-side** sensitivity engine for full VLM (slow, so heavily capped/limited).
    - A **GPU-side** engine only for **exportable subgraphs** (e.g., vision-only ONNX).
- For **cross-framework comparisons** (INC vs ModelOpt vs TensorRT-LLM):
  - We can still use INC sensitivity to design mixed-precision profiles, but:
    - The heavy lifting of final GPU quantization and deployment should be in GPU-native stacks (ModelOpt, Neural Speed, etc.).
    - INC plays more of a **reference/oracle** role than a production GPU optimizer for complex VLMs.


Possible directions / open questions
------------------------------------

- Track upstream improvements:
  - **Transformers / optimum-onnx**:
    - Wait for official `OnnxConfig` and export recipes for `qwen2_5_vl`.
    - See if future versions fix the `_Map_base::at` error during ONNX tracing.
  - **INC**:
    - Watch for better support and recipes for VLMs (Qwen2.5‑VL, Llava successors, etc.) in ONNXRuntime backends.
- Explore alternative strategies:
  - Use Qwen2‑VL ONNX community models as a **proxy**:
    - Run full ONNX+INC sensitivity on Qwen2‑VL.
    - Transfer the learned layer sensitivity patterns to Qwen2.5‑VL (similar architectures).
  - Use CPU-only INC sensitivity selectively:
    - Analyze only a subset of layers (e.g., specific blocks or projections) on CPU.
    - Use those insights to steer GPU quantization in other tools.
- Long term:
  - Consider a dedicated GPU-friendly sensitivity tool (ModelOpt internal metrics, custom probes) rather than forcing everything through ONNX+INC.

