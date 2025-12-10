# Subtask 1.2: Choose INC integration path for Qwen2.5-VL-3B

## Scope

Decide how Qwen2.5-VL-3B-Instruct will be integrated with Intel Neural Compressor: either directly as a PyTorch HF model using INC’s 3.x PyTorch API, or via an exported language-model-only ONNX graph using INC’s ONNX Runtime adaptor. This includes clarifying how to isolate the language-model component (and optionally keep the vision encoder in higher precision) and how to wrap the model in INC’s `Model` abstraction. The chosen path must make it easy to treat INC primarily as a **layer sensitivity engine** (via `calculate_op_sensitivity(...)` and related helpers) rather than depending on it to produce a high-accuracy quantized model.

## Planned outputs

- A documented choice between:
  - PyTorch-INC path (directly on `Qwen2_5_VLForConditionalGeneration`), or
  - ONNX-INC path (language-model-only ONNX graph with INC ONNXRT adaptor).
- A short description of how the language-model submodule is identified and isolated from the vision encoder.
- A sketch (code snippet or pseudo-code) of how to instantiate the INC adaptor for the chosen path.
- Updated main plan or a small note referencing this decision.

## TODOs

- [x] Job-001-102-001 Review INC docs and examples for LLM and VLM quantization (PyTorch vs ONNX Runtime paths).
- [x] Job-001-102-002 Inspect Qwen2.5-VL-3B HF model structure to identify the language-model component and vision encoder modules.
- [x] Job-001-102-003 Evaluate pros/cons of PyTorch-INC vs ONNX-INC for this use case (sensitivity analysis, export format, ease of deployment).
- [x] Job-001-102-004 Decide on a single integration path and write a short summary, including how the model will be wrapped for INC (adaptor, `Model` wrapper, etc.).

## Notes

- If the long-term deployment stack is ONNX/TensorRT, favoring an ONNX-INC path may make downstream integration easier; otherwise, PyTorch-INC can be simpler for rapid iteration.
- For this Qwen2.5-VL-3B INC sensitivity and W8A8 spectrum work, we **choose the PyTorch-INC path on the native HF model** and treat ONNX export as a later deployment step:
  - This keeps sensitivity analysis (`mse_v2`, optional `hawq_v2`) close to the original `Qwen2_5_VLForConditionalGeneration` and avoids introducing ONNX export issues while we are still iterating on op-wise precision schemes.
  - INC’s PyTorch adaptor already supports generic MSE-based per-op sensitivity (`calculate_op_sensitivity`) and HAWQ_V2 flows for LLM-style models, as described in `context/summaries/inc-kb/howto-inc-layer-sensitivity-for-mixed-precision.md`, and exposes the internal helpers (`get_fallback_order`, `get_mse_order_per_fp32`, `get_mse_order_per_int8`) that we can monkeypatch and call directly to obtain per-op rankings even when `quantization.fit` never finds an “acceptable” quantized model.
- Current implementation status:
  - All Qwen2.5-VL INC scripts in this repo (`scripts/qwen/inc_qwen2_5_vl_3b_sensitivity*.py`) use the PyTorch FX adaptor on CPU, treating INC as a layer-sensitivity engine that can be invoked directly via adaptor APIs.
- **Language-model vs vision separation** in the HF model:
  - The top-level module is `Qwen2_5_VLForConditionalGeneration` with children:
    - `model` → `Qwen2_5_VLModel`
    - `lm_head` → `torch.nn.Linear`
  - Inside `model`, we have:
    - `visual` → vision encoder (`Qwen2_5_VisionTransformerPretrainedModel`)
    - `language_model` → text decoder (`Qwen2_5_VLTextModel`)
  - For sensitivity and W8A8 quantization, we treat `model.language_model` (plus `lm_head`) as the **language-model component**, and either:
    - keep `model.visual` in higher precision (bf16/fp16), or
    - quantize only selected visual layers if we later extend sensitivity to the vision branch.
- **INC PyTorch adaptor wrapping sketch (pseudo-code)** for op-sensitivity and W8A8 PTQ on the language-model part:

  ```python
  from pathlib import Path
  from neural_compressor import quantization
  from neural_compressor.config import PostTrainingQuantConfig, TuningCriterion
  from neural_compressor.utils.pytorch import load, save
  from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

  model_dir = Path("models/qwen2_5_vl_3b_instruct/checkpoints/Qwen2.5-VL-3B-Instruct")
  model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_dir, torch_dtype=torch.bfloat16)

  # Optionally freeze / keep vision encoder in higher precision.
  for param in model.model.visual.parameters():
      param.requires_grad = False  # and/or exclude from quantization config

  # INC expects either a PyTorch nn.Module or a wrapped Model object.
  # For simple PTQ + op sensitivity, passing the module directly is sufficient.
  conf = PostTrainingQuantConfig(
      backend="pytorch",
      approach="static",
      tuning_criterion=TuningCriterion(
          strategy="mse_v2",
          strategy_kwargs={"confidence_batches": 2},
      ),
      # Later subtasks will map sensitivity results into op-wise precision constraints.
  )

  # calib_dataloader and eval_func will come from Subtask 1.3.
  q_model = quantization.fit(
      model=model,
      conf=conf,
      calib_dataloader=calib_dataloader,
      eval_func=eval_func,  # optional for sensitivity-only runs
  )

  # Save a quantized checkpoint or state_dict for downstream W8A8 experiments.
  save(q_model, "./tmp/qwen2_5_vl_3b_inc_w8a8.pt")
  ```

  - In later subtasks, we will:
    - Build calibration and eval loaders around the COCO captions / VLM calib DB.
    - Call the PyTorch adaptor’s sensitivity APIs (or parse INC logs) to obtain per-op MSE scores.
    - Translate those scores into mixed-precision profiles and incorporate them into a dedicated INC quantization driver script (`scripts/qwen/inc_qwen2_5_vl_3b_sensitivity.py` / `inc_qwen2_5_vl_3b_quantize.py`).
