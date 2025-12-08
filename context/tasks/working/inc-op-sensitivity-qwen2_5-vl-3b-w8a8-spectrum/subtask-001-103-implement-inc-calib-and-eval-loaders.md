# Subtask 1.3: Implement INC calibration and evaluation loaders for Qwen2.5-VL-3B

## Scope

Build calibration and evaluation dataloaders compatible with the chosen INC adaptor (PyTorch or ONNX) using existing datasets (e.g., COCO2017 captions, and optionally images). The goal is to provide realistic inputs for both sensitivity analysis and PTQ runs, while keeping runtime manageable on the RTX 5090 environment.

## Planned outputs

- A calibration dataloader that yields representative inputs for:
  - Text-only prompts (using COCO captions or a small text corpus).
  - Optionally image+text inputs if vision is included in sensitivity/quantization.
- An evaluation function (`eval_func` or equivalent) that:
  - Runs a small set of prompts and returns a simple scalar metric (e.g., average log-likelihood or a proxy score).
- Wiring of these loaders into the INC config or driver script so that sensitivity and PTQ runs can use them.

## TODOs

- [x] Job-001-103-001 Reuse or extend existing COCO2017 calibration assets (e.g., `datasets/vlm-quantize-calib/coco2017_captions.txt`) to build a text-focused calibration loader.
- [x] Job-001-103-002 Define input shapes and tokenization flow for calibration/eval consistent with the chosen INC integration path (PyTorch tensors or ONNX inputs).
- [x] Job-001-103-003 Implement a lightweight evaluation function that runs a small set of prompts and returns a scalar accuracy/quality proxy.
- [x] Job-001-103-004 Integrate the calibration and evaluation components into the planned INC driver script (e.g., `inc_qwen2_5_vl_3b_sensitivity.py`), and test them on a few batches.

## Notes

- Keep calibration and evaluation sample counts small enough to avoid long runtimes (e.g., a few hundred to a few thousand samples, depending on sequence length).
- Implemented CPU+PyTorch calibration/eval utilities in `src/auto_quantize_model/qwen2_5_vl_inc_data.py`:
  - `QwenCalibConfig` configures text-only calibration using `datasets/vlm-quantize-calib/coco2017_captions.txt` (one caption per line), with small default sample counts and batch sizes.
  - `build_qwen_calib_dataloader(...)` builds a `torch.utils.data.DataLoader` that:
    - Loads captions from `coco2017_captions.txt`, truncates to `max_samples`, and wraps them in a small `Dataset`.
    - Uses the Qwen2.5-VL tokenizer’s `apply_chat_template(..., add_generation_prompt=True)` so calibration traffic matches normal chat-style prompts.
    - Produces batched `input_ids` / `attention_mask` tensors (padded on the right with the tokenizer `pad_token_id` and 0 masks), plus the original `prompt_text` strings for debugging.
  - `build_qwen_eval_dataloader(...)` reuses the same pipeline with a smaller, deterministic prefix of captions for evaluation (no shuffle).
  - `make_qwen_eval_func(...)` returns an `eval_func(model) -> float` suitable for INC:
    - Runs the given model in `eval()` mode on CPU.
    - Computes standard causal LM loss by passing `labels=input_ids` and uses `attention_mask` to weight token counts.
    - Returns the **negative** average token-level loss as a scalar proxy (higher is better for INC’s objective).
- Wired these loaders into the PyTorch+INC sensitivity driver `scripts/qwen/inc_qwen2_5_vl_3b_sensitivity.py`:
  - The script targets **CPU-only PTQ** with `PostTrainingQuantConfig(backend="default", device="cpu", approach="static", tuning_criterion=TuningCriterion(strategy="mse_v2", ...))`, matching common INC usage for PyTorch.
  - It builds a calibration dataloader and a small eval dataloader from the captions file, prints their sizes, and instantiates `eval_func` via `make_qwen_eval_func`.
  - The HF `Qwen2_5_VLForConditionalGeneration` model is loaded in `torch.float32` and moved explicitly to CPU before calling `quantization.fit(...)`.
- Sanity checks:
  - Independent CPU-only tests confirm that `build_qwen_calib_dataloader` and `build_qwen_eval_dataloader` produce well-shaped batches for Qwen2.5-VL (e.g., `input_ids` `[B, L]`, `attention_mask` `[B, L]`), and that `make_qwen_eval_func` returns a finite scalar (negative average loss) on a few batches.
  - The INC driver script successfully constructs and uses the calibration/eval loaders for a short PTQ run; if no quantized model meets the accuracy criterion, it logs that `q_model` is `None` and exits cleanly without treating this as a loader failure.
