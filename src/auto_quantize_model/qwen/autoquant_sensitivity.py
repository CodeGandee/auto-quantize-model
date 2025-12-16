"""
Qwen multimodal (VLM) AutoQuant sensitivity helpers.

This module hosts reusable building blocks for Qwen LM-only AutoQuant flows,
including captions calibration datasets, loss functions, and a small wrapper
to run ModelOpt AutoQuant on the extracted language model component.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

import modelopt.torch.quantization as mtq  # type: ignore[import-untyped]

from auto_quantize_model.modelopt_autoquant import (
    AutoQuantSchemeConfig,
    build_quant_manifest,
    compute_num_score_steps,
    resolve_quantization_formats,
)


def extract_language_model_from_vl(full_model: torch.nn.Module) -> torch.nn.Module:
    """Best-effort extraction of the language model submodule from a VLM.

    We avoid relying on ModelOpt helper utilities here because different
    ModelOpt builds expose different export utilities across environments.
    """

    named_modules = dict(full_model.named_modules())
    for key in (
        "language_model",
        "model.language_model",
        "model.model.language_model",
    ):
        module = named_modules.get(key)
        if isinstance(module, torch.nn.Module):
            return module

    for path in (
        ("model", "language_model"),
        ("language_model",),
        ("text_model",),
        ("model", "text_model"),
    ):
        cursor: Any = full_model
        found = True
        for attr in path:
            if not hasattr(cursor, attr):
                found = False
                break
            cursor = getattr(cursor, attr)
        if found and isinstance(cursor, torch.nn.Module):
            return cursor

    raise RuntimeError(
        "Could not locate the language model submodule in the provided VLM; "
        "expected an attribute like `model.language_model` or `language_model`."
    )


class CocoCaptionsDataset(Dataset[Mapping[str, torch.Tensor]]):
    """Text-only calibration dataset built from COCO captions (one caption per line)."""

    def __init__(
        self,
        captions_path: Path,
        tokenizer: Any,
        max_samples: Optional[int] = None,
        max_length: int = 512,
    ) -> None:
        if not captions_path.is_file():
            raise FileNotFoundError(f"Calibration captions file not found: {captions_path}")

        self._tokenizer = tokenizer
        self._max_length = int(max_length)

        lines: List[str] = []
        with captions_path.open("r", encoding="utf-8") as file:
            for line in file:
                text = line.strip()
                if not text:
                    continue
                lines.append(text)
                if max_samples is not None and len(lines) >= max_samples:
                    break

        if not lines:
            raise RuntimeError(f"No non-empty captions found in {captions_path}")

        self._captions = lines

    def __len__(self) -> int:
        return len(self._captions)

    def __getitem__(self, index: int) -> Mapping[str, torch.Tensor]:
        text = self._captions[index]
        tokenized = self._tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self._max_length,
        )
        features: Dict[str, torch.Tensor] = {key: value.squeeze(0) for key, value in tokenized.items()}
        input_ids = features.get("input_ids")
        if input_ids is not None:
            features["labels"] = input_ids.clone()
        return features


def build_lm_calib_dataloader(
    captions_path: Path,
    tokenizer: Any,
    batch_size: int,
    max_samples: Optional[int],
    max_length: int,
) -> DataLoader[Mapping[str, torch.Tensor]]:
    """Build a deterministic text-only calibration DataLoader for LM AutoQuant."""

    dataset = CocoCaptionsDataset(
        captions_path=captions_path,
        tokenizer=tokenizer,
        max_samples=max_samples,
        max_length=max_length,
    )
    return DataLoader(dataset, batch_size=max(batch_size, 1), shuffle=False)


def create_lm_loss_func(
    device: torch.device,
    lm_head: torch.nn.Module,
    pad_token_id: Optional[int],
) -> Callable[[Any, Mapping[str, torch.Tensor]], torch.Tensor]:
    """Loss function for Qwen LM-only AutoQuant using `lm_head(last_hidden_state)`."""

    ignore_index = -100 if pad_token_id is None else pad_token_id
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    def _loss_func(output: Any, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        labels = batch["labels"].to(device)
        if not hasattr(output, "last_hidden_state"):
            raise ValueError("Expected output with last_hidden_state for loss computation.")

        hidden_states = output.last_hidden_state
        logits = lm_head(hidden_states)

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        return loss

    return _loss_func


def autoquant_lm(
    lm_model: torch.nn.Module,
    scheme: AutoQuantSchemeConfig,
    calib_loader: Iterable[Mapping[str, torch.Tensor]],
    batch_size: int,
    device: torch.device,
    lm_head: torch.nn.Module,
    pad_token_id: Optional[int],
    *,
    verbose: bool = True,
) -> Tuple[torch.nn.Module, Mapping[str, Any]]:
    """Run ModelOpt AutoQuant on a language model submodule."""

    quantization_formats = resolve_quantization_formats(scheme.quant_formats)

    calib_batches: List[Mapping[str, torch.Tensor]] = list(calib_loader)
    if not calib_batches:
        raise RuntimeError("Calibration dataloader is empty.")

    def _forward_step_gradient(model: torch.nn.Module, batch: Mapping[str, torch.Tensor]) -> Any:
        batch_on_device = {key: value.to(device) for key, value in batch.items()}
        return model(**batch_on_device)

    def _forward_step_kl_div(model: torch.nn.Module, batch: Mapping[str, torch.Tensor]) -> Any:
        batch_on_device = {key: value.to(device) for key, value in batch.items()}
        output = model(**batch_on_device)
        if not hasattr(output, "last_hidden_state"):
            raise ValueError("Expected output with last_hidden_state for kl_div scoring.")
        return lm_head(output.last_hidden_state)

    if scheme.auto_quantize_method == "gradient":
        forward_step = _forward_step_gradient
        loss_fn: Optional[Callable[[Any, Mapping[str, torch.Tensor]], torch.Tensor]] = create_lm_loss_func(
            device=device,
            lm_head=lm_head,
            pad_token_id=pad_token_id,
        )
    elif scheme.auto_quantize_method == "kl_div":
        forward_step = _forward_step_kl_div
        loss_fn = None
    else:
        raise ValueError(
            f"Unsupported auto_quantize_method: {scheme.auto_quantize_method}. "
            "Expected 'gradient' or 'kl_div'."
        )
    num_score_steps = compute_num_score_steps(
        scheme.auto_quantize_score_size,
        batch_size=max(batch_size, 1),
        num_batches=len(calib_batches),
    )

    quantized_model, state_dict = mtq.auto_quantize(
        lm_model,
        constraints={"effective_bits": scheme.auto_quantize_bits},
        quantization_formats=quantization_formats,
        data_loader=calib_batches,
        forward_step=forward_step,
        loss_func=loss_fn,
        num_calib_steps=len(calib_batches),
        num_score_steps=num_score_steps,
        verbose=bool(verbose),
    )

    return quantized_model, state_dict


def run_qwen3_vl_lm_autoquant_sensitivity(
    *,
    model_dir: Path,
    captions_path: Path,
    scheme: AutoQuantSchemeConfig,
    max_calib_samples: int,
    calib_seq_len: int,
    batch_size: int,
    device: str,
    torch_dtype: Optional[torch.dtype] = None,
) -> Tuple[Dict[str, Any], Mapping[str, Any]]:
    """Load Qwen3-VL, extract LM, run AutoQuant, and return the manifest/state."""

    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    from transformers import (  # type: ignore[import-untyped]
        AutoModelForImageTextToText,
        AutoTokenizer,
    )

    torch_device = torch.device(device)
    if torch_dtype is None:
        torch_dtype = torch.bfloat16 if torch_device.type == "cuda" else torch.float32

    full_model = AutoModelForImageTextToText.from_pretrained(
        str(model_dir),
        torch_dtype=torch_dtype,
        device_map=None,
        trust_remote_code=True,
    ).to(torch_device)
    full_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
    )
    tokenizer.padding_side = "left"

    language_model = extract_language_model_from_vl(full_model)

    lm_head = getattr(full_model, "lm_head", None)
    if lm_head is None:
        raise RuntimeError("Expected full Qwen3-VL model to have an `lm_head` attribute.")
    lm_head = lm_head.to(torch_device)

    calib_loader = build_lm_calib_dataloader(
        captions_path=captions_path,
        tokenizer=tokenizer,
        batch_size=max(batch_size, 1),
        max_samples=max_calib_samples,
        max_length=calib_seq_len,
    )

    quantized_lm, state_dict = autoquant_lm(
        lm_model=language_model,
        scheme=scheme,
        calib_loader=calib_loader,
        batch_size=max(batch_size, 1),
        device=torch_device,
        lm_head=lm_head,
        pad_token_id=tokenizer.pad_token_id,
    )

    manifest = build_quant_manifest(
        model=quantized_lm,
        scheme=scheme,
        state_dict=state_dict,
        model_id=str(model_dir),
    )
    return manifest, state_dict


def scheme_with_overrides(
    scheme: AutoQuantSchemeConfig,
    *,
    effective_bits: Optional[float] = None,
    score_size: Optional[int] = None,
) -> AutoQuantSchemeConfig:
    """Return a copy of `scheme` with optional overrides applied."""

    payload = asdict(scheme)
    if effective_bits is not None:
        payload["auto_quantize_bits"] = float(effective_bits)
    if score_size is not None:
        payload["auto_quantize_score_size"] = int(score_size)
    return AutoQuantSchemeConfig(**payload)
