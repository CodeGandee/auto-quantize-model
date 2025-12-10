#!/usr/bin/env python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchmetrics.text.perplexity import Perplexity
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration


@dataclass
class EvalConfig:
    fp16_model_dir: str
    quant_model_dir: str
    captions_path: str
    max_samples: int
    max_length: int
    batch_size: int
    device: str


class CaptionsEvalDataset(Dataset[Mapping[str, torch.Tensor]]):
    def __init__(
        self,
        captions_path: Path,
        max_samples: int,
    ) -> None:
        if not captions_path.is_file():
            raise FileNotFoundError(f"Captions file not found: {captions_path}")

        texts: List[str] = []
        with captions_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                text = line.strip()
                if not text:
                    continue
                texts.append(text)
                if len(texts) >= max_samples:
                    break

        if not texts:
            raise RuntimeError(f"No non-empty lines found in {captions_path}")
        self._texts = texts

    def __len__(self) -> int:
        return len(self._texts)

    def __getitem__(self, idx: int) -> Mapping[str, torch.Tensor]:
        return {"text": self._texts[idx]}


def build_eval_dataloader(
    captions_path: Path,
    tokenizer: AutoTokenizer,
    max_samples: int,
    max_length: int,
    batch_size: int,
) -> DataLoader[Mapping[str, torch.Tensor]]:
    dataset = CaptionsEvalDataset(
        captions_path=captions_path,
        max_samples=max_samples,
    )

    def collate_fn(batch: List[Mapping[str, torch.Tensor]]) -> Mapping[str, torch.Tensor]:
        texts = [item["text"] for item in batch]
        enc = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        features: Mapping[str, torch.Tensor] = {k: v for k, v in enc.items()}
        input_ids = features.get("input_ids")
        if input_ids is not None:
            # Teacher-forced language modeling: labels are input_ids shifted later.
            features = dict(features)
            features["labels"] = input_ids.clone()
        return features

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


def compute_perplexity_for_model(
    model: Qwen2_5_VLForConditionalGeneration,
    dataloader: Iterable[Mapping[str, torch.Tensor]],
    pad_token_id: Optional[int],
    device: torch.device,
) -> float:
    ignore_index = pad_token_id if pad_token_id is not None else -100
    ppl_metric = Perplexity(ignore_index=ignore_index)
    ppl_metric.to(device)

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids)
            logits = outputs.logits

            # Shift for teacher-forced causal LM perplexity.
            preds = logits[:, :-1, :]
            target = labels[:, 1:]

            ppl_metric.update(preds=preds, target=target)

    perp = float(ppl_metric.compute().item())
    return perp


def compute_logit_metrics(
    fp16_model: Qwen2_5_VLForConditionalGeneration,
    quant_model: Qwen2_5_VLForConditionalGeneration,
    dataloader: Iterable[Mapping[str, torch.Tensor]],
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Tuple[float, float]:
    fp16_model.eval()
    quant_model.eval()

    mse_sum = 0.0
    kl_sum = 0.0
    count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)

            out_fp16 = fp16_model(input_ids=input_ids)
            out_quant = quant_model(input_ids=input_ids)

            logits_fp16 = out_fp16.logits[:, -1, :]
            logits_quant = out_quant.logits[:, -1, :]

            mse = torch.mean((logits_quant - logits_fp16) ** 2).item()

            log_p = torch.log_softmax(logits_fp16, dim=-1)
            log_q = torch.log_softmax(logits_quant, dim=-1)
            kl = torch.sum(torch.exp(log_p) * (log_p - log_q), dim=-1).mean().item()

            mse_sum += mse
            kl_sum += kl
            count += 1

    if count == 0:
        return 0.0, 0.0
    return mse_sum / count, kl_sum / count


def upcast_fp8_weights_to_dtype(model: torch.nn.Module, target_dtype: torch.dtype) -> None:
    """
    Convert any FP8 weights in the given model to the specified floating dtype.

    This is a pragmatic helper for running evaluation metrics on ModelOpt-exported
    FP8 checkpoints using standard HF modules, which expect matching dtypes for
    inputs and weights in linear layers.
    """
    fp8_dtypes = {
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e4m3fnuz", None),
    }
    fp8_dtypes = {dt for dt in fp8_dtypes if dt is not None}

    if not fp8_dtypes:
        return

    for param in model.parameters():
        if param.dtype in fp8_dtypes:
            param.data = param.data.to(target_dtype)
