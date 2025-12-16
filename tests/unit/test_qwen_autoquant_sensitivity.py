from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch

from auto_quantize_model.modelopt_autoquant import AutoQuantSchemeConfig
from auto_quantize_model.qwen.autoquant_sensitivity import (
    CocoCaptionsDataset,
    build_lm_calib_dataloader,
    scheme_with_overrides,
)


class _DummyTokenizer:
    def __call__(
        self,
        text: str,
        *,
        return_tensors: str,
        padding: str,
        truncation: bool,
        max_length: int,
    ) -> Dict[str, torch.Tensor]:
        del text, return_tensors, padding, truncation
        input_ids = torch.arange(max_length, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def _write_captions(tmp_path: Path, lines: list[str]) -> Path:
    path = tmp_path / "captions.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_coco_captions_dataset_respects_max_samples_and_adds_labels(tmp_path: Path) -> None:
    captions_path = _write_captions(tmp_path, ["a", "b", "c"])
    dataset = CocoCaptionsDataset(
        captions_path=captions_path,
        tokenizer=_DummyTokenizer(),
        max_samples=2,
        max_length=8,
    )
    assert len(dataset) == 2
    sample = dataset[0]
    assert "input_ids" in sample
    assert "labels" in sample
    assert torch.equal(sample["labels"], sample["input_ids"])


def test_build_lm_calib_dataloader_batches(tmp_path: Path) -> None:
    captions_path = _write_captions(tmp_path, ["a", "b", "c", "d"])
    loader = build_lm_calib_dataloader(
        captions_path=captions_path,
        tokenizer=_DummyTokenizer(),
        batch_size=2,
        max_samples=4,
        max_length=8,
    )
    batch = next(iter(loader))
    assert batch["input_ids"].shape == (2, 8)
    assert batch["labels"].shape == (2, 8)


def test_scheme_with_overrides_updates_fields() -> None:
    scheme = AutoQuantSchemeConfig(
        name="test",
        auto_quantize_bits=8.0,
        auto_quantize_method="gradient",
        auto_quantize_score_size=128,
        quant_formats=["FP8_DEFAULT_CFG"],
    )
    updated = scheme_with_overrides(scheme, effective_bits=6.5, score_size=32)
    assert updated.name == scheme.name
    assert updated.auto_quantize_bits == 6.5
    assert updated.auto_quantize_score_size == 32
    assert updated.quant_formats == scheme.quant_formats
