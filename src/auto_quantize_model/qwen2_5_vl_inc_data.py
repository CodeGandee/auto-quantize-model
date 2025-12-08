from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase, Qwen2_5_VLForConditionalGeneration


DEFAULT_CAPTIONS_PATH = Path("datasets/vlm-quantize-calib/coco2017_captions.txt")


@dataclass
class QwenCalibConfig:
    """Configuration for Qwen2.5-VL calibration/eval dataloaders."""

    captions_path: Path = DEFAULT_CAPTIONS_PATH
    max_samples: int = 512
    max_seq_len: int = 256
    batch_size: int = 8
    num_workers: int = 0
    shuffle: bool = True


class _TextCaptionDataset(Dataset[dict]):
    """Simple Dataset that wraps a list of text captions and a tokenizer."""

    def __init__(
        self,
        captions: Sequence[str],
        tokenizer: PreTrainedTokenizerBase,
        max_seq_len: int,
    ) -> None:
        self._captions: List[str] = [c.strip() for c in captions if c.strip()]
        self._tokenizer = tokenizer
        self._max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self._captions)

    def __getitem__(self, idx: int) -> dict:
        prompt = self._captions[idx]
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ]
        chat_text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = self._tokenizer(
            chat_text,
            max_length=self._max_seq_len,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        # encoded tensors are 1 x L; squeeze to 1D for easier batching.
        item = {k: v[0] for k, v in encoded.items()}
        item["prompt_text"] = prompt
        return item


def _load_captions(path: Path, max_samples: int) -> List[str]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Calibration captions file not found: {path}. "
            "Build it via scripts/build_vlm_quantize_calib_coco2017_db.py."
        )
    lines = path.read_text(encoding="utf-8").splitlines()
    captions = [line.strip() for line in lines if line.strip()]
    if max_samples > 0:
        captions = captions[:max_samples]
    return captions


def build_qwen_calib_dataloader(
    model_dir: Path,
    config: Optional[QwenCalibConfig] = None,
) -> DataLoader:
    """Build a text-only calibration dataloader for Qwen2.5-VL.

    The dataloader yields dictionaries compatible with the Transformers
    Qwen2.5-VL model (``input_ids``, ``attention_mask``). Each sample is
    constructed using the model's chat template so calibration traffic
    matches normal inference requests.
    """
    cfg = config or QwenCalibConfig()
    captions = _load_captions(cfg.captions_path, cfg.max_samples)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    dataset = _TextCaptionDataset(
        captions=captions,
        tokenizer=tokenizer,
        max_seq_len=cfg.max_seq_len,
    )

    def collate_fn(batch: Sequence[dict]) -> dict:
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        prompts = [item["prompt_text"] for item in batch]

        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
        )
        attention_mask_padded = torch.nn.utils.rnn.pad_sequence(
            attention_mask,
            batch_first=True,
            padding_value=0,
        )

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "prompt_text": prompts,
        }

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )


def build_qwen_eval_dataloader(
    model_dir: Path,
    num_prompts: int = 32,
    max_seq_len: int = 256,
    batch_size: int = 4,
) -> DataLoader:
    """Build a small eval dataloader using a prefix of the captions file."""
    cfg = QwenCalibConfig(
        captions_path=DEFAULT_CAPTIONS_PATH,
        max_samples=num_prompts,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        shuffle=False,
    )
    return build_qwen_calib_dataloader(model_dir=model_dir, config=cfg)


def make_qwen_eval_func(
    model: Qwen2_5_VLForConditionalGeneration,
    dataloader: Iterable[dict],
    max_batches: Optional[int] = 8,
) -> Callable[[Qwen2_5_VLForConditionalGeneration], float]:
    """Create a lightweight eval_func compatible with INC quantization.fit.

    The returned callable runs a few batches in evaluation mode and returns
    the negative average token-level log-likelihood as a scalar proxy metric
    (higher is better when negated, lower raw loss is better).
    """

    def _eval(current_model: Qwen2_5_VLForConditionalGeneration) -> float:
        current_model.eval()
        device = next(current_model.parameters()).device

        total_loss = 0.0
        total_tokens = 0
        batches_seen = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                # Standard causal LM loss: shift labels by one.
                labels = input_ids.clone()
                outputs = current_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

                batch_tokens = attention_mask.sum().item()
                total_loss += float(loss.item()) * batch_tokens
                total_tokens += batch_tokens

                batches_seen += 1
                if max_batches is not None and batches_seen >= max_batches:
                    break

        if total_tokens == 0:
            return 0.0

        avg_loss = total_loss / float(total_tokens)
        # INC expects a scalar where higher is better, so return negative loss.
        return -avg_loss

    return _eval

