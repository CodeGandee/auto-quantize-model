#!/usr/bin/env python
"""
INT8 LM-only AutoQuant driver for Qwen3-VL-4B-Instruct.

This script:

- Loads the Qwen3-VL-4B-Instruct checkpoint.
- Extracts the language model component for LM-only AutoQuant, keeping the
  vision tower in higher precision.
- Builds a text-only calibration dataloader from COCO2017 captions.
- Runs NVIDIA ModelOpt AutoQuant with an INT8 configuration derived from
  ``INT8_LM_DEFAULT_CFG``.
- Emits per-layer sensitivity artifacts:
  - ``per-layer-sensitivity.md``
  - ``per-layer-sensitivity.json``

The resulting artifacts can be compared with FP8 all-layers runs to study
INT8 (W8A8) behavior for the text tower.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForImageTextToText, AutoTokenizer

import modelopt.torch.quantization as mtq
from auto_quantize_model.modelopt_configs import CUSTOM_QUANT_CONFIGS
from modelopt.torch.export.model_utils import get_language_model_from_vl

# Reuse shared AutoQuant helpers from the all-layers driver.
_THIS_DIR = Path(__file__).resolve().parent
_HELPERS_DIR = _THIS_DIR.parent
_ALL_LAYERS_DIR = _HELPERS_DIR / "qwen3_vl_4b_autoquant_all_layers"
_ALL_LAYERS_DIR_STR = str(_ALL_LAYERS_DIR)
if _ALL_LAYERS_DIR_STR not in sys.path:
    sys.path.insert(0, _ALL_LAYERS_DIR_STR)

from run_qwen3_vl_4b_autoquant_all_layers import (  # noqa: E402
    AutoQuantSchemeConfig,
    build_quant_manifest,
    create_forward_step,
    write_layer_sensitivity_json,
    write_layer_sensitivity_md,
)


class CocoCaptionsDataset(Dataset[Mapping[str, torch.Tensor]]):
    """
    Text-only calibration dataset built from COCO2017 captions.

    Each line in the captions file is treated as an independent sample
    and tokenized for causal language modeling.
    """

    def __init__(
        self,
        captions_path: Path,
        tokenizer: AutoTokenizer,
        max_samples: Optional[int] = None,
        max_length: int = 512,
    ) -> None:
        if not captions_path.is_file():
            raise FileNotFoundError(f"Calibration captions file not found: {captions_path}")

        self.m_tokenizer = tokenizer
        self.m_max_length = int(max_length)

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

        self.m_captions = lines

    def __len__(self) -> int:
        return len(self.m_captions)

    def __getitem__(self, index: int) -> Mapping[str, torch.Tensor]:
        text = self.m_captions[index]
        tokenized = self.m_tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.m_max_length,
        )
        features: Dict[str, torch.Tensor] = {key: value.squeeze(0) for key, value in tokenized.items()}
        input_ids = features.get("input_ids")
        if input_ids is not None:
            features["labels"] = input_ids.clone()
        return features


def build_lm_calib_dataloader(
    captions_path: Path,
    tokenizer: AutoTokenizer,
    batch_size: int,
    max_samples: Optional[int],
    max_length: int,
) -> DataLoader[Mapping[str, torch.Tensor]]:
    """Build a simple text-only calibration dataloader for LM AutoQuant.

    Parameters
    ----------
    captions_path :
        Path to the COCO2017 captions text file.
    tokenizer :
        Hugging Face tokenizer for Qwen3-VL.
    batch_size :
        Batch size used when constructing calibration batches.
    max_samples :
        Optional limit on the number of captions to load.
    max_length :
        Maximum sequence length for tokenization.

    Returns
    -------
    DataLoader
        Deterministic dataloader over calibration captions.
    """
    dataset = CocoCaptionsDataset(
        captions_path=captions_path,
        tokenizer=tokenizer,
        max_samples=max_samples,
        max_length=max_length,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def create_lm_loss_func(
    device: torch.device,
    lm_head: torch.nn.Module,
    pad_token_id: Optional[int],
) -> Callable[[Any, Mapping[str, torch.Tensor]], torch.Tensor]:
    """Create a loss function for gradient-based LM AutoQuant.

    The loss is a standard causal LM cross-entropy between the logits
    produced by ``lm_head(last_hidden_state)`` and the provided labels.

    Parameters
    ----------
    device :
        Target device to place labels and logits on.
    lm_head :
        Language modeling head that maps hidden states to vocabulary logits.
    pad_token_id :
        Token id used for padding; ignored in the loss.

    Returns
    -------
    callable
        Function that maps ``(output, batch)`` to a scalar loss tensor.
    """
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
) -> Tuple[torch.nn.Module, Mapping[str, Any]]:
    """Run ModelOpt AutoQuant on the LM-only module with INT8 formats.

    Parameters
    ----------
    lm_model :
        Language model submodule to quantize.
    scheme :
        AutoQuant configuration, including bits budget and formats.
    calib_loader :
        Iterable of calibration batches.
    batch_size :
        Logical calibration batch size.
    device :
        Target device for model execution.
    lm_head :
        Language modeling head used to compute logits.
    pad_token_id :
        Padding token id for the loss.

    Returns
    -------
    tuple
        Quantized LM module and the AutoQuant state dictionary.
    """
    quantization_formats: List[Union[Dict[str, Any], str]] = []
    for fmt_name in scheme.quant_formats:
        if fmt_name in CUSTOM_QUANT_CONFIGS:
            quantization_formats.append(CUSTOM_QUANT_CONFIGS[fmt_name])
            continue
        if hasattr(mtq, fmt_name):
            quantization_formats.append(getattr(mtq, fmt_name))
            continue
        raise ValueError(f"Unknown quantization format in scheme: {fmt_name}")

    calib_batches: List[Mapping[str, torch.Tensor]] = list(calib_loader)
    if not calib_batches:
        raise RuntimeError("Calibration dataloader is empty.")

    forward_step = create_forward_step(scheme.auto_quantize_method, device=device)
    loss_fn = create_lm_loss_func(device=device, lm_head=lm_head, pad_token_id=pad_token_id)

    num_score_steps = max(scheme.auto_quantize_score_size // max(batch_size, 1), 1)
    num_score_steps = min(num_score_steps, len(calib_batches))

    quantized_model, state_dict = mtq.auto_quantize(
        lm_model,
        constraints={"effective_bits": scheme.auto_quantize_bits},
        quantization_formats=quantization_formats,
        data_loader=calib_batches,
        forward_step=forward_step,
        loss_func=loss_fn,
        num_calib_steps=len(calib_batches),
        num_score_steps=num_score_steps,
        verbose=True,
    )
    return quantized_model, state_dict


AUTOQUANT_INT8_LM_DEFAULT = AutoQuantSchemeConfig(
    name="int8_autoquant_lm_default",
    auto_quantize_bits=8.0,
    auto_quantize_method="gradient",
    auto_quantize_score_size=128,
    coverage_mode="lm_default",
    coverage_fraction=1.0,
    quant_formats=["INT8_LM_DEFAULT_CFG"],
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the Qwen3-VL INT8 LM driver."""
    parser = argparse.ArgumentParser(
        description=(
            "Run ModelOpt AutoQuant INT8 LM-only sensitivity for Qwen3-VL-4B-Instruct "
            "and emit per-layer sensitivity artifacts."
        )
    )

    default_model_dir = (
        Path("models")
        / "qwen3_vl_4b_instruct"
        / "checkpoints"
        / "Qwen3-VL-4B-Instruct"
    )
    default_captions = (
        Path("datasets")
        / "vlm-quantize-calib"
        / "coco2017_captions_large.txt"
    )
    default_output_dir = Path("tmp") / "qwen3_vl_4b_autoquant_int8_lm"

    parser.add_argument(
        "--model-dir",
        type=Path,
        default=default_model_dir,
        help="Path to Qwen3-VL-4B-Instruct HF checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Directory to write the quantization manifest and artifacts.",
    )
    parser.add_argument(
        "--captions-path",
        type=Path,
        default=default_captions,
        help=(
            "Path to COCO2017 captions text file. Defaults to the shared "
            "large (512-sample) calibration subset."
        ),
    )
    parser.add_argument(
        "--max-calib-samples",
        type=int,
        default=512,
        help="Maximum number of text samples to use for calibration.",
    )
    parser.add_argument(
        "--calib-seq-len",
        type=int,
        default=512,
        help="Maximum sequence length for calibration tokens.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Calibration batch size for AutoQuant.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device to use (default: cuda).",
    )
    parser.add_argument(
        "--effective-bits",
        type=float,
        default=None,
        help="Override effective bits for AutoQuant (defaults to scheme value).",
    )
    parser.add_argument(
        "--auto-quantize-score-size",
        type=int,
        default=None,
        help="Override AutoQuant score size in samples (defaults to scheme value).",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        default=False,
        help=(
            "Do not run AutoQuant. Instead, read an existing quantization "
            "manifest from --output-dir and regenerate the per-layer "
            "sensitivity Markdown and JSON reports."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the Qwen3-VL INT8 LM-only AutoQuant driver."""
    args = parse_args(argv)

    scheme = AUTOQUANT_INT8_LM_DEFAULT
    if args.effective_bits is not None:
        scheme = AutoQuantSchemeConfig(
            **{
                **asdict(scheme),
                "auto_quantize_bits": args.effective_bits,
            }
        )
    if args.auto_quantize_score_size is not None:
        scheme = AutoQuantSchemeConfig(
            **{
                **asdict(scheme),
                "auto_quantize_score_size": args.auto_quantize_score_size,
            }
        )

    if args.report_only:
        if not args.output_dir.is_dir():
            print(
                f"[ERROR] Report-only mode requested but output dir does not exist: "
                f"{args.output_dir}",
                file=sys.stderr,
            )
            return 1
        manifest_path = args.output_dir / f"{scheme.name}_quant_manifest.json"
        if not manifest_path.is_file():
            print(
                "[ERROR] Report-only mode requested but manifest JSON not found at: "
                f"{manifest_path}",
                file=sys.stderr,
            )
            return 1
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(
                f"[ERROR] Failed to read manifest JSON at {manifest_path}: {exc}",
                file=sys.stderr,
            )
            return 1

        if "layer_sensitivity" not in manifest or "autoquant_state" not in manifest:
            print(
                "[ERROR] Manifest JSON is missing required keys "
                "`layer_sensitivity` or `autoquant_state`.",
                file=sys.stderr,
            )
            return 1

        sensitivity_md_path = args.output_dir / "per-layer-sensitivity.md"
        sensitivity_json_path = args.output_dir / "per-layer-sensitivity.json"
        model_id: Optional[str] = None
        model_meta = manifest.get("model") or {}
        if isinstance(model_meta, dict):
            model_id = model_meta.get("id")

        write_layer_sensitivity_md(
            layer_sensitivity=manifest["layer_sensitivity"],
            scheme=scheme,
            autoquant_state=manifest["autoquant_state"],
            out_path=sensitivity_md_path,
            model_id=model_id,
        )
        write_layer_sensitivity_json(
            manifest=manifest,
            out_path=sensitivity_json_path,
        )
        print(
            "[INFO] Report-only mode: regenerated per-layer sensitivity artifacts at "
            f"{args.output_dir}",
        )
        return 0

    if not args.model_dir.is_dir():
        print(
            f"[ERROR] Model directory not found: {args.model_dir}\n"
            "Hint: run models/qwen3_vl_4b_instruct/bootstrap.sh first.",
            file=sys.stderr,
        )
        return 1

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        print(
            "[WARN] CUDA is not available; running on CPU will be extremely slow.",
            file=sys.stderr,
        )

    device = torch.device(args.device)
    torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print(f"[INFO] Loading Qwen3-VL-4B-Instruct from {args.model_dir}")
    full_model = AutoModelForImageTextToText.from_pretrained(
        str(args.model_dir),
        torch_dtype=torch_dtype,
        device_map=None,
        trust_remote_code=True,
    ).to(device)
    full_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_dir),
        trust_remote_code=True,
    )
    tokenizer.padding_side = "left"

    lineage = get_language_model_from_vl(full_model)
    if lineage is None or len(lineage) < 2:
        print(
            "[ERROR] Could not extract language model from Qwen3-VL model.",
            file=sys.stderr,
        )
        return 1

    language_model = list(lineage)[-1]

    lm_head = getattr(full_model, "lm_head", None)
    if lm_head is None:
        print(
            "[ERROR] Expected full Qwen3-VL model to have an `lm_head` attribute "
            "for loss computation.",
            file=sys.stderr,
        )
        return 1
    lm_head = lm_head.to(device)

    print(f"[INFO] Building LM calibration dataloader from {args.captions_path}")
    calib_loader = build_lm_calib_dataloader(
        captions_path=args.captions_path,
        tokenizer=tokenizer,
        batch_size=max(args.batch_size, 1),
        max_samples=args.max_calib_samples,
        max_length=args.calib_seq_len,
    )

    print(f"[INFO] Running AutoQuant LM-only scheme: {scheme.name}")
    quantized_lm, state_dict = autoquant_lm(
        lm_model=language_model,
        scheme=scheme,
        calib_loader=calib_loader,
        batch_size=max(args.batch_size, 1),
        device=device,
        lm_head=lm_head,
        pad_token_id=tokenizer.pad_token_id,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir / f"{scheme.name}_autoquant_state.pt"
    torch.save(state_dict, state_path)

    manifest_path = args.output_dir / f"{scheme.name}_quant_manifest.json"
    print(f"[INFO] Building quantization manifest at {manifest_path}")
    manifest = build_quant_manifest(
        model=quantized_lm,
        scheme=scheme,
        state_dict=state_dict,
        model_id=str(args.model_dir),
    )
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    sensitivity_md_path = args.output_dir / "per-layer-sensitivity.md"
    sensitivity_json_path = args.output_dir / "per-layer-sensitivity.json"
    model_meta = manifest.get("model") or {}
    model_id = model_meta.get("id") if isinstance(model_meta, dict) else None

    write_layer_sensitivity_md(
        layer_sensitivity=manifest["layer_sensitivity"],
        scheme=scheme,
        autoquant_state=manifest["autoquant_state"],
        out_path=sensitivity_md_path,
        model_id=model_id,
    )
    write_layer_sensitivity_json(
        manifest=manifest,
        out_path=sensitivity_json_path,
    )

    print("[INFO] AutoQuant INT8 LM-only run completed successfully.")
    print(f"[INFO] Quantization manifest written to: {manifest_path}")
    print(f"[INFO] AutoQuant state written to: {state_path}")
    print(f"[INFO] Per-layer sensitivity report: {sensitivity_md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
