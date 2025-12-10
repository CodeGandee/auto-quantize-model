#!/usr/bin/env python
"""
AutoQuant FP8 schemes for Qwen2.5-VL-3B (LM-only).

This driver:
  - Loads the Qwen2.5-VL-3B-Instruct checkpoint.
  - Extracts the language model for LM-only AutoQuant, keeping the vision
    tower in BF16/FP16 (see context/plans/plan-modelopt-autoquant-fp8-qwen2_5-vl-mixed-schemes.md).
  - Builds a text-only calibration dataloader from COCO captions.
  - Runs ModelOpt AutoQuant with scheme-specific defaults (effective bits,
    score size, etc.).
  - Emits a JSON manifest describing per-layer quantization choices.

Checkpoint export is handled in a separate subtask; this script focuses on
search + manifest generation.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from mdutils import MdUtils
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration

import modelopt.torch.quantization as mtq
from auto_quantize_model.modelopt_configs import CUSTOM_QUANT_CONFIGS
from modelopt.torch.export.model_utils import get_language_model_from_vl
from modelopt.torch.quantization.utils import is_quantized_linear
from qwen_vl_utils import process_vision_info


@dataclass
class AutoQuantSchemeConfig:
    name: str
    auto_quantize_bits: float
    auto_quantize_method: str
    auto_quantize_score_size: int
    coverage_mode: str
    coverage_fraction: float
    quant_formats: List[str]


AUTOQUANT_FP8_SCHEMES: Dict[str, AutoQuantSchemeConfig] = {
    "fp8_autoquant_top25": AutoQuantSchemeConfig(
        name="fp8_autoquant_top25",
        auto_quantize_bits=13.0,
        auto_quantize_method="gradient",
        auto_quantize_score_size=128,
        coverage_mode="top_k_blocks",
        coverage_fraction=0.25,
        quant_formats=["FP8_DEFAULT_CFG"],
    ),
    "fp8_autoquant_top50": AutoQuantSchemeConfig(
        name="fp8_autoquant_top50",
        auto_quantize_bits=11.0,
        auto_quantize_method="gradient",
        auto_quantize_score_size=128,
        coverage_mode="top_k_blocks",
        coverage_fraction=0.50,
        quant_formats=["FP8_DEFAULT_CFG"],
    ),
    "fp8_autoquant_full": AutoQuantSchemeConfig(
        name="fp8_autoquant_full",
        auto_quantize_bits=9.0,
        auto_quantize_method="gradient",
        auto_quantize_score_size=96,
        coverage_mode="full",
        coverage_fraction=1.0,
        quant_formats=["FP8_DEFAULT_CFG"],
    ),
    "fp8_autoquant_all_layers_fp8": AutoQuantSchemeConfig(
        name="fp8_autoquant_all_layers_fp8",
        auto_quantize_bits=11.0,
        auto_quantize_method="gradient",
        auto_quantize_score_size=128,
        coverage_mode="full",
        coverage_fraction=1.0,
        quant_formats=["FP8_ALL_LAYERS_CFG"],
    ),
}


class CocoCaptionsDataset(Dataset[Mapping[str, torch.Tensor]]):
    """
    Simple text-only dataset that reads one caption per line and tokenizes it.
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
        self.tokenizer = tokenizer
        self.max_length = max_length

        lines: List[str] = []
        with captions_path.open("r", encoding="utf-8") as f:
            for line in f:
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

    def __getitem__(self, idx: int) -> Mapping[str, torch.Tensor]:
        text = self._captions[idx]
        tokenized = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        # Remove batch dimension and add labels for causal LM loss.
        features = {k: v.squeeze(0) for k, v in tokenized.items()}
        input_ids = features.get("input_ids")
        if input_ids is not None:
            features["labels"] = input_ids.clone()
        return features


def build_calib_dataloader(
    captions_path: Path,
    tokenizer: AutoTokenizer,
    batch_size: int,
    max_samples: Optional[int],
    max_length: int,
) -> DataLoader[Mapping[str, torch.Tensor]]:
    dataset = CocoCaptionsDataset(
        captions_path=captions_path,
        tokenizer=tokenizer,
        max_samples=max_samples,
        max_length=max_length,
    )
    # Simple sequential sampler; AutoQuant only needs representative batches.
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


class CocoVlmDataset(Dataset[Mapping[str, torch.Tensor]]):
    """
    Image+text calibration dataset built from a COCO2017 VLM calib DB.

    Each item is converted into processor-ready tensors (text + images)
    and includes labels for a simple causal LM loss.
    """

    def __init__(
        self,
        calib_db: Path,
        coco_root: Path,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        max_samples: int,
        max_length: int,
    ) -> None:
        if not calib_db.is_file():
            raise FileNotFoundError(f"Calibration DB not found: {calib_db}")
        if not coco_root.is_dir():
            raise FileNotFoundError(f"COCO root directory not found: {coco_root}")

        # Load a small set of (image_relpath, caption) pairs from the DB.
        import sqlite3

        conn = sqlite3.connect(str(calib_db))
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT image_relpath, caption
                FROM vlm_calib_samples
                ORDER BY id ASC
                LIMIT ?
                """,
                (max_samples,),
            )
            rows = cursor.fetchall()
        finally:
            conn.close()

        self.m_samples: List[Tuple[Path, str]] = []
        for image_relpath, caption in rows:
            image_path = coco_root / str(image_relpath)
            self.m_samples.append((image_path, str(caption)))

        if not self.m_samples:
            raise RuntimeError(
                f"No VLM calibration samples loaded from {calib_db}; "
                "ensure the DB is populated."
            )

        self.m_tokenizer = tokenizer
        self.m_processor = processor
        # Use an explicit maximum sequence length so we can keep VLM
        # calibration sequences bounded and avoid OOM in attention.
        self.m_max_length: int = int(max_length)

    def __len__(self) -> int:
        return len(self.m_samples)

    def __getitem__(self, idx: int) -> Mapping[str, torch.Tensor]:
        image_path, caption = self.m_samples[idx]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": caption},
                ],
            }
        ]
        text = self.m_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.m_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding="max_length",
            max_length=self.m_max_length,
            return_tensors="pt",
        )
        # Keep the tensor shapes as produced by the processor so that
        # keys like `image_grid_thw` retain their expected structure
        # (e.g., shape [batch, 3] instead of being squeezed to a 1-D
        # tensor). Add labels mirroring input_ids for a causal LM loss.
        features = dict(inputs)
        input_ids = features.get("input_ids")
        if input_ids is not None:
            features["labels"] = input_ids.clone()
        return features


def build_vlm_calib_dataloader(
    calib_db: Path,
    coco_root: Path,
    tokenizer: AutoTokenizer,
    processor: AutoProcessor,
    batch_size: int,  # noqa: ARG001 - kept for interface symmetry
    max_samples: int,
    max_length: int,
) -> Iterable[Mapping[str, torch.Tensor]]:
    dataset = CocoVlmDataset(
        calib_db=calib_db,
        coco_root=coco_root,
        tokenizer=tokenizer,
        processor=processor,
        max_samples=max_samples,
        max_length=max_length,
    )
    # For VLM calibration we keep things simple and treat each sample
    # as its own "batch" to avoid padding/collation issues across images.
    return [dataset[index] for index in range(len(dataset))]


def create_forward_step(auto_quantize_method: str, device: torch.device):
    """
    Build a forward_step callable for AutoQuant that ensures batches
    are placed on the same device as the model.
    """

    if auto_quantize_method == "gradient":

        def forward_step(model, batch):
            batch_on_device = {k: v.to(device) for k, v in batch.items()}
            return model(**batch_on_device)

    elif auto_quantize_method == "kl_div":

        def forward_step(model, batch):
            batch_on_device = {k: v.to(device) for k, v in batch.items()}
            return model(**batch_on_device).logits

    else:
        raise ValueError(
            f"Unsupported auto_quantize_method: {auto_quantize_method}. "
            "Expected 'gradient' or 'kl_div'."
        )
    return forward_step


def create_loss_func(
    device: torch.device,
    lm_head: torch.nn.Module,
    pad_token_id: Optional[int],
) -> Callable[[Any, Any], torch.Tensor]:
    """
    Build a loss function for gradient-based AutoQuant.

    The loss is a standard causal LM cross-entropy between the logits
    produced by `lm_head(last_hidden_state)` and the provided labels.
    """
    ignore_index = -100 if pad_token_id is None else pad_token_id
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    def _loss_func(output, batch):
        labels = batch["labels"].to(device)
        if hasattr(output, "last_hidden_state"):
            hidden_states = output.last_hidden_state
        else:
            raise ValueError("Expected output with last_hidden_state for loss computation.")

        logits = lm_head(hidden_states)
        # Shift for causal LM loss.
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        return loss

    return _loss_func


def extract_lm_only(model: Qwen2_5_VLForConditionalGeneration) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """
    Extract the language model from a Qwen2.5-VL model.

    This driver focuses on LM-only AutoQuant search and manifest
    generation, so we simply return the full VLM and its language
    model submodule. Vision and other components remain in their
    original (BF16/FP16) form.
    """
    full_model: torch.nn.Module = model
    lineage = get_language_model_from_vl(full_model)
    if lineage is None or len(lineage) < 2:
        raise RuntimeError("Could not extract language model from Qwen2.5-VL model.")

    lineage_list = list(lineage)
    language_model = lineage_list[-1]
    return full_model, language_model


def autoquant_lm(
    lm_model: torch.nn.Module,
    scheme: AutoQuantSchemeConfig,
    calib_loader: Iterable[Mapping[str, torch.Tensor]],
    batch_size: int,
    device: torch.device,
    lm_head: torch.nn.Module,
    pad_token_id: Optional[int],
) -> Tuple[torch.nn.Module, Mapping[str, Any]]:
    """
    Run ModelOpt AutoQuant on the LM-only module with FP8 formats.
    Returns the quantized model and AutoQuant state dict.
    """
    quantization_formats: List[Union[Dict[str, Any], str]] = []
    for fmt in scheme.quant_formats:
        if fmt in CUSTOM_QUANT_CONFIGS:
            quantization_formats.append(CUSTOM_QUANT_CONFIGS[fmt])
            continue
        if hasattr(mtq, fmt):
            quantization_formats.append(getattr(mtq, fmt))
            continue
        raise ValueError(f"Unknown quantization format in scheme: {fmt}")

    calib_loader_list = list(calib_loader)
    if not calib_loader_list:
        raise RuntimeError("Calibration dataloader is empty.")

    forward_step = create_forward_step(scheme.auto_quantize_method, device=device)
    loss_fn = create_loss_func(device=device, lm_head=lm_head, pad_token_id=pad_token_id)

    # AutoQuant scoring size is expressed in samples; convert to steps.
    num_score_steps = max(scheme.auto_quantize_score_size // max(batch_size, 1), 1)
    num_score_steps = min(num_score_steps, len(calib_loader_list))

    model, state_dict = mtq.auto_quantize(
        lm_model,
        constraints={"effective_bits": scheme.auto_quantize_bits},
        quantization_formats=quantization_formats,
        data_loader=calib_loader_list,
        forward_step=forward_step,
        loss_func=loss_fn,
        num_calib_steps=len(calib_loader_list),
        num_score_steps=num_score_steps,
        verbose=True,
    )

    return model, state_dict


def build_quant_manifest(
    model: torch.nn.Module,
    scheme: AutoQuantSchemeConfig,
    state_dict: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Build a simple manifest describing per-layer quantization status.
    This is intentionally conservative and focuses on whether a module
    is quantized and which scheme was requested.
    """
    layers: Dict[str, Dict[str, Any]] = {}

    for name, module in model.named_modules():
        if is_quantized_linear(module):
            layers[name] = {
                "quantized": True,
                "module_type": type(module).__name__,
            }

    # Extract a JSON-friendly view of layer sensitivity stats.
    def _to_float_list(values: Iterable[Any]) -> List[float]:
        result: List[float] = []
        for v in values:
            if isinstance(v, (float, int)):
                result.append(float(v))
            elif hasattr(v, "item"):
                try:
                    result.append(float(v.item()))
                except Exception:
                    continue
            else:
                # Best effort conversion; may raise if not numeric-like.
                try:
                    result.append(float(v))
                except Exception:
                    continue
        return result

    layer_sensitivity: Dict[str, Dict[str, Any]] = {}
    candidate_stats = state_dict.get("candidate_stats", {})
    for name, stats in candidate_stats.items():
        formats = stats.get("formats", [])
        scores = stats.get("scores", [])
        costs = stats.get("costs", [])
        layer_sensitivity[name] = {
            "formats": [str(fmt) for fmt in formats],
            "scores": _to_float_list(scores),
            "costs": _to_float_list(costs),
        }

    best = state_dict.get("best", {})
    autoquant_state_summary = {
        "keys": list(state_dict.keys()),
        "constraints": best.get("constraints"),
        "score": best.get("score"),
        "is_satisfied": best.get("is_satisfied"),
    }

    manifest: Dict[str, Any] = {
        "scheme": asdict(scheme),
        "num_quantized_layers": len(layers),
        "layers": layers,
        "autoquant_state": autoquant_state_summary,
        "layer_sensitivity": layer_sensitivity,
    }
    return manifest


def write_layer_sensitivity_md(
    layer_sensitivity: Mapping[str, Mapping[str, Any]],
    scheme: AutoQuantSchemeConfig,
    autoquant_state: Mapping[str, Any],
    out_path: Path,
) -> None:
    """
    Write a Markdown summary of per-layer AutoQuant sensitivity.

    The table includes, for each layer quant_recipe key:
    layer name, candidate formats, scores, and costs.
    """
    md_file = MdUtils(
        file_name=str(out_path.with_suffix("")),
        title=f"AutoQuant Layer Sensitivity ({scheme.name})",
    )

    md_file.new_paragraph(f"**Scheme:** `{scheme.name}`")
    constraints = autoquant_state.get("constraints") or {}
    eff_bits = constraints.get("effective_bits")
    score = autoquant_state.get("score")
    is_satisfied = autoquant_state.get("is_satisfied")

    if eff_bits is not None:
        md_file.new_paragraph(f"**Effective bits (from search):** `{eff_bits:.4f}`")
    if score is not None:
        md_file.new_paragraph(f"**Total AutoQuant score:** `{score:.6e}`")
    if is_satisfied is not None:
        md_file.new_paragraph(
            f"**Constraint satisfied:** `{bool(is_satisfied)}`",
        )

    md_file.new_header(
        level=2,
        title="Per-layer sensitivity table",
        add_table_of_contents="n",
    )

    md_file.new_paragraph(
        "- **Layer**: Name of the quant_recipe handle for a group of "
        "quantizable modules (e.g., attention or MLP projections).\n"
        "- **Num Bits**: Effective number of bits allocated for the "
        "quantized recipe(s) considered at this layer.\n"
        "- **Sensitivity**: AutoQuant sensitivity score for the "
        "quantized recipe(s). Higher values indicate that quantizing this "
        "layer is more harmful to model quality.\n"
        "- **Costs**: Approximate compressed weight size contribution of "
        "the layer under the corresponding recipe(s). Higher values indicate "
        "more memory usage.\n"
        "\n"
        "Note: In the JSON manifest, layer keys end with "
        "`.quant_recipe` (e.g., `language_model.layers.0.mlp.gate_proj.quant_recipe`). "
        "This suffix is added by ModelOpt to represent the AutoQuant hyperparameter "
        "attached to that module. In this table we strip the `.quant_recipe` suffix "
        "for readability; the underlying module path is the part before that suffix."
    )

    headers = ["Layer", "Num Bits", "Sensitivity", "Costs"]
    rows: List[str] = []
    row_entries: List[tuple[str, List[float], List[float], List[float]]] = []

    for layer_name, entry in layer_sensitivity.items():
        entry = layer_sensitivity[layer_name]
        formats = entry.get("formats", [])
        scores = entry.get("scores", [])
        costs = entry.get("costs", [])

        # Filter out the unquantized NONE(...) recipe when reporting sensitivity.
        filtered: List[tuple[str, float, float]] = []
        for fmt, score, cost in zip(formats, scores, costs):
            if fmt.startswith("NONE("):
                continue
            filtered.append((fmt, float(score), float(cost)))

        # Skip layers that only had a NONE recipe.
        if not filtered:
            continue

        fmt_values = [f for f, _, _ in filtered]
        score_values = [s for _, s, _ in filtered]
        cost_values = [c for _, _, c in filtered]

        row_entries.append((layer_name, fmt_values, score_values, cost_values))

    # Sort layers from highest to lowest sensitivity (max score per layer).
    row_entries.sort(key=lambda x: max(x[2]) if x[2] else 0.0, reverse=True)

    for layer_name, fmt_values, score_values, cost_values in row_entries:
        # Derive effective bits from format strings (e.g., "...effective-bits: 8.0)").
        num_bits_values: List[float] = []
        for fmt in fmt_values:
            marker = "effective-bits:"
            bits_val: Optional[float] = None
            if marker in fmt:
                try:
                    suffix = fmt.split(marker, 1)[1]
                    num_str = suffix.split(")", 1)[0].strip()
                    bits_val = float(num_str)
                except Exception:
                    bits_val = None
            if bits_val is not None:
                num_bits_values.append(bits_val)

        num_bits_str = ", ".join(f"{b:.1f}" for b in num_bits_values) if num_bits_values else ""
        scores_str = ", ".join(f"{s:.3e}" for s in score_values)
        costs_str = ", ".join(f"{c:.3e}" for c in cost_values)

        # Strip the AutoQuant-specific 'quant_recipe' suffix for readability.
        display_name = layer_name.replace(".quant_recipe", "")

        rows.extend([display_name, num_bits_str, scores_str, costs_str])

    # Ensure there is a blank line before the table so Markdown
    # previewers render it correctly.
    md_file.new_line("")

    md_file.new_table(
        columns=4,
        rows=len(row_entries) + 1,
        text=headers + rows,
        text_align="left",
    )

    md_file.create_md_file()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run ModelOpt AutoQuant FP8 search for Qwen2.5-VL-3B LM-only "
            "according to a named scheme and emit a quantization manifest."
        )
    )
    default_model_dir = (
        Path("models")
        / "qwen2_5_vl_3b_instruct"
        / "checkpoints"
        / "Qwen2.5-VL-3B-Instruct"
    )
    default_captions = (
        Path("datasets")
        / "vlm-quantize-calib"
        / "coco2017_captions.txt"
    )
    parser.add_argument(
        "--scheme-name",
        type=str,
        required=True,
        choices=sorted(AUTOQUANT_FP8_SCHEMES.keys()),
        help="Name of the AutoQuant scheme to run.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=default_model_dir,
        help="Path to Qwen2.5-VL-3B-Instruct HF checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write the quantization manifest and artifacts.",
    )
    parser.add_argument(
        "--captions-path",
        type=Path,
        default=default_captions,
        help="Path to COCO2017 captions text file.",
    )
    parser.add_argument(
        "--max-calib-samples",
        type=int,
        default=4096,
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
        "--auto-quantize-method",
        type=str,
        default=None,
        choices=["gradient", "kl_div"],
        help="Override AutoQuant method (defaults to scheme value).",
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
            "sensitivity Markdown report. Fails if the manifest is missing."
        ),
    )
    parser.add_argument(
        "--vlm-calib-db",
        type=Path,
        default=Path("datasets")
        / "vlm-quantize-calib"
        / "coco2017_vlm_calib.db",
        help=(
            "Path to the COCO2017 VLM calibration SQLite DB containing "
            "image+caption samples (used for all-layers analysis schemes)."
        ),
    )
    parser.add_argument(
        "--coco-root",
        type=Path,
        default=Path("datasets") / "coco2017" / "source-data",
        help=(
            "Root directory of COCO2017 (must contain train2017/, val2017/, "
            "annotations/). Used to resolve image_relpath entries in the "
            "VLM calibration DB."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    scheme = AUTOQUANT_FP8_SCHEMES[args.scheme_name]

    # Apply CLI overrides.
    if args.effective_bits is not None:
        scheme = AutoQuantSchemeConfig(
            **{
                **asdict(scheme),
                "auto_quantize_bits": args.effective_bits,
            }
        )
    if args.auto_quantize_method is not None:
        scheme = AutoQuantSchemeConfig(
            **{
                **asdict(scheme),
                "auto_quantize_method": args.auto_quantize_method,
            }
        )
    if args.auto_quantize_score_size is not None:
        scheme = AutoQuantSchemeConfig(
            **{
                **asdict(scheme),
                "auto_quantize_score_size": args.auto_quantize_score_size,
            }
        )

    # Report-only mode: only regenerate Markdown from existing manifest.
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
        write_layer_sensitivity_md(
            layer_sensitivity=manifest["layer_sensitivity"],
            scheme=scheme,
            autoquant_state=manifest["autoquant_state"],
            out_path=sensitivity_md_path,
        )
        print(
            "[INFO] Report-only mode: regenerated per-layer sensitivity Markdown at "
            f"{sensitivity_md_path}"
        )
        return 0

    if scheme.auto_quantize_method != "gradient":
        print(
            "[ERROR] This ModelOpt version only supports gradient-based "
            "AutoQuant (no 'method' argument). Please use "
            "--auto-quantize-method gradient.",
            file=sys.stderr,
        )
        return 1

    if not args.model_dir.is_dir():
        print(
            f"[ERROR] Model directory not found: {args.model_dir}\n"
            "Hint: run models/qwen2_5_vl_3b_instruct/bootstrap.sh first.",
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

    print(f"[INFO] Loading Qwen2.5-VL-3B-Instruct from {args.model_dir}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(args.model_dir),
        torch_dtype=torch_dtype,
        device_map=None,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir))
    tokenizer.padding_side = "left"
    processor = AutoProcessor.from_pretrained(str(args.model_dir))

    print("[INFO] Extracting language model for LM-only AutoQuant ...")
    full_model, lm_model = extract_lm_only(model)
    lm_model.to(device)
    lm_head = getattr(full_model, "lm_head", None)
    if lm_head is None:
        print(
            "[ERROR] Expected full Qwen2.5-VL model to have an `lm_head` "
            "attribute for loss computation.",
            file=sys.stderr,
        )
        return 1
    lm_head = lm_head.to(device)

    use_vlm_calib = "FP8_ALL_LAYERS_CFG" in scheme.quant_formats
    if use_vlm_calib:
        print(
            f"[INFO] Building VLM calibration dataloader from {args.vlm_calib_db} "
            f"and COCO root {args.coco_root}"
        )
        calib_loader = build_vlm_calib_dataloader(
            calib_db=args.vlm_calib_db,
            coco_root=args.coco_root,
            tokenizer=tokenizer,
            processor=processor,
            batch_size=max(args.batch_size, 1),
            max_samples=args.max_calib_samples,
            max_length=args.calib_seq_len,
        )
    else:
        print(f"[INFO] Building calibration dataloader from {args.captions_path}")
        calib_loader = build_calib_dataloader(
            captions_path=args.captions_path,
            tokenizer=tokenizer,
            batch_size=max(args.batch_size, 1),
            max_samples=args.max_calib_samples,
            max_length=args.calib_seq_len,
        )

    print(f"[INFO] Running AutoQuant scheme: {scheme.name}")
    quantized_lm, state_dict = autoquant_lm(
        lm_model=lm_model,
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

    # Update full VLM to point to quantized LM for potential downstream use.
    if hasattr(full_model, "language_model"):
        full_model.language_model = quantized_lm
    manifest_path = args.output_dir / f"{scheme.name}_quant_manifest.json"

    print(f"[INFO] Building quantization manifest at {manifest_path}")
    manifest = build_quant_manifest(
        model=quantized_lm,
        scheme=scheme,
        state_dict=state_dict,
    )

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Per-layer sensitivity Markdown summary.
    sensitivity_md_path = args.output_dir / "per-layer-sensitivity.md"
    write_layer_sensitivity_md(
        layer_sensitivity=manifest["layer_sensitivity"],
        scheme=scheme,
        autoquant_state=manifest["autoquant_state"],
        out_path=sensitivity_md_path,
    )

    print("[INFO] AutoQuant completed successfully.")
    print(f"[INFO] Quantization manifest written to: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
