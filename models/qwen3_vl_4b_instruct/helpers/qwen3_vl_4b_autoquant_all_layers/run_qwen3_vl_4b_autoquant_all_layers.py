#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from mdutils import MdUtils
from torch.utils.data import Dataset
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

import modelopt.torch.quantization as mtq
from auto_quantize_model.modelopt_configs import CUSTOM_QUANT_CONFIGS
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


AUTOQUANT_FP8_ALL_LAYERS = AutoQuantSchemeConfig(
    name="fp8_autoquant_all_layers_fp8",
    auto_quantize_bits=11.0,
    auto_quantize_method="gradient",
    auto_quantize_score_size=128,
    coverage_mode="full",
    coverage_fraction=1.0,
    quant_formats=["FP8_ALL_LAYERS_CFG"],
)

AUTOQUANT_INT8_ALL_LAYERS = AutoQuantSchemeConfig(
    name="int8_autoquant_all_layers_int8",
    auto_quantize_bits=8.0,
    auto_quantize_method="gradient",
    auto_quantize_score_size=128,
    coverage_mode="full",
    coverage_fraction=1.0,
    quant_formats=["INT8_ALL_LAYERS_CFG"],
)


def select_autoquant_scheme(quant_format: str) -> AutoQuantSchemeConfig:
    """Select an AutoQuant scheme configuration based on the quant format.

    Parameters
    ----------
    quant_format :
        Name of the quantization format family to use. Supported values
        are ``\"fp8\"`` and ``\"int8\"``.

    Returns
    -------
    AutoQuantSchemeConfig
        Scheme describing bits budget, score size, and quant formats.

    Raises
    ------
    ValueError
        If an unsupported quant format is requested.
    """
    if quant_format == "fp8":
        return AUTOQUANT_FP8_ALL_LAYERS
    if quant_format == "int8":
        return AUTOQUANT_INT8_ALL_LAYERS
    raise ValueError(f"Unsupported quant_format: {quant_format!r}")


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

        self._samples: List[Tuple[Path, str]] = []
        for image_relpath, caption in rows:
            image_path = coco_root / str(image_relpath)
            self._samples.append((image_path, str(caption)))

        if not self._samples:
            raise RuntimeError(
                f"No VLM calibration samples loaded from {calib_db}; "
                "ensure the DB is populated."
            )

        self._tokenizer = tokenizer
        self._processor = processor
        self._max_length: int = int(max_length)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Mapping[str, torch.Tensor]:
        image_path, caption = self._samples[index]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": caption},
                ],
            }
        ]
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding="max_length",
            max_length=self._max_length,
            return_tensors="pt",
        )

        features: Dict[str, torch.Tensor] = dict(inputs)
        input_ids = features.get("input_ids")
        if input_ids is not None:
            features["labels"] = input_ids.clone()
        return features


def build_vlm_calib_dataloader(
    calib_db: Path,
    coco_root: Path,
    tokenizer: AutoTokenizer,
    processor: AutoProcessor,
    batch_size: int,
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


def create_forward_step(auto_quantize_method: str, device: torch.device) -> Callable:
    """
    Build a forward_step callable for AutoQuant that ensures batches
    are placed on the same device as the model.
    """

    if auto_quantize_method == "gradient":

        def forward_step(model: torch.nn.Module, batch: Mapping[str, torch.Tensor]) -> Any:
            batch_on_device = {key: value.to(device) for key, value in batch.items()}
            return model(**batch_on_device)

    elif auto_quantize_method == "kl_div":

        def forward_step(model: torch.nn.Module, batch: Mapping[str, torch.Tensor]) -> Any:
            batch_on_device = {key: value.to(device) for key, value in batch.items()}
            return model(**batch_on_device).logits

    else:
        raise ValueError(
            f"Unsupported auto_quantize_method: {auto_quantize_method}. "
            "Expected 'gradient' or 'kl_div'."
        )
    return forward_step


def create_loss_func(
    device: torch.device,
    pad_token_id: Optional[int],
) -> Callable[[Any, Mapping[str, torch.Tensor]], torch.Tensor]:
    """
    Build a loss function for gradient-based AutoQuant.

    The loss is a standard causal LM cross-entropy between the logits
    produced by the model and the provided labels.
    """
    ignore_index = -100 if pad_token_id is None else pad_token_id
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    def _loss_func(output: Any, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        labels = batch["labels"].to(device)
        if hasattr(output, "logits"):
            logits = output.logits
        elif isinstance(output, torch.Tensor):
            logits = output
        else:
            raise ValueError("Expected output with logits for loss computation.")

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        return loss

    return _loss_func


def autoquant_model(
    model: torch.nn.Module,
    scheme: AutoQuantSchemeConfig,
    calib_loader: Iterable[Mapping[str, torch.Tensor]],
    batch_size: int,
    device: torch.device,
    pad_token_id: Optional[int],
) -> Tuple[torch.nn.Module, Mapping[str, Any]]:
    """
    Run ModelOpt AutoQuant on the full VLM module with FP8 formats.
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

    calib_batches: List[Mapping[str, torch.Tensor]] = list(calib_loader)
    if not calib_batches:
        raise RuntimeError("Calibration dataloader is empty.")

    forward_step = create_forward_step(scheme.auto_quantize_method, device=device)
    loss_fn = create_loss_func(device=device, pad_token_id=pad_token_id)

    # AutoQuant scoring size is expressed in samples; convert to steps.
    num_score_steps = max(scheme.auto_quantize_score_size // max(batch_size, 1), 1)
    num_score_steps = min(num_score_steps, len(calib_batches))

    quantized_model, state_dict = mtq.auto_quantize(
        model,
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


def build_quant_manifest(
    model: torch.nn.Module,
    scheme: AutoQuantSchemeConfig,
    state_dict: Mapping[str, Any],
    model_id: Optional[str] = None,
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

    def _to_float_list(values: Iterable[Any]) -> List[float]:
        result: List[float] = []
        for value in values:
            if isinstance(value, (float, int)):
                result.append(float(value))
            elif hasattr(value, "item"):
                try:
                    result.append(float(value.item()))
                except Exception:
                    continue
            else:
                try:
                    result.append(float(value))
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

    sensitivity_ranking: List[Dict[str, Any]] = []
    for name, entry in layer_sensitivity.items():
        scores = entry.get("scores") or []
        importance = max(scores) if scores else 0.0
        entry["importance"] = float(importance)
        sensitivity_ranking.append({"name": name, "importance": float(importance)})

    sensitivity_ranking.sort(key=lambda item: item["importance"])
    for rank, item in enumerate(sensitivity_ranking, start=1):
        layer_name = item["name"]
        layer_sensitivity[layer_name]["rank"] = rank

    best = state_dict.get("best", {})
    autoquant_state_summary = {
        "keys": list(state_dict.keys()),
        "constraints": best.get("constraints"),
        "score": best.get("score"),
        "is_satisfied": best.get("is_satisfied"),
    }

    manifest: Dict[str, Any] = {
        "scheme": asdict(scheme),
        "model": {"id": model_id} if model_id is not None else {},
        "num_quantized_layers": len(layers),
        "layers": layers,
        "autoquant_state": autoquant_state_summary,
        "layer_sensitivity": layer_sensitivity,
        "sensitivity_ranking": sensitivity_ranking,
    }
    return manifest


def write_layer_sensitivity_md(
    layer_sensitivity: Mapping[str, Mapping[str, Any]],
    scheme: AutoQuantSchemeConfig,
    autoquant_state: Mapping[str, Any],
    out_path: Path,
    model_id: Optional[str] = None,
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
    if model_id:
        md_file.new_paragraph(f"**Model:** `{model_id}`")
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
        "- **Size Cost**: Approximate compressed weight size contribution of "
        "the layer under the corresponding recipe(s). Higher values indicate "
        "more memory usage.\n"
        "\n"
        "Note: In the JSON manifest, layer keys may end with "
        "`.quant_recipe` (e.g., `language_model.layers.0.mlp.gate_proj.quant_recipe`). "
        "This suffix is added by ModelOpt to represent the AutoQuant hyperparameter "
        "attached to that module. In this table we strip the `.quant_recipe` suffix "
        "for readability; the underlying module path is the part before that suffix."
    )

    headers = ["Layer", "Num Bits", "Sensitivity", "Size Cost"]
    rows: List[str] = []
    row_entries: List[Tuple[str, List[str], List[float], List[float]]] = []

    for layer_name, entry in layer_sensitivity.items():
        formats = entry.get("formats", [])
        scores = entry.get("scores", [])
        costs = entry.get("costs", [])

        filtered: List[Tuple[str, float, float]] = []
        for fmt, score_value, cost_value in zip(formats, scores, costs):
            if str(fmt).startswith("NONE("):
                continue
            filtered.append((str(fmt), float(score_value), float(cost_value)))

        if not filtered:
            continue

        fmt_values = [fmt for fmt, _, _ in filtered]
        score_values = [score_value for _, score_value, _ in filtered]
        cost_values = [cost_value for _, _, cost_value in filtered]

        row_entries.append((layer_name, fmt_values, score_values, cost_values))

    row_entries.sort(key=lambda item: max(item[2]) if item[2] else 0.0, reverse=True)

    for layer_name, fmt_values, score_values, cost_values in row_entries:
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

        num_bits_str = ", ".join(f"{bits:.1f}" for bits in num_bits_values) if num_bits_values else ""
        scores_str = ", ".join(f"{score_value:.3e}" for score_value in score_values)
        costs_str = ", ".join(f"{cost_value:.3e}" for cost_value in cost_values)

        display_name = layer_name.replace(".quant_recipe", "")

        rows.extend([display_name, num_bits_str, scores_str, costs_str])

    md_file.new_line("")

    md_file.new_table(
        columns=4,
        rows=len(row_entries) + 1,
        text=headers + rows,
        text_align="left",
    )

    md_file.create_md_file()


def write_layer_sensitivity_json(
    manifest: Mapping[str, Any],
    out_path: Path,
) -> None:
    """
    Write a JSON summary of per-layer AutoQuant sensitivity.

    The structure mirrors the Markdown table but is machine-readable:
    - Top-level keys: scheme, model, autoquant_state, layer_sensitivity.
    - Each entry in layer_sensitivity has: layer, num_bits, sensitivity, size_cost.
    """
    scheme = manifest.get("scheme", {})
    model_meta = manifest.get("model", {})
    autoquant_state = manifest.get("autoquant_state", {})
    layer_sensitivity = manifest.get("layer_sensitivity", {})

    rows: List[Dict[str, Any]] = []
    for name, entry in layer_sensitivity.items():
        formats = entry.get("formats") or []
        scores = entry.get("scores") or []
        costs = entry.get("costs") or []

        filtered: List[Tuple[str, float, float]] = []
        for fmt, score, cost in zip(formats, scores, costs):
            fmt_str = str(fmt)
            if fmt_str.startswith("NONE("):
                continue
            filtered.append((fmt_str, float(score), float(cost)))

        if not filtered:
            continue

        num_bits_vals: List[float] = []
        for fmt_str, _, _ in filtered:
            marker = "effective-bits:"
            if marker in fmt_str:
                try:
                    suffix = fmt_str.split(marker, 1)[1]
                    num_str = suffix.split(")", 1)[0].strip()
                    num_bits_vals.append(float(num_str))
                except Exception:
                    continue

        display_name = name.replace(".quant_recipe", "")

        num_bits = num_bits_vals[0] if num_bits_vals else None
        sensitivity = filtered[0][1] if filtered else None
        size_cost = filtered[0][2] if filtered else None

        rows.append(
            {
                "layer": display_name,
                "num_bits": num_bits,
                "sensitivity": sensitivity,
                "size_cost": size_cost,
            }
        )

    rows.sort(key=lambda item: item["sensitivity"] if item["sensitivity"] is not None else 0.0, reverse=True)

    payload: Dict[str, Any] = {
        "scheme": scheme,
        "model": model_meta,
        "autoquant_state": autoquant_state,
        "layer_sensitivity": rows,
    }

    with out_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run ModelOpt AutoQuant all-layer sensitivity for Qwen3-VL-4B-Instruct "
            "and emit a quantization manifest."
        )
    )
    default_model_dir = (
        Path("models")
        / "qwen3_vl_4b_instruct"
        / "checkpoints"
        / "Qwen3-VL-4B-Instruct"
    )
    default_output_dir = Path("tmp") / "qwen3_vl_4b_autoquant_all_layers_fp8"
    default_vlm_calib_db = (
        Path("datasets")
        / "vlm-quantize-calib"
        / "coco2017_vlm_calib_large.db"
    )
    default_coco_root = Path("datasets") / "coco2017" / "source-data"

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
        "--vlm-calib-db",
        type=Path,
        default=default_vlm_calib_db,
        help=(
            "Path to the COCO2017 VLM calibration SQLite DB. Defaults to the "
            "shared large (512-sample) subset."
        ),
    )
    parser.add_argument(
        "--coco-root",
        type=Path,
        default=default_coco_root,
        help=(
            "Root directory of COCO2017 (must contain train2017/, val2017/, "
            "annotations/). Used to resolve image_relpath entries in the "
            "VLM calibration DB."
        ),
    )
    parser.add_argument(
        "--max-calib-samples",
        type=int,
        default=512,
        help="Maximum number of image+caption samples to use for calibration.",
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
        default=1,
        help="Logical calibration batch size (used for score-size normalization).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device to use (default: cuda).",
    )
    parser.add_argument(
        "--quant-format",
        type=str,
        default="fp8",
        choices=["fp8", "int8"],
        help=(
            "Quantization format family to use. "
            "'fp8' selects FP8_ALL_LAYERS_CFG, 'int8' selects INT8_ALL_LAYERS_CFG."
        ),
    )
    parser.add_argument(
        "--effective-bits",
        type=float,
        default=None,
        help=(
            "Optional override for the AutoQuant effective_bits constraint. "
            "Defaults to the scheme's configured value."
        ),
    )
    parser.add_argument(
        "--auto-quantize-score-size",
        type=int,
        default=None,
        help=(
            "Optional override for the AutoQuant score size in samples. "
            "Defaults to the scheme's configured value."
        ),
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
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        scheme = select_autoquant_scheme(args.quant_format)
    except ValueError as exc:  # noqa: BLE001
        print(f"[ERROR] {exc}")
        return 1

    # Apply CLI overrides to the selected scheme in an immutable way.
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
            )
            return 1
        manifest_path = args.output_dir / f"{scheme.name}_quant_manifest.json"
        if not manifest_path.is_file():
            print(
                "[ERROR] Report-only mode requested but manifest JSON not found at: "
                f"{manifest_path}",
            )
            return 1
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(
                f"[ERROR] Failed to read manifest JSON at {manifest_path}: {exc}",
            )
            return 1

        if "layer_sensitivity" not in manifest or "autoquant_state" not in manifest:
            print(
                "[ERROR] Manifest JSON is missing required keys "
                "`layer_sensitivity` or `autoquant_state`.",
            )
            return 1

        sensitivity_md_path = args.output_dir / "per-layer-sensitivity.md"
        sensitivity_json_path = args.output_dir / "per-layer-sensitivity.json"
        model_id = None
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
            "[INFO] Report-only mode: regenerated per-layer sensitivity Markdown at "
            f"{sensitivity_md_path}",
        )
        return 0

    if not args.model_dir.is_dir():
        print(
            f"[ERROR] Model directory not found: {args.model_dir}\n"
            "Hint: run models/qwen3_vl_4b_instruct/bootstrap.sh first.",
        )
        return 1

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        print(
            "[WARN] CUDA is not available; running on CPU will be extremely slow.",
        )

    device = torch.device(args.device)
    torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print(f"[INFO] Loading Qwen3-VL-4B-Instruct from {args.model_dir}")
    model = AutoModelForImageTextToText.from_pretrained(
        str(args.model_dir),
        torch_dtype=torch_dtype,
        device_map=None,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_dir),
        trust_remote_code=True,
    )
    tokenizer.padding_side = "left"
    processor = AutoProcessor.from_pretrained(
        str(args.model_dir),
        trust_remote_code=True,
    )

    print(
        f"[INFO] Building VLM calibration dataloader from {args.vlm_calib_db} "
        f"and COCO root {args.coco_root}",
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

    print(
        "[INFO] Running AutoQuant all-layers scheme "
        f"{scheme.name} for Qwen3-VL-4B-Instruct ..."
    )
    quantized_model, state_dict = autoquant_model(
        model=model,
        scheme=scheme,
        calib_loader=calib_loader,
        batch_size=max(args.batch_size, 1),
        device=device,
        pad_token_id=tokenizer.pad_token_id,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir / f"{scheme.name}_autoquant_state.pt"
    torch.save(state_dict, state_path)

    manifest_path = args.output_dir / f"{scheme.name}_quant_manifest.json"
    print(f"[INFO] Building quantization manifest at {manifest_path}")
    manifest = build_quant_manifest(
        model=quantized_model,
        scheme=scheme,
        state_dict=state_dict,
        model_id=str(args.model_dir),
    )
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    sensitivity_md_path = args.output_dir / "per-layer-sensitivity.md"
    sensitivity_json_path = args.output_dir / "per-layer-sensitivity.json"
    model_id = None
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

    print("[INFO] AutoQuant all-layers FP8 completed successfully.")
    print(f"[INFO] Quantization manifest written to: {manifest_path}")
    print(f"[INFO] AutoQuant state written to: {state_path}")
    print(f"[INFO] Per-layer sensitivity report: {sensitivity_md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
