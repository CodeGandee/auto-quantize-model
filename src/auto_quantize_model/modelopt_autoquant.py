"""
Shared helpers for NVIDIA ModelOpt AutoQuant workflows.

This module centralizes utilities that were previously duplicated across
model-specific driver scripts (e.g., Qwen AutoQuant sensitivity runners):

- Quantization format resolution via `auto_quantize_model.modelopt_configs`.
- AutoQuant `forward_step` and loss function builders.
- Quantization manifest construction and per-layer sensitivity report writers.

The helpers here are intentionally framework-agnostic (no Hydra) so they can be
used from both CLI scripts and Hydra runners.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from mdutils import MdUtils  # type: ignore[import-untyped]

from auto_quantize_model.modelopt_configs import resolve_quant_config

from modelopt.torch.quantization.utils import is_quantized_linear  # type: ignore[import-untyped]


@dataclass(frozen=True)
class AutoQuantSchemeConfig:
    """Configuration container for a single AutoQuant run."""

    name: str
    auto_quantize_bits: float
    auto_quantize_method: str
    auto_quantize_score_size: int
    coverage_mode: str = "full"
    coverage_fraction: float = 1.0
    quant_formats: List[str] = field(default_factory=list)


def resolve_quantization_formats(format_names: Sequence[str]) -> List[Dict[str, Any]]:
    """Resolve a sequence of ModelOpt format names to config dicts."""

    resolved: List[Dict[str, Any]] = []
    for name in format_names:
        resolved.append(resolve_quant_config(name))
    return resolved


def compute_num_score_steps(score_size: int, batch_size: int, num_batches: int) -> int:
    """Convert AutoQuant score size (in samples) into score steps."""

    safe_batch = max(int(batch_size), 1)
    safe_score_size = max(int(score_size), 1)
    steps = max(safe_score_size // safe_batch, 1)
    return min(steps, max(int(num_batches), 1))


def create_forward_step(auto_quantize_method: str, device: torch.device) -> Callable:
    """Build a forward_step callable for AutoQuant.

    The returned callable places the batch tensors on `device` before calling
    the model.
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


def create_causal_lm_loss_func(
    device: torch.device,
    pad_token_id: Optional[int],
) -> Callable[[Any, Mapping[str, torch.Tensor]], torch.Tensor]:
    """Loss function for models that return `logits` (standard causal LM loss)."""

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


def build_quant_manifest(
    model: torch.nn.Module,
    scheme: AutoQuantSchemeConfig,
    state_dict: Mapping[str, Any],
    model_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a JSON-serializable manifest describing AutoQuant results."""

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
    """Write a Markdown summary of per-layer AutoQuant sensitivity."""

    def _format_effective_bits(value: Any) -> str:
        if value is None:
            return "None"
        try:
            return f"{float(value):.4f}"
        except Exception:
            return str(value)

    def _format_total_score(value: Any) -> str:
        if value is None:
            return "None"
        try:
            return f"{float(value):.6e}"
        except Exception:
            return str(value)

    md_file = MdUtils(
        file_name=str(out_path.with_suffix("")),
        title=f"AutoQuant Layer Sensitivity ({scheme.name})",
    )

    md_file.new_line("")
    md_file.new_line(f"**Scheme:** `{scheme.name}`")
    if model_id is not None:
        md_file.new_line("")
        md_file.new_line(f"**Model:** `{model_id}`")

    constraints = autoquant_state.get("constraints") or {}
    eff_bits = None
    if isinstance(constraints, dict):
        eff_bits = constraints.get("effective_bits")

    score = autoquant_state.get("score")
    is_satisfied = autoquant_state.get("is_satisfied")

    md_file.new_line("")
    md_file.new_line(f"**Effective bits (from search):** `{_format_effective_bits(eff_bits)}`")
    md_file.new_line("")
    md_file.new_line(f"**Total AutoQuant score:** `{_format_total_score(score)}`")
    md_file.new_line("")
    md_file.new_line(f"**Constraint satisfied:** `{is_satisfied}`")

    md_file.new_header(level=2, title="Per-layer sensitivity table", add_table_of_contents="n")
    md_file.new_line("")

    md_file.new_line(
        "- **Layer**: Name of the quant_recipe handle for a group of quantizable modules (e.g., attention or MLP projections)."
    )
    md_file.new_line(
        "- **Num Bits**: Effective number of bits allocated for the quantized recipe(s) considered at this layer."
    )
    md_file.new_line(
        "- **Sensitivity**: AutoQuant sensitivity score for the quantized recipe(s). Higher values indicate that quantizing this layer is more harmful to model quality."
    )
    md_file.new_line(
        "- **Size Cost**: Approximate compressed weight size contribution of the layer under the corresponding recipe(s). Higher values indicate more memory usage."
    )
    md_file.new_line("")
    md_file.new_line(
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
            for fmt, score_value, cost_value in zip(formats, scores, costs):
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
    """Write a JSON summary of per-layer AutoQuant sensitivity."""

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
            for fmt, score, cost in zip(formats, scores, costs):
                filtered.append((str(fmt), float(score), float(cost)))

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
