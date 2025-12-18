"""
Shared helpers for NVIDIA ModelOpt AutoQuant workflows.

This module centralizes utilities that were previously duplicated across
model-specific driver scripts (e.g., Qwen AutoQuant sensitivity runners):

- Quantization format resolution via `auto_quantize_model.modelopt_configs`.
- AutoQuant `forward_step` and loss function builders.
- Quantization manifest construction and layer sensitivity report writers.

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

from modelopt.torch.quantization.utils import (  # type: ignore[import-untyped]
    is_quantized,
    is_quantized_linear,
)


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
        if not name:
            continue
        if is_quantized_linear(module) or (
            isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d))
            and is_quantized(module)
        ):
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
    dataset: Optional[Mapping[str, Any]] = None,
    quantization: Optional[Mapping[str, Any]] = None,
    run_config: Optional[Mapping[str, Any]] = None,
) -> None:
    """Write a Markdown summary of AutoQuant layer sensitivity."""

    def _kv_table(title: str, entries: Sequence[tuple[str, str]]) -> None:
        md_file.new_header(level=2, title=title, add_table_of_contents="n")
        md_file.new_line("")
        table_text: List[str] = ["Key", "Value"]
        for key, value in entries:
            table_text.extend([key, value])
        md_file.new_table(
            columns=2,
            rows=len(entries) + 1,
            text=table_text,
            text_align="left",
        )

    def _code_block(language: str, text: str) -> None:
        md_file.new_line("")
        md_file.new_line(f"```{language}")
        for line in text.rstrip().splitlines():
            md_file.new_line(line)
        md_file.new_line("```")

    def _format_json_inline(value: Any) -> str:
        try:
            return json.dumps(value, sort_keys=True)
        except Exception:
            return str(value)

    def _summarize_quantization(meta: Mapping[str, Any]) -> Dict[str, Any]:
        base_format = meta.get("base_format_name") or meta.get("format_name")

        weight_dtype = meta.get("weight_dtype")
        act_dtype = meta.get("act_dtype")
        pair = meta.get("quant_pair")
        if isinstance(pair, Mapping):
            weight_dtype = pair.get("weight", weight_dtype)
            act_dtype = pair.get("activation", act_dtype)

        granularity_name: Optional[str] = None
        overrides: Any = None
        granularity = meta.get("quant_granularity")
        if isinstance(granularity, Mapping):
            granularity_name = granularity.get("name")  # type: ignore[assignment]
            overrides = granularity.get("quant_cfg_overrides")
        else:
            granularity_name = meta.get("granularity") or meta.get("granularity_name") or meta.get("variant_name")
            overrides = meta.get("quant_cfg_overrides")

        summary: Dict[str, Any] = {}
        if base_format is not None:
            summary["base_format_name"] = base_format
        if weight_dtype is not None or act_dtype is not None:
            summary["weight_dtype"] = weight_dtype
            summary["act_dtype"] = act_dtype
        if granularity_name is not None:
            summary["granularity"] = granularity_name
        if overrides is not None:
            summary["quant_cfg_overrides"] = overrides
        return summary

    def _read_composed_yaml(meta: Mapping[str, Any]) -> tuple[str | None, str | None]:
        yaml_text = meta.get("composed_yaml")
        if isinstance(yaml_text, str) and yaml_text.strip():
            return yaml_text, meta.get("composed_yaml_path") if isinstance(meta.get("composed_yaml_path"), str) else None

        path_value = meta.get("composed_yaml_path") or meta.get("composed_config_path") or meta.get("path")
        if not isinstance(path_value, str) or not path_value.strip():
            return None, None

        config_path = Path(path_value)
        if not config_path.is_absolute():
            config_path = out_path.parent / config_path

        try:
            return config_path.read_text(encoding="utf-8"), config_path.name
        except Exception:
            return None, config_path.name

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

    constraints = autoquant_state.get("constraints") or {}
    eff_bits = None
    if isinstance(constraints, dict):
        eff_bits = constraints.get("effective_bits")

    score = autoquant_state.get("score")
    is_satisfied = autoquant_state.get("is_satisfied")

    md_file = MdUtils(
        file_name=str(out_path.with_suffix("")),
        title=f"AutoQuant Layer Sensitivity ({scheme.name})",
    )

    summary_entries: List[tuple[str, str]] = [
        ("Scheme", f"`{scheme.name}`"),
        ("Effective bits (from search)", f"`{_format_effective_bits(eff_bits)}`"),
        ("Total AutoQuant score", f"`{_format_total_score(score)}`"),
        ("Constraint satisfied", f"`{is_satisfied}`"),
    ]
    if model_id is not None:
        summary_entries.insert(1, ("Model", f"`{model_id}`"))

    _kv_table("Summary", summary_entries)

    if dataset:
        dataset_entries: List[tuple[str, str]] = []
        for key, label in (
            ("name", "Name"),
            ("size", "Size"),
            ("root", "Root"),
            ("captions_path", "Captions path"),
            ("image_list", "Image list"),
            ("imgsz", "Image size"),
            ("calib_seq_len", "Calibration seq len"),
            ("batch_size", "Batch size"),
            ("num_calib_batches", "Calibration batches"),
        ):
            value = dataset.get(key)
            if value is not None:
                dataset_entries.append((label, f"`{value}`"))

        max_calib_samples = dataset.get("max_calib_samples")
        num_calib_samples = dataset.get("num_calib_samples")
        if max_calib_samples is not None or num_calib_samples is not None:
            used_display = num_calib_samples if num_calib_samples is not None else "unknown"
            max_display = max_calib_samples if max_calib_samples is not None else "unknown"
            dataset_entries.append(
                ("Calibration samples (used / max)", f"`{used_display}` / `{max_display}`")
            )

        if dataset_entries:
            _kv_table("Dataset", dataset_entries)

    if quantization:
        summary = _summarize_quantization(quantization)
        if summary:
            quant_entries: List[tuple[str, str]] = []
            if "base_format_name" in summary:
                quant_entries.append(("Base format", f"`{summary['base_format_name']}`"))
            if "weight_dtype" in summary or "act_dtype" in summary:
                quant_entries.append(
                    ("Dtypes", f"`W={summary.get('weight_dtype')}` / `A={summary.get('act_dtype')}`")
                )
            if "granularity" in summary:
                quant_entries.append(("Granularity", f"`{summary['granularity']}`"))
            if "quant_cfg_overrides" in summary:
                quant_entries.append(("Quant cfg overrides", "`see below`"))

            _kv_table("Quantization", quant_entries)

            overrides = summary.get("quant_cfg_overrides")
            if isinstance(overrides, Mapping) and overrides:
                _code_block("json", json.dumps(overrides, indent=2, sort_keys=True))

    md_file.new_header(level=2, title="Layer Sensitivity Table", add_table_of_contents="n")
    md_file.new_paragraph(
        "Sorted by sensitivity (descending). Layer names are AutoQuant recipe handles; "
        "a trailing `.quant_recipe` suffix (if present) is stripped for readability."
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

    if run_config:
        yaml_text, yaml_name = _read_composed_yaml(run_config)
        if yaml_text is not None:
            title_suffix = f" (`{yaml_name}`)" if yaml_name else ""
            md_file.new_header(level=2, title=f"Composed Config{title_suffix}", add_table_of_contents="n")
            _code_block("yaml", yaml_text)

    md_file.create_md_file()


def write_layer_sensitivity_json(
    manifest: Mapping[str, Any],
    out_path: Path,
) -> None:
    """Write a JSON summary of AutoQuant layer sensitivity."""

    scheme = manifest.get("scheme", {})
    model_meta = manifest.get("model", {})
    dataset_meta = manifest.get("dataset", {})
    quantization_meta = manifest.get("quantization", {})
    run_config_meta = manifest.get("run_config", {})
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
        "dataset": dataset_meta,
        "quantization": quantization_meta,
        "run_config": run_config_meta,
        "autoquant_state": autoquant_state,
        "layer_sensitivity": rows,
    }

    with out_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
