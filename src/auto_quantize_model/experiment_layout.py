from __future__ import annotations

from pathlib import Path
from typing import Optional


def resolve_scheme_name(quant_pair_name: str, scheme_name_override: Optional[str] = None) -> str:
    """Resolve the AutoQuant scheme name used for filenames and report titles."""

    if scheme_name_override:
        return str(scheme_name_override)
    return f"{quant_pair_name}_autoquant_lm"


def resolve_pair_dir(weight: str, activation: str, pair_dir_override: Optional[str] = None) -> str:
    """Resolve the `weight-<w>-act-<a>` folder name used for published artifacts."""

    if pair_dir_override:
        return str(pair_dir_override)
    return f"weight-{weight}-act-{activation}"


def resolve_lm_run_dir(
    model_name: str,
    quant_pair_name: str,
    dataset_size: str,
    run_dir_override: Optional[str] = None,
) -> str:
    """Resolve the run directory name for an LM-only sensitivity run."""

    if run_dir_override:
        return str(run_dir_override)
    return f"{model_name}_autoquant_{quant_pair_name}_lm_{dataset_size}"


def resolve_publish_output_dir(
    root_dir: Path,
    *,
    weight: str,
    activation: str,
    model_name: str,
    quant_pair_name: str,
    quant_granularity_name: str,
    dataset_size: str,
    pair_dir_override: Optional[str] = None,
    run_dir_override: Optional[str] = None,
) -> Path:
    """Resolve the output directory for published LM-only sensitivity artifacts."""

    pair_dir = resolve_pair_dir(weight, activation, pair_dir_override=pair_dir_override)
    run_dir = resolve_lm_run_dir(
        model_name=model_name,
        quant_pair_name=quant_pair_name,
        dataset_size=dataset_size,
        run_dir_override=run_dir_override,
    )
    granularity_dir = str(quant_granularity_name)
    return root_dir / pair_dir / granularity_dir / run_dir
