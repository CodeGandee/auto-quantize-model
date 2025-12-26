"""Config and run-summary helpers for YOLOv10 W4A16 (EMA + QC) validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig


@dataclass(frozen=True)
class ValidationConfig:
    """Resolved config for one validation run (variant + method + profile)."""

    variant: str
    method: str
    profile: str
    coco_root: Path
    run_root: Path
    imgsz: int
    epochs: int
    batch: int
    device: str
    workers: int
    seed: int
    amp: bool


@dataclass(frozen=True)
class RunMetricsSummary:
    """Primary metric values used for stability checks."""

    primary_name: str
    best_value: float
    final_value: float


@dataclass(frozen=True)
class RunStability:
    """Collapse detection outputs."""

    collapse_threshold_ratio: float
    final_over_best_ratio: float
    collapsed: bool


def _normalize_method_group(method: str) -> str:
    if method == "ema+qc":
        return "ema_qc"
    return method


def _quote_override_value(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def compose_validation_cfg(*, config_dir: Path, overrides: dict[str, Any]) -> DictConfig:
    """Hydra-compose the validation config from `config_dir`, applying overrides."""

    config_dir = Path(config_dir).expanduser().resolve()
    if not config_dir.is_dir():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")

    override_list: list[str] = []

    profile = overrides.get("profile")
    if profile is not None:
        override_list.append(f"profile={profile}")

    variant = overrides.get("variant")
    if variant is not None:
        override_list.append(f"variant={variant}")

    method = overrides.get("method")
    if method is not None:
        override_list.append(f"method={_normalize_method_group(str(method))}")

    coco_root = overrides.get("coco_root")
    if coco_root is not None:
        override_list.append(f"dataset.coco_root={_quote_override_value(str(Path(coco_root).expanduser()))}")

    imgsz = overrides.get("imgsz")
    if imgsz is not None:
        override_list.append(f"training.imgsz={int(imgsz)}")

    epochs = overrides.get("epochs")
    if epochs is not None:
        override_list.append(f"training.epochs={int(epochs)}")

    batch = overrides.get("batch")
    if batch is not None:
        override_list.append(f"training.batch={int(batch)}")

    device = overrides.get("device")
    if device is not None:
        override_list.append(f"training.device={_quote_override_value(str(device))}")

    workers = overrides.get("workers")
    if workers is not None:
        override_list.append(f"training.workers={int(workers)}")

    seed = overrides.get("seed")
    if seed is not None:
        override_list.append(f"training.seed={int(seed)}")

    amp = overrides.get("amp")
    if amp is not None:
        override_list.append(f"training.amp={bool(amp)}")

    from hydra import compose, initialize_config_dir  # type: ignore[import-untyped]
    from hydra.core.global_hydra import GlobalHydra  # type: ignore[import-untyped]

    global_hydra = GlobalHydra.instance()
    if global_hydra.is_initialized():
        global_hydra.clear()

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(config_name="config", overrides=override_list)


def load_validation_config(*, config_dir: Path, overrides: dict[str, Any]) -> ValidationConfig:
    """Load and resolve configs from `conf/cv-models/yolov10_w4a16_validation/`."""

    cfg = compose_validation_cfg(config_dir=config_dir, overrides=overrides)

    run_root = overrides.get("run_root")
    if run_root is None:
        raise ValueError("Missing required override: run_root")

    return ValidationConfig(
        variant=str(cfg.model.variant),
        method=str(cfg.method.variant),
        profile=str(overrides.get("profile") or "smoke"),
        coco_root=Path(str(cfg.dataset.coco_root)).expanduser(),
        run_root=Path(run_root).expanduser(),
        imgsz=int(cfg.training.imgsz),
        epochs=int(cfg.training.epochs),
        batch=int(cfg.training.batch),
        device=str(cfg.training.device),
        workers=int(cfg.training.workers),
        seed=int(cfg.training.seed),
        amp=bool(cfg.training.amp),
    )


def write_run_summary_json(*, out_path: Path, payload: dict[str, Any]) -> None:
    """Write `run_summary.json` to disk."""

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _md_escape(value: Any) -> str:
    text = str(value)
    return text.replace("|", "\\|")


def write_run_summary_markdown(*, out_path: Path, payload: dict[str, Any]) -> None:
    """Write a human-readable `summary.md` describing stability and key metrics."""

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model_raw = payload.get("model")
    model: dict[str, Any] = model_raw if isinstance(model_raw, dict) else {}

    method_raw = payload.get("method")
    method: dict[str, Any] = method_raw if isinstance(method_raw, dict) else {}

    dataset_raw = payload.get("dataset")
    dataset: dict[str, Any] = dataset_raw if isinstance(dataset_raw, dict) else {}

    artifacts_raw = payload.get("artifacts")
    artifacts: dict[str, Any] = artifacts_raw if isinstance(artifacts_raw, dict) else {}

    metrics_raw = payload.get("metrics")
    metrics: dict[str, Any] = metrics_raw if isinstance(metrics_raw, dict) else {}

    stability_raw = payload.get("stability")
    stability: dict[str, Any] = stability_raw if isinstance(stability_raw, dict) else {}

    lines: list[str] = []
    lines.append("# YOLOv10 W4A16 QAT Validation Run")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    lines.append(f"| run_id | `{_md_escape(payload.get('run_id'))}` |")
    lines.append(f"| created_at | `{_md_escape(payload.get('created_at'))}` |")
    lines.append(f"| model.variant | `{_md_escape(model.get('variant'))}` |")
    lines.append(f"| method.variant | `{_md_escape(method.get('variant'))}` |")
    lines.append(f"| status | `{_md_escape(payload.get('status'))}` |")
    lines.append("")

    lines.append("## Metrics")
    lines.append("")
    lines.append("| Metric | Best | Final | Final/Best | Collapsed |")
    lines.append("|---|---:|---:|---:|---|")
    primary_name = metrics.get("primary_name")
    best_value = metrics.get("best_value")
    final_value = metrics.get("final_value")
    ratio = stability.get("final_over_best_ratio")
    collapsed = stability.get("collapsed")
    lines.append(
        "| "
        f"`{_md_escape(primary_name)}` | "
        f"{_md_escape(best_value)} | "
        f"{_md_escape(final_value)} | "
        f"{_md_escape(ratio)} | "
        f"`{_md_escape(collapsed)}` |"
    )
    lines.append("")

    lines.append("## Dataset")
    lines.append("")
    lines.append(f"- coco_root: `{_md_escape(dataset.get('coco_root'))}`")
    lines.append(f"- dataset_yaml: `{_md_escape(dataset.get('dataset_yaml'))}`")
    lines.append(f"- train_images: `{_md_escape(dataset.get('train_images'))}`")
    lines.append(f"- val_images: `{_md_escape(dataset.get('val_images'))}`")
    lines.append("")

    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- run_root: `{_md_escape(artifacts.get('run_root'))}`")
    if artifacts.get("results_csv"):
        lines.append(f"- results_csv: `{_md_escape(artifacts.get('results_csv'))}`")
    if artifacts.get("tensorboard_log_dir"):
        lines.append(f"- tensorboard_log_dir: `{_md_escape(artifacts.get('tensorboard_log_dir'))}`")
    lines.append("")

    if payload.get("error"):
        lines.append("## Error")
        lines.append("")
        lines.append("```text")
        lines.append(str(payload.get("error")))
        lines.append("```")
        lines.append("")

    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
