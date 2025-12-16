#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

import hydra  # type: ignore[import-untyped]
import torch
from omegaconf import DictConfig

from auto_quantize_model.experiment_layout import resolve_publish_output_dir, resolve_scheme_name
from auto_quantize_model.modelopt_autoquant import AutoQuantSchemeConfig, write_layer_sensitivity_json, write_layer_sensitivity_md
from auto_quantize_model.modelopt_configs import resolve_quant_config
from auto_quantize_model.qwen.autoquant_sensitivity import run_qwen3_vl_lm_autoquant_sensitivity


def _resolve_max_calib_samples(cfg: DictConfig) -> int:
    override = cfg.dataset.get("max_calib_samples")
    if override is not None:
        return int(override)

    size_key = str(cfg.dataset.size)
    size_map = cfg.dataset.get("size_to_max_samples") or {}
    if size_key not in size_map:
        raise ValueError(
            f"Unknown dataset.size {size_key!r}; expected one of {sorted(list(size_map.keys()))}."
        )
    return int(size_map[size_key])


def _resolve_output_dir(cfg: DictConfig) -> Path:
    override = cfg.runner.get("output_dir")
    if override:
        return Path(str(override)).expanduser()

    mode = str(cfg.output_layout.mode)
    if mode == "tmp":
        return Path.cwd()

    if mode == "publish":
        root_dir = Path(str(cfg.output_layout.root_dir)).expanduser()
        pair_override = cfg.quant_pair.get("publish_pair_dir")
        run_override = cfg.quant_pair.get("publish_run_dir")
        return resolve_publish_output_dir(
            root_dir,
            weight=str(cfg.quant_pair.weight),
            activation=str(cfg.quant_pair.activation),
            model_name=str(cfg.model.name),
            quant_pair_name=str(cfg.quant_pair.name),
            dataset_size=str(cfg.dataset.size),
            pair_dir_override=str(pair_override) if pair_override is not None else None,
            run_dir_override=str(run_override) if run_override is not None else None,
        )

    raise ValueError(f"Unsupported output_layout.mode: {mode!r}")


def _build_scheme(cfg: DictConfig) -> AutoQuantSchemeConfig:
    fmt_name = str(cfg.quant_pair.format_name)
    _ = resolve_quant_config(fmt_name)

    scheme_override = cfg.quant_pair.get("scheme_name")
    scheme_name = resolve_scheme_name(
        quant_pair_name=str(cfg.quant_pair.name),
        scheme_name_override=str(scheme_override) if scheme_override is not None else None,
    )

    coverage_mode_override = cfg.quant_pair.get("coverage_mode")
    coverage_mode = "lm_only" if coverage_mode_override is None else str(coverage_mode_override)
    coverage_fraction_override = cfg.quant_pair.get("coverage_fraction")
    coverage_fraction = 1.0 if coverage_fraction_override is None else float(coverage_fraction_override)

    return AutoQuantSchemeConfig(
        name=scheme_name,
        auto_quantize_bits=float(cfg.autoquant.effective_bits),
        auto_quantize_method=str(cfg.autoquant.method),
        auto_quantize_score_size=int(cfg.autoquant.score_size),
        coverage_mode=coverage_mode,
        coverage_fraction=coverage_fraction,
        quant_formats=[fmt_name],
    )


def _load_manifest(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to read manifest JSON at {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise RuntimeError(f"Manifest JSON is not an object: {path}")
    if "layer_sensitivity" not in payload or "autoquant_state" not in payload:
        raise RuntimeError(
            "Manifest JSON is missing required keys `layer_sensitivity` or `autoquant_state`: "
            f"{path}"
        )
    return payload


@hydra.main(config_path="../../conf", config_name="preset/qwen3_lm_sensitivity", version_base=None)
def main(cfg: DictConfig) -> None:
    scheme = _build_scheme(cfg)

    output_dir = _resolve_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / f"{scheme.name}_quant_manifest.json"
    state_path = output_dir / f"{scheme.name}_autoquant_state.pt"
    sensitivity_md_path = output_dir / "per-layer-sensitivity.md"
    sensitivity_json_path = output_dir / "per-layer-sensitivity.json"

    if bool(cfg.quant_pair.get("experimental", False)):
        print(f"[WARN] quant_pair {cfg.quant_pair.name!r} is marked experimental.")

    if bool(cfg.runner.report_only):
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Report-only mode requested but manifest JSON not found: {manifest_path}")

        manifest = _load_manifest(manifest_path)
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
        print(f"[INFO] Report-only mode: regenerated sensitivity artifacts at {output_dir}")
        return

    max_calib_samples = _resolve_max_calib_samples(cfg)

    if not torch.cuda.is_available() and str(cfg.autoquant.device).startswith("cuda"):
        print("[WARN] CUDA is not available; running on CPU will be extremely slow.")

    model_dir = Path(str(cfg.model.path))
    captions_path = Path(str(cfg.dataset.captions_path))

    print(f"[INFO] Running Qwen3-VL LM-only AutoQuant: {scheme.name}")
    manifest, state_dict = run_qwen3_vl_lm_autoquant_sensitivity(
        model_dir=model_dir,
        captions_path=captions_path,
        scheme=scheme,
        max_calib_samples=max_calib_samples,
        calib_seq_len=int(cfg.dataset.calib_seq_len),
        batch_size=int(cfg.autoquant.batch_size),
        device=str(cfg.autoquant.device),
    )

    torch.save(state_dict, state_path)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

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

    print("[INFO] AutoQuant LM-only sensitivity completed.")
    print(f"[INFO] Output dir: {output_dir}")
    print(f"[INFO] Manifest: {manifest_path}")
    print(f"[INFO] State: {state_path}")
    print(f"[INFO] Report: {sensitivity_md_path}")


if __name__ == "__main__":
    main()
