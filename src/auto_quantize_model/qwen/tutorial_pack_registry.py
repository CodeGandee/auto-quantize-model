"""
Model registry for the shared Qwen3-VL tutorial pack runner.

To add a new Qwen3-VL tutorial pack without copy/paste orchestration:

1) Add a new `ModelConfig` entry in `build_model_registry()`.
2) Create a thin tutorial wrapper under `docs/tutorial/howto/<pack>/run_demo.sh`
   that delegates to `auto_quantize_model.qwen.cli_tutorial_pack_runner` with
   `--model-id` pointing at the new entry.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    """Model-specific configuration for running tutorial pack scenarios."""

    model_id: str
    workspace_slug: str
    checkpoint_dir: Path
    bootstrap_script: Path
    all_layers_script: Path
    lm_only_script: Path
    lm_only_model_name: str
    lm_only_model_variant: str
    allowed_quant_pairs: tuple[str, ...]


def build_model_registry(repo_root: Path) -> dict[str, ModelConfig]:
    """Return the built-in model registry for Qwen3-VL tutorial packs."""

    all_layers_script = (
        repo_root
        / "models"
        / "qwen3_vl_4b_instruct"
        / "helpers"
        / "qwen3_vl_4b_autoquant_all_layers"
        / "run_qwen3_vl_4b_autoquant_all_layers.py"
    )
    lm_only_script = repo_root / "scripts" / "qwen" / "qwen3_lm_sensitivity.py"

    models: list[ModelConfig] = [
        ModelConfig(
            model_id="qwen3_vl_4b_instruct",
            workspace_slug="qwen3_vl_4b_layer_sensitivity",
            checkpoint_dir=repo_root
            / "models"
            / "qwen3_vl_4b_instruct"
            / "checkpoints"
            / "Qwen3-VL-4B-Instruct",
            bootstrap_script=repo_root / "models" / "qwen3_vl_4b_instruct" / "bootstrap.sh",
            all_layers_script=all_layers_script,
            lm_only_script=lm_only_script,
            lm_only_model_name="qwen3_vl_4b_instruct",
            lm_only_model_variant="3-vl-4b-instruct",
            allowed_quant_pairs=("wint4_afp16", "wint4_aint8"),
        ),
        ModelConfig(
            model_id="qwen3_vl_8b_instruct",
            workspace_slug="qwen3_vl_8b_layer_sensitivity",
            checkpoint_dir=repo_root
            / "models"
            / "qwen3_vl_8b_instruct"
            / "checkpoints"
            / "Qwen3-VL-8B-Instruct",
            bootstrap_script=repo_root / "models" / "qwen3_vl_8b_instruct" / "bootstrap.sh",
            all_layers_script=all_layers_script,
            lm_only_script=lm_only_script,
            lm_only_model_name="qwen3_vl_8b_instruct",
            lm_only_model_variant="3-vl-8b-instruct",
            allowed_quant_pairs=("wint4_afp16", "wint4_aint8"),
        ),
    ]
    return {model.model_id: model for model in models}

