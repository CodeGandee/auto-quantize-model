"""
CLI wrapper for generating schema-locked tutorial pack summaries.

This script reads a scenario manifest JSON and emits:

- `summary.json`
- `summary.md`

Both outputs are deterministic and used by `run_demo.sh` for verification.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import cast

from auto_quantize_model.qwen.tutorial_pack_summary import (
    DatasetSize,
    Mode,
    TutorialPackScenarioSummary,
    write_summary_json,
    write_summary_md,
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Generate schema-locked tutorial pack summaries from a scenario manifest JSON.",
    )
    parser.add_argument("manifest_json", type=Path, help="Path to the scenario manifest JSON.")
    parser.add_argument("output_dir", type=Path, help="Output directory for summary.json and summary.md.")
    parser.add_argument("--scenario-id", required=True, help="Stable scenario id (e.g., all_layers/wint4_afp16).")
    parser.add_argument("--mode", required=True, choices=["all_layers", "lm_only"], help="Scenario mode.")
    parser.add_argument("--quant-pair", required=True, help="Quant-pair identifier (e.g., wint4_afp16).")
    parser.add_argument(
        "--dataset-size",
        required=True,
        choices=["small", "medium", "large"],
        help="Dataset preset key (small|medium|large).",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    """Generate `summary.json` and `summary.md` from a manifest JSON."""
    args = _parse_args(argv[1:])

    manifest_path: Path = args.manifest_json
    out_dir: Path = args.output_dir
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError("Manifest JSON must be an object.")

    summary = TutorialPackScenarioSummary.from_manifest(
        manifest,
        scenario_id=str(args.scenario_id),
        mode=cast(Mode, str(args.mode)),
        quant_pair=str(args.quant_pair),
        dataset_size=cast(DatasetSize, str(args.dataset_size)),
    )

    write_summary_json(out_dir / "summary.json", summary)
    write_summary_md(out_dir / "summary.md", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
