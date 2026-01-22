"""
CLI frontend for the shared Qwen3-VL tutorial pack runner.

This CLI is invoked by the tutorial packs' `run_demo.sh` wrappers and preserves
the existing user-facing flags:

- `--snapshot-report`
- `--device`
- `--dataset-size`
- `--modes`
- `--quant-pairs`

It additionally requires:

- `--model-id` (selects the tutorial/model configuration)
- `--expected-report-dir` (pack-local `expected_report/` directory)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from auto_quantize_model.qwen.tutorial_pack_runner import (
    TutorialPackRunnerError,
    TutorialPackRunRequest,
    run_tutorial_pack,
    parse_modes_csv,
    parse_quant_pairs_csv,
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""

    parser = argparse.ArgumentParser(
        prog="qwen-tutorial-pack-runner",
        description="Run Qwen3-VL tutorial packs (snapshot or verify) via a shared runner.",
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help="Model registry identifier (e.g., qwen3_vl_4b_instruct).",
    )
    parser.add_argument(
        "--expected-report-dir",
        required=True,
        type=Path,
        help="Path to the tutorial pack's expected_report/ directory.",
    )
    parser.add_argument(
        "--snapshot-report",
        action="store_true",
        help="Refresh expected_report/ summaries (summary-only) instead of verifying.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device selector (default: cuda:0).",
    )
    parser.add_argument(
        "--dataset-size",
        default="medium",
        choices=["small", "medium", "large"],
        help="Dataset preset key (default: medium).",
    )
    parser.add_argument(
        "--modes",
        default="all_layers,lm_only",
        help="Comma-separated modes to run (default: all_layers,lm_only).",
    )
    parser.add_argument(
        "--quant-pairs",
        default="wint4_afp16,wint4_aint8",
        help="Comma-separated quant pairs to run (default: wint4_afp16,wint4_aint8).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    modes = parse_modes_csv(str(args.modes))
    quant_pairs = parse_quant_pairs_csv(str(args.quant_pairs))

    request = TutorialPackRunRequest(
        model_id=str(args.model_id),
        expected_report_dir=Path(args.expected_report_dir),
        snapshot_report=bool(args.snapshot_report),
        device=str(args.device),
        dataset_size=str(args.dataset_size),  # type: ignore[arg-type]
        modes=tuple(modes),
        quant_pairs=tuple(quant_pairs),
    )

    try:
        run_tutorial_pack(request)
    except TutorialPackRunnerError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
