#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI args for summarizing multiple run roots."""

    parser = argparse.ArgumentParser(description="Summarize YOLOv10 W4A16 QAT validation runs into a comparison report.")
    parser.add_argument("--run-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--out-path", type=Path, required=True)
    return parser.parse_args(argv)


def _load_run_summary(run_root: Path) -> dict[str, Any]:
    run_root = Path(run_root)
    path = run_root / "run_summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing run_summary.json under {run_root}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"run_summary.json is not an object: {path}")
    return payload


def _render_comparison_markdown(run_summaries: list[dict[str, Any]]) -> str:
    rows: list[dict[str, Any]] = []
    for summary in run_summaries:
        model_raw = summary.get("model")
        model: dict[str, Any] = model_raw if isinstance(model_raw, dict) else {}

        method_raw = summary.get("method")
        method: dict[str, Any] = method_raw if isinstance(method_raw, dict) else {}

        training_raw = summary.get("training")
        training: dict[str, Any] = training_raw if isinstance(training_raw, dict) else {}

        metrics_raw = summary.get("metrics")
        metrics: dict[str, Any] = metrics_raw if isinstance(metrics_raw, dict) else {}

        stability_raw = summary.get("stability")
        stability: dict[str, Any] = stability_raw if isinstance(stability_raw, dict) else {}

        artifacts_raw = summary.get("artifacts")
        artifacts: dict[str, Any] = artifacts_raw if isinstance(artifacts_raw, dict) else {}

        rows.append(
            {
                "variant": model.get("variant", ""),
                "method": method.get("variant", ""),
                "seed": training.get("seed", ""),
                "status": summary.get("status", ""),
                "collapsed": stability.get("collapsed", ""),
                "best": metrics.get("best_value", ""),
                "final": metrics.get("final_value", ""),
                "ratio": stability.get("final_over_best_ratio", ""),
                "run_root": artifacts.get("run_root", ""),
            }
        )

    rows_sorted = sorted(
        rows,
        key=lambda r: (
            str(r.get("variant", "")),
            str(r.get("method", "")),
            int(r.get("seed", 0)) if str(r.get("seed", "")).isdigit() else 0,
        ),
    )

    lines: list[str] = []
    lines.append("# YOLOv10 W4A16 QAT Validation Summary")
    lines.append("")
    lines.append(f"- Runs: `{len(rows_sorted)}`")
    lines.append("")
    lines.append("| Variant | Method | Seed | Status | Collapsed | Best | Final | Final/Best | Run Root |")
    lines.append("|---|---|---:|---|---|---:|---:|---:|---|")
    for row in rows_sorted:
        lines.append(
            "| "
            f"`{row['variant']}` | "
            f"`{row['method']}` | "
            f"`{row['seed']}` | "
            f"`{row['status']}` | "
            f"`{row['collapsed']}` | "
            f"{row['best']} | "
            f"{row['final']} | "
            f"{row['ratio']} | "
            f"`{row['run_root']}` |"
        )
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    """Read run_summary.json files and write a comparison summary.md.

    Contract: `specs/001-yolov10-qat-validation/contracts/artifacts.md`.
    """

    args = parse_args(argv)
    out_path = Path(args.out_path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    run_summaries: list[dict[str, Any]] = []
    for run_root in args.run_roots:
        run_summaries.append(_load_run_summary(Path(run_root)))

    out_path.write_text(_render_comparison_markdown(run_summaries), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
