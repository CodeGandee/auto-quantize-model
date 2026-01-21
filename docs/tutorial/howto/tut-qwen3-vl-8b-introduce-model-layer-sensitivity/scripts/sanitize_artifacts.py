from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


def _sanitize_str(value: str) -> str:
    if value.startswith("/"):
        return "<ABSOLUTE_PATH>"
    return value


def _sanitize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _sanitize_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_json(v) for v in value]
    if isinstance(value, str):
        return _sanitize_str(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sanitize_text(text: str) -> str:
    # Replace obvious absolute paths without trying to fully parse Markdown/YAML.
    return re.sub(r"(?P<path>/[A-Za-z0-9._\\-~/]+)", "<ABSOLUTE_PATH>", text)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _copy_sanitized_json(src: Path, dst: Path) -> None:
    payload = json.loads(src.read_text(encoding="utf-8"))
    _write_json(dst, _sanitize_json(payload))


def _copy_sanitized_text(src: Path, dst: Path) -> None:
    _write_text(dst, _sanitize_text(src.read_text(encoding="utf-8")))


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(
            "Usage: sanitize_artifacts.py <run_output_dir> <expected_case_dir>",
            file=sys.stderr,
        )
        return 2

    run_dir = Path(argv[1])
    expected_dir = Path(argv[2])

    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")

    expected_dir.mkdir(parents=True, exist_ok=True)

    report_md = run_dir / "layer-sensitivity-report.md"
    report_json = run_dir / "layer-sensitivity-report.json"

    if report_md.is_file():
        _copy_sanitized_text(report_md, expected_dir / "layer-sensitivity-report.md")
    if report_json.is_file():
        _copy_sanitized_json(report_json, expected_dir / "layer-sensitivity-report.json")

    composed_yaml = run_dir / "composed-config.yaml"
    if composed_yaml.is_file():
        _copy_sanitized_text(composed_yaml, expected_dir / "composed-config.yaml")

    for manifest_path in sorted(run_dir.glob("*_quant_manifest.json")):
        _copy_sanitized_json(manifest_path, expected_dir / "quant_manifest.json")
        break

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
