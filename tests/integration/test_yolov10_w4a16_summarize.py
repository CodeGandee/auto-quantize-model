from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_summarizer_emits_comparison_markdown(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    fixtures_root = repo_root / "tests" / "integration" / "fixtures" / "yolov10_w4a16"

    run_roots = [
        fixtures_root / "yolo10n" / "baseline" / "seed0",
        fixtures_root / "yolo10n" / "ema" / "seed0",
        fixtures_root / "yolo10n" / "ema-qc" / "seed0",
    ]
    out_path = tmp_path / "summary.md"

    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "cv-models" / "summarize_yolov10_w4a16_qat_validation.py"),
        "--run-roots",
        *[str(p) for p in run_roots],
        "--out-path",
        str(out_path),
    ]
    proc = subprocess.run(cmd, cwd=repo_root, check=True, capture_output=True, text=True)
    assert proc.returncode == 0

    text = out_path.read_text(encoding="utf-8")
    assert "YOLOv10 W4A16 QAT Validation Summary" in text
    assert "`yolo10n`" in text
    assert "`baseline`" in text
    assert "`ema+qc`" in text

