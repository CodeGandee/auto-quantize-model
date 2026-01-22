"""
Tutorial-pack summary builder for Qwen3-VL layer sensitivity demos.

This module builds a deterministic, schema-locked per-scenario summary used by
the tutorial pack verifier (`run_demo.sh`). The summary is intentionally small
and stable across machines: it avoids absolute paths and diffs only the fields
described in `specs/002-revise-qwen3-vl-tutorial/contracts/summary.schema.json`.

Classes
-------
TutorialPackScenarioSummary
    Schema-locked summary used for tutorial verification.

Functions
---------
write_summary_json
    Write stable JSON for verification diffs.
write_summary_md
    Write stable Markdown for human review and diffs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Optional, Sequence, cast

Mode = Literal["all_layers", "lm_only"]
DatasetSize = Literal["small", "medium", "large"]


def _sorted_manifest_keys(manifest: Mapping[str, Any]) -> list[str]:
    return sorted(str(key) for key in manifest.keys())


def _require_mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Manifest key {key!r} must be a JSON object.")
    return cast(Mapping[str, Any], value)


def _require_int(parent: Mapping[str, Any], key: str) -> int:
    value = parent.get(key)
    if isinstance(value, bool):
        raise ValueError(f"Manifest key {key!r} must be an integer (got bool).")
    if isinstance(value, (int, float)):
        as_int = int(value)
        if as_int < 1:
            raise ValueError(f"Manifest key {key!r} must be >= 1 (got {value!r}).")
        return as_int
    raise ValueError(f"Manifest key {key!r} must be an integer (got {type(value).__name__}).")


def _iter_layer_sensitivities(layer_sensitivity: Any) -> Iterable[float]:
    if isinstance(layer_sensitivity, Sequence) and not isinstance(layer_sensitivity, (str, bytes)):
        for item in layer_sensitivity:
            if not isinstance(item, Mapping):
                continue
            value = item.get("sensitivity")
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                yield float(value)
                continue
            try:
                yield float(value)  # type: ignore[arg-type]
            except Exception:
                continue
        return

    if isinstance(layer_sensitivity, Mapping):
        for entry in layer_sensitivity.values():
            if not isinstance(entry, Mapping):
                continue
            formats = entry.get("formats") or []
            scores = entry.get("scores") or []

            filtered_scores: list[float] = []
            if isinstance(formats, Sequence) and isinstance(scores, Sequence):
                for fmt, score in zip(formats, scores):
                    fmt_str = str(fmt)
                    if fmt_str.startswith("NONE("):
                        continue
                    try:
                        filtered_scores.append(float(score))
                    except Exception:
                        continue

            if filtered_scores:
                yield from filtered_scores
                continue

            if isinstance(scores, Sequence):
                for score in scores:
                    try:
                        yield float(score)
                    except Exception:
                        continue


def _has_nonzero_sensitivity(manifest: Mapping[str, Any]) -> bool:
    layer_sensitivity = manifest.get("layer_sensitivity")
    for sensitivity in _iter_layer_sensitivities(layer_sensitivity):
        if float(sensitivity) != 0.0:
            return True
    return False


@dataclass(frozen=True)
class TutorialPackScenarioSummary:
    """Schema-locked summary used for tutorial verification."""

    scenario_id: str
    mode: Mode
    quant_pair: str
    dataset_size: DatasetSize

    dataset_calib_seq_len: int
    dataset_batch_size: int
    dataset_num_calib_batches: int
    dataset_num_calib_samples: int
    dataset_max_calib_samples: int

    auto_quantize_score_size: int

    has_layer_sensitivity: bool
    has_autoquant_state: bool
    has_nonzero_sensitivity: bool

    manifest_keys: list[str]

    scheme_name: Optional[str] = None
    quant_formats: Optional[list[str]] = None

    @classmethod
    def from_manifest(
        cls,
        manifest: Mapping[str, Any],
        *,
        scenario_id: str,
        mode: Mode,
        quant_pair: str,
        dataset_size: DatasetSize,
    ) -> TutorialPackScenarioSummary:
        """Create a stable summary from a manifest JSON object.

        Parameters
        ----------
        manifest:
            Parsed JSON object for a scenario manifest.
        scenario_id:
            Stable scenario identifier (e.g., ``"all_layers/wint4_afp16"``).
        mode:
            Scenario mode (``"all_layers"`` or ``"lm_only"``).
        quant_pair:
            Quant-pair identifier (e.g., ``"wint4_afp16"``).
        dataset_size:
            Dataset preset key (``"small"``, ``"medium"``, or ``"large"``).
        """

        dataset = _require_mapping(manifest, "dataset")
        scheme = manifest.get("scheme")

        scheme_name: Optional[str] = None
        quant_formats: Optional[list[str]] = None
        auto_quantize_score_size: Optional[int] = None
        if isinstance(scheme, Mapping):
            scheme_name = str(scheme.get("name")) if scheme.get("name") is not None else None
            raw_formats = scheme.get("quant_formats")
            if isinstance(raw_formats, Sequence) and not isinstance(raw_formats, (str, bytes)):
                quant_formats = [str(item) for item in raw_formats]
            if scheme.get("auto_quantize_score_size") is not None:
                auto_quantize_score_size = int(scheme["auto_quantize_score_size"])

        if auto_quantize_score_size is None:
            auto_quantize_score_size = _require_int(manifest, "auto_quantize_score_size")

        has_layer_sensitivity = isinstance(manifest.get("layer_sensitivity"), (Mapping, Sequence))
        has_autoquant_state = isinstance(manifest.get("autoquant_state"), Mapping)

        return cls(
            scenario_id=scenario_id,
            mode=mode,
            quant_pair=quant_pair,
            dataset_size=dataset_size,
            dataset_calib_seq_len=_require_int(dataset, "calib_seq_len"),
            dataset_batch_size=_require_int(dataset, "batch_size"),
            dataset_num_calib_batches=_require_int(dataset, "num_calib_batches"),
            dataset_num_calib_samples=_require_int(dataset, "num_calib_samples"),
            dataset_max_calib_samples=_require_int(dataset, "max_calib_samples"),
            auto_quantize_score_size=int(auto_quantize_score_size),
            has_layer_sensitivity=bool(has_layer_sensitivity),
            has_autoquant_state=bool(has_autoquant_state),
            has_nonzero_sensitivity=_has_nonzero_sensitivity(manifest),
            manifest_keys=_sorted_manifest_keys(manifest),
            scheme_name=scheme_name,
            quant_formats=quant_formats,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict with schema keys only."""

        return {
            "scenario_id": self.scenario_id,
            "mode": self.mode,
            "quant_pair": self.quant_pair,
            "dataset_size": self.dataset_size,
            "dataset_calib_seq_len": int(self.dataset_calib_seq_len),
            "dataset_batch_size": int(self.dataset_batch_size),
            "dataset_num_calib_batches": int(self.dataset_num_calib_batches),
            "dataset_num_calib_samples": int(self.dataset_num_calib_samples),
            "dataset_max_calib_samples": int(self.dataset_max_calib_samples),
            "auto_quantize_score_size": int(self.auto_quantize_score_size),
            "scheme_name": self.scheme_name,
            "quant_formats": self.quant_formats,
            "has_layer_sensitivity": bool(self.has_layer_sensitivity),
            "has_autoquant_state": bool(self.has_autoquant_state),
            "has_nonzero_sensitivity": bool(self.has_nonzero_sensitivity),
            "manifest_keys": list(self.manifest_keys),
        }


def write_summary_json(path: Path, summary: TutorialPackScenarioSummary) -> None:
    """Write schema-locked JSON with stable ordering/newlines."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = summary.to_dict()
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")


def write_summary_md(path: Path, summary: TutorialPackScenarioSummary) -> None:
    """Write a stable Markdown table for human review + diffs."""

    payload = summary.to_dict()
    ordered_keys = [
        "scenario_id",
        "mode",
        "quant_pair",
        "dataset_size",
        "dataset_calib_seq_len",
        "dataset_batch_size",
        "dataset_num_calib_batches",
        "dataset_num_calib_samples",
        "dataset_max_calib_samples",
        "auto_quantize_score_size",
        "scheme_name",
        "quant_formats",
        "has_layer_sensitivity",
        "has_autoquant_state",
        "has_nonzero_sensitivity",
        "manifest_keys",
    ]

    lines: list[str] = []
    lines.append("# Tutorial Pack Scenario Summary")
    lines.append("")
    lines.append("| Key | Value |")
    lines.append("|---|---|")
    for key in ordered_keys:
        value = payload.get(key)
        rendered = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
        lines.append(f"| {key} | `{rendered}` |")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")

