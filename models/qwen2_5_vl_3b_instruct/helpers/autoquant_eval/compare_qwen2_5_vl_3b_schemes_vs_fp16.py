#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from mdutils import MdUtils
from torchmetrics.text.perplexity import Perplexity
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

from text_eval_common import build_eval_dataloader, upcast_fp8_weights_to_dtype


@dataclass
class SchemeMetrics:
    name: str
    quant_model_dir: str
    perplexity_quant: float
    perplexity_ratio_quant_over_fp16: Optional[float]
    logit_mse_last_token: float
    logit_kl_last_token_fp16_to_quant: float


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Qwen2.5-VL-3B FP16 vs one or more FP8 quantized schemes "
            "using perplexity and logit-based metrics on a shared caption set."
        )
    )
    base_dir = Path("models") / "qwen2_5_vl_3b_instruct"
    default_fp16 = base_dir / "checkpoints" / "Qwen2.5-VL-3B-Instruct"
    default_quant_root = base_dir / "quantized"
    default_quant_dirs = [
        default_quant_root / "fp8_autoquant_all_layers_top10_coco2017",
    ]
    default_captions = (
        Path("datasets") / "vlm-quantize-calib" / "coco2017_captions.txt"
    )
    default_out = Path("tmp") / "modelopt-autoquant-fp8" / "eval-all-layers-schemes"

    parser.add_argument(
        "--fp16-model-dir",
        type=Path,
        default=default_fp16,
        help="Path to the FP16/BF16 base model checkpoint.",
    )
    parser.add_argument(
        "--quant-model-dirs",
        type=Path,
        nargs="+",
        default=default_quant_dirs,
        help=(
            "One or more quantized model checkpoint directories to compare "
            "against FP16 (default: top-10% all-layers AutoQuant scheme)."
        ),
    )
    parser.add_argument(
        "--scheme-names",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Optional human-readable names for each quantized scheme; "
            "defaults to the basename of each quant model directory."
        ),
    )
    parser.add_argument(
        "--captions-path",
        type=Path,
        default=default_captions,
        help="Path to a text file with one caption per line for evaluation.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1024,
        help="Maximum number of evaluation samples to use.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum sequence length for tokenization.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device to use (default: cuda).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=default_out,
        help=(
            "Output directory for aggregated metrics JSON and Markdown summary "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--max-logit-batches",
        type=int,
        default=64,
        help=(
            "Maximum number of batches to use for logit-based metrics. "
            "This applies independently to each model pass."
        ),
    )
    return parser.parse_args(argv)


def _resolve_scheme_names(
    quant_model_dirs: Sequence[Path],
    scheme_names: Optional[Sequence[str]],
) -> List[str]:
    if scheme_names is None:
        return [p.name for p in quant_model_dirs]
    if len(scheme_names) != len(quant_model_dirs):
        raise ValueError(
            "--scheme-names must have the same length as --quant-model-dirs"
        )
    return list(scheme_names)


def _init_sqlite(out_dir: Path) -> sqlite3.Connection:
    db_path = out_dir / "metrics.db"
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS fp16_logits (
            sample_index INTEGER PRIMARY KEY,
            logits       BLOB NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS aggregate_metrics (
            key   TEXT PRIMARY KEY,
            value REAL NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS scheme_metrics (
            scheme_name TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            value       REAL NOT NULL,
            PRIMARY KEY (scheme_name, metric_name)
        )
        """
    )
    conn.commit()
    return conn


def _run_fp16_teacher_pass(
    model: Qwen2_5_VLForConditionalGeneration,
    dataloader: torch.utils.data.DataLoader,
    pad_token_id: Optional[int],
    device: torch.device,
    conn: sqlite3.Connection,
    max_logit_batches: Optional[int],
) -> Tuple[float, int]:
    ignore_index = pad_token_id if pad_token_id is not None else -100
    ppl_metric = Perplexity(ignore_index=ignore_index).to(device)

    cur = conn.cursor()
    sample_index = 0
    batch_count = 0

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            if max_logit_batches is not None and batch_count >= max_logit_batches:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids)
            logits = outputs.logits

            preds = logits[:, :-1, :]
            target = labels[:, 1:]
            ppl_metric.update(preds=preds, target=target)

            last_logits = logits[:, -1, :].detach().to(torch.float16).cpu().numpy()
            for row in last_logits:
                cur.execute(
                    "INSERT OR REPLACE INTO fp16_logits (sample_index, logits) "
                    "VALUES (?, ?)",
                    (sample_index, row.tobytes()),
                )
                sample_index += 1

            conn.commit()
            batch_count += 1

    ppl_fp16 = float(ppl_metric.compute().item())
    cur.execute(
        "INSERT OR REPLACE INTO aggregate_metrics (key, value) VALUES (?, ?)",
        ("perplexity_fp16", ppl_fp16),
    )
    cur.execute(
        "INSERT OR REPLACE INTO aggregate_metrics (key, value) VALUES (?, ?)",
        ("num_samples", float(sample_index)),
    )
    conn.commit()
    return ppl_fp16, sample_index


def _run_quant_pass_for_scheme(
    scheme_name: str,
    quant_model_dir: Path,
    tokenizer: AutoTokenizer,
    captions_path: Path,
    max_samples: int,
    max_length: int,
    batch_size: int,
    device: torch.device,
    conn: sqlite3.Connection,
    max_logit_batches: Optional[int],
) -> SchemeMetrics:
    dataloader = build_eval_dataloader(
        captions_path=captions_path,
        tokenizer=tokenizer,
        max_samples=max_samples,
        max_length=max_length,
        batch_size=batch_size,
    )

    print(f"[INFO] Loading quantized model '{scheme_name}' from {quant_model_dir}")
    quant_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(quant_model_dir)
    )
    upcast_fp8_weights_to_dtype(quant_model, target_dtype=torch.float32)
    quant_model.to(device)

    ignore_index = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100
    ppl_metric = Perplexity(ignore_index=ignore_index).to(device)

    mse_sum = 0.0
    kl_sum = 0.0
    count = 0

    cur = conn.cursor()
    teacher_cursor = conn.cursor()
    teacher_cursor.execute(
        "SELECT sample_index, logits FROM fp16_logits ORDER BY sample_index ASC"
    )
    expected_sample_index = 0
    batch_count = 0

    quant_model.eval()
    with torch.no_grad():
        for batch in dataloader:
            if max_logit_batches is not None and batch_count >= max_logit_batches:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = quant_model(input_ids=input_ids)
            logits_quant = outputs.logits

            preds = logits_quant[:, :-1, :]
            target = labels[:, 1:]
            ppl_metric.update(preds=preds, target=target)

            last_logits_quant = logits_quant[:, -1, :]
            batch_size_actual = last_logits_quant.size(0)

            teacher_logits_batch: List[torch.Tensor] = []
            for _ in range(batch_size_actual):
                row = teacher_cursor.fetchone()
                if row is None:
                    raise RuntimeError(
                        "Insufficient FP16 logits stored for quantized evaluation."
                    )
                sample_index_db, blob = row
                if sample_index_db != expected_sample_index:
                    raise RuntimeError(
                        "FP16 logits sample_index mismatch "
                        f"(expected {expected_sample_index}, got {sample_index_db})."
                    )
                vec = np.frombuffer(blob, dtype=np.float16).astype("float32")
                if vec.shape[0] != last_logits_quant.size(-1):
                    raise RuntimeError(
                        "FP16 logits vocab size mismatch for sample "
                        f"index {sample_index_db}."
                    )
                teacher_logits_batch.append(
                    torch.from_numpy(vec).to(device=device, dtype=last_logits_quant.dtype)
                )
                expected_sample_index += 1

            logits_fp16 = torch.stack(teacher_logits_batch, dim=0)

            mse = torch.mean((last_logits_quant - logits_fp16) ** 2).item()
            log_p = torch.log_softmax(logits_fp16, dim=-1)
            log_q = torch.log_softmax(last_logits_quant, dim=-1)
            kl = torch.sum(torch.exp(log_p) * (log_p - log_q), dim=-1).mean().item()

            mse_sum += mse
            kl_sum += kl
            count += 1
            batch_count += 1

    ppl_quant = float(ppl_metric.compute().item())
    mse_avg = float(mse_sum / max(count, 1))
    kl_avg = float(kl_sum / max(count, 1))

    ratio = None
    cur.execute(
        "SELECT value FROM aggregate_metrics WHERE key = ?", ("perplexity_fp16",)
    )
    row = cur.fetchone()
    if row is not None and row[0] > 0:
        ratio = float(ppl_quant) / float(row[0])

    for metric_name, value in (
        ("perplexity_quant", ppl_quant),
        ("perplexity_ratio_quant_over_fp16", ratio if ratio is not None else float("nan")),
        ("logit_mse_last_token", mse_avg),
        ("logit_kl_last_token_fp16_to_quant", kl_avg),
    ):
        cur.execute(
            """
            INSERT OR REPLACE INTO scheme_metrics (scheme_name, metric_name, value)
            VALUES (?, ?, ?)
            """,
            (scheme_name, metric_name, float(value)),
        )
    conn.commit()

    return SchemeMetrics(
        name=scheme_name,
        quant_model_dir=str(quant_model_dir),
        perplexity_quant=ppl_quant,
        perplexity_ratio_quant_over_fp16=ratio,
        logit_mse_last_token=mse_avg,
        logit_kl_last_token_fp16_to_quant=kl_avg,
    )


def _write_markdown_summary(
    out_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
    ppl_fp16: float,
    scheme_results: Sequence[SchemeMetrics],
) -> None:
    md_path = out_dir / "summary.md"
    md = MdUtils(
        file_name=str(md_path.with_suffix("")),
        title="Qwen2.5-VL-3B FP16 vs FP8 schemes comparison",
    )

    md.new_header(level=2, title="Configuration", add_table_of_contents="n")
    md.new_paragraph(
        "\n".join(
            [
                f"- FP16 model: `{args.fp16_model_dir}`",
                f"- Eval data: `{args.captions_path}`",
                f"- Max samples: `{args.max_samples}`",
                f"- Max length: `{args.max_length}`",
                f"- Batch size: `{args.batch_size}`",
                f"- Device: `{device}`",
            ]
        )
    )

    md.new_header(level=2, title="Baseline", add_table_of_contents="n")
    md.new_paragraph(f"- FP16 perplexity: `{ppl_fp16:.4f}`")

    md.new_header(level=2, title="Schemes", add_table_of_contents="n")

    headers = [
        "Scheme",
        "Quant dir",
        "Perplexity (quant)",
        "PPL ratio (quant/FP16)",
        "Logit MSE (last token)",
        "Logit KL (KL(fp16 || quant))",
    ]
    table_cells: List[str] = list(headers)

    for result in scheme_results:
        ratio_str = (
            f"{result.perplexity_ratio_quant_over_fp16:.4f}"
            if result.perplexity_ratio_quant_over_fp16 is not None
            else "N/A"
        )
        table_cells.extend(
            [
                f"`{result.name}`",
                f"`{result.quant_model_dir}`",
                f"`{result.perplexity_quant:.4f}`",
                f"`{ratio_str}`",
                f"`{result.logit_mse_last_token:.6e}`",
                f"`{result.logit_kl_last_token_fp16_to_quant:.6e}`",
            ]
        )

    md.new_line("")
    md.new_table(
        columns=len(headers),
        rows=len(scheme_results) + 1,
        text=table_cells,
        text_align="left",
    )

    md.create_md_file()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if not args.fp16_model_dir.is_dir():
        raise FileNotFoundError(f"FP16 model directory not found: {args.fp16_model_dir}")
    if not args.captions_path.is_file():
        raise FileNotFoundError(f"Captions file not found: {args.captions_path}")
    for path in args.quant_model_dirs:
        if not path.is_dir():
            raise FileNotFoundError(f"Quantized model directory not found: {path}")

    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    conn = _init_sqlite(args.out_dir)

    scheme_names = _resolve_scheme_names(args.quant_model_dirs, args.scheme_names)

    print(f"[INFO] Loading tokenizer from {args.fp16_model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(args.fp16_model_dir))
    tokenizer.padding_side = "left"

    print(f"[INFO] Building evaluation dataloader from {args.captions_path} (FP16 pass)")
    dataloader_fp16 = build_eval_dataloader(
        captions_path=args.captions_path,
        tokenizer=tokenizer,
        max_samples=args.max_samples,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    print(f"[INFO] Loading FP16/BF16 base model from {args.fp16_model_dir}")
    fp16_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(args.fp16_model_dir)
    ).to(device)

    print("[INFO] Running FP16 teacher pass (perplexity + teacher logits)")
    ppl_fp16, num_samples = _run_fp16_teacher_pass(
        model=fp16_model,
        dataloader=dataloader_fp16,
        pad_token_id=tokenizer.pad_token_id,
        device=device,
        conn=conn,
        max_logit_batches=args.max_logit_batches,
    )
    print(f"[INFO] FP16 perplexity: {ppl_fp16:.4f}")
    print(f"[INFO] Stored FP16 last-token logits for {num_samples} samples.")

    # Free FP16 model from device memory before loading quantized schemes.
    del fp16_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    scheme_results: List[SchemeMetrics] = []

    for name, quant_dir in zip(scheme_names, args.quant_model_dirs):
        print(f"[INFO] Evaluating quantized scheme '{name}'")
        scheme_result = _run_quant_pass_for_scheme(
            scheme_name=name,
            quant_model_dir=quant_dir,
            tokenizer=tokenizer,
            captions_path=args.captions_path,
            max_samples=args.max_samples,
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=device,
            conn=conn,
            max_logit_batches=args.max_logit_batches,
        )
        print(
            f"[INFO] Scheme '{name}': "
            f"ppl={scheme_result.perplexity_quant:.4f}, "
            f"mse={scheme_result.logit_mse_last_token:.6e}, "
            f"kl={scheme_result.logit_kl_last_token_fp16_to_quant:.6e}"
        )
        scheme_results.append(scheme_result)

    metrics: Dict[str, object] = {
        "config": {
            "fp16_model_dir": str(args.fp16_model_dir),
            "captions_path": str(args.captions_path),
            "max_samples": args.max_samples,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "device": str(device),
            "quant_model_dirs": [str(p) for p in args.quant_model_dirs],
            "scheme_names": scheme_names,
        },
        "metrics": {
            "perplexity_fp16": float(ppl_fp16),
        },
        "schemes": {
            result.name: {
                "quant_model_dir": result.quant_model_dir,
                "perplexity_quant": result.perplexity_quant,
                "perplexity_ratio_quant_over_fp16": result.perplexity_ratio_quant_over_fp16,
                "logit_mse_last_token": result.logit_mse_last_token,
                "logit_kl_last_token_fp16_to_quant": result.logit_kl_last_token_fp16_to_quant,
            }
            for result in scheme_results
        },
    }

    json_path = args.out_dir / "metrics.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    _write_markdown_summary(
        out_dir=args.out_dir,
        args=args,
        device=device,
        ppl_fp16=ppl_fp16,
        scheme_results=scheme_results,
    )

    print(f"[INFO] Wrote aggregated metrics JSON to: {json_path}")
    print(f"[INFO] Wrote Markdown summary to: {args.out_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
