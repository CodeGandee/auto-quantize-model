#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from mdutils import MdUtils
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration

from text_eval_common import upcast_fp8_weights_to_dtype


@dataclass
class VlmEvalSample:
    split: str
    image_relpath: str
    caption: str


@dataclass
class VlmEvalConfig:
    fp16_model_dir: str
    quant_model_dir: str
    eval_jsonl: str
    coco_root: str
    max_samples: int
    batch_size: int
    device: str


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Qwen2.5-VL-3B FP16 vs a 10-percent-split FP8 quantized "
            "model on a small COCO2017 VLM eval subset using logit-level metrics."
        )
    )
    base_dir = Path("models") / "qwen2_5_vl_3b_instruct"
    default_fp16 = base_dir / "checkpoints" / "Qwen2.5-VL-3B-Instruct"
    default_quant = (
        base_dir / "quantized" / "fp8_autoquant_all_layers_top10_coco2017"
    )
    default_eval_jsonl = (
        Path("datasets")
        / "vlm-quantize-calib"
        / "coco2017_vlm_eval_100.jsonl"
    )
    default_coco_root = Path("datasets") / "coco2017" / "source-data"
    default_out = (
        Path("tmp")
        / "modelopt-autoquant-fp8"
        / "eval-all-layers-top10-vlm"
    )

    parser.add_argument(
        "--fp16-model-dir",
        type=Path,
        default=default_fp16,
        help="Path to the FP16/BF16 base model checkpoint.",
    )
    parser.add_argument(
        "--quant-model-dir",
        type=Path,
        default=default_quant,
        help=(
            "Path to the 10-percent split quantized model checkpoint "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--eval-jsonl",
        type=Path,
        default=default_eval_jsonl,
        help=(
            "Path to the COCO2017 VLM eval JSONL file built by "
            "build_coco2017_vlm_eval_subset.py."
        ),
    )
    parser.add_argument(
        "--coco-root",
        type=Path,
        default=default_coco_root,
        help=(
            "Root directory of COCO2017 (must contain train2017/, val2017/, "
            "annotations/). Used to resolve image_relpath entries."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Maximum number of (image, caption) pairs to use from the eval JSONL.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for VLM evaluation.",
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
            "Output directory for metrics JSON and Markdown summary "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help=(
            "Maximum number of batches to use for logit-based metrics "
            "(to keep runtime manageable). If None, use all batches."
        ),
    )
    return parser.parse_args(argv)


def load_vlm_eval_samples(
    eval_jsonl: Path,
    max_samples: int,
) -> List[VlmEvalSample]:
    if not eval_jsonl.is_file():
        raise FileNotFoundError(f"Eval JSONL not found: {eval_jsonl}")

    samples: List[VlmEvalSample] = []
    with eval_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            samples.append(
                VlmEvalSample(
                    split=str(data.get("split", "")),
                    image_relpath=str(data["image_relpath"]),
                    caption=str(data["caption"]),
                )
            )
            if len(samples) >= max_samples:
                break

    if not samples:
        raise RuntimeError(f"No samples found in eval JSONL: {eval_jsonl}")
    return samples


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
        CREATE TABLE IF NOT EXISTS vlm_metrics (
            key   TEXT PRIMARY KEY,
            value REAL NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def _run_vlm_fp16_teacher_pass(
    model: Qwen2_5_VLForConditionalGeneration,
    samples: Sequence[VlmEvalSample],
    tokenizer: AutoTokenizer,
    processor: AutoProcessor,
    coco_root: Path,
    device: torch.device,
    batch_size: int,
    max_batches: Optional[int] = None,
) -> Tuple[int, int]:
    model.eval()

    conn = _run_vlm_fp16_teacher_pass.conn  # type: ignore[attr-defined]
    cur = conn.cursor()

    sample_index = 0
    num_batches = 0

    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            if max_batches is not None and num_batches >= max_batches:
                break

            batch_samples = samples[start : start + batch_size]
            texts: List[str] = []
            image_inputs_all: List[dict] = []

            for sample in batch_samples:
                image_path = coco_root / sample.image_relpath
                if not image_path.is_file():
                    print(f"[WARN] Image not found, skipping: {image_path}")
                    continue

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_path.as_posix()},
                            {"type": "text", "text": sample.caption},
                        ],
                    }
                ]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                image_inputs, _video_inputs = process_vision_info(messages)
                texts.append(text)
                image_inputs_all.append(image_inputs)

            if not texts:
                continue

            inputs = processor(
                text=texts,
                images=image_inputs_all,
                videos=None,
                padding=True,
                return_tensors="pt",
            ).to(device)

            out_fp16 = model(**inputs)
            logits_fp16 = out_fp16.logits[:, -1, :].detach().to(torch.float16).cpu().numpy()

            for row in logits_fp16:
                cur.execute(
                    "INSERT OR REPLACE INTO fp16_logits (sample_index, logits) VALUES (?, ?)",
                    (sample_index, row.tobytes()),
                )
                sample_index += 1
            conn.commit()

            num_batches += 1

    return sample_index, num_batches


def _run_vlm_quant_pass(
    model: Qwen2_5_VLForConditionalGeneration,
    samples: Sequence[VlmEvalSample],
    tokenizer: AutoTokenizer,
    processor: AutoProcessor,
    coco_root: Path,
    device: torch.device,
    batch_size: int,
    conn: sqlite3.Connection,
    max_batches: Optional[int] = None,
) -> Tuple[float, float, int]:
    model.eval()

    mse_sum = 0.0
    kl_sum = 0.0
    count = 0
    num_batches = 0

    teacher_cursor = conn.cursor()
    teacher_cursor.execute(
        "SELECT sample_index, logits FROM fp16_logits ORDER BY sample_index ASC"
    )
    expected_sample_index = 0

    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            if max_batches is not None and num_batches >= max_batches:
                break

            batch_samples = samples[start : start + batch_size]
            texts: List[str] = []
            image_inputs_all: List[dict] = []

            for sample in batch_samples:
                image_path = coco_root / sample.image_relpath
                if not image_path.is_file():
                    print(f"[WARN] Image not found, skipping: {image_path}")
                    continue

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_path.as_posix()},
                            {"type": "text", "text": sample.caption},
                        ],
                    }
                ]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                image_inputs, _video_inputs = process_vision_info(messages)
                texts.append(text)
                image_inputs_all.append(image_inputs)

            if not texts:
                continue

            inputs = processor(
                text=texts,
                images=image_inputs_all,
                videos=None,
                padding=True,
                return_tensors="pt",
            ).to(device)

            out_quant = model(**inputs)
            logits_quant = out_quant.logits[:, -1, :]

            batch_size_actual = logits_quant.size(0)
            teacher_logits_batch: List[torch.Tensor] = []
            for _ in range(batch_size_actual):
                row = teacher_cursor.fetchone()
                if row is None:
                    raise RuntimeError(
                        "Insufficient FP16 logits stored for VLM quantized evaluation."
                    )
                sample_index_db, blob = row
                if sample_index_db != expected_sample_index:
                    raise RuntimeError(
                        "FP16 logits sample_index mismatch "
                        f"(expected {expected_sample_index}, got {sample_index_db})."
                    )
                vec = np.frombuffer(blob, dtype=np.float16).astype("float32")
                if vec.shape[0] != logits_quant.size(-1):
                    raise RuntimeError(
                        "FP16 logits vocab size mismatch for sample "
                        f"index {sample_index_db}."
                    )
                teacher_logits_batch.append(
                    torch.from_numpy(vec).to(device=device, dtype=logits_quant.dtype)
                )
                expected_sample_index += 1

            logits_fp16 = torch.stack(teacher_logits_batch, dim=0)

            mse = torch.mean((logits_quant - logits_fp16) ** 2).item()
            log_p = torch.log_softmax(logits_fp16, dim=-1)
            log_q = torch.log_softmax(logits_quant, dim=-1)
            kl = torch.sum(torch.exp(log_p) * (log_p - log_q), dim=-1).mean().item()

            mse_sum += mse
            kl_sum += kl
            count += 1
            num_batches += 1

    if count == 0:
        return 0.0, 0.0, 0

    mse_avg = float(mse_sum / count)
    kl_avg = float(kl_sum / count)

    cur = conn.cursor()
    for key, value in (
        ("logit_mse_last_token", mse_avg),
        ("logit_kl_last_token_fp16_to_quant", kl_avg),
        ("vlm_num_batches", float(num_batches)),
    ):
        cur.execute(
            "INSERT OR REPLACE INTO vlm_metrics (key, value) VALUES (?, ?)",
            (key, float(value)),
        )
    conn.commit()

    return mse_avg, kl_avg, num_batches


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if not args.fp16_model_dir.is_dir():
        raise FileNotFoundError(f"FP16 model directory not found: {args.fp16_model_dir}")
    if not args.quant_model_dir.is_dir():
        raise FileNotFoundError(
            f"Quantized model directory not found: {args.quant_model_dir}"
        )
    if not args.eval_jsonl.is_file():
        raise FileNotFoundError(f"Eval JSONL file not found: {args.eval_jsonl}")
    if not args.coco_root.is_dir():
        raise FileNotFoundError(f"COCO root directory not found: {args.coco_root}")

    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    conn = _init_sqlite(args.out_dir)

    print(f"[INFO] Loading tokenizer from {args.fp16_model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(args.fp16_model_dir))
    tokenizer.padding_side = "left"

    print(f"[INFO] Loading processor from {args.fp16_model_dir}")
    processor = AutoProcessor.from_pretrained(str(args.fp16_model_dir))

    print(f"[INFO] Loading VLM eval samples from {args.eval_jsonl}")
    samples = load_vlm_eval_samples(
        eval_jsonl=args.eval_jsonl,
        max_samples=args.max_samples,
    )
    print(f"[INFO] Loaded {len(samples)} eval samples.")

    print(f"[INFO] Loading FP16/BF16 base model from {args.fp16_model_dir}")
    fp16_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(args.fp16_model_dir)
    ).to(device)

    # Attach the SQLite connection to the function so it can be accessed
    # without threading it through every call.
    _run_vlm_fp16_teacher_pass.conn = conn  # type: ignore[attr-defined]

    print("[INFO] Running FP16 teacher pass for VLM eval")
    num_samples_with_logits, num_batches_fp16 = _run_vlm_fp16_teacher_pass(
        model=fp16_model,
        samples=samples,
        tokenizer=tokenizer,
        processor=processor,
        coco_root=args.coco_root,
        device=device,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
    )
    print(
        f"[INFO] Stored FP16 last-token logits for "
        f"{num_samples_with_logits} samples across {num_batches_fp16} batches."
    )

    del fp16_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"[INFO] Loading 10-percent split quantized model from {args.quant_model_dir}")
    quant_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(args.quant_model_dir)
    )
    upcast_fp8_weights_to_dtype(quant_model, target_dtype=torch.float32)
    quant_model.to(device)

    eval_config = VlmEvalConfig(
        fp16_model_dir=str(args.fp16_model_dir),
        quant_model_dir=str(args.quant_model_dir),
        eval_jsonl=str(args.eval_jsonl),
        coco_root=str(args.coco_root),
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        device=str(device),
    )

    print("[INFO] Computing VLM logit MSE and KL divergence")
    mse, kl, num_batches = _run_vlm_quant_pass(
        model=quant_model,
        samples=samples,
        tokenizer=tokenizer,
        processor=processor,
        coco_root=args.coco_root,
        device=device,
        batch_size=args.batch_size,
        conn=conn,
        max_batches=args.max_batches,
    )
    print(f"[INFO] Used {num_batches} batches for metrics.")
    print(f"[INFO] VLM logit MSE (last token): {mse:.6e}")
    print(
        "[INFO] VLM logit KL (last token, KL(fp16 || quant)): "
        f"{kl:.6e}"
    )

    metrics = {
        "config": asdict(eval_config),
        "metrics": {
            "logit_mse_last_token": mse,
            "logit_kl_last_token_fp16_to_quant": kl,
            "num_batches": num_batches,
            "num_samples": len(samples),
        },
    }

    json_path = args.out_dir / "metrics.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    md_path = args.out_dir / "summary.md"
    md = MdUtils(
        file_name=str(md_path.with_suffix("")),
        title="Qwen2.5-VL-3B FP16 vs 10%-split FP8 VLM eval",
    )

    md.new_header(level=2, title="Configuration", add_table_of_contents="n")
    md.new_paragraph(
        "\n".join(
            [
                f"- FP16 model: `{args.fp16_model_dir}`",
                f"- Quantized model: `{args.quant_model_dir}`",
                f"- Eval JSONL: `{args.eval_jsonl}`",
                f"- COCO root: `{args.coco_root}`",
                f"- Max samples: `{args.max_samples}`",
                f"- Batch size: `{args.batch_size}`",
                f"- Device: `{device}`",
            ]
        )
    )

    md.new_header(level=2, title="Metrics", add_table_of_contents="n")
    md.new_paragraph(
        "\n".join(
            [
                f"- Num samples (loaded): `{len(samples)}`",
                f"- Num batches (used): `{num_batches}`",
                f"- VLM logit MSE (last token): `{mse:.6e}`",
                "- VLM logit KL (last token, KL(fp16 || quant)): "
                f"`{kl:.6e}`",
            ]
        )
    )

    md.create_md_file()

    print(f"[INFO] Wrote VLM metrics JSON to: {json_path}")
    print(f"[INFO] Wrote VLM Markdown summary to: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
