#!/usr/bin/env python
"""
Quantize Qwen2.5-VL-3B-Instruct to FP8 using NVIDIA ModelOpt,
calibrating on COCO2017 image+caption pairs.

This script is a VLM-focused complement to
`quantize_qwen2_5_vl_3b_fp8_coco2017.sh`, which uses text-only
captions. Here we run calibration on full VLM inputs (image + text)
to better exercise the multimodal pathway during quantization.

Important:
  - Community / official FP8 recipes for Qwen2.5-VL (ModelOpt and
    LLM-Compressor) deliberately keep the vision tower in BF16/FP16
    and quantize only the language model. vLLM is currently wired to
    those LM-only FP8 layouts.
  - This script *does* attempt to quantize the vision stack as well
    and is therefore experimental and **not** vLLM-compatible as of
    vLLM 0.10.x. See:
      models/qwen2_5_vl_3b_instruct/reports/fp8-vlm-vs-textonly-vllm-compat.md

The pipeline:
  - Reads (image_relpath, caption) pairs from
      datasets/vlm-quantize-calib/coco2017_vlm_calib.db
  - Resolves images under a COCO root (default:
      datasets/coco2017/source-data)
  - Uses Qwen2.5-VL's processor and `qwen_vl_utils.process_vision_info`
    to build image+text inputs.
  - Applies ModelOpt FP8 quantization (`FP8_DEFAULT_CFG` plus FP8 KV).
  - Exports a unified HF checkpoint under:
      models/qwen2_5_vl_3b_instruct/quantized/fp8_fp8_coco2017_vlm
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoConfig, AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration

import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint


@dataclass
class VlmCalibSample:
    split: str
    image_relpath: str
    caption: str


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Quantize Qwen2.5-VL-3B-Instruct to FP8 using ModelOpt, "
            "calibrating on COCO2017 image+caption pairs."
        )
    )
    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        default=Path("models")
        / "qwen2_5_vl_3b_instruct"
        / "checkpoints"
        / "Qwen2.5-VL-3B-Instruct",
        help=(
            "Path to the base Qwen2.5-VL-3B-Instruct checkpoint directory "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--calib-db",
        type=Path,
        default=Path("datasets")
        / "vlm-quantize-calib"
        / "coco2017_vlm_calib.db",
        help=(
            "SQLite DB built by build_vlm_quantize_calib_coco2017_db.py "
            "containing COCO image/caption calibration samples."
        ),
    )
    parser.add_argument(
        "--coco-root",
        type=Path,
        default=Path("datasets") / "coco2017" / "source-data",
        help=(
            "Root directory of COCO2017 (must contain train2017/, val2017/, "
            "annotations/). Used to resolve image_relpath entries."
        ),
    )
    parser.add_argument(
        "--export-path",
        type=Path,
        default=Path("models")
        / "qwen2_5_vl_3b_instruct"
        / "quantized"
        / "fp8_fp8_coco2017_vlm",
        help="Output directory for the FP8-quantized HF checkpoint.",
    )
    parser.add_argument(
        "--calib-size",
        type=int,
        default=4096,
        help=(
            "Number of COCO image+caption samples to use for calibration "
            "(max; truncated if DB has fewer rows)."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for quantization (default: cuda).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Calibration batch size (number of image+caption pairs per "
            "model.forward call). Keeping this small avoids OOM."
        ),
    )
    parser.add_argument(
        "--kv-cache-fp8",
        action="store_true",
        default=True,
        help=(
            "Enable FP8 KV cache quantization. This mirrors the fp8_fp8 "
            "text-only script (weights+activations+KV all FP8)."
        ),
    )
    return parser.parse_args(argv)


def load_vlm_calib_samples(calib_db: Path, max_samples: int) -> List[VlmCalibSample]:
    if not calib_db.is_file():
        raise FileNotFoundError(f"Calibration DB not found: {calib_db}")

    conn = sqlite3.connect(calib_db)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT split, image_relpath, caption
            FROM vlm_calib_samples
            ORDER BY id ASC
            LIMIT ?
            """,
            (max_samples,),
        )
        rows = cursor.fetchall()
    finally:
        conn.close()

    samples: List[VlmCalibSample] = []
    for split, image_relpath, caption in rows:
        samples.append(
            VlmCalibSample(
                split=str(split),
                image_relpath=str(image_relpath),
                caption=str(caption),
            )
        )
    return samples


def build_calib_dataset(
    samples: Iterable[VlmCalibSample],
    coco_root: Path,
) -> List[Tuple[Path, str]]:
    dataset: List[Tuple[Path, str]] = []
    for sample in samples:
        image_path = coco_root / sample.image_relpath
        dataset.append((image_path, sample.caption))
    return dataset


def create_vlm_forward_loop(
    dataset: List[Tuple[Path, str]],
    tokenizer: AutoTokenizer,
    processor: AutoProcessor,
    device: torch.device,
    batch_size: int,
):
    """
    Create a forward loop that runs image+text batches through the full
    Qwen2.5-VL model. This is used as the `forward_loop` for ModelOpt
    FP8 quantization.
    """

    def forward_loop(model: Qwen2_5_VLForConditionalGeneration) -> None:
        model.eval()
        from math import ceil

        total = len(dataset)
        num_batches = ceil(total / batch_size) if batch_size > 0 else 1

        with torch.inference_mode():
            for b in range(num_batches):
                batch = dataset[b * batch_size : (b + 1) * batch_size]
                if not batch:
                    continue

                texts: List[str] = []
                image_inputs_all: List[dict] = []

                for image_path, caption in batch:
                    if not image_path.is_file():
                        # Skip missing images but keep calibration robust.
                        continue

                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": str(image_path)},
                                {"type": "text", "text": caption},
                            ],
                        }
                    ]
                    text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    image_inputs, _video_inputs = process_vision_info(messages)
                    texts.append(text)
                    image_inputs_all.append(image_inputs)

                if not texts:
                    continue

                # Qwen2.5-VL processor can handle lists of text/images. We pass
                # videos=None here since the calibration dataset contains only
                # static images.
                inputs = processor(
                    text=texts,
                    images=image_inputs_all,
                    videos=None,
                    padding=True,
                    return_tensors="pt",
                ).to(device)

                _ = model(**inputs)

    return forward_loop


def build_fp8_quant_cfg(enable_kv_cache: bool) -> dict:
    """
    Build an FP8 quantization config for ModelOpt, with optional FP8 KV cache.
    This mirrors the `fp8` + `fp8` KV setup used in hf_ptq for text-only.
    """
    quant_cfg = mtq.FP8_DEFAULT_CFG

    if enable_kv_cache:
        # Some installed ModelOpt versions do not expose the helper
        # `update_quant_cfg_with_kv_cache_quant` on the public `mtq`
        # namespace. Re-implement the small helper locally to stay
        # compatible.
        fp8_kv_cfg = mtq.FP8_KV_CFG["quant_cfg"]

        quant_cfg = dict(quant_cfg)  # shallow copy
        quant_cfg["quant_cfg"] = quant_cfg.get("quant_cfg", {"default": {"enable": False}})
        quant_cfg["quant_cfg"].update(fp8_kv_cfg)
        if not quant_cfg.get("algorithm"):
            quant_cfg["algorithm"] = "max"

    return quant_cfg


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        print(
            "[ERROR] CUDA is not available; FP8 quantization for Qwen2.5-VL "
            "requires a GPU.",
            file=sys.stderr,
        )
        return 1

    ckpt_dir = args.ckpt_dir
    if not ckpt_dir.is_dir():
        print(
            f"[ERROR] Qwen2.5-VL-3B-Instruct checkpoint not found at: {ckpt_dir}\n"
            "Hint: run models/qwen2_5_vl_3b_instruct/bootstrap.sh first.",
            file=sys.stderr,
        )
        return 1

    if not args.coco_root.is_dir():
        print(
            f"[ERROR] COCO root directory not found at: {args.coco_root}\n"
            "Hint: ensure datasets/coco2017/source-data is populated.",
            file=sys.stderr,
        )
        return 1

    print(f"[INFO] Using Qwen2.5-VL checkpoint: {ckpt_dir}")
    print(f"[INFO] Using COCO calibration DB: {args.calib_db}")
    print(f"[INFO] Using COCO root for images: {args.coco_root}")
    print(f"[INFO] Calibration samples requested: {args.calib_size}")

    samples = load_vlm_calib_samples(args.calib_db, args.calib_size)
    if not samples:
        print(
            "[ERROR] No calibration samples found in DB. "
            "Make sure coco2017_vlm_calib.db was created.",
            file=sys.stderr,
        )
        return 1

    vlm_dataset = build_calib_dataset(samples, args.coco_root)
    print(f"[INFO] Loaded {len(vlm_dataset)} image+caption calibration samples.")

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print("[INFO] Loading Qwen2.5-VL model, tokenizer, and processor ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(ckpt_dir),
        torch_dtype=dtype,
        device_map=None,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir))

    # For Qwen2.5-VL, AutoProcessor handles image+text inputs.
    processor = AutoProcessor.from_pretrained(str(ckpt_dir))

    forward_loop = create_vlm_forward_loop(
        dataset=vlm_dataset,
        tokenizer=tokenizer,
        processor=processor,
        device=device,
        batch_size=max(args.batch_size, 1),
    )

    print("[INFO] Building FP8 quantization config (with FP8 KV cache).")
    quant_cfg = build_fp8_quant_cfg(enable_kv_cache=args.kv_cache_fp8)

    print("[INFO] Starting ModelOpt FP8 quantization on VLM inputs ...")

    with torch.inference_mode():
        quantized_model = mtq.quantize(
            model,
            quant_cfg,
            forward_loop=forward_loop,
        )

    export_path = args.export_path
    export_path.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Exporting FP8-quantized HF checkpoint to: {export_path}")

    # Save full config / processor for VLMs (mirroring hf_ptq VLM export).
    AutoConfig.from_pretrained(str(ckpt_dir), trust_remote_code=True).save_pretrained(
        export_path
    )
    try:
        AutoProcessor.from_pretrained(str(ckpt_dir), trust_remote_code=True).save_pretrained(
            export_path
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Could not save processor config: {exc}", file=sys.stderr)

    # Export quantized HF checkpoint (weights + quantization metadata).
    export_hf_checkpoint(
        quantized_model,
        export_dir=export_path,
    )

    # Restore tokenizer padding side (if changed) and save tokenizer.
    tokenizer.save_pretrained(export_path)

    print(
        "[INFO] Done. FP8-quantized Qwen2.5-VL-3B-Instruct (VLM-calibrated) "
        f"exported to: {export_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
