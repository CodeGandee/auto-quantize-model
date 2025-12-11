#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Optional, Sequence

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration


def _find_default_image() -> Optional[Path]:
    """Best-effort search for a sample image in the workspace."""
    candidates = [
        Path("datasets/coco2017/source-data/val2017"),
        Path("datasets/coco2017/source-data/train2017"),
        Path("tmp/yolo10-infer"),
    ]
    for base in candidates:
        if not base.is_dir():
            continue
        for pattern in ("*.jpg", "*.jpeg", "*.png"):
            images = sorted(base.glob(pattern))
            if images:
                return images[0]
    return None


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sanity-check Qwen2.5-VL-3B-Instruct with one text-only and one "
            "image+text query using GPUs when available."
        )
    )
    default_model_dir = (
        Path("models")
        / "qwen2_5_vl_3b_instruct"
        / "checkpoints"
        / "Qwen2.5-VL-3B-Instruct"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=default_model_dir,
        help=(
            "Path to the local Qwen2.5-VL-3B-Instruct snapshot directory "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help=(
            "Transformers device_map argument (default: '%(default)s'). "
            "Use 'auto' to shard across available GPUs (requires accelerate)."
        ),
    )
    parser.add_argument(
        "--text-prompt",
        type=str,
        default="Write a short haiku about GPUs and quantization.",
        help="Prompt for the text-only sanity check.",
    )
    parser.add_argument(
        "--image-path",
        type=Path,
        default=None,
        help=(
            "Path to an image for the image+text sanity check. "
            "If omitted, a sample image will be searched under "
            "datasets/coco2017/source-data or tmp/yolo10-infer."
        ),
    )
    parser.add_argument(
        "--image-prompt",
        type=str,
        default="Describe this image briefly.",
        help="Prompt for the image+text sanity check.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate for each query.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tmp/qwen2_5_vl_3b_sanity"),
        help="Directory to save sanity-check inputs and outputs.",
    )
    return parser.parse_args(argv)


def load_model_and_tokenizer(
    model_dir: Path, device_map: str
) -> tuple[Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor]:
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"Model directory not found: {model_dir} "
            "(run models/qwen2_5_vl_3b_instruct/bootstrap.sh first to create the symlink)."
        )

    if not torch.cuda.is_available():
        print(
            "[WARN] CUDA is not available; running on CPU. This will be very slow "
            "for Qwen2.5-VL-3B. Ensure you are using pixi and have a GPU-enabled "
            "environment.",
            file=sys.stderr,
        )

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"[INFO] Loading Qwen2.5-VL-3B-Instruct from {model_dir} ...", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_dir),
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    processor = AutoProcessor.from_pretrained(str(model_dir))
    return model, tokenizer, processor


def run_text_only_sanity(
    model: Qwen2_5_VLForConditionalGeneration,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    out_dir: Optional[Path],
) -> None:
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    device = model.device if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(text, return_tensors="pt").to(device)

    print("[INFO] Running text-only generation ...", flush=True)
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_text = tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0]

    print("\n=== Text-only sanity check ===")
    print(f"Prompt: {prompt}")
    print("Response:")
    print(generated_text)
    print("=============================\n")

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "text_only.txt"
        with out_path.open("w", encoding="utf-8") as f:
            f.write("=== Text-only sanity check ===\n")
            f.write(f"Prompt:\n{prompt}\n\n")
            f.write("Response:\n")
            f.write(generated_text)
            f.write("\n")


def run_image_text_sanity(
    model: Qwen2_5_VLForConditionalGeneration,
    tokenizer: AutoTokenizer,
    processor: AutoProcessor,
    image_path: Path,
    prompt: str,
    max_new_tokens: int,
    out_dir: Optional[Path],
) -> None:
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(f"[INFO] Using image: {image_path}", flush=True)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
    ).to(device)

    print("[INFO] Running image+text generation ...", flush=True)
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_text = tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0]

    print("\n=== Image+text sanity check ===")
    print(f"Image: {image_path}")
    print(f"Prompt: {prompt}")
    print("Response:")
    print(generated_text)
    print("================================\n")

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "image_text.txt"
        with out_path.open("w", encoding="utf-8") as f:
            f.write("=== Image+text sanity check ===\n")
            f.write(f"Image: {image_path}\n")
            f.write(f"Prompt:\n{prompt}\n\n")
            f.write("Response:\n")
            f.write(generated_text)
            f.write("\n")

        # Also copy the input image into the output directory for inspection.
        try:
            if image_path.is_file():
                image_dest = out_dir / image_path.name
                if image_dest.resolve() != image_path.resolve():
                    shutil.copy2(image_path, image_dest)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[WARN] Failed to copy image to output dir: {exc}",
                file=sys.stderr,
            )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        model, tokenizer, processor = load_model_and_tokenizer(
            args.model_dir, args.device_map
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Failed to load model or tokenizer: {exc}", file=sys.stderr)
        return 1

    # Text-only sanity check.
    run_text_only_sanity(
        model=model,
        tokenizer=tokenizer,
        prompt=args.text_prompt,
        max_new_tokens=args.max_new_tokens,
        out_dir=args.out_dir,
    )

    # Image+text sanity check.
    image_path: Optional[Path] = args.image_path
    if image_path is None:
        image_path = _find_default_image()
        if image_path is None:
            print(
                "[WARN] No image path provided and no default image found under "
                "datasets/coco2017/source-data or tmp/yolo10-infer; "
                "skipping image+text sanity check.",
                file=sys.stderr,
            )
            return 0

    try:
        run_image_text_sanity(
            model=model,
            tokenizer=tokenizer,
            processor=processor,
            image_path=image_path,
            prompt=args.image_prompt,
            max_new_tokens=args.max_new_tokens,
            out_dir=args.out_dir,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Image+text sanity check failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
