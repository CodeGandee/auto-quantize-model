#!/usr/bin/env python
"""
Run a Qwen2.5-VL-3B-Instruct multimodal (image + text) forward pass using ONNX Runtime on GPU.

This script composes:
- Vision ONNX:  qwen2_5_vl_3b_vision_672_fp32.onnx
- Text+image ONNX: qwen2_5_vl_3b_text_with_image_fp16.onnx

High-level flow:
1. Load and preprocess an image to 672x672, run the vision ONNX to get vision features.
2. Tokenize a text prompt with the HF tokenizer.
3. Build `input_ids`, `attention_mask`, and `image_embeds`:
   - A contiguous block of positions at the start of the sequence are image tokens.
   - Those positions in `input_ids` are set to `image_token_id`.
   - `image_embeds` is filled with vision features at those positions (zeros elsewhere).
4. Run the text+image ONNX decoder on GPU and print the top-1 token prediction and a small decoded string.

This is a simple demonstration, not a full chat/generation loop.

Run from repo root, for example:

  pixi run -e rtx5090-vllm python models/qwen2_5_vl_3b_instruct/helpers/run_qwen2_5_vl_onnx_multimodal_demo.py \\
      --image path/to/image.jpg \\
      --prompt "Describe the image." \\
      --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Qwen2.5-VL-3B multimodal inference using ONNX Runtime (vision + text_with_image decoders)."
    )
    default_ckpt = (
        Path("models")
        / "qwen2_5_vl_3b_instruct"
        / "checkpoints"
        / "Qwen2.5-VL-3B-Instruct"
    )
    default_onnx_dir = Path("models") / "qwen2_5_vl_3b_instruct" / "onnx"

    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        default=default_ckpt,
        help=f"HF checkpoint directory (for tokenizer/config). Default: {default_ckpt}",
    )
    parser.add_argument(
        "--onnx-dir",
        type=Path,
        default=default_onnx_dir,
        help=f"Directory containing ONNX models. Default: {default_onnx_dir}",
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to an input image file.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe the image.",
        help="Text prompt to condition on.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="ONNX Runtime device to run on (default: cuda).",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=640,
        help=(
            "Maximum total sequence length (image tokens + text tokens). "
            "Must be large enough to hold all image tokens plus the prompt."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=0,
        help=(
            "Optional number of new tokens to greedily generate beyond the prompt. "
            "If 0, only a single forward pass is run and the next-token prediction is printed."
        ),
    )
    return parser.parse_args()


def load_onnx_session(model_path: Path, device: str) -> ort.InferenceSession:
    if device == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(str(model_path), providers=providers)


def preprocess_image_to_672(image_path: Path) -> np.ndarray:
    """Load an image and convert to pixel_values of shape (1, 3, 672, 672) in float32."""
    image = Image.open(image_path).convert("RGB")
    image = image.resize((672, 672))
    image_np = np.array(image).astype("float32") / 255.0  # (H, W, C)
    image_np = np.transpose(image_np, (2, 0, 1))  # (C, H, W)
    image_np = np.expand_dims(image_np, 0)  # (1, C, H, W)
    return image_np


def run_vision_onnx(
    session: ort.InferenceSession,
    pixel_values: np.ndarray,
) -> np.ndarray:
    """Run the vision ONNX model and return vision features as (1, T_img, D)."""
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: pixel_values})
    vision_features = outputs[0]  # expected shape: (T_img, D)
    if vision_features.ndim == 2:
        vision_features = np.expand_dims(vision_features, 0)  # (1, T_img, D)
    return vision_features


def build_multimodal_inputs(
    tokenizer: AutoTokenizer,
    prompt: str,
    image_token_id: int,
    vision_features: np.ndarray,
    max_seq_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (input_ids, attention_mask, image_embeds) for the text+image ONNX decoder.

    Strategy:
    - Place all image tokens at the start of the sequence.
    - Use as many image tokens as fit before the text prompt.
    - Pad text/truncate so that total length <= max_seq_len.
    """
    batch_size, num_image_tokens, hidden_size = vision_features.shape
    assert batch_size == 1, "Demo currently supports batch_size=1 only."

    # Reserve room for image tokens first, then text.
    max_text_tokens = max_seq_len - num_image_tokens
    if max_text_tokens <= 0:
        raise ValueError(
            f"max_seq_len={max_seq_len} is too small for num_image_tokens={num_image_tokens}. "
            "Increase --max-seq-len or reduce image resolution."
        )

    encoded = tokenizer(
        prompt,
        return_tensors="np",
        max_length=max_text_tokens,
        padding=False,
        truncation=True,
    )
    text_input_ids: np.ndarray = encoded["input_ids"]  # (1, L_text)
    text_length = text_input_ids.shape[1]

    total_seq_len = num_image_tokens + text_length
    if total_seq_len > max_seq_len:
        raise ValueError(
            f"Total sequence length {total_seq_len} exceeds max_seq_len={max_seq_len}. "
            "Increase --max-seq-len or shorten the prompt."
        )

    input_ids = np.full((1, total_seq_len), fill_value=image_token_id, dtype=np.int64)
    # Place text tokens after the image token block.
    input_ids[:, num_image_tokens:] = text_input_ids

    attention_mask = np.ones_like(input_ids, dtype=np.int64)

    image_embeds = np.zeros(
        (1, total_seq_len, hidden_size),
        dtype=vision_features.dtype,
    )
    image_embeds[:, :num_image_tokens, :] = vision_features

    return input_ids, attention_mask, image_embeds


def run_decoder_once(
    session: ort.InferenceSession,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    image_embeds: np.ndarray,
) -> np.ndarray:
    """Run a single forward pass of the text+image ONNX decoder and return logits."""
    input_map = {inp.name: None for inp in session.get_inputs()}
    # Align by name; scripts use these canonical names but we guard in case order differs.
    for inp in session.get_inputs():
        if inp.name == "input_ids":
            input_map[inp.name] = input_ids
        elif inp.name == "attention_mask":
            input_map[inp.name] = attention_mask
        elif inp.name == "image_embeds":
            input_map[inp.name] = image_embeds.astype(np.float16)

    outputs = session.run(None, input_map)
    logits = outputs[0]
    return logits


def greedy_generate(
    session: ort.InferenceSession,
    tokenizer: AutoTokenizer,
    image_token_id: int,
    vision_features: np.ndarray,
    prompt: str,
    max_seq_len: int,
    max_new_tokens: int,
) -> str:
    """Simple greedy decoding loop on top of the text+image ONNX decoder."""
    input_ids, attention_mask, image_embeds = build_multimodal_inputs(
        tokenizer=tokenizer,
        prompt=prompt,
        image_token_id=image_token_id,
        vision_features=vision_features,
        max_seq_len=max_seq_len,
    )

    for _ in range(max_new_tokens):
        logits = run_decoder_once(session, input_ids, attention_mask, image_embeds)
        # Take the last token's distribution.
        last_logits = logits[0, -1]
        next_token_id = int(np.argmax(last_logits))

        # Append next token if there is room.
        if input_ids.shape[1] >= max_seq_len:
            break

        input_ids = np.concatenate(
            [input_ids, np.array([[next_token_id]], dtype=np.int64)],
            axis=1,
        )
        attention_mask = np.concatenate(
            [attention_mask, np.ones((1, 1), dtype=np.int64)],
            axis=1,
        )
        # New positions have no image embedding.
        pad_image_embed = np.zeros((1, 1, image_embeds.shape[-1]), dtype=image_embeds.dtype)
        image_embeds = np.concatenate([image_embeds, pad_image_embed], axis=1)

        if next_token_id == tokenizer.eos_token_id:
            break

    decoded = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return decoded


def main() -> int:
    args = parse_args()

    ckpt_dir = args.ckpt_dir
    onnx_dir = args.onnx_dir

    vision_path = onnx_dir / "qwen2_5_vl_3b_vision_672_fp32.onnx"
    decoder_path = onnx_dir / "qwen2_5_vl_3b_text_with_image_fp16.onnx"

    if not vision_path.is_file():
        print(f"Error: vision ONNX model not found at {vision_path}")
        return 1
    if not decoder_path.is_file():
        print(f"Error: text+image ONNX model not found at {decoder_path}")
        return 1

    print(f"[QWEN2.5-VL] Loading tokenizer from {ckpt_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir))

    print(f"[QWEN2.5-VL] Loading ONNX models from {onnx_dir} ...")
    vision_sess = load_onnx_session(vision_path, args.device)
    decoder_sess = load_onnx_session(decoder_path, args.device)

    print(f"[QWEN2.5-VL] Preprocessing image {args.image} ...")
    pixel_values = preprocess_image_to_672(args.image)

    print("[QWEN2.5-VL] Running vision ONNX ...")
    vision_features = run_vision_onnx(vision_sess, pixel_values)
    print(f"[QWEN2.5-VL] Vision features shape: {vision_features.shape}")

    image_token_id = int(getattr(tokenizer, "image_token_id", None) or 151859)
    # If tokenizer does not expose image_token_id, fallback to config from checkpoint via HF.

    if args.max_new_tokens > 0:
        print("[QWEN2.5-VL] Running greedy generation via ONNX decoder ...")
        decoded = greedy_generate(
            session=decoder_sess,
            tokenizer=tokenizer,
            image_token_id=image_token_id,
            vision_features=vision_features,
            prompt=args.prompt,
            max_seq_len=args.max_seq_len,
            max_new_tokens=args.max_new_tokens,
        )
        print("=== Decoded output ===")
        print(decoded)
    else:
        print("[QWEN2.5-VL] Running single-step ONNX decoder forward ...")
        input_ids, attention_mask, image_embeds = build_multimodal_inputs(
            tokenizer=tokenizer,
            prompt=args.prompt,
            image_token_id=image_token_id,
            vision_features=vision_features,
            max_seq_len=args.max_seq_len,
        )
        logits = run_decoder_once(decoder_sess, input_ids, attention_mask, image_embeds)
        last_logits = logits[0, -1]
        next_token_id = int(np.argmax(last_logits))
        next_token = tokenizer.decode([next_token_id], skip_special_tokens=False)

        print("=== Single-step ONNX decoder output ===")
        print(f"Next token id: {next_token_id}")
        print(f"Next token (decoded): {next_token!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

