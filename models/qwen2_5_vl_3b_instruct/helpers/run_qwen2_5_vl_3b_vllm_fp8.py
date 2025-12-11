#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from vllm import LLM, SamplingParams


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run text-only inference on the FP8-quantized "
            "Qwen2.5-VL-3B-Instruct checkpoint using vLLM + ModelOpt."
        )
    )

    default_model_dir = (
        Path("models")
        / "qwen2_5_vl_3b_instruct"
        / "quantized"
        / "fp8_fp8_coco2017"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=default_model_dir,
        help=(
            "Path to the FP8-quantized Qwen2.5-VL-3B-Instruct checkpoint "
            "(default: %(default)s). "
            "Note: as of vLLM 0.10.x, vLLM expects an LM-only FP8 "
            "ModelOpt checkpoint (vision tower in BF16/FP16). "
            "Using a VLM-quantized FP8 checkpoint (e.g. *_vlm) is "
            "experimental and currently not supported; see "
            "models/qwen2_5_vl_3b_instruct/reports/"
            "fp8-vlm-vs-textonly-vllm-compat.md."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum number of new tokens to generate for each prompt.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling parameter.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tmp") / "qwen2_5_vl_3b_vllm_fp8",
        help=(
            "Directory to save prompts and responses "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--prompt",
        dest="prompts",
        action="append",
        help=(
            "Text prompt to run through the model. "
            "Can be specified multiple times. "
            "If omitted, a small default prompt set is used."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    model_dir: Path = args.model_dir
    if not model_dir.is_dir():
        print(
            f"[ERROR] Quantized model directory not found: {model_dir}\n"
            "Hint: run the FP8 PTQ script first, e.g.:\n"
            "  pixi run -e rtx5090 "
            "bash scripts/qwen/quantize_qwen2_5_vl_3b_fp8_coco2017.sh",
            file=sys.stderr,
        )
        return 1

    prompts = args.prompts
    if not prompts:
        prompts = [
            "Write a short haiku about GPUs and quantization.",
            "Explain FP8 quantization to a senior ML engineer "
            "in three sentences.",
            "List three practical benefits of using TensorRT-LLM "
            "with FP8 quantization.",
            "Describe a cute animal in one sentence.",
            "Generate a short conversation between a data scientist "
            "and a GPU about model compression.",
        ]

    print(
        "[INFO] Loading FP8-quantized Qwen2.5-VL-3B-Instruct with vLLM "
        f"from: {model_dir}",
        flush=True,
    )

    llm = LLM(
        model=str(model_dir),
        quantization="modelopt",
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print(f"[INFO] Running vLLM generation on {len(prompts)} prompts ...")
    outputs = llm.generate(prompts, sampling_params)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, output in enumerate(outputs, start=1):
        prompt_text = output.prompt
        generated_text = output.outputs[0].text
        print("\n=== vLLM FP8 inference ===")
        print(f"Prompt: {prompt_text!r}")
        print("Response:")
        print(generated_text)
        print("==========================")

        out_path = out_dir / f"sample_{idx:02d}.txt"
        with out_path.open("w", encoding="utf-8") as f:
            f.write("=== vLLM FP8 inference ===\n")
            f.write("Prompt:\n")
            f.write(prompt_text)
            f.write("\n\n")
            f.write("Response:\n")
            f.write(generated_text)
            f.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
