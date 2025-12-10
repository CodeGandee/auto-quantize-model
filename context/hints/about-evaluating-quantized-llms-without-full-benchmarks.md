# About evaluating quantized LLMs without full benchmarks

This note summarizes practical ways to compare a quantized Large Language Model (LLM) to its original FP16/BF16 version without running full-scale benchmarks like MMLU, C-EVAL, etc. The focus is on inexpensive, repeatable metrics you can compute on a modest calibration or eval corpus.

The ideas here are consistent with practices discussed in recent quantization papers and guides, for example:
- Evaluating Quantized Large Language Models (arXiv:2402.18158): https://arxiv.org/abs/2402.18158
- A Comprehensive Evaluation of Quantization Strategies for Large Language Models (Findings of ACL 2024): https://aclanthology.org/2024.findings-acl.726.pdf
- NVIDIA ModelOpt docs and examples (HF PTQ / AutoQuant): https://github.com/NVIDIA/TensorRT-Model-Optimizer

## 1. Use a shared evaluation corpus

Pick a small but representative text corpus that both the FP16 and quantized models can process:
- You can usually reuse your calibration data (e.g., Wikipedia paragraphs, COCO captions, web text).
- Aim for a few thousand sequences rather than millions; 1–10k examples is often enough for relative comparisons.
- Fix tokenization, max sequence length, and sampling rules so that both models see identical token inputs.

For VLMs, you can do the same with a modest set of image+caption or mixed multimodal prompts, but most metrics below are easiest to implement for text-only.

## 2. Perplexity on a held-out text set

> **What is perplexity?**  
> Perplexity is a standard language modeling metric that measures how “surprised” a model is by a sequence of tokens. For a causal LM and a token sequence `X = (x₁, x₂, ..., x_T)`, the model factorizes the sequence probability as  
> `P(X) = ∏_{k=1..T} P(x_k | x₁, ..., x_{k-1})`, i.e., it predicts each token `x_k` given the prefix `x₁...x_{k-1}` (teacher forcing). The average negative log-likelihood over tokens is  
> `NLL = -(1/T) * Σ_k log P(x_k | x₁...x_{k-1})`, and perplexity is defined as  
> `perplexity = exp(NLL)`.  
> Lower perplexity means the model assigns higher probability to the observed text (it is less surprised). When comparing FP16 vs a quantized model on the same evaluation corpus, an increase in perplexity indicates that the quantized model’s predictions have drifted away from the original model or the data distribution.

Even though perplexity is a “language modeling” metric, you can compute it on a relatively small corpus as a quick proxy for quantization loss:
- Procedure:
  - Run both FP16 and quantized models in teacher-forcing mode on the same tokenized sequences.
  - Compute negative log-likelihood for each token, then perplexity.
  - Compare:
    - Perplexity of FP16 vs quantized.
    - Per-token or per-sequence NLL deltas (e.g., mean NLL difference).
- Interpretation:
  - A small perplexity increase (e.g., <5–10% relative) usually indicates mild degradation.
  - Very large deltas often correlate with quality loss on downstream tasks.

Minimal sketch (PyTorch / HF style, not full code):

```python
def compute_perplexity(model, tokenizer, texts, max_length=512, device="cuda"):
    model.eval().to(device)
    total_neg_log_likelihood = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(device)
            input_ids = enc["input_ids"]
            labels = input_ids.clone()
            out = model(input_ids=input_ids, labels=labels)
            # Hugging Face causal LMs usually expose loss already averaged over tokens
            loss = out.loss
            num_tokens = input_ids.numel()
            total_neg_log_likelihood += float(loss) * num_tokens
            total_tokens += num_tokens

    avg_nll = total_neg_log_likelihood / max(total_tokens, 1)
    perplexity = math.exp(avg_nll)
    return perplexity
```

You can run this once with the FP16 model and once with the quantized model, then log `ppl_fp16`, `ppl_quant`, and their ratio.

## 3. Logit-level comparison (MSE / KL on next-token distribution)

Instead of or in addition to perplexity, you can compare the raw next-token distributions for the two models on the same inputs:
- For each sequence in your eval corpus:
  - Run FP16 model to get logits `logits_fp16` (shape `[batch, seq, vocab]`).
  - Run quantized model to get `logits_quant`.
  - Restrict to the last token or a prefix of tokens if you want to reduce cost.
- Metrics:
  - Logit MSE: `mean((logits_quant - logits_fp16) ** 2)`
  - Logit MAE: `mean(|logits_quant - logits_fp16|)`
  - Distribution KL divergence:
    - `p = softmax(logits_fp16 / T)` and `q = softmax(logits_quant / T)` for some temperature `T` (often `T=1`).
    - Compute `KL(p || q)` averaged over positions and examples.

Example sketch:

```python
def compute_logit_metrics(model_fp16, model_q, tokenizer, texts, max_length=512, device="cuda"):
    model_fp16.eval().to(device)
    model_q.eval().to(device)
    mse_sum = 0.0
    kl_sum = 0.0
    count = 0

    with torch.no_grad():
        for text in texts:
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(device)

            out_fp16 = model_fp16(**enc)
            out_q = model_q(**enc)

            # Compare only last-token logits for simplicity
            logits_fp16 = out_fp16.logits[:, -1, :]  # [1, vocab]
            logits_q = out_q.logits[:, -1, :]

            mse = torch.mean((logits_q - logits_fp16) ** 2).item()

            p = torch.log_softmax(logits_fp16, dim=-1)
            q = torch.log_softmax(logits_q, dim=-1)
            kl = torch.sum(torch.exp(p) * (p - q), dim=-1).mean().item()

            mse_sum += mse
            kl_sum += kl
            count += 1

    return {
        "logit_mse": mse_sum / max(count, 1),
        "logit_kl": kl_sum / max(count, 1),
    }
```

These metrics tell you how close the quantized model’s next-token beliefs are to the FP16 model, without needing any labeled tasks.

## 4. Hidden-state / embedding similarity

You can also compare intermediate representations:
- Capture hidden states from one or more transformer layers using forward hooks or HF’s `output_hidden_states=True`.
- For each layer and position (or pooled embedding), compute:
  - Cosine similarity between FP16 and quantized embeddings.
  - L2 / L∞ norm of the difference, optionally normalized by FP16 norm.
- Summarize per layer:
  - Mean and standard deviation of cosine similarity.
  - Percentiles (e.g., 5th/50th/95th) to see worst-case deviations.

This is useful for diagnosing which layers are most sensitive to quantization and correlates well with layer-level AutoQuant sensitivity statistics.

Minimal hook-based sketch:

```python
def collect_hidden_states(model, tokenizer, texts, layer_idx, max_length=512, device="cuda"):
    model.eval().to(device)
    activations = []

    def hook(module, inp, out):
        activations.append(out.detach().cpu())

    handle = model.model.layers[layer_idx].register_forward_hook(hook)
    try:
        with torch.no_grad():
            for text in texts:
                enc = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                ).to(device)
                _ = model(**enc)
    finally:
        handle.remove()

    return activations
```

You can run this for FP16 and quantized models, then compare the resulting activation tensors with cosine similarity or norm metrics.

## 5. Small probe tasks and curated prompts

Instead of full benchmarks, you can build a small, curated test set:
- A few dozen or hundred prompts that reflect your intended use (e.g., instruction-following tasks, simple QA, coding snippets).
- For each prompt, compare:
  - Model outputs qualitatively (side-by-side inspection).
  - Optional automatic scores:
    - BLEU / ROUGE against a reference if you have one.
    - Simple heuristic checks (e.g., whether an answer contains the correct numerical result).
- This is more manual but often very informative for “does this feel obviously worse?” checks.

For VLMs, you can do the same with a small set of images and prompts (e.g., COCO val images + captions or questions).

## 6. Putting it together for layer-split experiments

When you have multiple quantized variants (e.g., top-10%, 20%, …, 100% of layers quantized by sensitivity), a practical evaluation loop is:
- Fix:
  - A shared evaluation corpus (e.g., the same captions or text dataset used for calibration, plus a small set of curated prompts).
  - A reference FP16/BF16 model.
- For each quantized scheme:
  1. Compute perplexity on the eval text set.
  2. Compute logit MSE and/or KL divergence on the same inputs.
  3. Optionally sample a subset of prompts and compare outputs qualitatively.
- Plot or tabulate:
  - Perplexity vs. fraction of layers quantized.
  - Logit MSE / KL vs. fraction of layers quantized.
  - Any simple task-level scores you have.

This gives you a compact picture of how quantization aggressiveness (e.g., 10% vs 50% vs 100% of selected layers quantized) affects model behavior, without running large external benchmarks.

## 7. Helpful libraries and tools

You do not have to implement all metrics from scratch. A few libraries are commonly used for perplexity and related comparisons:

- **TorchMetrics (PyTorch)**  
  - Provides a `Perplexity` metric out of the box for language modeling: https://lightning.ai/docs/torchmetrics/stable/text/perplexity.html and https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/functional/text/perplexity.py  
  - You can feed logits or probabilities and targets to compute perplexity directly, which is convenient if you already have a training/eval loop in PyTorch.

- **Hugging Face `evaluate` library**  
  - Includes a `perplexity` metric implementation: https://github.com/huggingface/evaluate/blob/main/metrics/perplexity/perplexity.py  
  - Works well with 🤗 Datasets and 🤗 Transformers; you can pass a dataset and a model (including quantized models) and get perplexity with only a few lines of code.

- **EleutherAI `lm-evaluation-harness`**  
  - Project: https://github.com/EleutherAI/lm-evaluation-harness  
  - Designed for full benchmarks, but can also be used to run smaller subsets (e.g., Wikitext perplexity or a few tasks) to compare FP16 vs quantized models under the same evaluation harness.

- **Framework-specific tools (ModelOpt, vLLM, LLMC, etc.)**  
  - Vendor and runtime toolkits often include example scripts for evaluating quantized models (e.g., ModelOpt PTQ examples, vLLM’s `llm-compressor` utilities). These can be adapted to compute KL divergence, logit MSE, or perplexity vs a base model.

These tools can save time and reduce implementation errors, especially when you want to compare many quantized variants against a single FP16 baseline in a consistent way.

## 8. Practical tips

- Always compare models under the same decoding settings (temperature, top-p, max_new_tokens) when looking at outputs.
- Be careful about batch size and max length when comparing activations and logits, to avoid shape mismatches.
- When using KL on logits, consider clipping or ignoring extremely low-probability tokens to reduce numerical noise.
- For reproducibility, fix random seeds and log all configuration knobs (model version, quantization config, calibration/eval dataset slices).
