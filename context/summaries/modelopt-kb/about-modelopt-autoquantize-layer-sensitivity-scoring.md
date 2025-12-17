About: ModelOpt AutoQuant per-layer sensitivity scoring (gradient default + KL-div)

## HEADER
- **Purpose**: Answer “what algorithm does ModelOpt use by default for per-layer sensitivity?” and document the scoring formulas used by `modelopt.torch.quantization.auto_quantize(method="gradient" | "kl_div")`.
- **Status**: Draft (source-backed)
- **Date**: 2025-12-16
- **Owner**: AI assistant (Codex CLI)
- **Source**:
  - ModelOpt source (this repo checkout): `extern/TensorRT-Model-Optimizer/modelopt/torch/quantization/model_quant.py`, `extern/TensorRT-Model-Optimizer/modelopt/torch/quantization/algorithms.py`
  - Upstream (pinned to local checkout commit `5a4242faf4147fb0688bb73e10ca30b8ad3aabb3`):
    - https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/5a4242faf4147fb0688bb73e10ca30b8ad3aabb3/modelopt/torch/quantization/model_quant.py
    - https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/5a4242faf4147fb0688bb73e10ca30b8ad3aabb3/modelopt/torch/quantization/algorithms.py
  - ModelOpt docs (API): https://nvidia.github.io/TensorRT-Model-Optimizer/reference/generated/modelopt.torch.quantization.model_quant.html

## 1. The default algorithm (short answer)

When you call `modelopt.torch.quantization.auto_quantize(...)` without specifying `method=...`, ModelOpt defaults to:

- `method="gradient"` (gradient-based sensitivity scoring), implemented by `AutoQuantizeGradientSearcher`

This is explicit in the API signature and method selection logic:

- `extern/TensorRT-Model-Optimizer/modelopt/torch/quantization/model_quant.py` (`auto_quantize(..., method: str = "gradient", ...)` and the `if method == "gradient": searcher = AutoQuantizeGradientSearcher()` branch)

## 2. Sensitivity Analysis Algorithms

ModelOpt AutoQuant supports two sensitivity scoring methods:

- `method="gradient"`: gradient/Fisher-based (default; requires backward / a loss).
- `method="kl_div"`: KL-divergence-based (label-free; forward-only logits).

### 2.1 Gradient-based sensitivity scoring (`method="gradient"`, default)

ModelOpt’s gradient method estimates per-layer “sensitivity” as an approximation of loss increase caused by quantizing that layer, using a 2nd-order Taylor expansion around the unquantized output and substituting Fisher information for the Hessian (see `AutoQuantizeGradientSearcher` docstring in `algorithms.py`). Importantly, the “output” being perturbed here is an intermediate module output activation, not necessarily the model’s final output logits.

> **Fisher information (intuition + why it’s used here)**: In log-likelihood settings, Fisher information is a measure of how “sharp” the loss landscape is—i.e., how sensitive the loss is to small changes in some quantity. Formally, if $g = \partial L / \partial Y$ then $\mathcal{I} = \mathbb{E}[g g^\top]$ (expected outer product of gradients). Intuitively, larger $\mathcal{I}$ means that small perturbations to $Y$ are expected to cause larger changes in the loss (higher sensitivity). For negative log-likelihood losses, and under standard regularity conditions, the expected Hessian equals Fisher ($\mathbb{E}[H] = \mathcal{I}$), so Fisher is a convenient positive semi-definite curvature proxy for $H$; a common simplification is a diagonal Fisher approximation using $\mathbb{E}[g_i^2]$, which leads directly to the implemented weighting term “squared gradient times squared perturbation”.

#### Shared notation (KaTeX notation)

Let:
- each scoring sample (batch) $b$ contain model inputs $x^{(b)}$ and targets $y^{(b)}$ (for example, token IDs for language modeling, or class labels for classification).
- $L^{(b)}$ be the scalar objective used for sensitivity scoring, defined as:

$$
L^{(b)} \;=\; \ell\!\left(f(x^{(b)}),\, y^{(b)}\right)
$$

where $f(\cdot)$ denotes the model prediction used by the task (typically logits), and $\ell(\cdot,\cdot)$ is a differentiable task loss.

Examples of $\ell$ used in common ModelOpt workflows:
- **Causal language modeling (LLM PTQ flow)**: next-token negative log-likelihood / cross-entropy (the standard LM training loss). For a token sequence $t_1,\dots,t_T$ and predicted distributions $p_\theta(\cdot \mid t_{\le k})$:

$$
L^{(b)} \;=\; -\frac{1}{T-1}\sum_{k=1}^{T-1} \log p_\theta\!\left(t_{k+1} \mid t_{\le k}\right)
$$

> Source (LLM PTQ example): `extern/TensorRT-Model-Optimizer/examples/llm_ptq/hf_ptq.py:128` uses the model-provided causal-LM objective for scoring; the calibration dataloader constructs token labels from the input tokens (masking padding) in `extern/TensorRT-Model-Optimizer/modelopt/torch/utils/dataset_utils.py:278` so Hugging Face computes the standard next-token cross-entropy internally.

- **Image classification (vision auto-quant example)**: multiclass cross-entropy. For logits $z \in \mathbb{R}^C$ and ground-truth class $y$:

$$
L^{(b)} \;=\; -z_y + \log\sum_{c=1}^{C} e^{z_c}
$$

> Source (vision auto-quant example): `extern/TensorRT-Model-Optimizer/examples/onnx_ptq/torch_quant_to_onnx.py:116` uses multiclass cross-entropy as the scoring objective.

Other supervised losses are also valid (e.g., mean-squared error for regression), as long as they produce a single scalar objective that is differentiable w.r.t. the model’s intermediate activations being scored.
- A **score module** is the `nn.Module` whose output activation is used for sensitivity scoring. By default, the score module is the quantized module itself (so $Y$ is a layer output), but it can be overridden to a different module via `score_module_rules` (e.g., score at an MoE MLP output). In general, $Y$ is not the final model output unless the score module is itself the final output module.
- $Y$ be the (unquantized) output tensor of the score module for a given batch.
- $Y_r$ be the output tensor when the same module is evaluated under a candidate quantization recipe $r$.
- $\Delta Y_r = Y_r - Y$ be the output perturbation introduced by recipe $r$.
- $G^{(b)} = \frac{\partial L^{(b)}}{\partial Y^{(b)}}$ be the gradient of the objective w.r.t. the chosen score module output for batch $b$ (this is score-module-specific; different score modules generally have different gradients).

#### Score formula (Taylor/Fisher approximation)

> **Implementation names (ModelOpt variable mapping)**: In the gradient scorer, $Y$ is `output` from the score module run with “no quant” (`hparam.active = NONE`); $\Delta Y_r$ is cached as `output_diff` (computed as quantized output minus unquantized output for each candidate recipe); and $G^{(b)}$ arrives as `grad_output[0]` in the score module backward hook. Source: `extern/TensorRT-Model-Optimizer/modelopt/torch/quantization/algorithms.py` (`AutoQuantizeGradientSearcher._estimate_auto_quantize_scores`, `auto_quantize_score_estimate_forward`, `backward_hook`).

ModelOpt’s gradient-based per-batch score for recipe $r$ is the elementwise squared product, summed over all tensor elements:

$$
s_r \;=\; \sum_i \left(G_i\right)^2 \left(\Delta Y_{r,i}\right)^2
$$

> Source (score implementation): `extern/TensorRT-Model-Optimizer/modelopt/torch/quantization/algorithms.py` defines `_get_auto_quantize_score(grad_output, output_diff)`:
>
> ```python
> def _get_auto_quantize_score(grad_output, output_diff):
>     return ((grad_output.float() ** 2) * (output_diff.float() ** 2)).sum()
> ```

Note: The derivatives (and the Taylor/Fisher “Hessian” approximation) here are with respect to the score-module output $Y$ (an activation), not the layer weights. ModelOpt does not use $\,\partial L / \partial W\,$ or a weight-space Hessian for sensitivity scoring; weight quantization affects the score only through the induced activation/output perturbation $\Delta Y_r$.

> Source (ModelOpt intent): `extern/TensorRT-Model-Optimizer/modelopt/torch/quantization/algorithms.py:806` (“AutoQuantize only needs activation gradients to be computed (not weight gradients).”)

Across `num_score_steps` batches, ModelOpt accumulates:

$$
S_r \;=\; \sum_{b=1}^{B} \sum_i \left(G^{(b)}_i\right)^2 \left(\Delta Y^{(b)}_{r,i}\right)^2
$$

Connection to the docstring (“Taylor expansion + Fisher for Hessian”): a standard 2nd-order approximation is

$$
\Delta L \;\approx\; \frac{1}{2}\,\Delta Y_r^\top H\,\Delta Y_r
$$

and for log-likelihood losses ModelOpt’s docstring motivates approximating $H$ with Fisher information $\mathcal{I}$, then (implicitly) taking a diagonal approximation:

$$
H \;\approx\; \mathcal{I} \;\approx\; \mathbb{E}[G G^\top]
\quad\Rightarrow\quad
\Delta L \;\approx\; \frac{1}{2}\sum_i \mathbb{E}\!\left[G_i^2\right]\left(\Delta Y_{r,i}\right)^2
$$

The implementation uses a finite-sample estimate and drops constant factors, yielding the practical scoring rule “larger $S_r$ ⇒ more sensitive”.

#### How the score is computed (what ModelOpt actually does)

In `AutoQuantizeGradientSearcher._estimate_auto_quantize_scores(...)`:

1. For each “score module” (typically a quantizable Linear/Conv), ModelOpt monkey-patches `module.forward` to:
   - Force “no quantization” for the scored hparams and run the normal forward to produce `Y`
   - Under `torch.no_grad()`, temporarily enable each candidate recipe and re-run the forward to compute and cache `ΔY = Q(Y) - Y` for that recipe
2. It runs your `forward_backward_step` (or `forward_step` + `loss_func`) for `num_score_steps` batches.
3. A backward hook receives `grad_output = ∂L/∂Y` and accumulates `sum(grad_output^2 * ΔY^2)` into an internal per-layer-per-recipe importance dict.

Important implications:
- **This default method requires a backward pass**, so you must provide labels (via `loss_func`) or implement a `forward_backward_step`.
- Sensitivity is measured in **activation/output space** (gradients w.r.t. a score-module output), not in weight space (no $\,\partial L / \partial W\,$ is used for scoring).
- “Higher score” means “more sensitive to quantization” for that layer/recipe.

#### Pseudocode: AutoQuantize end-to-end flow (calibrate → score → search)

```text
function AUTO_QUANTIZE(model, constraints, quantization_formats, data_loader, forward_step,
                       loss_func?, forward_backward_step?, method="gradient", checkpoint?):
    model = apply_mode(model, mode="auto_quantize")        # inserts modelopt hparam machinery
    disable_all_quantizers(model)                          # AutoQuantize enables as-needed

    if method == "gradient":
        searcher = AutoQuantizeGradientSearcher()
    else if method == "kl_div":
        searcher = AutoQuantizeKLDivSearcher()
    else:
        error("Invalid method")

    searcher.before_search():
        recipes = normalize(quantization_formats) + [NONE]
        insert QuantRecipeHparam on quant_modules grouped by quant_grouping_rules
        (gradient only) optionally score at score_modules via score_module_rules

        for recipe in recipes where recipe != NONE:
            set all hparams active = recipe
            calibrate_quantizers(model, algorithm=recipe.algorithm, num_calib_steps)

        if checkpoint has candidate_stats:
            restore candidate_stats; skip scoring
        else:
            estimate_sensitivity_scores()                  # method-specific
            initialize_candidate_stats()                    # formats / scores / costs per hparam
            save checkpoint

    searcher.run_search():
        pick one recipe per hparam under effective_bits constraint
        apply selected recipes to model
        fold_pre_quant_scale_to_weights(model)

    return model, searcher.state_dict()
```

Notes (source):
- Calibration loop is in `extern/TensorRT-Model-Optimizer/modelopt/torch/quantization/algorithms.py` (`_AutoQuantizeBaseSearcher.before_search()`).
- Sensitivity scoring is `AutoQuantizeGradientSearcher.estimate_sensitivity_scores()` / `AutoQuantizeKLDivSearcher.estimate_sensitivity_scores()`.
- Search differs by method: gradient uses an LP solver; KLDiv uses a threshold-based binary search (see `run_search_with_stats` in each searcher).

#### Pseudocode: Gradient (“Taylor/Fisher”) sensitivity scoring mechanics

```text
function ESTIMATE_SENSITIVITY_SCORES_GRADIENT(model, data_loader, forward_step,
                                              loss_func or forward_backward_step,
                                              num_score_steps):
    model.eval()

    (grad_ckpt_context, is_param_grad_enabled) = maybe_select_custom_support(model)
    with grad_ckpt_context(model) if provided:
        score_modules = {m | hasattr(m, "_hparams_for_scoring") and any(hparam.is_configurable)}

        for each score_module in score_modules:
            patch score_module.forward(input):
                set all configurable hparams active = NONE
                Y = original_forward(input)
                if torch.is_grad_enabled() is false:
                    return Y
                under no_grad():
                    for each configurable hparam:
                        for each recipe in hparam.choices where recipe != NONE:
                            hparam.active = recipe
                            Yq = original_forward(input)
                            cache ΔY[hparam][recipe] = Yq - Y   # handle tuple outputs as needed
                        hparam.active = NONE
                return Y

            register backward_hook(score_module):
                grad_output = ∂L/∂Y
                for each cached (hparam, recipe, ΔY):
                    importance[hparam][recipe][score_module] += Σ(grad_output^2 * ΔY^2)

        for each parameter p in model.parameters:
            p.requires_grad = is_param_grad_enabled(p.name, model)  # minimize grad compute
            install hook to clear p.grad ASAP

        repeat num_score_steps:
            run forward_backward_step(model, batch)  # or forward_step + loss_func + loss.backward()

        restore original forwards, remove hooks, restore requires_grad
```

Key modelopt details reflected above (source-backed):
- Score modules are not always the quantized modules: ModelOpt can estimate a group’s sensitivity at a higher-level “score module” (e.g., MoE MLP output) via `score_module_rules`; the hparam attaches itself to `score_module._hparams_for_scoring` (see `QuantRecipeHparam.__init__` and `_AutoQuantizeBaseSearcher.insert_hparams_after_merge_rules()` in `extern/TensorRT-Model-Optimizer/modelopt/torch/quantization/algorithms.py`).
- The exact score function is `sum((grad_output^2) * (ΔY^2))` (`_get_auto_quantize_score` in `extern/TensorRT-Model-Optimizer/modelopt/torch/quantization/algorithms.py`).

### 2.2 KL-divergence sensitivity scoring (`method="kl_div"`, label-free)

ModelOpt also supports `method="kl_div"`:
- It measures sensitivity via **KL divergence** between unquantized and quantized model outputs (logits), and does not require labels/backward.
- It uses a different search strategy (threshold-based binary search) than the gradient method (linear programming).

#### Math (KaTeX notation)

Let:
- $z^{(b)}$ be the model logits for batch $b$ (the output of your `forward_step(model, batch)` for this method).
- $p^{(b)} = \mathrm{softmax}(z^{(b)})$ be the unquantized output distribution (computed once per batch).
- $z^{(b)}_r$ be the logits when evaluating under a candidate quantization recipe $r$.
- $q^{(b)}_r = \mathrm{softmax}(z^{(b)}_r)$ be the corresponding quantized output distribution.

ModelOpt’s per-batch KL-div score for recipe $r$ is:

$$
s_r^{(b)} \;=\; D_{\mathrm{KL}}\!\left(p^{(b)} \,\|\, q^{(b)}_r\right)
\;=\; \sum_i p^{(b)}_i \left(\log p^{(b)}_i - \log q^{(b)}_{r,i}\right)
$$

Across `num_score_steps` batches, ModelOpt accumulates:

$$
S_r \;=\; \sum_{b=1}^{B} s_r^{(b)}
$$

> **Implementation note (why ModelOpt computes only `-p*log q`)**: Since $p^{(b)}$ is fixed for a given batch, the term $\sum_i p^{(b)}_i \log p^{(b)}_i$ does not depend on recipe $r$. ModelOpt therefore scores using only the cross-entropy term $-\sum_i p^{(b)}_i \log q^{(b)}_{r,i}$ (see `_get_kl_div_loss`), which is equivalent for ranking recipes.

> **Implementation names (ModelOpt variable mapping)**: $p^{(b)}$ is `prob_unquant`; $z^{(b)}_r$ is `logits_quant`; and `score = _get_kl_div_loss(prob_unquant, logits_quant, ...)` is accumulated into `hparam._importance_dict[recipe][hparam.score_modules[0]]`. Source: `extern/TensorRT-Model-Optimizer/modelopt/torch/quantization/algorithms.py` (`AutoQuantizeKLDivSearcher.estimate_sensitivity_scores`, `_get_prob_from_logits`, `_get_kl_div_loss`).

See:
- `extern/TensorRT-Model-Optimizer/modelopt/torch/quantization/model_quant.py` (method selection)
- `extern/TensorRT-Model-Optimizer/modelopt/torch/quantization/algorithms.py` (`AutoQuantizeKLDivSearcher`)

#### Pseudocode: KL-divergence sensitivity scoring mechanics (label-free)

```text
function ESTIMATE_SENSITIVITY_SCORES_KLDIV(model, data_loader, forward_step, num_score_steps):
    model.eval()
    inference_mode():
        for batch in first num_score_steps batches:
            set all configurable QuantRecipeHparam.active = NONE
            logits_unquant = forward_step(model, batch)     # must return logits
            prob_unquant = softmax(logits_unquant)          # handles TP via helper

            for each configurable QuantRecipeHparam hparam:
                for each recipe in hparam.choices where recipe != NONE:
                    hparam.active = recipe
                    logits_quant = forward_step(model, batch)
                    score = KL(prob_unquant || logits_quant)  # implemented as -Σ p*log softmax(logits_quant)
                    importance[hparam][recipe][hparam.score_modules[0]] += score
                hparam.active = NONE
```

## 3. Minimal usage snippet (explicitly pin the method)

```python
import modelopt.torch.quantization as mtq

quantized_model, state = mtq.auto_quantize(
    model,
    constraints={"effective_bits": 4.8},
    quantization_formats=[mtq.NVFP4_AWQ_LITE_CFG, mtq.FP8_DEFAULT_CFG],
    data_loader=calib_loader,
    forward_step=forward_step,
    loss_func=loss_func,     # required for method="gradient"
    num_calib_steps=512,
    num_score_steps=128,
    method="gradient",       # default; set explicitly for reproducibility
)
```

If you need label-free scoring (logits only):

```python
quantized_model, state = mtq.auto_quantize(
    model,
    constraints={"effective_bits": 4.8},
    quantization_formats=[mtq.NVFP4_AWQ_LITE_CFG, mtq.FP8_DEFAULT_CFG],
    data_loader=calib_loader,
    forward_step=forward_step,  # must return logits
    num_calib_steps=512,
    num_score_steps=128,
    method="kl_div",
)
```
