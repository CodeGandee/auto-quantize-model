# How to Speed Up Intel Neural Compressor Quantization for LLMs

Intel Neural Compressor (INC) quantization can be slow for large language models, especially when using accuracy-driven auto-tuning. This guide covers well-known configurations and techniques to significantly reduce quantization time while maintaining acceptable accuracy.

INC source in this repo: `extern/neural-compressor/`

## Overview of Speed Bottlenecks

INC's quantization slowness typically comes from:
1. **Calibration phase**: Running many samples through the model to collect statistics
2. **Auto-tuning phase**: Iteratively trying different quantization configurations to meet accuracy criteria
3. **Evaluation phase**: Repeatedly measuring accuracy on validation dataset after each quantization trial
4. **Per-layer sensitivity analysis**: Computing impact scores for each layer (when using advanced strategies like MSE_V2, HAWQ_V2)

## Quick Wins: Essential Speed Optimizations

### 1. Reduce Calibration Sample Size

The `sampling_size` parameter controls how many calibration samples are used. Default values are often 500+, but you can reduce to 50-200 for LLMs without significant accuracy loss.

```python
from neural_compressor.config import PostTrainingQuantConfig

conf = PostTrainingQuantConfig(
    approach="static",
    calibration=CalibrationConfig(
        sampling_size=100,  # Reduce from default (often 500+)
    ),
)
```

YAML configuration:
```yaml
quantization:
  calibration:
    sampling_size: 100  # 50-200 recommended for LLMs
```

**Impact**: Can reduce calibration time by 5-10x depending on original sample count.

### 2. Limit Auto-Tuning Trials

Configure exit policy to stop tuning early using `TuningCriterion`:

```python
from neural_compressor.config import TuningCriterion

tuning_criterion = TuningCriterion(
    timeout=0,        # 0 = early stopping when first good config found
                      # Or set specific timeout in seconds (e.g., 3600 for 1 hour)
    max_trials=1,     # Limit tuning attempts (1 = no tuning, just quantize)
    strategy="basic"
)
```

**For fastest results (no tuning)**:
```python
tuning_criterion = TuningCriterion(
    timeout=0,
    max_trials=1,  # Single-shot quantization
)
```

**For moderate tuning**:
```python
tuning_criterion = TuningCriterion(
    timeout=3600,     # Stop after 1 hour
    max_trials=10,    # Or after 10 trials
)
```

**Impact**: Setting `max_trials=1` eliminates all tuning overhead. Setting `timeout=0` with higher `max_trials` allows early exit when accuracy goal is met.

### 3. Use Faster Quantization Approaches

Different quantization approaches have different speed characteristics:

**Dynamic Quantization** (fastest, no calibration):
```python
conf = PostTrainingQuantConfig(
    approach="dynamic",  # Weights quantized at conversion, activations at runtime
)
```
- No calibration dataset needed
- No accuracy evaluation during quantization
- Best for initial experiments

**Weight-Only Quantization** (very fast):
```python
conf = PostTrainingQuantConfig(
    approach="weight_only",  # Only quantize weights, activations stay FP16/FP32
)
```
- Minimal calibration needed
- Common for LLM deployment (W8A16, W4A16)

**Static Quantization** (slowest but best accuracy):
```python
conf = PostTrainingQuantConfig(
    approach="static",  # Full W8A8 quantization
)
```
- Requires full calibration
- Use only when necessary

**Impact**: Dynamic quantization can be 10-50x faster than static for large models.

## Advanced Optimizations

### 4. Reduce Tuning Space

Specify which operations or layers to quantize, reducing search space:

```python
conf = PostTrainingQuantConfig(
    approach="static",
    op_type_list=["Linear", "Conv2d"],  # Only quantize these op types
    # OR
    op_name_list=["model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.k_proj"],  # Specific layers
)
```

**For LLMs, focus on high-impact layers**:
```python
# Example: quantize only attention and MLP projections
op_type_list=["Linear"]  # Most LLM compute is in Linear layers
```

**Impact**: Can reduce tuning time by 2-5x by eliminating low-impact operations from search space.

### 5. Use O0 Quantization Level

The `quant_level=0` mode starts with FP32 and selectively quantizes operations:

```python
conf = PostTrainingQuantConfig(
    quant_level=0,  # Conservative, selective quantization
    tuning_criterion=TuningCriterion(
        timeout=0,
        max_trials=100,  # Allows early stopping
    ),
)
```

Compared to default `quant_level=1` which tries to quantize everything first, then falls back.

**Impact**: Faster convergence for models where only partial quantization is acceptable.

### 6. Reduce MSE_V2 Confidence Batches

If using the `mse_v2` strategy for sensitivity-based tuning, reduce `confidence_batches`:

```python
tuning_criterion = TuningCriterion(
    strategy="mse_v2",
    strategy_kwargs={"confidence_batches": 1},  # Default is 2+
)
```

Each confidence batch requires a full forward pass through the model for every layer being scored.

**Impact**: Can reduce MSE_V2 strategy time by 2-4x (proportional to batch reduction).

### 7. Skip Sensitivity Analysis

For fastest quantization, use `basic` strategy instead of advanced strategies:

```python
tuning_criterion = TuningCriterion(
    strategy="basic",  # Fastest strategy (default)
)
```

Avoid these slower strategies unless needed:
- `mse_v2`: Requires per-layer MSE computation
- `hawq_v2`: Requires Hessian trace computation (very slow)
- `bayesian`, `exhaustive`, `random`: Explore large search spaces
- `tpe`: Requires 200+ iterations for convergence

**Impact**: Can be 5-20x faster than MSE_V2 or HAWQ_V2.

### 8. Parallelize Tuning Across Multiple Nodes

For production workloads, use Neural Solution service to distribute tuning:

```python
# Neural Solution allows parallel tuning across workers
# Specify number of workers in task request
# Requires separate Neural Solution setup
```

Reference: INC Neural Solution documentation for distributed tuning setup.

**Impact**: Near-linear speedup with number of worker nodes.

### 9. Reduce Evaluation Dataset Size

If you must use accuracy-driven tuning, use a smaller evaluation dataset:

```python
# Use subset of validation data for faster eval_func
eval_dataloader = DataLoader(val_dataset[:1000], batch_size=32)  # Instead of full dataset

def eval_func(model):
    # Evaluate on subset only
    return evaluate_accuracy(model, eval_dataloader)

q_model = quantization.fit(
    model,
    conf=conf,
    calib_dataloader=calib_loader,
    eval_func=eval_func,  # Fast evaluation
)
```

**Impact**: Proportional to dataset size reduction (e.g., 10x smaller dataset = 10x faster evaluation).

## Recommended Fast Configurations for LLMs

### Configuration 1: Fastest (No Tuning, Dynamic Quantization)

```python
from neural_compressor import quantization
from neural_compressor.config import PostTrainingQuantConfig, TuningCriterion

conf = PostTrainingQuantConfig(
    approach="dynamic",
    tuning_criterion=TuningCriterion(
        timeout=0,
        max_trials=1,
    ),
)

q_model = quantization.fit(model, conf=conf)
```

**Use case**: Quick baseline, initial experiments, CI/CD testing
**Speed**: Minutes for billion-parameter models
**Accuracy**: May have 1-3% degradation vs. static

### Configuration 2: Balanced (Weight-Only, Minimal Tuning)

```python
from neural_compressor.config import PostTrainingQuantConfig, TuningCriterion, AccuracyCriterion

conf = PostTrainingQuantConfig(
    approach="weight_only",
    quant_level=1,
    tuning_criterion=TuningCriterion(
        timeout=1800,      # 30 minute timeout
        max_trials=5,      # Try up to 5 configs
        strategy="basic",
    ),
    accuracy_criterion=AccuracyCriterion(
        tolerable_loss=0.01,  # 1% accuracy loss acceptable
    ),
)

q_model = quantization.fit(
    model,
    conf=conf,
    calib_dataloader=calib_loader,
    eval_func=eval_func,
)
```

**Use case**: Production deployment preparation
**Speed**: 30-60 minutes for billion-parameter models
**Accuracy**: Typically <1% loss with proper calibration

### Configuration 3: Quality-Focused (Static, Reduced Samples)

```python
conf = PostTrainingQuantConfig(
    approach="static",
    calibration=CalibrationConfig(
        sampling_size=100,  # Reduced from 500+
    ),
    tuning_criterion=TuningCriterion(
        timeout=3600,       # 1 hour max
        max_trials=20,
        strategy="basic",
    ),
    accuracy_criterion=AccuracyCriterion(
        tolerable_loss=0.005,  # 0.5% accuracy loss
    ),
)

q_model = quantization.fit(
    model,
    conf=conf,
    calib_dataloader=calib_loader,
    eval_func=eval_func,
)
```

**Use case**: High-quality quantization with acceptable time budget
**Speed**: 1-3 hours for billion-parameter models
**Accuracy**: Best possible within time budget

## Alternative: AutoRound (Newer, Faster)

Intel recently integrated **AutoRound** (December 2024) into LLM Compressor, which is significantly faster than traditional INC for LLMs:

```python
# Using AutoRound via LLM Compressor (not standard INC)
from llmcompressor.modifiers.quantization import AutoRoundModifier

# AutoRound provides faster convergence for LLMs
# with better accuracy/speed tradeoff
```

AutoRound benefits:
- Tuning-based approach optimized for LLMs
- Faster convergence than traditional PTQ
- Better W4A16 accuracy than RTN (round-to-nearest)
- Native integration with vLLM for deployment

Reference: https://blog.vllm.ai/2025/12/09/intel-autoround-llmc.html

## YAML Configuration Examples

### Fast Configuration (YAML)

```yaml
model:
  name: llm_model
  framework: pytorch

quantization:
  approach: dynamic
  calibration:
    sampling_size: 100
  tuning:
    strategy:
      name: basic
    accuracy_criterion:
      relative: 0.01
    exit_policy:
      timeout: 0
      max_trials: 1
```

### Balanced Configuration (YAML)

```yaml
model:
  name: llm_model
  framework: pytorch

quantization:
  approach: weight_only
  calibration:
    sampling_size: 100
  tuning:
    strategy:
      name: basic
    accuracy_criterion:
      relative: 0.01
    exit_policy:
      timeout: 1800
      max_trials: 10
```

## Speed Impact Summary

| Optimization | Typical Speedup | Accuracy Impact |
|--------------|-----------------|-----------------|
| `sampling_size=100` (from 500) | 5x | Minimal (<0.1%) |
| `max_trials=1` | 10-50x | 1-3% loss |
| `approach="dynamic"` | 10-50x | 1-3% loss |
| `approach="weight_only"` | 5-20x | 0.5-2% loss |
| `strategy="basic"` (vs MSE_V2) | 5-10x | Varies |
| `confidence_batches=1` (from 2) | 2x | Minimal |
| Reduced eval dataset | 5-10x | None (tuning only) |
| Combined optimizations | 50-100x+ | 1-5% loss |

## Common Pitfalls

### 1. Setting `max_trials` too high with `timeout=0`

```python
# BAD: Can run indefinitely if accuracy goal not met
tuning_criterion = TuningCriterion(
    timeout=0,        # Early stopping only
    max_trials=1000,  # But will try 1000 times if needed
)

# GOOD: Combine timeout with max_trials
tuning_criterion = TuningCriterion(
    timeout=3600,     # 1 hour hard limit
    max_trials=100,   # Or 100 trials, whichever comes first
)
```

### 2. Using `hawq_v2` without loss function

`hawq_v2` requires a loss function and is very slow (Hessian computation):

```python
# SLOW: Only use if you need Hessian-based sensitivity
def model_loss(output, target, criterion):
    return criterion(output, target)

tuning_criterion = TuningCriterion(
    strategy="hawq_v2",
    strategy_kwargs={"hawq_v2_loss": model_loss},
)
```

For most LLM use cases, `basic` or `mse_v2` (with reduced confidence batches) is sufficient.

### 3. Not reducing evaluation overhead

```python
# BAD: Full evaluation on every trial
def eval_func(model):
    return evaluate_on_full_dataset(model)  # Slow!

# GOOD: Fast proxy evaluation
def eval_func(model):
    return evaluate_on_subset(model, n_samples=1000)  # Fast proxy
```

## References

- INC Tuning Strategies: https://intel.github.io/neural-compressor/2.0/tuning_strategies.html
- INC PyTorch Quantization: https://www.intel.com/content/www/us/en/developer/articles/technical/pytorch-quantization-using-intel-neural-compressor.html
- AutoRound Integration: https://blog.vllm.ai/2025/12/09/intel-autoround-llmc.html
- INC GitHub: https://github.com/intel/neural-compressor
- Neural Solution (Distributed Tuning): https://medium.com/intel-analytics-software/streamlining-model-optimization-as-a-service-with-intel-neural-compressor-fd970fdb2928

## Related Hints in this Repository

- `howto-inc-layer-sensitivity-for-mixed-precision.md`: Using MSE_V2 and HAWQ_V2 for sensitivity analysis
- `intro-inc-w8a8-quantization-fp8-int8.md`: FP8 and INT8 W8A8 quantization basics
