# auto-quantize-model

A project to test and evaluate different DNN model quantization tools and techniques.

## Overview

This repository serves as a testing ground for various neural network quantization methods, including:

- **QAT (Quantization-Aware Training)**: Training models with quantization in mind
- **PTQ (Post-Training Quantization)**: Quantizing pre-trained models
- **Automatic Mixed-Precision Scheme Selection**: Intelligently selecting precision for different layers
- **LLM Quantization**: Weight and activation quantization (WxAy) for large language models

## Purpose

The goal is to compare and benchmark different quantization approaches to understand their trade-offs in terms of:
- Model accuracy
- Inference speed
- Memory footprint
- Ease of implementation

## Getting Started

Coming soon...

## VS Code settings

If VS Code warns that it’s “unable to watch for file changes” (common on Linux
when this repo contains many vendored files under `extern/`), exclude the heavy
directories from file watching by adding this to `.vscode/settings.json`:

```json
{
	"files.watcherExclude": {
		"**/extern/**": true,
		"**/custom-build/**": true
	}
}
```

## License

TBD
