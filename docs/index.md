# auto-quantize-model docs

This site documents the Hydra-based experiment runners in this repository, plus a small set of reproducible CV/ONNX quantization workflows.

## Quickstart

Serve the docs locally:

```bash
pixi run mkdocs serve
```

Build static docs into `tmp/mkdocs-site/`:

```bash
pixi run mkdocs build
```

## Layer sensitivity workflows

- Start here: `Workflows → Layer Sensitivity (Hydra)`
- Qwen3-VL LM-only runner: `Workflows → Qwen3-VL LM-only Sensitivity`

## CV / ONNX workflows

- YOLOv10m low-bit quantization (ModelOpt ONNX PTQ): `Workflows → YOLOv10m Low-bit (ONNX PTQ)`
