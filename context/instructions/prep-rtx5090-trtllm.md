# Prep Instructions: RTX 5090 TensorRT-LLM Session

This file describes how to work in the current dialog session for this project when using the TensorRT-LLM–focused Pixi environment.

- If you want to run Python, **do not** use the system `python`.
- Always run Python via Pixi with the RTX 5090 TensorRT-LLM environment:

  - `pixi run -e rtx5090-trtllm python ...`

All Python code changes should be checked with:

- `pixi run -e rtx5090-trtllm mypy .`
- `pixi run -e rtx5090-trtllm ruff check .`

For debugging, any temporary scripts, notebooks, logs, or outputs should be saved under a task-specific subdirectory of `tmp/`, for example: `tmp/quantization-debug/` or `tmp/rtx5090-trtllm-exp-001/`.

Follow the existing repository guidelines in `AGENTS.md` and prefer Pixi-managed tools and environments over any system-level installations.

If any rules in `AGENTS.md` conflict with this prep document, prefer this prep document for this session.

