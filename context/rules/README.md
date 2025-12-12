# Rules Directory

## HEADER
- **Purpose**: Store task-specific rules and constraints for AI coding agents
- **Status**: Active
- **Date**: 2025-12-12
- **Dependencies**: Task documents under `context/tasks/` and project codebase
- **Target**: AI assistants and developers

## Content
Use this directory for markdown files that define rules for a specific task, experiment, or workflow. These rules should be concise, actionable, and written for AI coding agents to consume before or during implementation.

Typical contents include:
- Environment or tooling constraints for a task (e.g., allowed libraries, required commands)
- Behavioral guardrails (e.g., what not to change, how to validate results)
- Output formats and reporting expectations
- Task-local conventions that override or extend repo-wide instructions

Prefer naming files after the task or area they apply to (for example, `quantize-qwen-vl.md`). Reference these rules from the related task document in `context/tasks/`.
