# Subtask 4.7: Add documentation and KB cross-links for AutoQuant FP8 schemes

## Scope

Update relevant documentation and knowledge-base files so that the new AutoQuant FP8 schemes, scripts, and reports are discoverable and easy to reuse for future experiments, both for Qwen2.5-VL-3B and potentially other models.

## Planned outputs

- Short additions or cross-links in existing docs (e.g., under `context/summaries/modelopt-kb/` and related Qwen2.5-VL summaries) that reference:
  - The AutoQuant FP8 plan and this working task directory.
  - The `qwen2_5_vl_3b_autoquant_fp8_schemes.py` driver script and how to invoke it.
  - The FP8 AutoQuant schemes report and checkpoint locations.
- Optional: a brief “advanced usage” note in any vLLM or ModelOpt setup docs indicating how to try the new schemes.

## TODOs

- [ ] Job-004-107-001 Survey existing ModelOpt and Qwen2.5-VL documentation under `context/summaries/` and `context/plans/` to identify natural places to mention AutoQuant FP8 schemes.
- [ ] Job-004-107-002 Add concise cross-links to the new AutoQuant driver script, this working directory, and the FP8 AutoQuant schemes report in the appropriate KB or how-to files.
- [ ] Job-004-107-003 If applicable, extend any vLLM or ModelOpt environment setup docs to include a short “advanced: AutoQuant FP8 schemes” section with invocation examples.
- [ ] Job-004-107-004 Verify that new cross-links render correctly and do not break existing doc structure or narratives.

## Notes

- Keep documentation changes incremental and aligned with existing style; avoid duplicating full explanations that already live in the ModelOpt KB.

