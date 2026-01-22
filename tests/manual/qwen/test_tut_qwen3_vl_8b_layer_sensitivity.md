# Manual Test: Qwen3-VL-8B tutorial pack (layer sensitivity)

This manual test validates the end-to-end tutorial pack workflow (GPU run) for:

- `docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/run_demo.sh`

## Prerequisites

- `pixi install` completed
- CUDA GPU available (CPU runs are allowed but may be extremely slow)
- Model checkpoint link exists:
  - `models/qwen3_vl_8b_instruct/checkpoints/Qwen3-VL-8B-Instruct`
- Dataset assets exist:
  - `datasets/coco2017/source-data/`
  - `datasets/vlm-quantize-calib/coco2017_vlm_calib_medium.db`
  - `datasets/vlm-quantize-calib/coco2017_captions_medium.txt`

## Verify mode (default)

From repo root:

```bash
bash docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/run_demo.sh
```

Confirm:

- A new workspace is created under `tmp/tutorial_workspace_qwen3_vl_8b_layer_sensitivity_*`
- Per-scenario outputs exist under:
  - `tmp/.../outputs/all_layers/<quant_pair>/`
  - `tmp/.../outputs/lm_only/<quant_pair>/`
- Per-scenario summaries exist under:
  - `tmp/.../summaries/all_layers/<quant_pair>/summary.json`
  - `tmp/.../summaries/lm_only/<quant_pair>/summary.json`
- Verification passes (script exits 0) and no `diff` output is shown.

If verification fails due to degeneracy:

- Re-run with a larger preset (e.g., `--dataset-size medium` or `--dataset-size large`) and consult:
  - `docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/README.md`

## Snapshot mode

From repo root:

```bash
bash docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/run_demo.sh --snapshot-report
```

Confirm:

- `docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/expected_report/` is updated
- Re-running verify mode passes with no diffs
