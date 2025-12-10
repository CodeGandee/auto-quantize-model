# MSE_V2 Op Sensitivity Sanity Check (PyTorch FX, CPU)

- Calib samples: 3
- Calib batch size: 1
- Max seq len: 128

## INT8-op fallback sensitivity (lower MSE = less sensitive)

| Rank | Op Name | Op Type | MSE |
| --- | --- | --- | --- |
| 1 | `model.visual.patch_embed.proj` | `Conv3d` | 2.397563e+08 |
| 2 | `model.visual.blocks.0.attn.qkv.module` | `Linear` | 2.397563e+08 |
| 3 | `model.visual.blocks.0.attn.proj.module` | `Linear` | 2.397563e+08 |
