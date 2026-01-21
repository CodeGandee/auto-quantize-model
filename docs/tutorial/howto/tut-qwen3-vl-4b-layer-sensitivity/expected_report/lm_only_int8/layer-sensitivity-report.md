
AutoQuant Layer Sensitivity (int8_autoquant_lm_default)
=======================================================

## Summary


|Key|Value|
| :--- | :--- |
|Scheme|`int8_autoquant_lm_default`|
|Model|`<ABSOLUTE_PATH>-quantize-model<ABSOLUTE_PATH>-VL-4B-Instruct`|
|Effective bits (from search)|`16.0000`|
|Total AutoQuant score|`0.000000e+00`|
|Constraint satisfied|`False`|

## Dataset


|Key|Value|
| :--- | :--- |
|Name|`coco2017_captions_small`|
|Captions path|`<ABSOLUTE_PATH>-quantize-model<ABSOLUTE_PATH>
|Calibration seq len|`64`|
|Batch size|`1`|
|Calibration batches|`1`|
|Calibration samples (used / max)|`1` / `1`|

## Layer Sensitivity Table


Sorted by sensitivity (descending). Layer names are AutoQuant recipe handles; a trailing `.quant_recipe` suffix (if present) is stripped for readability.

|Layer|Num Bits|Sensitivity|Size Cost|
| :--- | :--- | :--- | :--- |
|model.visual.patch_embed.proj|16.0|0.000e+00|1.573e+06|
|model.visual.blocks.0.attn.qkv|16.0|0.000e+00|3.146e+06|
|model.visual.blocks.0.attn.proj|16.0|0.000e+00|1.049e+06|
|model.visual.blocks.0.mlp.linear_fc1|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.0.mlp.linear_fc2|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.1.attn.qkv|16.0|0.000e+00|3.146e+06|
|model.visual.blocks.1.attn.proj|16.0|0.000e+00|1.049e+06|
|model.visual.blocks.1.mlp.linear_fc1|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.1.mlp.linear_fc2|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.2.attn.qkv|16.0|0.000e+00|3.146e+06|
|model.visual.blocks.2.attn.proj|16.0|0.000e+00|1.049e+06|
|model.visual.blocks.2.mlp.linear_fc1|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.2.mlp.linear_fc2|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.3.attn.qkv|16.0|0.000e+00|3.146e+06|
|model.visual.blocks.3.attn.proj|16.0|0.000e+00|1.049e+06|
|model.visual.blocks.3.mlp.linear_fc1|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.3.mlp.linear_fc2|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.4.attn.qkv|16.0|0.000e+00|3.146e+06|
|model.visual.blocks.4.attn.proj|16.0|0.000e+00|1.049e+06|
|model.visual.blocks.4.mlp.linear_fc1|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.4.mlp.linear_fc2|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.5.attn.qkv|16.0|0.000e+00|3.146e+06|
|model.visual.blocks.5.attn.proj|16.0|0.000e+00|1.049e+06|
|model.visual.blocks.5.mlp.linear_fc1|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.5.mlp.linear_fc2|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.6.attn.qkv|16.0|0.000e+00|3.146e+06|
|model.visual.blocks.6.attn.proj|16.0|0.000e+00|1.049e+06|
|model.visual.blocks.6.mlp.linear_fc1|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.6.mlp.linear_fc2|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.7.attn.qkv|16.0|0.000e+00|3.146e+06|
|model.visual.blocks.7.attn.proj|16.0|0.000e+00|1.049e+06|
|model.visual.blocks.7.mlp.linear_fc1|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.7.mlp.linear_fc2|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.8.attn.qkv|16.0|0.000e+00|3.146e+06|
|model.visual.blocks.8.attn.proj|16.0|0.000e+00|1.049e+06|
|model.visual.blocks.8.mlp.linear_fc1|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.8.mlp.linear_fc2|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.9.attn.qkv|16.0|0.000e+00|3.146e+06|
|model.visual.blocks.9.attn.proj|16.0|0.000e+00|1.049e+06|
|model.visual.blocks.9.mlp.linear_fc1|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.9.mlp.linear_fc2|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.10.attn.qkv|16.0|0.000e+00|3.146e+06|
|model.visual.blocks.10.attn.proj|16.0|0.000e+00|1.049e+06|
|model.visual.blocks.10.mlp.linear_fc1|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.10.mlp.linear_fc2|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.11.attn.qkv|16.0|0.000e+00|3.146e+06|
|model.visual.blocks.11.attn.proj|16.0|0.000e+00|1.049e+06|
|model.visual.blocks.11.mlp.linear_fc1|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.11.mlp.linear_fc2|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.12.attn.qkv|16.0|0.000e+00|3.146e+06|
|model.visual.blocks.12.attn.proj|16.0|0.000e+00|1.049e+06|
|model.visual.blocks.12.mlp.linear_fc1|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.12.mlp.linear_fc2|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.13.attn.qkv|16.0|0.000e+00|3.146e+06|
|model.visual.blocks.13.attn.proj|16.0|0.000e+00|1.049e+06|
|model.visual.blocks.13.mlp.linear_fc1|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.13.mlp.linear_fc2|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.14.attn.qkv|16.0|0.000e+00|3.146e+06|
|model.visual.blocks.14.attn.proj|16.0|0.000e+00|1.049e+06|
|model.visual.blocks.14.mlp.linear_fc1|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.14.mlp.linear_fc2|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.15.attn.qkv|16.0|0.000e+00|3.146e+06|
|model.visual.blocks.15.attn.proj|16.0|0.000e+00|1.049e+06|
|model.visual.blocks.15.mlp.linear_fc1|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.15.mlp.linear_fc2|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.16.attn.qkv|16.0|0.000e+00|3.146e+06|
|model.visual.blocks.16.attn.proj|16.0|0.000e+00|1.049e+06|
|model.visual.blocks.16.mlp.linear_fc1|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.16.mlp.linear_fc2|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.17.attn.qkv|16.0|0.000e+00|3.146e+06|
|model.visual.blocks.17.attn.proj|16.0|0.000e+00|1.049e+06|
|model.visual.blocks.17.mlp.linear_fc1|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.17.mlp.linear_fc2|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.18.attn.qkv|16.0|0.000e+00|3.146e+06|
|model.visual.blocks.18.attn.proj|16.0|0.000e+00|1.049e+06|
|model.visual.blocks.18.mlp.linear_fc1|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.18.mlp.linear_fc2|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.19.attn.qkv|16.0|0.000e+00|3.146e+06|
|model.visual.blocks.19.attn.proj|16.0|0.000e+00|1.049e+06|
|model.visual.blocks.19.mlp.linear_fc1|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.19.mlp.linear_fc2|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.20.attn.qkv|16.0|0.000e+00|3.146e+06|
|model.visual.blocks.20.attn.proj|16.0|0.000e+00|1.049e+06|
|model.visual.blocks.20.mlp.linear_fc1|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.20.mlp.linear_fc2|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.21.attn.qkv|16.0|0.000e+00|3.146e+06|
|model.visual.blocks.21.attn.proj|16.0|0.000e+00|1.049e+06|
|model.visual.blocks.21.mlp.linear_fc1|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.21.mlp.linear_fc2|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.22.attn.qkv|16.0|0.000e+00|3.146e+06|
|model.visual.blocks.22.attn.proj|16.0|0.000e+00|1.049e+06|
|model.visual.blocks.22.mlp.linear_fc1|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.22.mlp.linear_fc2|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.23.attn.qkv|16.0|0.000e+00|3.146e+06|
|model.visual.blocks.23.attn.proj|16.0|0.000e+00|1.049e+06|
|model.visual.blocks.23.mlp.linear_fc1|16.0|0.000e+00|4.194e+06|
|model.visual.blocks.23.mlp.linear_fc2|16.0|0.000e+00|4.194e+06|
|model.visual.merger.linear_fc1|16.0|0.000e+00|1.678e+07|
|model.visual.merger.linear_fc2|16.0|0.000e+00|1.049e+07|
|model.visual.deepstack_merger_list.0.linear_fc1|16.0|0.000e+00|1.678e+07|
|model.visual.deepstack_merger_list.0.linear_fc2|16.0|0.000e+00|1.049e+07|
|model.visual.deepstack_merger_list.1.linear_fc1|16.0|0.000e+00|1.678e+07|
|model.visual.deepstack_merger_list.1.linear_fc2|16.0|0.000e+00|1.049e+07|
|model.visual.deepstack_merger_list.2.linear_fc1|16.0|0.000e+00|1.678e+07|
|model.visual.deepstack_merger_list.2.linear_fc2|16.0|0.000e+00|1.049e+07|
|model.language_model.layers.0.self_attn.q_proj|16.0|0.000e+00|1.573e+07|
|model.language_model.layers.0.self_attn.o_proj|16.0|0.000e+00|1.049e+07|
|model.language_model.layers.0.mlp.gate_proj|16.0|0.000e+00|4.981e+07|
|model.language_model.layers.0.mlp.down_proj|16.0|0.000e+00|2.490e+07|
|model.language_model.layers.1.self_attn.q_proj|16.0|0.000e+00|1.573e+07|
|model.language_model.layers.1.self_attn.o_proj|16.0|0.000e+00|1.049e+07|
|model.language_model.layers.1.mlp.gate_proj|16.0|0.000e+00|4.981e+07|
|model.language_model.layers.1.mlp.down_proj|16.0|0.000e+00|2.490e+07|
|model.language_model.layers.2.self_attn.q_proj|16.0|0.000e+00|1.573e+07|
|model.language_model.layers.2.self_attn.o_proj|16.0|0.000e+00|1.049e+07|
|model.language_model.layers.2.mlp.gate_proj|16.0|0.000e+00|4.981e+07|
|model.language_model.layers.2.mlp.down_proj|16.0|0.000e+00|2.490e+07|
|model.language_model.layers.3.self_attn.q_proj|16.0|0.000e+00|1.573e+07|
|model.language_model.layers.3.self_attn.o_proj|16.0|0.000e+00|1.049e+07|
|model.language_model.layers.3.mlp.gate_proj|16.0|0.000e+00|4.981e+07|
|model.language_model.layers.3.mlp.down_proj|16.0|0.000e+00|2.490e+07|
|model.language_model.layers.4.self_attn.q_proj|16.0|0.000e+00|1.573e+07|
|model.language_model.layers.4.self_attn.o_proj|16.0|0.000e+00|1.049e+07|
|model.language_model.layers.4.mlp.gate_proj|16.0|0.000e+00|4.981e+07|
|model.language_model.layers.4.mlp.down_proj|16.0|0.000e+00|2.490e+07|
|model.language_model.layers.5.self_attn.q_proj|16.0|0.000e+00|1.573e+07|
|model.language_model.layers.5.self_attn.o_proj|16.0|0.000e+00|1.049e+07|
|model.language_model.layers.5.mlp.gate_proj|16.0|0.000e+00|4.981e+07|
|model.language_model.layers.5.mlp.down_proj|16.0|0.000e+00|2.490e+07|
|model.language_model.layers.6.self_attn.q_proj|16.0|0.000e+00|1.573e+07|
|model.language_model.layers.6.self_attn.o_proj|16.0|0.000e+00|1.049e+07|
|model.language_model.layers.6.mlp.gate_proj|16.0|0.000e+00|4.981e+07|
|model.language_model.layers.6.mlp.down_proj|16.0|0.000e+00|2.490e+07|
|model.language_model.layers.7.self_attn.q_proj|16.0|0.000e+00|1.573e+07|
|model.language_model.layers.7.self_attn.o_proj|16.0|0.000e+00|1.049e+07|
|model.language_model.layers.7.mlp.gate_proj|16.0|0.000e+00|4.981e+07|
|model.language_model.layers.7.mlp.down_proj|16.0|0.000e+00|2.490e+07|
|model.language_model.layers.8.self_attn.q_proj|16.0|0.000e+00|1.573e+07|
|model.language_model.layers.8.self_attn.o_proj|16.0|0.000e+00|1.049e+07|
|model.language_model.layers.8.mlp.gate_proj|16.0|0.000e+00|4.981e+07|
|model.language_model.layers.8.mlp.down_proj|16.0|0.000e+00|2.490e+07|
|model.language_model.layers.9.self_attn.q_proj|16.0|0.000e+00|1.573e+07|
|model.language_model.layers.9.self_attn.o_proj|16.0|0.000e+00|1.049e+07|
|model.language_model.layers.9.mlp.gate_proj|16.0|0.000e+00|4.981e+07|
|model.language_model.layers.9.mlp.down_proj|16.0|0.000e+00|2.490e+07|
|model.language_model.layers.10.self_attn.q_proj|16.0|0.000e+00|1.573e+07|
|model.language_model.layers.10.self_attn.o_proj|16.0|0.000e+00|1.049e+07|
|model.language_model.layers.10.mlp.gate_proj|16.0|0.000e+00|4.981e+07|
|model.language_model.layers.10.mlp.down_proj|16.0|0.000e+00|2.490e+07|
|model.language_model.layers.11.self_attn.q_proj|16.0|0.000e+00|1.573e+07|
|model.language_model.layers.11.self_attn.o_proj|16.0|0.000e+00|1.049e+07|
|model.language_model.layers.11.mlp.gate_proj|16.0|0.000e+00|4.981e+07|
|model.language_model.layers.11.mlp.down_proj|16.0|0.000e+00|2.490e+07|
|model.language_model.layers.12.self_attn.q_proj|16.0|0.000e+00|1.573e+07|
|model.language_model.layers.12.self_attn.o_proj|16.0|0.000e+00|1.049e+07|
|model.language_model.layers.12.mlp.gate_proj|16.0|0.000e+00|4.981e+07|
|model.language_model.layers.12.mlp.down_proj|16.0|0.000e+00|2.490e+07|
|model.language_model.layers.13.self_attn.q_proj|16.0|0.000e+00|1.573e+07|
|model.language_model.layers.13.self_attn.o_proj|16.0|0.000e+00|1.049e+07|
|model.language_model.layers.13.mlp.gate_proj|16.0|0.000e+00|4.981e+07|
|model.language_model.layers.13.mlp.down_proj|16.0|0.000e+00|2.490e+07|
|model.language_model.layers.14.self_attn.q_proj|16.0|0.000e+00|1.573e+07|
|model.language_model.layers.14.self_attn.o_proj|16.0|0.000e+00|1.049e+07|
|model.language_model.layers.14.mlp.gate_proj|16.0|0.000e+00|4.981e+07|
|model.language_model.layers.14.mlp.down_proj|16.0|0.000e+00|2.490e+07|
|model.language_model.layers.15.self_attn.q_proj|16.0|0.000e+00|1.573e+07|
|model.language_model.layers.15.self_attn.o_proj|16.0|0.000e+00|1.049e+07|
|model.language_model.layers.15.mlp.gate_proj|16.0|0.000e+00|4.981e+07|
|model.language_model.layers.15.mlp.down_proj|16.0|0.000e+00|2.490e+07|
|model.language_model.layers.16.self_attn.q_proj|16.0|0.000e+00|1.573e+07|
|model.language_model.layers.16.self_attn.o_proj|16.0|0.000e+00|1.049e+07|
|model.language_model.layers.16.mlp.gate_proj|16.0|0.000e+00|4.981e+07|
|model.language_model.layers.16.mlp.down_proj|16.0|0.000e+00|2.490e+07|
|model.language_model.layers.17.self_attn.q_proj|16.0|0.000e+00|1.573e+07|
|model.language_model.layers.17.self_attn.o_proj|16.0|0.000e+00|1.049e+07|
|model.language_model.layers.17.mlp.gate_proj|16.0|0.000e+00|4.981e+07|
|model.language_model.layers.17.mlp.down_proj|16.0|0.000e+00|2.490e+07|
|model.language_model.layers.18.self_attn.q_proj|16.0|0.000e+00|1.573e+07|
|model.language_model.layers.18.self_attn.o_proj|16.0|0.000e+00|1.049e+07|
|model.language_model.layers.18.mlp.gate_proj|16.0|0.000e+00|4.981e+07|
|model.language_model.layers.18.mlp.down_proj|16.0|0.000e+00|2.490e+07|
|model.language_model.layers.19.self_attn.q_proj|16.0|0.000e+00|1.573e+07|
|model.language_model.layers.19.self_attn.o_proj|16.0|0.000e+00|1.049e+07|
|model.language_model.layers.19.mlp.gate_proj|16.0|0.000e+00|4.981e+07|
|model.language_model.layers.19.mlp.down_proj|16.0|0.000e+00|2.490e+07|
|model.language_model.layers.20.self_attn.q_proj|16.0|0.000e+00|1.573e+07|
|model.language_model.layers.20.self_attn.o_proj|16.0|0.000e+00|1.049e+07|
|model.language_model.layers.20.mlp.gate_proj|16.0|0.000e+00|4.981e+07|
|model.language_model.layers.20.mlp.down_proj|16.0|0.000e+00|2.490e+07|
|model.language_model.layers.21.self_attn.q_proj|16.0|0.000e+00|1.573e+07|
|model.language_model.layers.21.self_attn.o_proj|16.0|0.000e+00|1.049e+07|
|model.language_model.layers.21.mlp.gate_proj|16.0|0.000e+00|4.981e+07|
|model.language_model.layers.21.mlp.down_proj|16.0|0.000e+00|2.490e+07|
|model.language_model.layers.22.self_attn.q_proj|16.0|0.000e+00|1.573e+07|
|model.language_model.layers.22.self_attn.o_proj|16.0|0.000e+00|1.049e+07|
|model.language_model.layers.22.mlp.gate_proj|16.0|0.000e+00|4.981e+07|
|model.language_model.layers.22.mlp.down_proj|16.0|0.000e+00|2.490e+07|
|model.language_model.layers.23.self_attn.q_proj|16.0|0.000e+00|1.573e+07|
|model.language_model.layers.23.self_attn.o_proj|16.0|0.000e+00|1.049e+07|
|model.language_model.layers.23.mlp.gate_proj|16.0|0.000e+00|4.981e+07|
|model.language_model.layers.23.mlp.down_proj|16.0|0.000e+00|2.490e+07|
|model.language_model.layers.24.self_attn.q_proj|16.0|0.000e+00|1.573e+07|
|model.language_model.layers.24.self_attn.o_proj|16.0|0.000e+00|1.049e+07|
|model.language_model.layers.24.mlp.gate_proj|16.0|0.000e+00|4.981e+07|
|model.language_model.layers.24.mlp.down_proj|16.0|0.000e+00|2.490e+07|
|model.language_model.layers.25.self_attn.q_proj|16.0|0.000e+00|1.573e+07|
|model.language_model.layers.25.self_attn.o_proj|16.0|0.000e+00|1.049e+07|
|model.language_model.layers.25.mlp.gate_proj|16.0|0.000e+00|4.981e+07|
|model.language_model.layers.25.mlp.down_proj|16.0|0.000e+00|2.490e+07|
|model.language_model.layers.26.self_attn.q_proj|16.0|0.000e+00|1.573e+07|
|model.language_model.layers.26.self_attn.o_proj|16.0|0.000e+00|1.049e+07|
|model.language_model.layers.26.mlp.gate_proj|16.0|0.000e+00|4.981e+07|
|model.language_model.layers.26.mlp.down_proj|16.0|0.000e+00|2.490e+07|
|model.language_model.layers.27.self_attn.q_proj|16.0|0.000e+00|1.573e+07|
|model.language_model.layers.27.self_attn.o_proj|16.0|0.000e+00|1.049e+07|
|model.language_model.layers.27.mlp.gate_proj|16.0|0.000e+00|4.981e+07|
|model.language_model.layers.27.mlp.down_proj|16.0|0.000e+00|2.490e+07|
|model.language_model.layers.28.self_attn.q_proj|16.0|0.000e+00|1.573e+07|
|model.language_model.layers.28.self_attn.o_proj|16.0|0.000e+00|1.049e+07|
|model.language_model.layers.28.mlp.gate_proj|16.0|0.000e+00|4.981e+07|
|model.language_model.layers.28.mlp.down_proj|16.0|0.000e+00|2.490e+07|
|model.language_model.layers.29.self_attn.q_proj|16.0|0.000e+00|1.573e+07|
|model.language_model.layers.29.self_attn.o_proj|16.0|0.000e+00|1.049e+07|
|model.language_model.layers.29.mlp.gate_proj|16.0|0.000e+00|4.981e+07|
|model.language_model.layers.29.mlp.down_proj|16.0|0.000e+00|2.490e+07|
|model.language_model.layers.30.self_attn.q_proj|16.0|0.000e+00|1.573e+07|
|model.language_model.layers.30.self_attn.o_proj|16.0|0.000e+00|1.049e+07|
|model.language_model.layers.30.mlp.gate_proj|16.0|0.000e+00|4.981e+07|
|model.language_model.layers.30.mlp.down_proj|16.0|0.000e+00|2.490e+07|
|model.language_model.layers.31.self_attn.q_proj|16.0|0.000e+00|1.573e+07|
|model.language_model.layers.31.self_attn.o_proj|16.0|0.000e+00|1.049e+07|
|model.language_model.layers.31.mlp.gate_proj|16.0|0.000e+00|4.981e+07|
|model.language_model.layers.31.mlp.down_proj|16.0|0.000e+00|2.490e+07|
|model.language_model.layers.32.self_attn.q_proj|16.0|0.000e+00|1.573e+07|
|model.language_model.layers.32.self_attn.o_proj|16.0|0.000e+00|1.049e+07|
|model.language_model.layers.32.mlp.gate_proj|16.0|0.000e+00|4.981e+07|
|model.language_model.layers.32.mlp.down_proj|16.0|0.000e+00|2.490e+07|
|model.language_model.layers.33.self_attn.q_proj|16.0|0.000e+00|1.573e+07|
|model.language_model.layers.33.self_attn.o_proj|16.0|0.000e+00|1.049e+07|
|model.language_model.layers.33.mlp.gate_proj|16.0|0.000e+00|4.981e+07|
|model.language_model.layers.33.mlp.down_proj|16.0|0.000e+00|2.490e+07|
|model.language_model.layers.34.self_attn.q_proj|16.0|0.000e+00|1.573e+07|
|model.language_model.layers.34.self_attn.o_proj|16.0|0.000e+00|1.049e+07|
|model.language_model.layers.34.mlp.gate_proj|16.0|0.000e+00|4.981e+07|
|model.language_model.layers.34.mlp.down_proj|16.0|0.000e+00|2.490e+07|
|model.language_model.layers.35.self_attn.q_proj|16.0|0.000e+00|1.573e+07|
|model.language_model.layers.35.self_attn.o_proj|16.0|0.000e+00|1.049e+07|
|model.language_model.layers.35.mlp.gate_proj|16.0|0.000e+00|4.981e+07|
|model.language_model.layers.35.mlp.down_proj|16.0|0.000e+00|2.490e+07|
|lm_head|16.0|0.000e+00|3.890e+08|
