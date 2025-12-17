This is qwen3-vl llm-compressor target quantization recipe

```
recipe = """
quant_stage:
    quant_modifiers:
        GPTQModifier:
            ignore: [
                        "re:.*lm_head.*",
                        "re:.*vpm.*",
                        "re:resampler.kv_proj",
                        "re:.*resampler.*"
                    ]
            config_groups:
                group_0:
                    weights:
                        num_bits: 4
                        type: int
                        strategy: channel
                        dynamic: false
                        symmetric: true
                    input_activations:
                        num_bits: 8
                        type: int
                        strategy: token
                        dynamic: true
                        symmetric: true
                    targets: ["re:.*self_attn.q_proj.*", "re:.*self_attn.k_proj.*", "re:.*self_attn.v_proj.*", "re:.*self_attn.o_proj.*", "re:.*mlp.gate_proj.*", "re:.*mlp.up_proj.*", "re:.*mlp.down_proj.*"]
"""
```