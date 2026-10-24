# 梯度一致性：A(复用) vs B(不复用)，含噪声底线

- 序列：满长度；dropout 全置 0；training 模式（gc 生效）

| 比较 | RMS 相对差（整体向量） | 说明 |
|---|---|---|
| A vs A（噪声） | 1.480e-02 | 同一路径重复两次，种子固定 |
| B vs B（噪声） | 1.404e-02 | 同上 |
| A vs B（信号） | 5.879e-01 | 与噪声底线比较 |

**判定**：信号 5.879e-01 vs 噪声底线 1.480e-02（×1.5 = 2.220e-02）→ FAIL（信号显著高于噪声）

## A vs B 逐张量（按相对差排序，前 15）

| param | ‖gA‖ | ‖gA−gB‖ | 相对差 | 差值/总范数 |
|---|---|---|---|---|
| mixer:fc2.weight | 1.0218e+03 | 6.2990e+02 | 6.165e-01 | 5.864e-01 |
| mixer:fc1.weight | 2.7719e+00 | 1.6928e+00 | 6.107e-01 | 1.576e-03 |
| mixer:type_emb | 3.6844e-01 | 2.1783e-01 | 5.912e-01 | 2.028e-04 |
| mixer:fc1.bias | 4.2944e-01 | 2.3549e-01 | 5.484e-01 | 2.192e-04 |
| mixer:fc2.bias | 6.9065e+01 | 3.7830e+01 | 5.477e-01 | 3.522e-02 |
| lora:base_model.model.model.language_model.layers.29.linear_attn.in_proj_a.lora_B.default.weight | 2.1813e-02 | 1.0439e-02 | 4.786e-01 | 9.718e-06 |
| lora:base_model.model.model.language_model.layers.30.linear_attn.out_proj.lora_B.default.weight | 1.0424e+00 | 4.2807e-01 | 4.106e-01 | 3.985e-04 |
| lora:base_model.model.model.language_model.layers.30.linear_attn.in_proj_qkv.lora_B.default.weight | 1.9071e+00 | 6.6114e-01 | 3.467e-01 | 6.155e-04 |
| lora:base_model.model.model.language_model.layers.28.mlp.down_proj.lora_B.default.weight | 7.1660e-01 | 2.4364e-01 | 3.400e-01 | 2.268e-04 |
| lora:base_model.model.model.language_model.layers.30.mlp.down_proj.lora_B.default.weight | 9.0488e-01 | 3.0762e-01 | 3.400e-01 | 2.864e-04 |
| lora:base_model.model.model.language_model.layers.29.mlp.gate_proj.lora_B.default.weight | 9.5061e-01 | 2.9038e-01 | 3.055e-01 | 2.703e-04 |
| lora:base_model.model.model.language_model.layers.31.self_attn.q_proj.lora_B.default.weight | 1.0935e+00 | 3.3397e-01 | 3.054e-01 | 3.109e-04 |
| lora:base_model.model.model.language_model.layers.29.linear_attn.in_proj_z.lora_B.default.weight | 3.1037e-01 | 9.0468e-02 | 2.915e-01 | 8.422e-05 |
| lora:base_model.model.model.language_model.layers.28.linear_attn.in_proj_b.lora_B.default.weight | 2.0560e-01 | 5.8174e-02 | 2.829e-01 | 5.416e-05 |
| lora:base_model.model.model.language_model.layers.31.self_attn.k_proj.lora_B.default.weight | 8.5504e-01 | 2.2522e-01 | 2.634e-01 | 2.097e-04 |

## B vs B 噪声底线（同序前 15 项对照）

| param | ‖gB1‖ | ‖gB1−gB2‖ | 相对差 | 差值/总范数 |
|---|---|---|---|---|
| lora:base_model.model.model.visual.patch_embed.proj.lora_B.default.weight | 6.6933e+01 | 4.5795e+00 | 6.842e-02 | 4.212e-03 |
| lora:base_model.model.model.visual.blocks.1.attn.qkv.lora_B.default.weight | 1.4808e+01 | 4.7224e-01 | 3.189e-02 | 4.343e-04 |
| lora:base_model.model.model.visual.blocks.0.attn.qkv.lora_B.default.weight | 4.9137e+01 | 1.4526e+00 | 2.956e-02 | 1.336e-03 |
| lora:base_model.model.model.visual.blocks.1.attn.proj.lora_B.default.weight | 1.5530e+01 | 4.3672e-01 | 2.812e-02 | 4.017e-04 |
| lora:base_model.model.model.visual.blocks.0.attn.proj.lora_B.default.weight | 3.0293e+01 | 8.0737e-01 | 2.665e-02 | 7.425e-04 |
| lora:base_model.model.model.visual.blocks.0.mlp.linear_fc2.lora_B.default.weight | 1.0637e+01 | 2.8236e-01 | 2.655e-02 | 2.597e-04 |
| lora:base_model.model.model.visual.blocks.0.mlp.linear_fc1.lora_B.default.weight | 2.1114e+01 | 5.5825e-01 | 2.644e-02 | 5.134e-04 |
| lora:base_model.model.model.visual.blocks.2.attn.qkv.lora_B.default.weight | 2.6738e+01 | 6.9937e-01 | 2.616e-02 | 6.432e-04 |
| lora:base_model.model.model.visual.blocks.3.attn.proj.lora_B.default.weight | 1.4519e+01 | 3.5143e-01 | 2.420e-02 | 3.232e-04 |
| lora:base_model.model.model.visual.blocks.1.mlp.linear_fc1.lora_B.default.weight | 1.3198e+01 | 3.1267e-01 | 2.369e-02 | 2.876e-04 |
| lora:base_model.model.model.visual.blocks.2.attn.proj.lora_B.default.weight | 2.0930e+01 | 4.9276e-01 | 2.354e-02 | 4.532e-04 |
| lora:base_model.model.model.visual.blocks.4.attn.qkv.lora_B.default.weight | 1.3486e+01 | 3.0880e-01 | 2.290e-02 | 2.840e-04 |
| lora:base_model.model.model.visual.blocks.5.attn.proj.lora_B.default.weight | 9.1417e+00 | 2.0925e-01 | 2.289e-02 | 1.924e-04 |
| lora:base_model.model.model.visual.blocks.6.attn.qkv.lora_B.default.weight | 1.0711e+01 | 2.4439e-01 | 2.282e-02 | 2.248e-04 |
| lora:base_model.model.model.visual.blocks.5.attn.qkv.lora_B.default.weight | 1.4980e+01 | 3.4117e-01 | 2.278e-02 | 3.138e-04 |
