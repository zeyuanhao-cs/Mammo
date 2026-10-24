# 前向一致性测试（新代码 vs LLaMA-Factory collator/forward）

- 适配器：`/home/z/zhao/Mammography/model/Qwen3.5-4B-lora-2view-os-sub-vt`（1-epoch 解冻版），模型 eval 模式，bf16，786432
- 样本：`train_llamafactory_os_sub_local.json` 前 3 条

| # | prompt_len | comp_len | CE(A 新代码) | CE(B1 LF+pos_ids) | CE(B2 LF 去 pos_ids) | logits 最大差 A-B1 | logits 最大差 A-B2 | position_ids shape |
|---|---|---|---|---|---|---|---|---|
| 0 | 1906 | 58 | 0.040389 | 0.040389 | 0.040389 | 0.000000 | 0.000000 | (4, 1, 1964) |
| 1 | 1906 | 58 | 0.046889 | 0.046889 | 0.046889 | 0.000000 | 0.000000 | (4, 1, 1964) |
| 2 | 1906 | 58 | 0.069892 | 0.069892 | 0.069892 | 0.000000 | 0.000000 | (4, 1, 1964) |

**结论**：A-B1 最大差 0.000000，A-B2 最大差 0.000000 → 一致（三条路径 logits 差都很小）

