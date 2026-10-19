# 反方向前缀干预分析 (B 顺序 · density → BI-RADS → findings)

- 数据目录: `../../outputs/prefix_probe_dualorder_B_smoke`
- 样本: greedy 20 / down 20 / up 20, 三条件共同 id **20**
- 模型: 双顺序 LoRA + `prompt_order_b.txt`；贪心预测取自 `predictions_qwen3.5-4b_dualorder_b_2000.jsonl`
- bootstrap: 按 study 配对重采样 **1000** 次, seed=42, 95% CI 取百分位 2.5/97.5
- 干预方式: 只把 `breast_birads` 改成 `down=1` / `up=4`, density 保持贪心; 前缀截断到 `"findings": {` 之后, **findings 由模型自己贪心续写**
- P(1) = 生成该 finding 数字那一步的 softmax 概率; 比例类指标按样本计数

## 子集 1 · BI-RADS 改为 1 (down) → findings 是否跟着塌成 0

筛选: `GT BI-RADS >= 3` 且 `贪心 BI-RADS >= 3` 且 至少一个 finding 真阳性(真值=1 且贪心=1)

样本为空。

## 子集 2 · BI-RADS 改为 4 (up) → findings 是否被凭空生成

筛选: `GT BI-RADS 1–2` 且 GT 与贪心的 findings 全 0 且 `贪心 BI-RADS 1–2`

- n = **17** 条, 10 个 study (筛选命中 19 条; 4 个 findings 未全部解析出: greedy 2 条 / up 0 条, 并集剔除 2 条)

| 指标 | 值 (95% CI) |
|---|---|
| up 后出现 **>=1 个 finding=1** 的比例 | 1.0000 [1.0000, 1.0000] |
| up 后 4 个 findings 仍全 0 的比例 | 0.0000 [0.0000, 0.0000] |

| finding | 平均 P(1) greedy | 平均 P(1) up | Δ | 生成=1 比例 greedy → up |
|---|---|---|---|---|
| mass | 0.0008 [0.0004, 0.0016] | 0.4367 [0.3909, 0.4869] | 0.4358 [0.3905, 0.4854] | 0.0000 → 0.1176 |
| suspicious_calcification | 0.0002 [0.0001, 0.0005] | 0.1170 [0.0859, 0.1510] | 0.1168 [0.0857, 0.1507] | 0.0000 → 0.0000 |
| asymmetry | 0.0013 [0.0009, 0.0020] | 0.5468 [0.4686, 0.6205] | 0.5455 [0.4675, 0.6194] | 0.0000 → 0.7647 |
| architectural_distortion | 0.0002 [0.0001, 0.0005] | 0.2762 [0.1549, 0.4841] | 0.2760 [0.1545, 0.4839] | 0.0000 → 0.1176 |


