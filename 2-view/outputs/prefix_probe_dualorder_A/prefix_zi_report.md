# zero / inject 前缀干预分析 (双顺序模型 · prompt_order_a)

- 数据目录: `/home/z/zhao/Mammography/Mammo/2-view/outputs/prefix_probe_dualorder_A`
- 样本: greedy 2000 / zero 2000 / inject 2000, 三条件共同 id **2000**
- bootstrap: 按 study 配对重采样 **1000** 次, seed=42, 95% CI 取百分位 2.5/97.5
- 前缀只改 findings 数字: `zero` = 4 个 findings 全 0; `inject` = 只把 `mass` 置 1; density 与 birads 文本保持贪心
- P(BI>=3) = P(3)+P(4)+P(5); 级别 = 5 类概率的 argmax

## 子集 1 · findings 全置 0 (zero) 对 BI-RADS 的降级影响

筛选: `GT BI-RADS >= 3` 且 至少一个 finding 真阳性(真值=1 且贪心=1) 且 `贪心 BI-RADS >= 3`

- n = **52** 条, 52 个 study

| 指标 | 值 (95% CI) |
|---|---|
| 平均 P(BI>=3) · greedy | 0.9862 [0.9760, 0.9946] |
| 平均 P(BI>=3) · zero | 0.0547 [0.0458, 0.0637] |
| Δ 平均 P(BI>=3) (zero − greedy) | -0.9315 [-0.9434, -0.9177] |
| 掉到 1–2 级 (argmax) 的比例 | 1.0000 [1.0000, 1.0000] |

解读: zero 后平均 P(BI>=3) 变化 -0.9315, 其中 100.0% 的样本 argmax 落回 1–2 级。

## 子集 2 · mass 置 1 (inject) 对 BI-RADS 的升级影响

筛选: `GT BI-RADS 1–2` 且 GT 与贪心的 findings 全 0 且 `贪心 BI-RADS 1–2`

- n = **1789** 条, 992 个 study

| 指标 | 值 (95% CI) |
|---|---|
| 平均 P(BI>=3) · greedy | 0.0020 [0.0018, 0.0022] |
| 平均 P(BI>=3) · inject | 0.9721 [0.9700, 0.9740] |
| Δ 平均 P(BI>=3) (inject − greedy) | 0.9702 [0.9680, 0.9721] |
| 升到 >=3 级 (argmax) 的比例 | 0.9978 [0.9949, 1.0000] |

解读: inject 后平均 P(BI>=3) 变化 0.9702, 其中 99.8% 的样本 argmax 升到 >=3 级。

