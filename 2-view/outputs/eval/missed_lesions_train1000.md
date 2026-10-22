# train1000 上 e3(b8) 漏检病灶分析

- 样本: `overfit_train_e3_b8_with_probs.jsonl` (n=1000, seed42 抽样)
- GT 有病灶乳房: **95**，检出 **32**，漏检 **63**

## 1. 漏检样本的 P(finding=1) 分布（仅 GT 阳性那些 finding）

| 统计口径 | n | 中位数 | Q1 | Q3 | >0.3 个数 | >0.4 个数 |
|---|---|---|---|---|---|---|
| 每个漏检乳房取其 GT 阳性 finding 的最大概率 | 63 | 0.1067 | 0.0397 | 0.2123 | 11 | 7 |
| 全部 (漏检乳房 × GT 阳性 finding) | 66 | 0.1067 | 0.0434 | 0.2018 | 11 | 7 |
|   └ mass | 38 | 0.1480 | 0.0430 | 0.3350 | 10 | 6 |
|   └ suspicious_calcification | 3 | 0.0534 | 0.0504 | 0.2304 | 1 | 1 |
|   └ asymmetry | 21 | 0.0851 | 0.0474 | 0.1480 | 0 | 0 |
|   └ architectural_distortion | 4 | 0.0439 | 0.0166 | 0.0719 | 0 | 0 |

## 2. 漏检样本的病灶类型分布

| 类型 | 漏检个数 | 该类 GT 阳性总数 | 漏检率 |
|---|---|---|---|
| mass | 38 | 65 | 0.5846 |
| suspicious_calcification | 3 | 17 | 0.1765 |
| asymmetry | 21 | 22 | 0.9545 |
| architectural_distortion | 4 | 4 | 1.0000 |

| 组合 (GT 阳性 finding) | 漏检个数 | 该类总数 |
|---|---|---|
| mass | 36 | 54 |
| asymmetry | 19 | 19 |
| architectural_distortion | 3 | 3 |
| suspicious_calcification | 2 | 7 |
| architectural_distortion+mass | 1 | 1 |
| asymmetry+mass | 1 | 1 |
| asymmetry+suspicious_calcification | 1 | 1 |

## 3. 判读

- 检出定义：贪心输出中任一 finding=1（不加阈值）。漏检 = 6 个字段全为 0。
- 若漏检样本的 GT 阳性 finding 概率中位数仍很低，说明是排序/证据不足而非阈值问题；若集中在 0.2–0.5，则降阈值有收益。
