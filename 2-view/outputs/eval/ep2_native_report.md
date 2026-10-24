# 原分辨率 ep2（896×1536）vs 普通分辨率 ep2（786432）· b8 测试集

| model | BI-RADS AUC | ≥3 AUC | BI-RADS F1 | acc | bACC | Density AUC | Density F1 | Density bACC | findings mean AUROC |
|---|---|---|---|---|---|---|---|---|---|
| ep2 (786432) | 0.838121 | 0.808185 | 0.484220 | 0.779500 | 0.458694 | 0.939795 | 0.592621 | 0.558442 | 0.835117 |
| ep2-native (896x1536) | 0.828932 | 0.805434 | 0.485103 | 0.782000 | 0.457052 | 0.939829 | 0.553336 | 0.551101 | 0.849380 |

## 各类 precision / recall（probs argmax / greedy）

| model | 类别 | precision | recall | F1 | support |
|---|---|---|---|---|---|
| ep2 (786432) | BI-RADS 1 | 0.813517 | 0.969426 | 0.884655 | 1341 |
| ep2 (786432) | BI-RADS 2 | 0.761246 | 0.471092 | 0.582011 | 467 |
| ep2 (786432) | BI-RADS 3 | 0.333333 | 0.107527 | 0.162602 | 93 |
| ep2 (786432) | BI-RADS 4 | 0.274194 | 0.223684 | 0.246377 | 76 |
| ep2 (786432) | BI-RADS 5 | 0.571429 | 0.521739 | 0.545455 | 23 |
| ep2 (786432) | Density A | 1.000000 | 0.100000 | 0.181818 | 10 |
| ep2 (786432) | Density B | 0.712500 | 0.600000 | 0.651429 | 190 |
| ep2 (786432) | Density C | 0.886507 | 0.918954 | 0.902439 | 1530 |
| ep2 (786432) | Density D | 0.656126 | 0.614815 | 0.634799 | 270 |
| ep2-native (896x1536) | BI-RADS 1 | 0.801095 | 0.982103 | 0.882412 | 1341 |
| ep2-native (896x1536) | BI-RADS 2 | 0.796935 | 0.445396 | 0.571429 | 467 |
| ep2-native (896x1536) | BI-RADS 3 | 0.363636 | 0.086022 | 0.139130 | 93 |
| ep2-native (896x1536) | BI-RADS 4 | 0.372549 | 0.250000 | 0.299213 | 76 |
| ep2-native (896x1536) | BI-RADS 5 | 0.545455 | 0.521739 | 0.533333 | 23 |
| ep2-native (896x1536) | Density A | 0.000000 | 0.000000 | 0.000000 | 10 |
| ep2-native (896x1536) | Density B | 0.670157 | 0.673684 | 0.671916 | 190 |
| ep2-native (896x1536) | Density C | 0.894466 | 0.908497 | 0.901427 | 1530 |
| ep2-native (896x1536) | Density D | 0.658824 | 0.622222 | 0.640000 | 270 |

## findings（AUC=AUROC，F1/P/R 用贪心输出）

| model | finding | AUC | precision | recall | F1 | TP/FP/FN |
|---|---|---|---|---|---|---|
| ep2 (786432) | mass | 0.870809 | 0.543478 | 0.420168 | 0.473934 | 50/42/69 |
| ep2 (786432) | suspicious_calcification | 0.964462 | 0.707317 | 0.537037 | 0.610526 | 29/12/25 |
| ep2 (786432) | asymmetry | 0.720018 | 0.000000 | 0.000000 | 0.000000 | 0/0/49 |
| ep2 (786432) | architectural_distortion | 0.785181 | 0.000000 | 0.000000 | 0.000000 | 0/0/13 |
| ep2-native (896x1536) | mass | 0.857782 | 0.567568 | 0.352941 | 0.435233 | 42/32/77 |
| ep2-native (896x1536) | suspicious_calcification | 0.972627 | 0.789474 | 0.555556 | 0.652174 | 30/8/24 |
| ep2-native (896x1536) | asymmetry | 0.789171 | 0.000000 | 0.000000 | 0.000000 | 0/0/49 |
| ep2-native (896x1536) | architectural_distortion | 0.777941 | 0.000000 | 0.000000 | 0.000000 | 0/0/13 |

## 病灶检出（GT 有病灶乳房，贪心任一 finding=1）

大小分档阈值（框面积/整图面积）：小 ≤ 0.4036%，中 ≤ 1.2400%，大 > 1.2400%；样本数 {'small': 64, 'medium': 64, 'large': 64}

| model | 总体 | 小 | 中 | 大 |
|---|---|---|---|---|
| ep2 (786432) | 73/192 | 10/64 | 24/64 | 39/64 |
| ep2-native (896x1536) | 66/192 | 10/64 | 21/64 | 35/64 |

## 三类分析（基于普通分辨率 ep2 的 b8 结果，GT BI-RADS ≥3 的 192 个乳房）

detected = 预测 BI-RADS ≥3；missed_sign = 预测 ≤2 且（预测=2 或 4 个 finding 的 probs_norm 最大值 >0.1）；missed_nosign = 其余。

| 类别 | 个数 | 原分辨率模型检出数（预测 BI-RADS ≥3） |
|---|---|---|
| detected | 72 | 60 |
| missed_sign | 59 | 5 |
| missed_nosign | 61 | 0 |

## 差值显著性（配对 bootstrap，study 级，1000 次，seed 42；Δ = native − ep2）

| 指标 | ΔAUC | CI_low | CI_high | 是否显著 |
|---|---|---|---|---|
| BI-RADS macro AUC | -0.009189 | -0.020140 | 0.000522 | 否 |
| BI-RADS ≥3 AUC | -0.002751 | -0.019161 | 0.013616 | 否 |
| Density macro AUC | 0.000034 | -0.002961 | 0.003644 | 否 |
| findings mean AUROC | 0.014263 | -0.005106 | 0.033818 | 否 |
| finding mass | -0.013027 | -0.028573 | 0.001407 | 否 |
| finding suspicious_calcification | 0.008165 | -0.000333 | 0.019991 | 否 |
| finding asymmetry | 0.069153* | 0.024025 | 0.119216 | 是 |
| finding architectural_distortion | -0.007239 | -0.064514 | 0.046229 | 否 |

## 口径

- 全部为 b8 流水线（`infer_2view_batched.py`），测试集 2000 条；原分辨率推理用 `--image-max-pixels 1376256`（896×1536）。
- macro-F1 / acc / bACC 基于 probs argmax；bACC = 各类 recall 的平均。
- findings 的 F1/P/R 用贪心输出（greedy_findings），AUC 用 prob。
- 病灶大小分档脚本：`scripts/bench/lesion_size_bins.py`（面积比三分位，报告只读 `size_bins.json`，换阈值不必重推理）。
