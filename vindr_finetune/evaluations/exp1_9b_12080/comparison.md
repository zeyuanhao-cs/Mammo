# 实验一：Qwen3.5-9B 与合作者 4B 2:1 对比

9B 作业 12080 已完成 2 epoch 训练与全量 4000 例推理；最终 adapter 非空，运行错误 0，解析失败 4。
训练时间约 1:57:06，测试推理时间约 1:37:30。训练与推理的图像像素上限为 786432，关闭思考。

## 与原表相同 500 例的配对比较

合作者 main@20e980b 的 round3.sh 明确使用 --limit 500；其 direct_full/predictions.jsonl 也只有索引 0–499 的 500 例。该子集包含 357 张有病灶图（71.4%），全量 4000 例有病灶图同为 357 张（8.93%）。
4B 与 9B 的配对真值和 9B 的图像索引检查通过。评测代码重现原表 2:1 列全部 10 项指标（按原表保留一位小数）。以下保留两位小数，差值单位为百分点。

| 指标 | 4B 2:1 | 9B 2:1 | 差值（百分点） |
|---|---:|---:|---:|
| JSON 有效率 | 99.40% | 99.40% | +0.00 |
| Breast BI-RADS Accuracy | 36.20% | 43.60% | +7.40 |
| Breast BI-RADS Macro-F1 | 24.50% | 31.52% | +7.02 |
| Breast Density Accuracy | 78.80% | 83.20% | +4.40 |
| Breast Density Macro-F1 | 49.37% | 51.57% | +2.20 |
| Category Macro-F1 @ IoU 0.3 | 18.98% | 22.19% | +3.21 |
| Localized F1 @ IoU 0.3 | 25.33% | 30.14% | +4.81 |
| Finding BI-RADS Accuracy | 42.53% | 52.54% | +10.01 |
| Finding BI-RADS Macro-F1 | 18.38% | 27.76% | +9.38 |
| Joint Finding F1 @ IoU 0.3 | 10.77% | 15.84% | +5.07 |

9B 在这批 500 例上定位真阳性由 87 增至 118，类别、定位和 BI-RADS 同时正确的病灶由 37 增至 62。JSON 有效率持平，其余汇总任务指标提高。
但正常图误报由 13/143（9.09%）增至 27/143（18.88%）；病灶类别收益主要来自 Mass 与 Calcification，Asymmetry 与 Architectural Distortion 的类别定位 F1 略降。
若比较原表所有训练策略的最佳值，9B 密度 Macro-F1 为 51.57%，仍低于 4B Full train 的 52.6%；JSON 有效率也未超过 Full train 的 100%。

## 9B 全量 4000 例

| 指标 | 9B 全量 |
|---|---:|
| JSON 有效率 | 99.90% |
| Breast BI-RADS Accuracy | 60.30% |
| Breast BI-RADS Macro-F1 | 27.62% |
| Breast Density Accuracy | 82.30% |
| Breast Density Macro-F1 | 54.28% |
| Category Macro-F1 @ IoU 0.3 | 13.23% |
| Localized F1 @ IoU 0.3 | 15.55% |
| Finding BI-RADS Accuracy | 52.54% |
| Finding BI-RADS Macro-F1 | 27.76% |
| Joint Finding F1 @ IoU 0.3 | 8.17% |

全量预测病灶 1030 个，定位真阳性 118 个，联合真阳性 62 个。3643 张无病灶图中 754 张输出病灶（20.70%），说明正常图误报仍是主要问题。
尚无本次 2:1 4B 对应的全量 4000 例预测，因此不能把其 500 例指标直接与这里全量数值相比较，也不能确认全量 9B 相对全量 4B 的提升。

## 评测口径

- 图像级 BI-RADS 固定五类 1–5；密度固定四类 A–D；无效 JSON 按分类未命中和空病灶处理。
- 病灶先按类别一致、IoU ≥ 0.3、IoU 降序做一对一贪心匹配；同 IoU 时按索引升序。
- Category Macro-F1 对 Mass、Calcification、Asymmetry、Architectural Distortion 四类的定位 F1 求均值。
- Localized F1 统计所有病灶的类别与位置匹配；Finding BI-RADS 仅在定位匹配且真值 BI-RADS 非 null 的病灶上计算，Macro-F1 按原表固定五类 1–5。
- Joint Finding F1 复用定位匹配，仅将 BI-RADS 也一致的匹配记为真阳性，不为联合分数另行重新匹配。
- 本表是已有两次运行结果的实测比较；4B 原始运行环境快照未完整提供，不能把全部差异严格归因于模型规模。

## 可复核产物

- evaluate_mammo.py：标准库评测实现，原始预测只在 H200 内读取。
- metrics.json：聚合指标、计数、4B 提交与 blob、9B 预测 SHA256、评测代码 SHA256。
- 9B 预测 SHA256：3f327ddab60e8e0427c393edec29e9c20ddd8fc1183f9300bda2243a84dcc417
- 4B 来源提交：20e980b591a666d95aa098bdb4709c51a4c3864a
- 未把原始预测内容下载或加入本次评测提交。
