# 遮挡实验报告（3ep 模型，本地 A40 全流程复现）

- 模型: `Qwen3.5-4B` + `Qwen3.5-4B-lora-2view-os-sub-vt-ep3`（3ep，checkpoint-1317），贪心解码，batch=8，默认分辨率 `image_max_pixels=786432`（672×1152 → 756 视觉 token/图）
- 样本: 测试集 GT BI-RADS≥3 的 192 个乳房 × 4 种情况（orig / lesion / ctrl1 / ctrl2），遮挡图 384 视角 × 3 条件 = **1152 张**（912×1520，应用清单生成）
- Δ 定义: **Δ = 本次 orig − 遮挡后**（正 = 概率下降）；配对 Wilcoxon（双侧），对照 Δ 取 ctrl1/ctrl2 平均
- 遮挡与 NeSI 冻结模型实验**完全一致**（同 192 乳房、同 rect/fill/Otsu，本地重算 0 差异），结果可直接对比

## 0. 结论（TL;DR）

1. **3ep 模型与冻结模型的遮挡响应模式一致**：detected 组遮病灶 `max_sign_p` 中位 Δ = **0.816**（对照 0.013；冻结模型 0.627 / 0.055），遮病灶后 detected 组 64/68 从 ≥3 跌到 ≤2。
2. **missed_sign 的「怀疑」仍来自病灶但强度低一个量级**：中位 Δ = **0.119**（对照 0.011，Wilcoxon p <1e-4，88% 乳房病灶下降 > 对照；冻结模型 0.095 / 0.017、p<1e-4、81%）——约 detected 的 0.15×；遮病灶后 27/74 从 2 退到 1（对照 6 / 7；冻结模型 20/67，对照 10）。
3. **missed_nosign 基本不来自病灶**：中位 Δ = **0.0127**（对照 0.0008；冻结模型 0.003），实质为零。
4. **单视角有框乳房会稀释 Δ**（30 个）；排除后结论不变，见各主表的 excl_sv 版本。

### 与 NeSI 冻结模型并排对比（max_sign_p 中位 Δ）

| 组 | 冻结模型 n | 冻结 Δlesion | 冻结 Δctrl | 3ep n | 3ep Δlesion | 3ep Δctrl | 3ep p | 3ep 病灶>对照比例 |
|---|---|---|---|---|---|---|---|---|
| detected | 54 | 0.627 | 0.055 | 68 | 0.816 | 0.013 | <1e-4 | 0.88 |
| missed_sign | 67 | 0.095 | 0.017 | 74 | 0.119 | 0.011 | <1e-4 | 0.88 |
| missed_nosign | 71 | 0.003 | — | 50 | 0.013 | 0.001 | <1e-4 | 0.80 |

冻结模型 missed_sign 2→1: 20/67（对照 10）；3ep: 27/74（对照 6 / 7）。

## 1. 推理管线一致性（本次 orig vs 既有 2000 条）

| 项 | 值 |
|---|---|
| 对比样本数 | 192 |
| 贪心 BI-RADS 预测一致率 | 1.0000 |
| 不一致比例（阈值 2%） | 0.0000 |
| max 征象 P(=1) 最大/中位绝对差 | 0.00e+00 / 0.00e+00 |
| P(BI-RADS≥3) 最大/中位绝对差 | 0.00e+00 / 0.00e+00 |
| 判定 | **一致** |

## 2. 分类（本次 orig，规则与既有口径完全一致）

| 组 | old（既有 2000 条） | new（本次 orig） | 差 |
|---|---|---|---|
| detected | 68 | 68 | +0 |
| missed_sign | 74 | 74 | +0 |
| missed_nosign | 50 | 50 | +0 |
| 合计 | 192 | 192 | — |

换组乳房数：0 / 192（预期三类约 68 / 74 / 50）

## 3a. Δ 表 — 全部 192 乳房

Δ = orig − 遮挡后（正 = 下降）；ctrl = ctrl1/ctrl2 的 Δ 平均。

### max 征象 P(=1)

| 组 | n | 原图中位 | Δlesion | Δctrl | Wilcoxon p | 病灶>对照比例 |
|---|---|---|---|---|---|---|
| detected (new) | 68 | 0.9047 | 0.8161 | 0.0134 | <1e-4 | 0.88 |
| missed_sign (new) | 74 | 0.1921 | 0.1186 | 0.0106 | <1e-4 | 0.88 |
| missed_nosign (new) | 50 | 0.0421 | 0.0127 | 0.0008 | <1e-4 | 0.80 |
| detected (old) | 68 | 0.9047 | 0.8161 | 0.0134 | <1e-4 | 0.88 |
| missed_sign (old) | 74 | 0.1921 | 0.1186 | 0.0106 | <1e-4 | 0.88 |
| missed_nosign (old) | 50 | 0.0421 | 0.0127 | 0.0008 | <1e-4 | 0.80 |

### P(BI-RADS≥2)

| 组 | n | 原图中位 | Δlesion | Δctrl | Wilcoxon p | 病灶>对照比例 |
|---|---|---|---|---|---|---|
| detected (new) | 68 | 1.0000 | 0.8355 | 0.0000 | <1e-4 | 0.88 |
| missed_sign (new) | 74 | 0.5312 | 0.2167 | 0.0518 | <1e-4 | 0.86 |
| missed_nosign (new) | 50 | 0.0851 | 0.0252 | 0.0067 | <1e-4 | 0.80 |
| detected (old) | 68 | 1.0000 | 0.8355 | 0.0000 | <1e-4 | 0.88 |
| missed_sign (old) | 74 | 0.5312 | 0.2167 | 0.0518 | <1e-4 | 0.86 |
| missed_nosign (old) | 50 | 0.0851 | 0.0252 | 0.0067 | <1e-4 | 0.80 |

### P(BI-RADS≥3)

| 组 | n | 原图中位 | Δlesion | Δctrl | Wilcoxon p | 病灶>对照比例 |
|---|---|---|---|---|---|---|
| detected (new) | 68 | 1.0000 | 0.9999 | 0.0000 | <1e-4 | 0.90 |
| missed_sign (new) | 74 | 0.0000 | 0.0000 | -0.0000 | <1e-4 | 0.91 |
| missed_nosign (new) | 50 | 0.0000 | 0.0000 | -0.0000 | <1e-4 | 0.82 |
| detected (old) | 68 | 1.0000 | 0.9999 | 0.0000 | <1e-4 | 0.90 |
| missed_sign (old) | 74 | 0.0000 | 0.0000 | -0.0000 | <1e-4 | 0.91 |
| missed_nosign (old) | 50 | 0.0000 | 0.0000 | -0.0000 | <1e-4 | 0.82 |

## 3b. Δ 表 — 排除单视角有框乳房

Δ = orig − 遮挡后（正 = 下降）；ctrl = ctrl1/ctrl2 的 Δ 平均。

### max 征象 P(=1)

| 组 | n | 原图中位 | Δlesion | Δctrl | Wilcoxon p | 病灶>对照比例 |
|---|---|---|---|---|---|---|
| detected (new) | 66 | 0.9047 | 0.8217 | 0.0134 | <1e-4 | 0.88 |
| missed_sign (new) | 61 | 0.2018 | 0.1224 | 0.0114 | <1e-4 | 0.89 |
| missed_nosign (new) | 35 | 0.0421 | 0.0128 | 0.0021 | 0.0002 | 0.77 |
| detected (old) | 66 | 0.9047 | 0.8217 | 0.0134 | <1e-4 | 0.88 |
| missed_sign (old) | 61 | 0.2018 | 0.1224 | 0.0114 | <1e-4 | 0.89 |
| missed_nosign (old) | 35 | 0.0421 | 0.0128 | 0.0021 | 0.0002 | 0.77 |

### P(BI-RADS≥2)

| 组 | n | 原图中位 | Δlesion | Δctrl | Wilcoxon p | 病灶>对照比例 |
|---|---|---|---|---|---|---|
| detected (new) | 66 | 1.0000 | 0.8437 | 0.0000 | <1e-4 | 0.88 |
| missed_sign (new) | 61 | 0.6225 | 0.2269 | 0.0565 | <1e-4 | 0.87 |
| missed_nosign (new) | 35 | 0.0851 | 0.0278 | 0.0177 | 0.0006 | 0.80 |
| detected (old) | 66 | 1.0000 | 0.8437 | 0.0000 | <1e-4 | 0.88 |
| missed_sign (old) | 61 | 0.6225 | 0.2269 | 0.0565 | <1e-4 | 0.87 |
| missed_nosign (old) | 35 | 0.0851 | 0.0278 | 0.0177 | 0.0006 | 0.80 |

### P(BI-RADS≥3)

| 组 | n | 原图中位 | Δlesion | Δctrl | Wilcoxon p | 病灶>对照比例 |
|---|---|---|---|---|---|---|
| detected (new) | 66 | 0.9999 | 0.9999 | 0.0000 | <1e-4 | 0.89 |
| missed_sign (new) | 61 | 0.0000 | 0.0000 | -0.0000 | <1e-4 | 0.92 |
| missed_nosign (new) | 35 | 0.0000 | 0.0000 | -0.0000 | 0.0005 | 0.77 |
| detected (old) | 66 | 0.9999 | 0.9999 | 0.0000 | <1e-4 | 0.89 |
| missed_sign (old) | 61 | 0.0000 | 0.0000 | -0.0000 | <1e-4 | 0.92 |
| missed_nosign (old) | 35 | 0.0000 | 0.0000 | -0.0000 | 0.0005 | 0.77 |

注：gt_sign_p（GT 类别征象概率）的完整数值在 `reports/step6.json`。

## 4. 遮病灶后 BI-RADS 迁移

| 分类 | 组 | n | 范围 | 2→1 | ≥3→≤2 |
|---|---|---|---|---|---|
| new/all | detected | 68 | lesion | 0 | 64 |
| new/all | detected | 68 | ctrl1 | 0 | 12 |
| new/all | detected | 68 | ctrl2 | 0 | 14 |
| new/all | missed_sign | 74 | lesion | 27 | 0 |
| new/all | missed_sign | 74 | ctrl1 | 6 | 0 |
| new/all | missed_sign | 74 | ctrl2 | 7 | 0 |
| new/all | missed_nosign | 50 | lesion | 0 | 0 |
| new/all | missed_nosign | 50 | ctrl1 | 0 | 0 |
| new/all | missed_nosign | 50 | ctrl2 | 0 | 0 |
| old/all | detected | 68 | lesion | 0 | 64 |
| old/all | detected | 68 | ctrl1 | 0 | 12 |
| old/all | detected | 68 | ctrl2 | 0 | 14 |
| old/all | missed_sign | 74 | lesion | 27 | 0 |
| old/all | missed_sign | 74 | ctrl1 | 6 | 0 |
| old/all | missed_sign | 74 | ctrl2 | 7 | 0 |
| old/all | missed_nosign | 50 | lesion | 0 | 0 |
| old/all | missed_nosign | 50 | ctrl1 | 0 | 0 |
| old/all | missed_nosign | 50 | ctrl2 | 0 | 0 |
| new/excl_sv | detected | 66 | lesion | 0 | 63 |
| new/excl_sv | detected | 66 | ctrl1 | 0 | 12 |
| new/excl_sv | detected | 66 | ctrl2 | 0 | 14 |
| new/excl_sv | missed_sign | 61 | lesion | 24 | 0 |
| new/excl_sv | missed_sign | 61 | ctrl1 | 4 | 0 |
| new/excl_sv | missed_sign | 61 | ctrl2 | 5 | 0 |
| new/excl_sv | missed_nosign | 35 | lesion | 0 | 0 |
| new/excl_sv | missed_nosign | 35 | ctrl1 | 0 | 0 |
| new/excl_sv | missed_nosign | 35 | ctrl2 | 0 | 0 |
| old/excl_sv | detected | 66 | lesion | 0 | 63 |
| old/excl_sv | detected | 66 | ctrl1 | 0 | 12 |
| old/excl_sv | detected | 66 | ctrl2 | 0 | 14 |
| old/excl_sv | missed_sign | 61 | lesion | 24 | 0 |
| old/excl_sv | missed_sign | 61 | ctrl1 | 4 | 0 |
| old/excl_sv | missed_sign | 61 | ctrl2 | 5 | 0 |
| old/excl_sv | missed_nosign | 35 | lesion | 0 | 0 |
| old/excl_sv | missed_nosign | 35 | ctrl1 | 0 | 0 |
| old/excl_sv | missed_nosign | 35 | ctrl2 | 0 | 0 |

## 5. missed_sign 内部：按预测 BI-RADS = 2 / = 1 拆分

| 分类 | 范围 | 子组 | n | 原图 max_sign | Δlesion | Δctrl | Wilcoxon p | 比例 |
|---|---|---|---|---|---|---|---|---|
| new/all | | pred=2 | 39 | 0.2451 | 0.1449 | 0.0045 | <1e-4 | 0.87 |
| new/all | | pred=1 | 35 | 0.1824 | 0.1100 | 0.0200 | <1e-4 | 0.89 |
| new/excl_sv | | pred=2 | 34 | 0.2451 | 0.1644 | 0.0060 | <1e-4 | 0.91 |
| new/excl_sv | | pred=1 | 27 | 0.1824 | 0.1133 | 0.0211 | <1e-4 | 0.85 |
| old/all | | pred=2 | 39 | 0.2451 | 0.1449 | 0.0045 | <1e-4 | 0.87 |
| old/all | | pred=1 | 35 | 0.1824 | 0.1100 | 0.0200 | <1e-4 | 0.89 |
| old/excl_sv | | pred=2 | 34 | 0.2451 | 0.1644 | 0.0060 | <1e-4 | 0.91 |
| old/excl_sv | | pred=1 | 27 | 0.1824 | 0.1133 | 0.0211 | <1e-4 | 0.85 |

| 分类 | 范围 | 子组 | Δlesion P(≥3) | Δctrl P(≥3) | Wilcoxon p |
|---|---|---|---|---|---|
| new/all | | pred=2 | 0.0000 | -0.0000 | <1e-4 |
| new/all | | pred=1 | 0.0000 | -0.0000 | <1e-4 |
| new/excl_sv | | pred=2 | 0.0000 | -0.0000 | <1e-4 |
| new/excl_sv | | pred=1 | 0.0000 | -0.0000 | <1e-4 |
| old/all | | pred=2 | 0.0000 | -0.0000 | <1e-4 |
| old/all | | pred=1 | 0.0000 | -0.0000 | <1e-4 |
| old/excl_sv | | pred=2 | 0.0000 | -0.0000 | <1e-4 |
| old/excl_sv | | pred=1 | 0.0000 | -0.0000 | <1e-4 |

## 6. 病灶 token 数（该视角最大病灶框(不外扩)面积/整图 × 756，每乳房取 CC/MLO 较大值）

| 分组 | n | 中位 | p25 | p75 |
|---|---|---|---|---|
| detected (new) | 68 | 40.1 | 18.2 | 108.4 |
| detected (new, excl_sv) | 66 | 40.1 | 18.6 | 111.0 |
| detected (old) | 68 | 40.1 | 18.2 | 108.4 |
| detected (old, excl_sv) | 66 | 40.1 | 18.6 | 111.0 |
| missed_sign (new) | 74 | 18.7 | 9.1 | 39.0 |
| missed_sign (new, excl_sv) | 61 | 22.2 | 9.0 | 41.5 |
| missed_sign (old) | 74 | 18.7 | 9.1 | 39.0 |
| missed_sign (old, excl_sv) | 61 | 22.2 | 9.0 | 41.5 |
| missed_nosign (new) | 50 | 9.0 | 5.5 | 25.3 |
| missed_nosign (new, excl_sv) | 35 | 8.6 | 4.8 | 26.8 |
| missed_nosign (old) | 50 | 9.0 | 5.5 | 25.3 |
| missed_nosign (old, excl_sv) | 35 | 8.6 | 4.8 | 26.8 |

| 病灶类型（全部 192） | n | 中位 | p25 | p75 | min | max |
|---|---|---|---|---|---|---|
| mass | 119 | 22.9 | 9.9 | 42.0 | 3.1 | 361.8 |
| calcification | 32 | 12.8 | 3.1 | 141.9 | 0.8 | 381.3 |
| other | 41 | 23.7 | 9.9 | 57.6 | 3.4 | 411.1 |

## 7. 只有一个视角有病灶框的乳房（30 个）

这些样本另一视角无遮挡，Δ 被稀释；各主表的 excl_sv 版本已排除它们。

| id | new 组 | old 组 |
|---|---|---|
| 02dba952e4ef2cf35613761a64bbc751_R | missed_nosign | missed_nosign |
| 043febde473009753d404fe89a41c395_R | missed_nosign | missed_nosign |
| 07889ee9b57d18e83cfd940b160965b6_R | missed_nosign | missed_nosign |
| 0fab8097f55f21d4bfa7667250122312_L | detected | detected |
| 1c41d5f075f0291af7f33545704df750_R | missed_nosign | missed_nosign |
| 25f7d988de28237bbf10273129097c9c_R | missed_sign | missed_sign |
| 268bbbd1932753cedf8846fe716c669b_L | missed_sign | missed_sign |
| 29bfec1e3b5a67751abc3dbf46df3e62_L | missed_nosign | missed_nosign |
| 31fa226bc9f87e0a2cb2de6dddbc2270_L | missed_nosign | missed_nosign |
| 5864ac95d2f1bcf325d36cbe8609297c_L | missed_nosign | missed_nosign |
| 66235fe8968ef266cf15b08cfaa4a40e_L | missed_nosign | missed_nosign |
| 6691601fba2d3233547361585c21fdde_L | missed_sign | missed_sign |
| 6e79c21e0593c8c459ef609b99dbfb70_L | missed_nosign | missed_nosign |
| 6e79c21e0593c8c459ef609b99dbfb70_R | missed_nosign | missed_nosign |
| 75b2eef816fe608734977d1ffb16eae9_R | missed_sign | missed_sign |
| 792610d6c4044d3a4e2cfdbc5b494768_R | missed_nosign | missed_nosign |
| 7b17b1d061b038ceee40c3811341b2fc_R | missed_nosign | missed_nosign |
| 805d72b89d7f73058bf420ef26caaff4_L | missed_sign | missed_sign |
| 8138169c67834e4e979fa17f38a826ad_R | missed_sign | missed_sign |
| 891ff1343a54c212f973f1b83ec4952c_L | missed_sign | missed_sign |
| 8e1fe4584f868271dce3e84e59b65388_L | missed_nosign | missed_nosign |
| 91489fc5a74d1452737d06d7113837f4_L | missed_sign | missed_sign |
| 968f72a200354163e45343aeeead090d_R | missed_sign | missed_sign |
| a1f8c1d32ac2da1967b3c6b3ee34616e_L | missed_nosign | missed_nosign |
| ad721e26365b1555dd539aed6123a851_R | missed_sign | missed_sign |
| ae38a60e17f0f0c0f48db63680d0f619_L | missed_sign | missed_sign |
| d0389f18021cd6e9294eef7d553039b3_R | missed_sign | missed_sign |
| d1808200af6ccda3be8abce5c9b4980f_L | detected | detected |
| ee117236523dbd94ef212e6b12eaca61_L | missed_sign | missed_sign |
| fc45fd8203dadd8899344e52935ff069_L | missed_nosign | missed_nosign |

## 8. 口径与已知限制

- 坐标: `vindr_detection_v1_folds.csv` 的 `resized_*` 即 912×1520 PNG 像素坐标（6 张叠加图人工核对，红框压病灶、按宽高比换算的绿框整体偏离；见 `check/ov*.png`）。
- 遮挡: lesion 框每边外扩 20%（裁到边界内）；对照与病灶框一一对应同尺寸，随机放在 Otsu 乳腺区内（≥80% 面积在乳腺内、不与任何外扩病灶框重叠）；填充 = 框外 16px 环中位灰度；seed 42。本地重算的 rect/fill/Otsu 与 NeSI manifest 384 视角 0 差异。
- 检查图: `check/ov*.png`（坐标验证）、`check/preview_*.png`（6 乳房 × 4 情况拼图）、`reports/step3_check.txt`（1152 张数量/尺寸校验）。
- 原始概率: `occlusion_probs/{{orig,lesion,ctrl1,ctrl2}}.jsonl`；完整数值统计: `reports/step6.json`。

