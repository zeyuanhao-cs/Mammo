# 校验 breast_level_vindr.json 的 5 项一致性检查
import json
import os
import pandas as pd
from collections import Counter, defaultdict

CSV = "/nesi/project/massey04900/Mammography/vindr_detection_v1_folds.csv"
J = "/nesi/project/massey04900/Mammography/breast_level_vindr.json"
IMG_DIR = "/nesi/project/massey04900/Mammography/images_png"

with open(J) as f:
    samples = json.load(f)
df = pd.read_csv(CSV, low_memory=False)

anomalies = defaultdict(list)

# ---------- 1. 路径存在性 ----------
n_missing_cc = n_missing_mlo = 0
for s in samples:
    for view, p in s["images"].items():
        if p is None:
            anomalies["missing_view_path"].append((s["patient_id"], s["laterality"], view, "null路径"))
            if view == "CC":
                n_missing_cc += 1
            else:
                n_missing_mlo += 1
        elif not os.path.exists(p):
            anomalies["missing_view_path"].append((s["patient_id"], s["laterality"], view, p))
            if view == "CC":
                n_missing_cc += 1
            else:
                n_missing_mlo += 1
print(f"[1] CC 路径不存在: {n_missing_cc}, MLO 路径不存在: {n_missing_mlo}")

# ---------- 2. study_id 列是否存在；用 series_id 对比分组 ----------
cols = df.columns.tolist()
if "study_id" in cols:
    study_col = "study_id"
else:
    study_col = None
    print(f"[2] CSV 无 study_id 列。列名: {cols[:6]}... 用最接近的 series_id 对比。")

key_col = "study_id" if study_col else "series_id"
g_pat = df.groupby(["patient_id", "laterality"]).size()
g_study = df.groupby([key_col, "laterality"]).size()
print(f"    (patient_id, laterality) 组数: {len(g_pat)}")
print(f"    ({key_col}, laterality) 组数: {len(g_study)}")

# 乳房 -> series 的映射：一个乳房内有多少个不同 series；一个 series 是否跨乳房
breast2series = df.groupby(["patient_id", "laterality"])[key_col].nunique()
n_multi = (breast2series > 1).sum()
print(f"    单个乳房内含多个 {key_col} 的样本数: {n_multi}")
for (pid, lat), c in breast2series[breast2series > 1].items():
    sers = df[(df.patient_id == pid) & (df.laterality == lat)][key_col].unique().tolist()
    anomalies["breast_with_multiple_series"].append((pid, lat, sers))
series2breast = df.groupby(key_col)[["patient_id", "laterality"]].agg(["nunique"])
n_cross = (series2breast[("patient_id", "nunique")] > 1) | (series2breast[("laterality", "nunique")] > 1)
print(f"    跨多个乳房（患者或侧别不唯一）的 {key_col} 数: {n_cross.sum()}")
for ser in series2breast[n_cross].index[:20]:
    sub = df[df[key_col] == ser]
    anomalies["series_crossing_breasts"].append(
        (ser, sub["patient_id"].unique().tolist(), sub["laterality"].unique().tolist()))

# ---------- 3. no_finding 与原始 No_Finding 乳房级对比 ----------
orig_nf = df.groupby(["patient_id", "laterality"]).apply(
    lambda g: bool((g["No_Finding"] == 1).any()), include_groups=False)
n_mismatch = 0
for s in samples:
    k = (s["patient_id"], s["laterality"])
    if s["no_finding"] != orig_nf[k]:
        n_mismatch += 1
        # 找出差异原因：该乳房有哪些超出四类的 finding
        g = df[(df.patient_id == k[0]) & (df.laterality == k[1])]
        other_cols = ["Nipple_Retraction", "Skin_Retraction", "Skin_Thickening", "Suspicious_Lymph_Node"]
        others = [c for c in other_cols if (g[c] == 1).any()]
        anomalies["no_finding_mismatch"].append((k[0], k[1], s["breast_birads"],
                                                 f"json.no_finding={s['no_finding']} 原始No_Finding={orig_nf[k]} 其他finding={others}"))
print(f"[3] no_finding 与原始 No_Finding 不一致的乳房数: {n_mismatch}")

# ---------- 4. BI-RADS 与 findings 规则 ----------
n_rule_violation = 0
for s in samples:
    b = int(s["breast_birads"].split()[-1])
    any_pos = any(s["findings"].values())
    if b in (1, 2) and any_pos:
        anomalies["birads_rule_violation"].append((s["patient_id"], s["laterality"], s["breast_birads"], "BI-RADS 1/2 但有阳性finding", s["findings"]))
        n_rule_violation += 1
    if b in (3, 4, 5) and not any_pos:
        # 检查是否只有四类之外的 finding
        g = df[(df.patient_id == s["patient_id"]) & (df.laterality == s["laterality"])]
        other_cols = ["Nipple_Retraction", "Skin_Retraction", "Skin_Thickening", "Suspicious_Lymph_Node"]
        others = [c for c in other_cols if (g[c] == 1).any()]
        anomalies["birads_rule_violation"].append((s["patient_id"], s["laterality"], s["breast_birads"], f"BI-RADS {b} 但四finding全零", f"其他finding={others}"))
        n_rule_violation += 1
print(f"[4] BI-RADS/findings 规则违例数: {n_rule_violation}")

# ---------- 5. 输出异常样本列表 ----------
print("\n[5] 异常样本列表")
total_anom = sum(len(v) for v in anomalies.values())
if total_anom == 0:
    print("    无异常")
for k, v in anomalies.items():
    print(f"\n  -- {k} ({len(v)}) --")
    for item in v[:30]:
        print(f"     {item}")
    if len(v) > 30:
        print(f"     ... 其余 {len(v) - 30} 条略")

# 保存完整异常清单
with open("/nesi/project/massey04900/Mammography/breast_level_anomalies.txt", "w") as f:
    for k, v in anomalies.items():
        f.write(f"== {k} ({len(v)}) ==\n")
        for item in v:
            f.write(f"  {item}\n")
print(f"\n共 {total_anom} 条异常，已保存: /nesi/project/massey04900/Mammography/breast_level_anomalies.txt")
