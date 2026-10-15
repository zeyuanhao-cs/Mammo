# 构建 VinDr-Mammo 同侧双视图乳房级样本
# 分组键: (patient_id, laterality)，合并 CC + MLO 两张图
import json
import pandas as pd
from collections import Counter

CSV = "/nesi/project/massey04900/Mammography/vindr_detection_v1_folds.csv"
IMG_DIR = "/nesi/project/massey04900/Mammography/images_png"
OUT_JSON = "/nesi/project/massey04900/Mammography/breast_level_vindr.json"
OUT_STATS = "/nesi/project/massey04900/Mammography/breast_level_stats.txt"

df = pd.read_csv(CSV, low_memory=False)

BIRADS_ORDER = ["BI-RADS 1", "BI-RADS 2", "BI-RADS 3", "BI-RADS 4", "BI-RADS 5"]
DENSITY_ORDER = ["DENSITY A", "DENSITY B", "DENSITY C", "DENSITY D"]
FINDINGS = ["mass", "suspicious_calcification", "asymmetry", "architectural_distortion"]

birads_rank = {b: i + 1 for i, b in enumerate(BIRADS_ORDER)}
dens_rank = {d: ord(d[-1]) for d in DENSITY_ORDER}  # A=65..D=68

n_birads_conflict = 0
n_density_conflict = 0
n_missing_cc = 0
n_missing_mlo = 0
n_split_conflict = 0

samples = []
for (pid, lat), g in df.groupby(["patient_id", "laterality"]):
    # 每张图取一行图像级标签（同一图多行标注时 breast_birads/density 相同）
    img_rows = g.drop_duplicates(subset="image_id")
    cc = img_rows[img_rows["view"] == "CC"]
    mlo = img_rows[img_rows["view"] == "MLO"]
    cc_row = cc.iloc[0] if len(cc) else None
    mlo_row = mlo.iloc[0] if len(mlo) else None

    if cc_row is None:
        n_missing_cc += 1
    if mlo_row is None:
        n_missing_mlo += 1

    # BI-RADS：两视图不一致记冲突，取 max
    birads_vals = [r["breast_birads"] for r in (cc_row, mlo_row) if r is not None]
    if len(birads_vals) > 1 and birads_vals[0] != birads_vals[1]:
        n_birads_conflict += 1
    breast_birads = max(birads_vals, key=lambda b: birads_rank.get(b, 0))

    # 密度：不一致记冲突，取多数（两视图各一票，冲突时取第一个）
    dens_vals = [r["breast_density"] for r in (cc_row, mlo_row) if r is not None]
    if len(dens_vals) > 1 and dens_vals[0] != dens_vals[1]:
        n_density_conflict += 1
        dens_count = Counter(dens_vals)
        breast_density = dens_count.most_common(1)[0][0]  # 平票时 Counter 保持插入顺序（第一个）
    else:
        breast_density = dens_vals[0]

    # findings：任一视图出现即为阳性
    findings = {
        "mass": bool((g["Mass"] == 1).any()),
        "suspicious_calcification": bool((g["Suspicious_Calcification"] == 1).any()),
        "asymmetry": bool((g[["Asymmetry", "Focal_Asymmetry", "Global_Asymmetry"]] == 1).any().any()),
        "architectural_distortion": bool((g["Architectural_Distortion"] == 1).any()),
    }
    no_finding = not any(findings.values())

    # split：组内不一致记冲突，取第一个
    splits = g["split"].unique().tolist()
    if len(splits) > 1:
        n_split_conflict += 1
    split = splits[0]

    samples.append({
        "patient_id": pid,
        "laterality": lat,
        "images": {
            "CC": f"{IMG_DIR}/{pid}/{cc_row['image_id']}" if cc_row is not None else None,
            "MLO": f"{IMG_DIR}/{pid}/{mlo_row['image_id']}" if mlo_row is not None else None,
        },
        "breast_birads": breast_birads,
        "breast_density": breast_density,
        "findings": findings,
        "no_finding": no_finding,
        "split": split,
    })

# ---------------- 统计 ----------------
lines = []
P = lines.append
n = len(samples)
P(f"VinDr-Mammo 乳房级样本统计（{n} 个乳房样本，patient_id + laterality 分组）")
P(f"样本总数: {n}")
tr = sum(1 for s in samples if s["split"] == "training")
te = sum(1 for s in samples if s["split"] == "test")
P(f"训练集: {tr}  测试集: {te}")

P("\n=== BI-RADS 1-5 分布 ===")
birads_dist = Counter(s["breast_birads"] for s in samples)
for b in BIRADS_ORDER:
    c = birads_dist.get(b, 0)
    P(f"{b}: {c} ({c / n * 100:.2f}%)")

P("\n=== breast_density A-D 分布 ===")
dens_dist = Counter(s["breast_density"] for s in samples)
for d in DENSITY_ORDER:
    c = dens_dist.get(d, 0)
    P(f"{d}: {c} ({c / n * 100:.2f}%)")

P("\n=== 四个 finding 阳性数和占比 ===")
for f in FINDINGS:
    c = sum(1 for s in samples if s["findings"][f])
    P(f"{f}: {c} ({c / n * 100:.2f}%)")

all_zero = sum(1 for s in samples if s["no_finding"])
P(f"\n四项全零 (no_finding): {all_zero} ({all_zero / n * 100:.2f}%)")

P("\n=== 各 finding 在 BI-RADS 1-5 中的分布（阳性数 / 该 BI-RADS 样本数） ===")
header = f"{'finding':<28s}" + "".join(f"{b.split()[-1]:>10s}" for b in BIRADS_ORDER)
P(header)
birads_n = {b: birads_dist.get(b, 0) for b in BIRADS_ORDER}
for f in FINDINGS:
    row = f"{f:<28s}"
    for b in BIRADS_ORDER:
        c = sum(1 for s in samples if s["breast_birads"] == b and s["findings"][f])
        row += f"{c:>7d}/{birads_n[b]:<3d}"
    P(row)
# no_finding 也按 BI-RADS 分布
row = f"{'no_finding':<28s}"
for b in BIRADS_ORDER:
    c = sum(1 for s in samples if s["breast_birads"] == b and s["no_finding"])
    row += f"{c:>7d}/{birads_n[b]:<3d}"
P(row)

P("\n=== 视图缺失与标签冲突 ===")
P(f"CC 缺失: {n_missing_cc}")
P(f"MLO 缺失: {n_missing_mlo}")
P(f"BI-RADS 两视图冲突（取 max）: {n_birads_conflict}")
P(f"密度两视图冲突（取多数/第一个）: {n_density_conflict}")
P(f"split 组内冲突: {n_split_conflict}")

stats_text = "\n".join(lines)
print(stats_text)

with open(OUT_JSON, "w") as f:
    json.dump(samples, f, indent=1, ensure_ascii=False)
with open(OUT_STATS, "w") as f:
    f.write(stats_text + "\n")
print(f"\nSaved: {OUT_JSON}")
print(f"Saved: {OUT_STATS}")
