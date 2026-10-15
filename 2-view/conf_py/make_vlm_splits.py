# 基于 breast_level_vindr.json 生成最终 VLM 训练文件
# 1) 删除 CC/MLO 为 null 的样本 2) test split 原样保留 3) training 按 patient 分组划分 train/val (10%)
import json
import random
from collections import Counter

SRC = "/nesi/project/massey04900/Mammography/breast_level_vindr.json"
OUT_DIR = "/nesi/project/massey04900/Mammography"
REMOVED_LOG = f"{OUT_DIR}/removed_samples.json"
SEED = 42
VAL_RATIO = 0.1

with open(SRC) as f:
    samples = json.load(f)

# ---------- 1. 删除 CC 或 MLO 为 null 的样本 ----------
kept, removed = [], []
for s in samples:
    if s["images"]["CC"] is None or s["images"]["MLO"] is None:
        removed.append({
            "patient_id": s["patient_id"],
            "laterality": s["laterality"],
            "reason": "CC is null" if s["images"]["CC"] is None else "MLO is null",
            "breast_birads": s["breast_birads"],
            "split": s["split"],
        })
    else:
        kept.append(s)
print(f"[1] 删除样本数: {len(removed)} / {len(samples)}")
for r in removed:
    print(f"    删除: {r['patient_id']} {r['laterality']} ({r['reason']}, {r['breast_birads']}, {r['split']})")
with open(REMOVED_LOG, "w") as f:
    json.dump(removed, f, indent=2, ensure_ascii=False)

# ---------- 2/3. 划分 train/val/test ----------
train_pool = [s for s in kept if s["split"] == "training"]
test_set = [s for s in kept if s["split"] == "test"]
assert len(train_pool) + len(test_set) == len(kept)

# ---------- 检查1: official train/test patient_id 泄漏 ----------
official_train_pids = {s["patient_id"] for s in samples if s["split"] == "training"}
official_test_pids = {s["patient_id"] for s in samples if s["split"] == "test"}
overlap = sorted(official_train_pids & official_test_pids)
print(f"[检查1] official train/test patient_id overlap: {len(overlap)}")
for p in overlap:
    print(f"    重叠 patient_id: {p}")

# ---------- 划分: 不再切分 val，training 全部作为训练集 ----------
FINDING_KEYS = ["mass", "suspicious_calcification", "asymmetry", "architectural_distortion"]
train_set = train_pool
print(f"[2] test 样本数: {len(test_set)} (官方 split 原样保留)")
print(f"[3] train 样本: {len(train_set)} (training split 全量, 不再划分 val)")

# ---------- 4/5/6. 转换为 VLM 训练格式，findings bool -> 0/1 ----------
def convert(s):
    return {
        "id": f"{s['patient_id']}_{s['laterality']}",
        "images": [s["images"]["CC"], s["images"]["MLO"]],
        "patient_id": s["patient_id"],
        "laterality": s["laterality"],
        "output": {
            "breast_birads": int(s["breast_birads"].split()[-1]),      # "BI-RADS 4" -> 4
            "breast_density": s["breast_density"].split()[-1],          # "DENSITY C" -> "C"
            "findings": {
                "mass": int(s["findings"]["mass"]),
                "suspicious_calcification": int(s["findings"]["suspicious_calcification"]),
                "asymmetry": int(s["findings"]["asymmetry"]),
                "architectural_distortion": int(s["findings"]["architectural_distortion"]),
            },
        },
    }

for name, data in [("train_raw", train_set), ("test", test_set)]:
    with open(f"{OUT_DIR}/{name}.json", "w") as f:
        json.dump([convert(s) for s in data], f, indent=1, ensure_ascii=False)
    print(f"[4] 已保存 {name}.json ({len(data)} 条)")

# ---------- 7. 分布统计 ----------
BIRADS = [1, 2, 3, 4, 5]
DENSITY = ["A", "B", "C", "D"]
FINDINGS = ["mass", "suspicious_calcification", "asymmetry", "architectural_distortion"]
lines = []
P = lines.append
P("=== train/test 分布统计（VLM 训练格式, 无验证集） ===")
P(f"official train/test patient_id overlap: {len(overlap)}")
for name, data in [("train_raw", train_set), ("test", test_set)]:
    n = len(data)
    conv = [convert(s) for s in data]
    P(f"\n----- {name} (n={n}) -----")
    P("BI-RADS: " + "  ".join(
        f"{b}:{sum(1 for c in conv if c['output']['breast_birads'] == b)}" for b in BIRADS))
    P("Density: " + "  ".join(
        f"{d}:{sum(1 for c in conv if c['output']['breast_density'] == d)}" for d in DENSITY))
    for fd in FINDINGS:
        c = sum(1 for x in conv if x["output"]["findings"][fd] == 1)
        P(f"{fd}: {c} ({c / n * 100:.2f}%)")
    nz = sum(1 for x in conv if not any(x["output"]["findings"].values()))
    P(f"no_finding(四项全零): {nz} ({nz / n * 100:.2f}%)")

stats_text = "\n".join(lines)
print("\n" + stats_text)
with open(f"{OUT_DIR}/train_val_test_stats.txt", "w") as f:
    f.write(stats_text + "\n")
print(f"\n统计已保存: {OUT_DIR}/train_val_test_stats.txt")
