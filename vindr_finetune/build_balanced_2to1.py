#!/usr/bin/env python3
"""
构建 2:1 (lesion-positive : normal) 且 finding_category 类别平衡的训练集。

来源: train_lesion_only_1to3.json (1396 lesion + 4188 normal, 无 Associated Feature)
策略:
  1. lesion-positive 内部对 Mass/Calcification/Asymmetry/Architectural Distortion
     做 record 级类别平衡: 以最大类记录数为 target, 对少数类记录 oversample
     (复制含少数类的记录, 优先复制"所有类别都未超标"的记录, 避免多数类被进一步放大)
  2. normal 数量 = lesion 总数(含复制) / 2, 从 normal 池无放回随机采样
  3. 全部打散后输出
输出: data/train_balanced_2to1.json
"""
import json
import random
from collections import Counter
from pathlib import Path

SEED = 42
TARGET_CATS = ["Mass", "Calcification", "Asymmetry", "Architectural Distortion"]

THIS_DIR = Path(__file__).resolve().parent
SRC = THIS_DIR / "data" / "train_lesion_only_1to3.json"
OUT = THIS_DIR / "data" / "train_balanced_2to1.json"

random.seed(SEED)

data = json.load(open(SRC, encoding="utf-8"))
lesion, normal = [], []
for r in data:
    out = json.loads(r["output"])
    (lesion if out.get("findings") else normal).append(r)

print(f"source: {len(data)} -> lesion={len(lesion)}, normal={len(normal)}")

# 每条 lesion 记录包含的类别集合 (record 级)
rec_cats = []
for r in lesion:
    out = json.loads(r["output"])
    cats = {f["finding_category"] for f in out["findings"] if f["finding_category"] in TARGET_CATS}
    rec_cats.append(cats)

def cat_counts(multiset_cats):
    c = Counter()
    for cats in multiset_cats:
        for k in cats:
            c[k] += 1
    return c

base = cat_counts(rec_cats)
target = max(base[c] for c in TARGET_CATS)
print("record-level counts (before):", {c: base[c] for c in TARGET_CATS}, f"target={target}")

# 贪心 oversample: 每轮补最欠缺的类别, 选一条含该类别且其他类别尽量不超标的记录复制
multiset = list(rec_cats)  # 与最终记录列表一一对应
final_records = list(lesion)
guard = 0
while True:
    cur = cat_counts(multiset)
    deficits = {c: target - cur[c] for c in TARGET_CATS if cur[c] < target}
    if not deficits:
        break
    guard += 1
    assert guard < 100000, "oversample loop runaway"
    need_cat = max(deficits, key=deficits.get)
    # 候选: 含 need_cat 的原始记录, 按"其他类别是否已超标"排序, 优先全不超标
    cand = [
        (sum(1 for c in cats if cur[c] >= target and c != need_cat), i)
        for i, cats in enumerate(rec_cats)
        if need_cat in cats
    ]
    cand.sort()
    # 均匀轮转复制, 避免同一条被连续复制
    pick = cand[guard % len(cand)]
    i = pick[1]
    multiset.append(rec_cats[i])
    final_records.append(lesion[i])

balanced = cat_counts(multiset)
L = len(final_records)
N = L // 2
print(f"after oversample: {dict(balanced)}")
print(f"lesion total (含复制) = {L}, normal 采样 = {N} (池: {len(normal)})")
assert N <= len(normal), "normal 池不足"

normal_sampled = random.sample(normal, N)

result = final_records + normal_sampled
random.shuffle(result)

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False)

# 校验输出
chk_lesion = chk_normal = 0
chk_cats = Counter()
for r in result:
    out = json.loads(r["output"])
    if out.get("findings"):
        chk_lesion += 1
        for fd in out["findings"]:
            chk_cats[fd["finding_category"]] += 1
    else:
        chk_normal += 1
print("===== 输出校验 =====")
print(f"total={len(result)}, lesion={chk_lesion}, normal={chk_normal}, ratio={chk_lesion/chk_normal:.2f}:1")
print("record-level category coverage:", {c: balanced[c] for c in TARGET_CATS})
print("finding-level counts:", dict(chk_cats))
print(f"saved -> {OUT}")
