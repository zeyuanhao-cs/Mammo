#!/usr/bin/env python3
"""
Study 级 train/val 划分 (validation-based threshold calibration 准备, 第 1 步).

规则:
  - 从 train_raw.json (7999 乳房级样本) 按 study (patient_id) 切 20% val
    同一 study 的 L/R 乳房必须在一起, 不分离
  - 多标签分层 (iterative stratification, 简化贪心实现, 候选随机洗牌):
      10 个 study 级标签 (L/R 取"或"):
        4 个 finding、BI-RADS>=3、BI-RADS 4/5、BI-RADS 5、
        density A、density D、multi_flag (任一乳腺 >=2 findings 阳性)
  - val 保持自然分布, 不做均衡、不重采样
  - seed=42
  - test.json 保持官方 test, 不动

输出 (2-view/data/):
  - split_study_ids.json   # {val_studies, train_studies, 元信息}
  - train_sub_raw.json     # 剩余 train 乳房级样本 (后续 Repeat-OS 重采样输入)
  - val_raw.json           # val 乳房级样本 (后续 scoring 用)
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

SEED = 42
VAL_RATIO = 0.2

SRC = Path("/nesi/project/massey04900/Mammography/train_raw.json")
THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR.parent / "data"
RAW_DIR = DATA_DIR / "raw"

FINDINGS = ["mass", "suspicious_calcification", "asymmetry", "architectural_distortion"]


def study_labels(samples):
    """study 级标签: L/R 侧取'或'."""
    labs = set()
    for k in FINDINGS:
        if any(s["output"]["findings"][k] == 1 for s in samples):
            labs.add(f"finding:{k}")
    b = [s["output"]["breast_birads"] for s in samples]
    if any(x >= 3 for x in b):
        labs.add("birads_ge3")
    if any(x >= 4 for x in b):
        labs.add("birads_45")
    if any(x == 5 for x in b):
        labs.add("birads_5")
    d = [s["output"]["breast_density"] for s in samples]
    if any(x == "A" for x in d):
        labs.add("density_A")
    if any(x == "D" for x in d):
        labs.add("density_D")
    # 复杂度: study 内任一乳腺有 >=2 个 findings 阳性
    if any(sum(1 for k in FINDINGS if s["output"]["findings"][k] == 1) >= 2 for s in samples):
        labs.add("multi_flag")
    return labs


def main():
    rng = random.Random(SEED)
    data = json.loads(SRC.read_text(encoding="utf-8"))

    # ---- 按 study 分组 (patient_id 即 study; id = <study>_<L|R>) ----
    studies = defaultdict(list)
    for s in data:
        studies[s["patient_id"]].append(s)
    n_studies = len(studies)

    study_lab = {sid: study_labels(ss) for sid, ss in studies.items()}
    label_sets = defaultdict(set)
    for sid, labs in study_lab.items():
        for l in labs:
            label_sets[l].add(sid)

    # ---- iterative stratification (rarest label first) ----
    unallocated = set(studies)
    val, train = set(), set()
    label_order = sorted(label_sets, key=lambda l: len(label_sets[l]))

    for l in label_order:
        cands = [sid for sid in label_sets[l] if sid in unallocated]
        if not cands:
            continue
        # 期望进入 val 的该标签数量 (扣除已通过更早标签进入 val 的)
        need = VAL_RATIO * len(label_sets[l]) - len(label_sets[l] & val)
        # 固定顺序后随机洗牌, 直接取前 need 个 (不按"剩余标签数"排序)
        cands.sort()
        rng.shuffle(cands)
        for i, sid in enumerate(cands):
            if i < need:
                val.add(sid)
            else:
                train.add(sid)
            unallocated.discard(sid)

    # ---- 无任何阳性标签的 study: 随机 20% 进 val ----
    rest = sorted(unallocated)
    rng.shuffle(rest)
    n_val_rest = round(len(rest) * VAL_RATIO)
    val.update(rest[:n_val_rest])
    train.update(rest[n_val_rest:])

    assert not (val & train)
    assert len(val) + len(train) == n_studies

    # ---- 乳房级样本划分 ----
    val_samples = [s for s in data if s["patient_id"] in val]
    train_samples = [s for s in data if s["patient_id"] in train]
    assert len(val_samples) + len(train_samples) == len(data)
    # 校验: 同一 study 的乳房没有被拆开 (由分组保证), val/train 乳房 id 无交集
    assert not ({s["id"] for s in val_samples} & {s["id"] for s in train_samples})

    # ---- 保存 ----
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_ids = {
        "seed": SEED,
        "val_ratio": VAL_RATIO,
        "source": str(SRC),
        "stratify_labels": sorted(label_sets.keys()),
        "n_studies_total": n_studies,
        "n_studies_val": len(val),
        "n_studies_train": len(train),
        "n_breasts_val": len(val_samples),
        "n_breasts_train": len(train_samples),
        "val_studies": sorted(val),
        "train_studies": sorted(train),
    }
    (RAW_DIR / "split_study_ids.json").write_text(
        json.dumps(out_ids, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (RAW_DIR / "train_sub_raw.json").write_text(
        json.dumps(train_samples, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (RAW_DIR / "val_raw.json").write_text(
        json.dumps(val_samples, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    # ---- 报告 ----
    print(f"studies: total={n_studies}  val={len(val)} ({len(val)/n_studies:.1%})  train={len(train)}")
    print(f"breasts: total={len(data)}  val={len(val_samples)}  train={len(train_samples)}")

    print("\n=== 分层标签 study 级阳性数 (val / train / val占比) ===")
    for l in label_order:
        v, t = len(label_sets[l] & val), len(label_sets[l] & train)
        print(f"  {l:28s} val={v:4d}  train={t:4d}  ({v/(v+t):.1%})")

    def dist(samples, name):
        birads = Counter(s["output"]["breast_birads"] for s in samples)
        dens = Counter(s["output"]["breast_density"] for s in samples)
        pos = {k: sum(1 for s in samples if s["output"]["findings"][k] == 1) for k in FINDINGS}
        z = sum(1 for s in samples if all(s["output"]["findings"][k] == 0 for k in FINDINGS))
        print(f"\n--- {name} (n={len(samples)}) 乳房级 ---")
        print(f"  BI-RADS:  {dict(sorted(birads.items()))}")
        print(f"  density:  {dict(sorted(dens.items()))}")
        print(f"  findings: {pos}")
        print(f"  all-zero: {z} ({z/len(samples):.1%})")

    dist(train_samples, "train 剩余")
    dist(val_samples, "val")

    # ---- 核对: 乳房级 val 占比 (BI-RADS 各类 + multi-flag) ----
    print("\n=== 核对: 乳房级 val 占比 ===")
    for c in [1, 2, 3, 4, 5]:
        v = sum(1 for s in val_samples if s["output"]["breast_birads"] == c)
        tot = sum(1 for s in data if s["output"]["breast_birads"] == c)
        print(f"  BI-RADS {c}: val={v}/{tot} ({v / tot:.1%})")
    v = sum(1 for s in val_samples if sum(1 for k in FINDINGS if s["output"]["findings"][k] == 1) >= 2)
    tot = sum(1 for s in data if sum(1 for k in FINDINGS if s["output"]["findings"][k] == 1) >= 2)
    print(f"  multi-flag (>=2 findings): val={v}/{tot} ({v / tot:.1%})")

    print(f"\nOutput: {DATA_DIR}/split_study_ids.json, train_sub_raw.json, val_raw.json")


if __name__ == "__main__":
    main()
