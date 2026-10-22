#!/usr/bin/env python3
"""对比两份 probs jsonl 的贪心输出（greedy_prediction），逐字段统计不同条数 +
给出差异样本的真实类别分布。"""
import json
import sys
from collections import Counter

FINDINGS = ["mass", "suspicious_calcification", "asymmetry", "architectural_distortion"]
FIELDS = ["breast_density", "breast_birads"] + [f"findings.{k}" for k in FINDINGS]


def load(path):
    d = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                d[r["id"]] = r
    return d


def get_val(pred, field):
    if not isinstance(pred, dict):
        return None
    if field.startswith("findings."):
        fnd = pred.get("findings") or {}
        return fnd.get(field.split(".", 1)[1])
    return pred.get(field)


def get_gt(gt, field):
    if not isinstance(gt, dict):
        return None
    if field.startswith("findings."):
        fnd = gt.get("findings") or {}
        return fnd.get(field.split(".", 1)[1])
    return gt.get(field)


def norm(v):
    if v is None:
        return "None"
    if isinstance(v, str):
        return v.upper() if v.upper() in ("A", "B", "C", "D") else v
    return str(v)


def main(pa, pb, na="原流水线", nb="batch=8"):
    A, B = load(pa), load(pb)
    ids = [i for i in A if i in B]
    print(f"A={na}  n={len(A)}")
    print(f"B={nb}  n={len(B)}")
    print(f"common ids={len(ids)}")
    missing = [i for i in A if i not in B]
    if missing:
        print(f"[warn] A 中有 {len(missing)} 条在 B 中缺失")

    print("\n================ 逐字段预测差异 ================")
    print(f"{'字段':<32}{'不同条数':>10}{'占比':>10}{'一致条数':>10}")
    diff_ids_by_field = {}
    for f in FIELDS:
        d = [i for i in ids if norm(get_val(A[i].get("greedy_prediction"), f))
             != norm(get_val(B[i].get("greedy_prediction"), f))]
        diff_ids_by_field[f] = d
        print(f"{f:<32}{len(d):>10}{len(d) / len(ids):>9.1%}{len(ids) - len(d):>10}")

    # 整条 JSON 完全一致
    same_all = sum(1 for i in ids if A[i].get("greedy_prediction") == B[i].get("greedy_prediction"))
    anydiff = len({i for d in diff_ids_by_field.values() for i in d})
    print(f"\n整条 prediction 完全相同: {same_all}/{len(ids)}")
    print(f"至少一个字段不同的样本数: {anydiff}")

    print("\n================ 差异样本的真实类别分布 ================")
    for f in FIELDS:
        d = diff_ids_by_field[f]
        if not d:
            print(f"\n[{f}] 无差异")
            continue
        gt_c = Counter(norm(get_gt(A[i].get("ground_truth"), f)) for i in d)
        pa_c = Counter(norm(get_val(A[i].get("greedy_prediction"), f)) for i in d)
        pb_c = Counter(norm(get_val(B[i].get("greedy_prediction"), f)) for i in d)
        print(f"\n[{f}] 差异 {len(d)} 条")
        print(f"  真实类别分布: {dict(sorted(gt_c.items()))}")
        print(f"  原流水线预测: {dict(sorted(pa_c.items()))}")
        print(f"  batch=8 预测: {dict(sorted(pb_c.items()))}")
        pair = Counter(
            (norm(get_val(A[i].get("greedy_prediction"), f)), norm(get_val(B[i].get("greedy_prediction"), f)))
            for i in d
        )
        print(f"  迁移 (原→b8): {dict(sorted(pair.items()))}")
        print(f"  样本 id (前 10): {d[:10]}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
