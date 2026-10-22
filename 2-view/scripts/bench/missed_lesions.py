#!/usr/bin/env python3
"""e3 (b8) 在 train1000 上：统计 95 个 GT 有病灶乳房中未被检出的那些，
给出 GT 阳性 finding 的 P(finding=1) 分布与病灶类型分布。纯 CPU。"""
import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path

THIS = Path(__file__).resolve().parent
PROJ = THIS.parent.parent
sys.path.insert(0, str(PROJ / "scripts" / "eval"))
import common as C  # noqa: E402

PROBS = PROJ / "outputs" / "scoring" / "overfit_train_e3_b8_with_probs.jsonl"
OUT = PROJ / "outputs" / "eval" / "missed_lesions_train1000.md"


def q(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def fmt(v):
    return "—" if v is None else f"{v:.4f}"


def main():
    recs = C.load_probs_jsonl(str(PROBS))
    lesion = [r for r in recs if any(r["gt_findings"][k] for k in C.FINDINGS)]
    missed = [r for r in lesion
              if not any(r["greedy_findings"].get(k, 0) == 1 for k in C.FINDINGS)]
    detected = len(lesion) - len(missed)

    lines = ["# train1000 上 e3(b8) 漏检病灶分析", ""]
    lines.append(f"- 样本: `{PROBS.name}` (n={len(recs)}, seed42 抽样)")
    lines.append(f"- GT 有病灶乳房: **{len(lesion)}**，检出 **{detected}**，漏检 **{len(missed)}**")
    lines.append("")

    # ---- 概率分布 ----
    max_p, all_p, per_type = [], [], {k: [] for k in C.FINDINGS}
    for r in missed:
        pos = [k for k in C.FINDINGS if r["gt_findings"][k]]
        ps = [r["p_findings"][k] for k in pos]
        max_p.append(max(ps))
        for k, p in zip(pos, ps):
            all_p.append(p)
            per_type[k].append(p)
    def block(name, vals):
        return (f"| {name} | {len(vals)} | {fmt(st.median(vals) if vals else None)} | "
                f"{fmt(q(vals, 0.25))} | {fmt(q(vals, 0.75))} | "
                f"{sum(1 for v in vals if v > 0.3)} | {sum(1 for v in vals if v > 0.4)} |")

    lines += ["## 1. 漏检样本的 P(finding=1) 分布（仅 GT 阳性那些 finding）", "",
              "| 统计口径 | n | 中位数 | Q1 | Q3 | >0.3 个数 | >0.4 个数 |", "|---|---|---|---|---|---|---|",
              block("每个漏检乳房取其 GT 阳性 finding 的最大概率", max_p),
              block("全部 (漏检乳房 × GT 阳性 finding)", all_p)]
    for k in C.FINDINGS:
        if per_type[k]:
            lines.append(block(f"  └ {k}", per_type[k]))
    lines.append("")

    # ---- 病灶类型分布 ----
    lines += ["## 2. 漏检样本的病灶类型分布", "", "| 类型 | 漏检个数 | 该类 GT 阳性总数 | 漏检率 |",
              "|---|---|---|---|"]
    gt_type = Counter()
    for r in lesion:
        for k in C.FINDINGS:
            if r["gt_findings"][k]:
                gt_type[k] += 1
    miss_type = Counter()
    for r in missed:
        for k in C.FINDINGS:
            if r["gt_findings"][k]:
                miss_type[k] += 1
    for k in C.FINDINGS:
        tot = gt_type[k]
        lines.append(f"| {k} | {miss_type[k]} | {tot} | "
                     f"{(miss_type[k] / tot if tot else 0):.4f} |")
    comb = Counter(tuple(sorted(k for k in C.FINDINGS if r["gt_findings"][k])) for r in missed)
    comb_all = Counter(tuple(sorted(k for k in C.FINDINGS if r["gt_findings"][k])) for r in lesion)
    lines += ["", "| 组合 (GT 阳性 finding) | 漏检个数 | 该类总数 |", "|---|---|---|"]
    for c, n in comb.most_common():
        lines.append(f"| {'+'.join(c)} | {n} | {comb_all[c]} |")
    lines.append("")
    lines += ["## 3. 判读", "",
              "- 检出定义：贪心输出中任一 finding=1（不加阈值）。漏检 = 6 个字段全为 0。",
              "- 若漏检样本的 GT 阳性 finding 概率中位数仍很低，说明是排序/证据不足而非阈值问题；"
              "若集中在 0.2–0.5，则降阈值有收益。"]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
