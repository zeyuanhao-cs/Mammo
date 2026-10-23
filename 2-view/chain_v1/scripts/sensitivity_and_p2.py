#!/usr/bin/env python3
"""补两组统计（纯 CPU）：
 1. 敏感性曲线数值：p=0,0.1,...,1 时 20 例 P(>=3) 的中位数/四分位数；每例首次 >0.5 的 p（中位数、范围）
 2. GT 征象全 0 的乳房：teacher 前缀下 BI-RADS 分布里 P2 的分布（真实 2 级 vs 1 级），
    以及真实 2 级中 P2 > P1 的比例
输出 chain_v1/outputs/soft_evidence_extra.md
"""
import json
import statistics as st
from pathlib import Path

CV1 = Path(__file__).resolve().parent.parent
OUT = CV1 / "outputs" / "soft_evidence_extra.md"
FIND = ["mass", "suspicious_calcification", "asymmetry", "architectural_distortion"]


def q(vals, p):
    s = sorted(vals)
    if not s:
        return None
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def f6(v):
    return "—" if v is None else f"{v:.6f}"


def load(split):
    p = CV1 / "outputs" / f"soft_evidence_{split}.jsonl"
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main():
    lines = ["# 软证据补充统计", ""]

    # ---------- 1. 敏感性曲线 ----------
    d = json.loads((CV1 / "outputs" / "soft_evidence_sensitivity.json").read_text(encoding="utf-8"))
    grid, rows = d["grid"], d["rows"]
    lines += ["## 1. 敏感性曲线（mass p: 0→1，20 例）", "",
              "| mass p | 中位数 P(>=3) | Q1 | Q3 |", "|---|---|---|---|"]
    for i, g in enumerate(grid):
        vals = [r["p_ge3"][i] for r in rows]
        lines.append(f"| {g:.1f} | {f6(st.median(vals))} | {f6(q(vals, 0.25))} | {f6(q(vals, 0.75))} |")

    cross = []
    never = 0
    for r in rows:
        hit = next((g for g, v in zip(grid, r["p_ge3"]) if v > 0.5), None)
        if hit is None:
            never += 1
        else:
            cross.append(hit)
    lines += ["", "| 首次 >0.5 的 p | 值 |", "|---|---|",
              f"| 例数（能跨过 0.5） | {len(cross)}/20 |",
              f"| 中位数 | {f6(st.median(cross)) if cross else '—'} |",
              f"| 最小 | {f6(min(cross)) if cross else '—'} |",
              f"| 最大 | {f6(max(cross)) if cross else '—'} |",
              f"| 从未跨过 0.5 的例数 | {never} |", ""]

    # ---------- 2. 全 0 征象组的 P2 ----------
    lines += ["## 2. GT 征象全 0 的乳房：teacher 前缀下 P2 的分布", "",
              "| split | n(gt1/gt2) | P2 中位数 (gt2) | P2 中位数 (gt1) | gt2 中 P2>P1 的比例 |",
              "|---|---|---|---|---|"]
    pooled2, pooled1 = [], []
    for split in ("test", "val"):
        rows2 = [r for r in load(split) if all(r["gt_findings"][k] == 0 for k in FIND)]
        g2 = [r for r in rows2 if r["gt_birads"] == 2]
        g1 = [r for r in rows2 if r["gt_birads"] == 1]
        p2_2 = [r["teacher_birads"][1] for r in g2]
        p2_1 = [r["teacher_birads"][1] for r in g1]
        pooled2 += p2_2
        pooled1 += p2_1
        frac = (sum(1 for r in g2 if r["teacher_birads"][1] > r["teacher_birads"][0]) / len(g2)) if g2 else None
        lines.append(f"| {split} | {len(g1)}/{len(g2)} | {f6(st.median(p2_2)) if p2_2 else '—'} | "
                     f"{f6(st.median(p2_1)) if p2_1 else '—'} | {f6(frac)} |")
    rows_all = [r for s in ("test", "val") for r in load(s)
                if all(r["gt_findings"][k] == 0 for k in FIND) and r["gt_birads"] == 2]
    frac_all = sum(1 for r in rows_all if r["teacher_birads"][1] > r["teacher_birads"][0]) / len(rows_all)
    lines.append(f"| test+val | {len(pooled1)}/{len(pooled2)} | {f6(st.median(pooled2))} | "
                 f"{f6(st.median(pooled1))} | {f6(frac_all)} |")
    lines += ["", "（P2 = teacher 前缀下 BI-RADS 分布里等级 2 的概率，P1 同理；teacher = 用 GT 征象构成的前缀。）", ""]

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
