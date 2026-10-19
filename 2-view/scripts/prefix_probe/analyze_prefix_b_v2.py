#!/usr/bin/env python3
"""反方向前缀干预分析 v2 (双顺序模型 · B 顺序 · prompt_order_b).

与 v1 的区别:
  - 前缀截到 `"findings":` 冒号之后 (token 边界), 由模型自己生成 ` {"mass": ...}`,
    不再把原文的 `{"` token 切成 ` {`;
  - 条件扩展为 greedy / down(BI=1) / up3(BI=3) / up4(BI=4) / up5(BI=5);
  - 前置检查: greedy 与原预测的一致率、各条件空对象 (`{}`) 比例。

输入 (run_probe.py --zi-b 产出, 默认目录 outputs/prefix_probe_dualorder_B_v2):
  probs_{greedy,down,up3,up4,up5}_test.jsonl

统计: 按 study 配对 bootstrap 1000 次 (seed=42), 95% CI 取百分位 2.5/97.5。

用法:
  python analyze_prefix_b_v2.py [--out-root ...] [--n-boot 1000] [--seed 42]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from analyze_prefix_b import (FINDINGS, boot_ci, f_p1, f_values, fmt, fmt_ci,
                              greedy_birads, greedy_findings, group_by_study,
                              gt_findings, load_mode, study_of, subset1_ids,
                              subset2_ids, analyse)

THIS_DIR = Path(__file__).resolve().parent
PROJ_DIR = THIS_DIR.parent.parent
DEFAULT_OUT = PROJ_DIR / "outputs" / "prefix_probe_dualorder_B_v2"
PREDS = PROJ_DIR / "outputs" / "predictions" / "predictions_qwen3.5-4b_dualorder_b_2000.jsonl"

MODES = ("greedy", "down", "up3", "up4", "up5")
UPS = ("up3", "up4", "up5")


def load_preds(path: Path) -> dict:
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                out[d["id"]] = d
    return out


def empty_mask(rows: dict, ids: list) -> np.ndarray:
    """该条件是否直接输出空 findings (`{}`, 4 个字段一个都没生成)."""
    return np.array([np.isnan(f_values(rows[s])).all() for s in ids])


def agree_mask(probe: dict, preds: dict) -> np.ndarray:
    """greedy 重新生成的 4 个 finding 是否与原预测完全一致."""
    out = []
    for sid in probe:
        fs = f_values(probe[sid])
        pf = ((preds[sid].get("prediction_json") or {}).get("findings") or {})
        pv = np.array([float(pf.get(k, np.nan)) for k in FINDINGS])
        out.append(bool((~np.isnan(fs)).all() and np.array_equal(fs, pv)))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=str(DEFAULT_OUT))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    data = {m: load_mode(out_root, m) for m in MODES}
    missing = [m for m in MODES if not data[m]]
    if missing:
        print(f"[error] 缺少 {missing} 的 jsonl (目录 {out_root})")
        return 1
    preds = load_preds(PREDS)

    ids = sorted(set.intersection(*[set(data[m]) for m in MODES]))
    ids = [i for i in ids if i in preds]
    print(f"[data] " + " ".join(f"{m}={len(data[m])}" for m in MODES)
          + f"  共同 id={len(ids)}")

    merged = {}
    for sid in ids:
        r = dict(data["greedy"][sid])
        for m in MODES[1:]:
            r[m] = data[m][sid]
        merged[sid] = r

    # ---------- 0. greedy 与原预测的一致率 ----------
    g = data["greedy"]
    ok_ids = [s for s in ids if (~np.isnan(f_values(g[s]))).all()]
    ag = agree_mask({s: g[s] for s in ok_ids}, preds)
    grp = group_by_study(ok_ids)
    agree_ci = boot_ci(grp, lambda ix: ag[ix].mean())
    print(f"[agree] 4 字段齐全 {len(ok_ids)} 条, 与原预测一致 {int(ag.sum())} 条 "
          f"({ag.mean():.4f})")

    # ---------- 1. 各条件空对象比例 ----------
    empty = {m: empty_mask(data[m], ids) for m in MODES}
    empty_ci = {m: boot_ci(group_by_study(ids), lambda ix, m=m: empty[m][ix].mean())
                for m in MODES}
    for m in MODES:
        print(f"[empty] {m:6s} {int(empty[m].sum()):5d}/{len(ids)} "
              f"= {empty[m].mean():.4f}")

    # ---------- 子集 ----------
    s1, s2 = subset1_ids(merged), subset2_ids(merged)
    print(f"[subset1] {len(s1)} 条 / {len(set(study_of(s) for s in s1))} studies")
    print(f"[subset2] {len(s2)} 条 / {len(set(study_of(s) for s in s2))} studies")

    r1 = analyse(s1, merged, "down", "down") if s1 else None
    r2 = {m: (analyse(s2, merged, m, "up") if s2 else None) for m in UPS}

    # ---------- 报告 ----------
    L = []
    L.append("# 反方向前缀干预分析 v2 (B 顺序 · density → BI-RADS → findings)")
    L.append("")
    L.append(f"- 数据目录: `{out_root}`")
    L.append(f"- 样本: " + " / ".join(f"{m} {len(data[m])}" for m in MODES)
             + f", 五条件共同 id **{len(ids)}**")
    L.append(f"- 贪心预测来源: `predictions_qwen3.5-4b_dualorder_b_2000.jsonl`")
    L.append("- 干预方式: 只改 `breast_birads` (down=1 / up3=3 / up4=4 / up5=5), "
             "density 保持贪心; 前缀截到 `\"findings\":` 冒号之后, "
             "` {\"mass\": ...}` 由模型自己贪心生成")
    L.append("- 前缀校验: 前缀 token 序列必须是完整输出 token 序列的严格前缀, "
             "不满足的样本打印并跳过 (v2 为 0 条)")
    L.append(f"- bootstrap: 按 study 配对重采样 **{args.n_boot}** 次, seed={args.seed}, "
             f"95% CI 取百分位 2.5/97.5")
    L.append("- P(1) = 生成该 finding 数字那一步的 softmax 概率; 比例类指标按样本计数")
    L.append("- 空对象 = 模型在 `\"findings\":` 后直接生成 `{}`, 4 个字段一个都没写出")
    L.append("")

    L.append("## 0. 前置检查")
    L.append("")
    L.append("| 检查 | 值 |")
    L.append("|---|---|")
    L.append(f"| greedy 重新生成 vs 原预测 一致率 (n={len(ok_ids)}) | "
             f"{int(ag.sum())}/{len(ok_ids)} = **{ag.mean():.4f}** "
             f"[{fmt(agree_ci[1])}, {fmt(agree_ci[2])}] |")
    for m in MODES:
        L.append(f"| 空对象比例 · {m} | {int(empty[m].sum())}/{len(ids)} = "
                 f"{fmt_ci(*empty_ci[m])} |")
    L.append("")

    # ---------- 子集 1 ----------
    L.append("## 子集 1 · BI-RADS 改为 1 (down) → findings 是否跟着塌成 0")
    L.append("")
    L.append("筛选: `GT BI-RADS >= 3` 且 `贪心 BI-RADS >= 3` 且 至少一个 finding 真阳性"
             "(真值=1 且贪心=1)")
    L.append("")
    if r1 is None:
        L.append("样本为空。")
    else:
        L.append(f"- n = **{r1['n']}** 条, {r1['n_studies']} 个 study "
                 f"(筛选命中 {r1['n_sel']} 条; down 空对象 {r1['n_empty_mode']} 条, "
                 f"greedy 空对象 {r1['n_empty_greedy']} 条)")
        L.append("")
        L.append("| 指标 | 值 (95% CI) |")
        L.append("|---|---|")
        L.append(f"| down 后 4 个 findings **全为 0** 的比例 | {fmt_ci(*r1['event'])} |")
        L.append(f"| 其中 down 直接输出空 `findings: {{}}` | {fmt_ci(*r1['empty_rate_mode'])} |")
        L.append(f"| down 后出现 >=1 个 finding=1 的比例 | {fmt_ci(*r1['any1_rate'])} |")
        L.append("")
        L.append("| finding | 平均 P(1) greedy | 平均 P(1) down | Δ | 生成=1 比例 greedy → down |")
        L.append("|---|---|---|---|---|")
        for k in FINDINGS:
            v = r1["per_finding"].get(k)
            if not v:
                L.append(f"| {k} | - | - | - | - |")
                continue
            L.append(f"| {k} | {fmt_ci(*v['p1_greedy'])} | {fmt_ci(*v['p1_mode'])} | "
                     f"{fmt_ci(*v['delta'])} | {fmt(v['rate1_greedy'][0])} → "
                     f"{fmt(v['rate1_mode'][0])} |")
        L.append("")
        ns = [v["n"] for v in r1["per_finding"].values() if v]
        if ns:
            L.append(f"- P(1) 三列基于 {min(ns)}–{max(ns)} 行 (空对象行的 P(1) 记为缺失)")
    L.append("")

    # ---------- 子集 2 ----------
    L.append("## 子集 2 · BI-RADS 抬高 (up3 / up4 / up5) → findings 是否被凭空生成")
    L.append("")
    L.append("筛选: `GT BI-RADS 1–2` 且 GT 与贪心的 findings 全 0 且 `贪心 BI-RADS 1–2`")
    L.append("")
    if not s2:
        L.append("样本为空。")
    else:
        base = r2[UPS[0]]
        L.append(f"- n = **{base['n']}** 条, {base['n_studies']} 个 study "
                 f"(筛选命中 {base['n_sel']} 条)")
        L.append("")
        L.append("| 指标 | " + " | ".join(UPS) + " |")
        L.append("|---|" + "---|" * len(UPS))
        L.append("| 出现 **>=1 个 finding=1** 的比例 | "
                 + " | ".join(fmt_ci(*r2[m]["event"]) for m in UPS) + " |")
        L.append("| 4 个 findings 仍全 0 的比例 | "
                 + " | ".join(fmt_ci(*r2[m]["all0_rate"]) for m in UPS) + " |")
        L.append("| 直接输出空 `findings: {}` 的比例 | "
                 + " | ".join(fmt_ci(*r2[m]["empty_rate_mode"]) for m in UPS) + " |")
        L.append("")
        L.append("### 各 finding 被标成 1 的比例")
        L.append("")
        L.append("| finding | greedy | " + " | ".join(UPS) + " |")
        L.append("|---|---|" + "---|" * len(UPS))
        for k in FINDINGS:
            v0 = base["per_finding"].get(k)
            cells = []
            for m in UPS:
                v = r2[m]["per_finding"].get(k)
                cells.append("-" if not v else fmt(v["rate1_mode"][0]))
            L.append(f"| {k} | " + ("-" if not v0 else fmt(v0["rate1_greedy"][0]))
                     + " | " + " | ".join(cells) + " |")
        L.append("")
        L.append("### 各 finding 的平均 P(1) (95% CI)")
        L.append("")
        L.append("| finding | greedy | " + " | ".join(UPS) + " |")
        L.append("|---|---|" + "---|" * len(UPS))
        for k in FINDINGS:
            v0 = base["per_finding"].get(k)
            cells = []
            for m in UPS:
                v = r2[m]["per_finding"].get(k)
                cells.append("-" if not v else fmt_ci(*v["p1_mode"]))
            L.append(f"| {k} | " + ("-" if not v0 else fmt_ci(*v0["p1_greedy"]))
                     + " | " + " | ".join(cells) + " |")
        L.append("")
        L.append("### 平均 P(1) 相对 greedy 的变化 Δ (95% CI)")
        L.append("")
        L.append("| finding | " + " | ".join(UPS) + " |")
        L.append("|---|" + "---|" * len(UPS))
        for k in FINDINGS:
            cells = []
            for m in UPS:
                v = r2[m]["per_finding"].get(k)
                cells.append("-" if not v else fmt_ci(*v["delta"]))
            L.append(f"| {k} | " + " | ".join(cells) + " |")
        L.append("")
        L.append("| 条件 | n | 空对象 |")
        L.append("|---|---|---|")
        for m in UPS:
            L.append(f"| {m} | {r2[m]['n']} | {r2[m]['n_empty_mode']} |")
    L.append("")

    md = out_root / "prefix_b_v2_report.md"
    md.write_text("\n".join(L) + "\n", encoding="utf-8")
    print()
    print("\n".join(L))
    print(f"[report] -> {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
