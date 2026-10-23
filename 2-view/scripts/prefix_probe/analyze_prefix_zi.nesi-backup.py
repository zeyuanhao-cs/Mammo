#!/usr/bin/env python3
"""zero/inject 前缀干预分析 (只读已有概率文件, CPU).

子集 1 (阳性确认组): GT BI-RADS>=3 & 模型贪心 findings 至少 1 个预测正确 & greedy 前缀 BI 预测>=3
  -> 比较 greedy vs zero 的 P(BI>=3) 均值/中位数, zero 下 argmax 掉到 1-2 的比例
子集 2 (阴性注入组): GT BI-RADS 1-2 & 模型贪心 findings 全 0 & greedy 前缀 BI 预测 1-2
  -> 比较 greedy vs inject 的 P(BI>=3) 均值/中位数, inject 下 argmax 升到 >=3 的比例

全部指标带 study 级配对 bootstrap 95% CI (1000 次, seed 42).
输出: outputs/prefix_probe/prefix_zero_inject_report.{md,json}
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PROJ_DIR = THIS_DIR.parent.parent
OUT_ROOT = PROJ_DIR / "outputs" / "prefix_probe"

FINDINGS = ["mass", "suspicious_calcification", "asymmetry", "architectural_distortion"]
SEED = 42
BOOT_N = 1000


def load_rows(name):
    rows = {}
    with open(OUT_ROOT / name, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows[r["id"]] = r
    return rows


def p_ge3(row):
    b = row["probs_norm"]["breast_birads"]
    return sum(b[str(k)] for k in (3, 4, 5))


def birads_argmax(row):
    b = row["probs_norm"]["breast_birads"]
    return int(max(b, key=lambda k: b[k]))


def study_of(bid):
    return re.sub(r"_L$", "", re.sub(r"_R$", "", bid))


def greedy_findings(pred):
    """从贪心预测 (dict 或原文) 提取 4 个 findings 的 0/1."""
    out = {}
    if isinstance(pred, dict):
        for k in FINDINGS:
            v = pred.get("findings", {}).get(k)
            out[k] = None if v is None else int(bool(v))
    else:
        for k in FINDINGS:
            m = re.search(rf'"{k}"\s*:\s*([01])', pred or "")
            out[k] = int(m.group(1)) if m else None
    return out


def boot_paired(fn, ids, studies_unique, id_by_study, n=BOOT_N, seed=SEED):
    """study 级配对 bootstrap: 对统计量 fn(subset_ids) 取 95% 分位区间."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        pick = rng.choice(len(studies_unique), size=len(studies_unique), replace=True)
        sub = [i for j in pick for i in id_by_study[studies_unique[j]]]
        vals.append(fn(sub))
    v = np.asarray(vals, dtype=float)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def mean_med(x):
    x = np.asarray(x, dtype=float)
    return float(x.mean()), float(np.median(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--greedy", default="probs_greedy_test.jsonl")
    ap.add_argument("--zero", default="probs_zerofind_test.jsonl")
    ap.add_argument("--inject", default="probs_injectfind_test.jsonl")
    ap.add_argument("--out-root", default=None,
                    help="覆盖输出目录 (默认 outputs/prefix_probe)")
    args = ap.parse_args()

    global OUT_ROOT
    if args.out_root:
        OUT_ROOT = Path(args.out_root)

    G = load_rows(args.greedy)
    Z = load_rows(args.zero)
    I = load_rows(args.inject)
    common = sorted(set(G) & set(Z) & set(I))
    print(f"三条件公共 id: {len(common)} (greedy={len(G)} zero={len(Z)} inject={len(I)})")
    assert len(common) >= 1900, "概率文件不完整, 先跑完 zero/inject 全量"

    # ---- 子集划分 ----
    sub1, sub2 = [], []
    for i in common:
        g, gt = G[i], G[i]["ground_truth"]
        gb = gt["breast_birads"]
        gf = greedy_findings(g.get("greedy_prediction"))
        assert all(v is not None for v in gf.values()), f"{i} 贪心 findings 解析失败"
        g_pred = birads_argmax(g)
        if gb >= 3 and any(gf[k] == 1 and int(bool(gt["findings"][k])) == 1 for k in FINDINGS) and g_pred >= 3:
            sub1.append(i)
        if gb in (1, 2) and all(gf[k] == 0 for k in FINDINGS) and g_pred <= 2:
            sub2.append(i)
    print(f"子集1 (阳性确认组): {len(sub1)}  子集2 (阴性注入组): {len(sub2)}")

    rep = {"n_common": len(common), "n_sub1": len(sub1), "n_sub2": len(sub2), "subsets": {}}

    def analyze(sub, other, other_name, rise_or_drop, tag):
        """rise_or_drop='drop': other 下 argmax<=2 比例; ='rise': other 下 argmax>=3 比例."""
        pg = np.array([p_ge3(G[i]) for i in sub])
        po = np.array([p_ge3(other[i]) for i in sub])
        mg, mdg = mean_med(pg)
        mo, mdo = mean_med(po)
        if rise_or_drop == "drop":
            flagged = np.array([birads_argmax(other[i]) <= 2 for i in sub], dtype=float)
        else:
            flagged = np.array([birads_argmax(other[i]) >= 3 for i in sub], dtype=float)
        studies_unique = sorted({study_of(i) for i in sub})
        id_by_study = {}
        for i in sub:
            id_by_study.setdefault(study_of(i), []).append(i)
        idx = {i: n for n, i in enumerate(sub)}

        def f_mean_diff(ids):
            a = [idx[i] for i in ids]
            return float(po[a].mean() - pg[a].mean())

        def f_flag(ids):
            a = [idx[i] for i in ids]
            return float(flagged[a].mean())

        ci_d = boot_paired(f_mean_diff, sub, studies_unique, id_by_study)
        ci_f = boot_paired(f_flag, sub, studies_unique, id_by_study)
        out = {
            "n": len(sub), "n_studies": len(studies_unique),
            "greedy_p_ge3_mean": mg, "greedy_p_ge3_median": mdg,
            f"{other_name}_p_ge3_mean": mo, f"{other_name}_p_ge3_median": mdo,
            "delta_mean": mo - mg, "delta_mean_ci95": ci_d,
            "flag_proportion": float(flagged.mean()), "flag_proportion_ci95": ci_f,
            "flag_definition": ("argmax BI-RADS <= 2 under " + other_name) if rise_or_drop == "drop"
                               else ("argmax BI-RADS >= 3 under " + other_name),
        }
        rep["subsets"][tag] = out
        return out

    s1 = analyze(sub1, Z, "zero", "drop", "sub1_positive_confirmed_zero")
    s2 = analyze(sub2, I, "inject", "rise", "sub2_negative_injected_mass")

    # ---- markdown ----
    L = ["# zero / inject 前缀干预报告 (os_sub, test 2000)", "",
         f"三条件公共样本 {rep['n_common']}; 子集1 (GT BI>=3 & >=1 finding 真阳性(真值=1且预测=1) & greedy 预测>=3) n={rep['n_sub1']}; "
         f"子集2 (GT BI 1-2 & findings 全 0 & greedy 预测 1-2) n={rep['n_sub2']}", "",
         "## 子集 1: 阳性确认组 — greedy vs zero (findings 全清零)", "",
         "| 指标 | greedy | zero |", "|---|---|---|",
         f"| P(BI-RADS>=3) 均值 | {s1['greedy_p_ge3_mean']:.4f} | {s1['zero_p_ge3_mean']:.4f} |",
         f"| P(BI-RADS>=3) 中位数 | {s1['greedy_p_ge3_median']:.4f} | {s1['zero_p_ge3_median']:.4f} |",
         f"| Δmean (zero-greedy) | | {s1['delta_mean']:+.4f} [{s1['delta_mean_ci95'][0]:+.4f}, {s1['delta_mean_ci95'][1]:+.4f}] |",
         f"| zero 下掉到 1-2 级比例 | | {s1['flag_proportion']:.1%} [{s1['flag_proportion_ci95'][0]:.1%}, {s1['flag_proportion_ci95'][1]:.1%}] |",
         "", "## 子集 2: 阴性注入组 — greedy vs inject (仅 mass 置 1)", "",
         "| 指标 | greedy | inject |", "|---|---|---|",
         f"| P(BI-RADS>=3) 均值 | {s2['greedy_p_ge3_mean']:.4f} | {s2['inject_p_ge3_mean']:.4f} |",
         f"| P(BI-RADS>=3) 中位数 | {s2['greedy_p_ge3_median']:.4f} | {s2['inject_p_ge3_median']:.4f} |",
         f"| Δmean (inject-greedy) | | {s2['delta_mean']:+.4f} [{s2['delta_mean_ci95'][0]:+.4f}, {s2['delta_mean_ci95'][1]:+.4f}] |",
         f"| inject 下升到 >=3 级比例 | | {s2['flag_proportion']:.1%} [{s2['flag_proportion_ci95'][0]:.1%}, {s2['flag_proportion_ci95'][1]:.1%}] |",
         "", "CI 为 study 级配对 bootstrap 95% (1000 次, seed 42)。", ""]
    (OUT_ROOT / "prefix_zero_inject_report.md").write_text("\n".join(L), encoding="utf-8")
    json.dump(rep, open(OUT_ROOT / "prefix_zero_inject_report.json", "w"),
              ensure_ascii=False, indent=1)
    print("\n".join(L))
    print(f"[saved] {OUT_ROOT/'prefix_zero_inject_report.md'}")


if __name__ == "__main__":
    main()
