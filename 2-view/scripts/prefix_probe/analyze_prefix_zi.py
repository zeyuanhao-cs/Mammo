#!/usr/bin/env python3
"""zero / inject 前缀干预的配对分析 (双顺序模型, prompt_order_a).

输入 (由 run_probe.py --zi 产出):
  outputs/prefix_probe_dualorder_A/probs_greedy_test.jsonl
  outputs/prefix_probe_dualorder_A/probs_zero_test.jsonl
  outputs/prefix_probe_dualorder_A/probs_inject_test.jsonl

每行: {id, mode, ground_truth, greedy(贪心 prediction_json), prefix, birads_probs, p_ge3, ...}
三个文件按 id 一一配对, 同一 id 的图像/prompt 完全相同, 只有 findings 前缀不同:
  greedy = 模型自己的 findings;  zero = findings 全 0;  inject = 只把 mass 置 1
因此 greedy↔zero / greedy↔inject 是**同一样本的配对比较**。

子集:
  子集1 (zero, 掉级风险): GT BI-RADS>=3, 且至少一个 finding 真阳性(真值=预测=1),
                          且贪心 BI-RADS>=3
  子集2 (inject, 升级风险): GT BI-RADS 1-2, 且 GT 与贪心的 findings 全 0,
                          且贪心 BI-RADS 1-2

统计: 按 study 配对 bootstrap 1000 次 (seed=42), 95% CI 取百分位 2.5/97.5。
输出: <out_root>/prefix_zi_report.md (+ stdout)

用法:
  python analyze_prefix_zi.py [--out-root ...] [--n-boot 1000] [--seed 42]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent          # 2-view/scripts/prefix_probe
PROJ_DIR = THIS_DIR.parent.parent                    # 2-view
DEFAULT_OUT = PROJ_DIR / "outputs" / "prefix_probe_dualorder_A"

FINDINGS = ["mass", "suspicious_calcification", "asymmetry", "architectural_distortion"]
MODES = ("greedy", "zero", "inject")


# ---------------- 读入 ----------------

def load_mode(out_root: Path, mode: str) -> dict:
    p = out_root / f"probs_{mode}_test.jsonl"
    rows = {}
    if not p.exists():
        return rows
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                rows[r["id"]] = r
    return rows


def probs5(row) -> np.ndarray:
    bi = (row or {}).get("birads_probs") or {}
    return np.array([float(bi.get(c, 0.0)) for c in "12345"])


def p_ge3(row) -> float:
    v = probs5(row)
    return float(v[2:].sum()) if v.sum() > 0 else float("nan")


def level(row):
    v = probs5(row)
    return int(np.argmax(v)) + 1 if v.sum() > 0 else None


def study_of(sid: str) -> str:
    return sid.rsplit("_", 1)[0]


def gt_findings(row) -> dict:
    return {k: int(row["ground_truth"]["findings"][k]) for k in FINDINGS}


def greedy_findings(row) -> dict:
    g = row.get("greedy") or {}
    f = g.get("findings") or {}
    return {k: int(f.get(k, 0)) for k in FINDINGS}


def greedy_birads(row):
    g = row.get("greedy") or {}
    v = g.get("breast_birads")
    return int(v) if v is not None else None


# ---------------- bootstrap ----------------

def group_by_study(ids):
    groups = {}
    for i, sid in enumerate(ids):
        groups.setdefault(study_of(sid), []).append(i)
    return {k: np.asarray(v) for k, v in groups.items()}


def boot_ci(groups, stat, n_boot=1000, seed=42):
    """study 层面有放回重采样; 返回 (point, lo, hi)."""
    names = list(groups)
    all_idx = np.concatenate([groups[s] for s in names])
    point = float(stat(all_idx))
    if len(names) < 2:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(names), size=len(names))
        vals[b] = stat(np.concatenate([groups[names[i]] for i in pick]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, float(lo), float(hi)


def fmt(v, nd=4):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    return f"{v:.{nd}f}"


def fmt_ci(point, lo, hi, nd=4):
    if np.isnan(lo):
        return fmt(point, nd)
    return f"{fmt(point, nd)} [{fmt(lo, nd)}, {fmt(hi, nd)}]"


# ---------------- 子集 ----------------

def subset1_ids(rows):
    """GT>=3 且 至少一个 finding 真阳性(真值=预测=1) 且 贪心 BI-RADS>=3."""
    keep = []
    for sid, r in rows.items():
        gt_b = r["ground_truth"]["breast_birads"]
        gf, pf = gt_findings(r), greedy_findings(r)
        tp = any(gf[k] == 1 and pf[k] == 1 for k in FINDINGS)
        gb = greedy_birads(r)
        if gt_b >= 3 and tp and gb is not None and gb >= 3:
            keep.append(sid)
    return sorted(keep)


def subset2_ids(rows):
    """GT 1-2 且 GT/贪心 findings 全 0 且 贪心 BI-RADS 1-2."""
    keep = []
    for sid, r in rows.items():
        gt_b = r["ground_truth"]["breast_birads"]
        gf, pf = gt_findings(r), greedy_findings(r)
        gb = greedy_birads(r)
        if (gt_b in (1, 2) and all(v == 0 for v in gf.values())
                and all(v == 0 for v in pf.values())
                and gb in (1, 2)):
            keep.append(sid)
    return sorted(keep)


def analyse(ids, rows, mode, primary_rate):
    """返回指标 dict; primary_rate = 'down' (zero 后 <=2) 或 'up' (inject 后 >=3)."""
    n = len(ids)
    if n == 0:
        return None
    p_g = np.array([p_ge3(rows[s]) for s in ids])
    p_m = np.array([p_ge3(rows[s][mode]) if mode in rows[s] else np.nan for s in ids])
    lv_m = np.array([(level(rows[s][mode]) if mode in rows[s] else 0) or 0 for s in ids])

    ok = ~np.isnan(p_g) & ~np.isnan(p_m)
    sub = [s for s, k in zip(ids, ok) if k]
    p_g, p_m, lv_m = p_g[ok], p_m[ok], lv_m[ok]
    groups = group_by_study(sub)

    rate = (lv_m <= 2) if primary_rate == "down" else (lv_m >= 3)
    delta = p_m - p_g

    out = {"n": len(sub), "n_studies": len(groups)}
    out["p3_greedy"] = boot_ci(groups, lambda ix: p_g[ix].mean())
    out["p3_mode"] = boot_ci(groups, lambda ix: p_m[ix].mean())
    out["p3_delta"] = boot_ci(groups, lambda ix: delta[ix].mean())
    out["rate"] = boot_ci(groups, lambda ix: rate[ix].mean())
    return out


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

    ids = sorted(set(data["greedy"]) & set(data["zero"]) & set(data["inject"]))
    print(f"[data] greedy={len(data['greedy'])} zero={len(data['zero'])} "
          f"inject={len(data['inject'])} 共同 id={len(ids)}")

    merged = {}
    for sid in ids:
        r = dict(data["greedy"][sid])
        r["zero"] = data["zero"][sid]
        r["inject"] = data["inject"][sid]
        merged[sid] = r

    s1 = subset1_ids(merged)
    s2 = subset2_ids(merged)
    print(f"[subset1] GT>=3 且 finding 真阳性 且 贪心>=3: {len(s1)} 条 / "
          f"{len(set(study_of(s) for s in s1))} studies")
    print(f"[subset2] GT 1-2 且 findings 全 0 且 贪心 1-2: {len(s2)} 条 / "
          f"{len(set(study_of(s) for s in s2))} studies")

    r1 = analyse(s1, merged, "zero", "down") if s1 else None
    r2 = analyse(s2, merged, "inject", "up") if s2 else None

    L = []
    L.append("# zero / inject 前缀干预分析 (双顺序模型 · prompt_order_a)")
    L.append("")
    L.append(f"- 数据目录: `{out_root}`")
    L.append(f"- 样本: greedy {len(data['greedy'])} / zero {len(data['zero'])} / "
             f"inject {len(data['inject'])}, 三条件共同 id **{len(ids)}**")
    L.append(f"- bootstrap: 按 study 配对重采样 **{args.n_boot}** 次, seed={args.seed}, "
             f"95% CI 取百分位 2.5/97.5")
    L.append("- 前缀只改 findings 数字: `zero` = 4 个 findings 全 0; "
             "`inject` = 只把 `mass` 置 1; density 与 birads 文本保持贪心")
    L.append("- P(BI>=3) = P(3)+P(4)+P(5); 级别 = 5 类概率的 argmax")
    L.append("")

    # ---------------- 子集 1 ----------------
    L.append("## 子集 1 · findings 全置 0 (zero) 对 BI-RADS 的降级影响")
    L.append("")
    L.append("筛选: `GT BI-RADS >= 3` 且 至少一个 finding 真阳性(真值=1 且贪心=1) "
             "且 `贪心 BI-RADS >= 3`")
    L.append("")
    if r1 is None:
        L.append("样本为空。")
    else:
        L.append(f"- n = **{r1['n']}** 条, {r1['n_studies']} 个 study")
        L.append("")
        L.append("| 指标 | 值 (95% CI) |")
        L.append("|---|---|")
        L.append(f"| 平均 P(BI>=3) · greedy | {fmt_ci(*r1['p3_greedy'])} |")
        L.append(f"| 平均 P(BI>=3) · zero | {fmt_ci(*r1['p3_mode'])} |")
        L.append(f"| Δ 平均 P(BI>=3) (zero − greedy) | {fmt_ci(*r1['p3_delta'])} |")
        L.append(f"| 掉到 1–2 级 (argmax) 的比例 | {fmt_ci(*r1['rate'])} |")
        L.append("")
        L.append(f"解读: zero 后平均 P(BI>=3) 变化 {fmt(r1['p3_delta'][0])}, "
                 f"其中 {fmt(r1['rate'][0] * 100, 1)}% 的样本 argmax 落回 1–2 级。")
    L.append("")

    # ---------------- 子集 2 ----------------
    L.append("## 子集 2 · mass 置 1 (inject) 对 BI-RADS 的升级影响")
    L.append("")
    L.append("筛选: `GT BI-RADS 1–2` 且 GT 与贪心的 findings 全 0 且 `贪心 BI-RADS 1–2`")
    L.append("")
    if r2 is None:
        L.append("样本为空。")
    else:
        L.append(f"- n = **{r2['n']}** 条, {r2['n_studies']} 个 study")
        L.append("")
        L.append("| 指标 | 值 (95% CI) |")
        L.append("|---|---|")
        L.append(f"| 平均 P(BI>=3) · greedy | {fmt_ci(*r2['p3_greedy'])} |")
        L.append(f"| 平均 P(BI>=3) · inject | {fmt_ci(*r2['p3_mode'])} |")
        L.append(f"| Δ 平均 P(BI>=3) (inject − greedy) | {fmt_ci(*r2['p3_delta'])} |")
        L.append(f"| 升到 >=3 级 (argmax) 的比例 | {fmt_ci(*r2['rate'])} |")
        L.append("")
        L.append(f"解读: inject 后平均 P(BI>=3) 变化 {fmt(r2['p3_delta'][0])}, "
                 f"其中 {fmt(r2['rate'][0] * 100, 1)}% 的样本 argmax 升到 >=3 级。")
    L.append("")

    md = out_root / "prefix_zi_report.md"
    md.write_text("\n".join(L) + "\n", encoding="utf-8")
    print()
    print("\n".join(L))
    print(f"[report] -> {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
