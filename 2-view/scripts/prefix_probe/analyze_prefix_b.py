#!/usr/bin/env python3
"""反方向前缀干预分析 (双顺序模型 · B 顺序 · prompt_order_b).

输入 (由 run_probe.py --zi-b 产出):
  outputs/prefix_probe_dualorder_B/probs_{greedy,down,up}_test.jsonl

每行: {id, mode, ground_truth, greedy(贪心 prediction_json), prefix, gen_text,
       findings: {field: {value, p1}}, n_findings_1, seconds}
  greedy = BI-RADS 不改;  down = BI-RADS 改成 1;  up = BI-RADS 改成 4
前缀停在 `"findings": {` 之后, 4 个 findings 由模型自己贪心续写 (不 teacher-force)。

子集:
  子集1 (down, 追问"findings 是否跟着塌"): GT BI-RADS>=3, 贪心 BI-RADS>=3,
                                         且至少一个 finding 真阳性(真值=预测=1)
  子集2 (up, 追问"findings 是否被凭空生成"): GT BI-RADS 1-2, GT 与贪心 findings 全 0,
                                          贪心 BI-RADS 1-2

统计: 按 study 配对 bootstrap 1000 次 (seed=42), 95% CI 取百分位 2.5/97.5。
输出: <out_root>/prefix_b_report.md (+ stdout)

用法:
  python analyze_prefix_b.py [--out-root ...] [--n-boot 1000] [--seed 42]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent          # 2-view/scripts/prefix_probe
PROJ_DIR = THIS_DIR.parent.parent                    # 2-view
DEFAULT_OUT = PROJ_DIR / "outputs" / "prefix_probe_dualorder_B"

FINDINGS = ["mass", "suspicious_calcification", "asymmetry", "architectural_distortion"]
MODES = ("greedy", "down", "up")


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


def f_values(row) -> np.ndarray:
    """4 个 finding 生成出的 0/1 (缺失记 NaN)."""
    fs = (row or {}).get("findings") or {}
    out = []
    for k in FINDINGS:
        v = (fs.get(k) or {}).get("value")
        out.append(np.nan if v is None else float(v))
    return np.array(out)


def f_p1(row) -> np.ndarray:
    """4 个 finding 生成时的 P(1) (缺失记 NaN)."""
    fs = (row or {}).get("findings") or {}
    out = []
    for k in FINDINGS:
        v = (fs.get(k) or {}).get("p1")
        out.append(np.nan if v is None else float(v))
    return np.array(out)


def study_of(sid: str) -> str:
    return sid.rsplit("_", 1)[0]


def gt_findings(row) -> dict:
    return {k: int(row["ground_truth"]["findings"][k]) for k in FINDINGS}


def greedy_findings(row) -> dict:
    g = row.get("greedy") or {}
    f = g.get("findings") or {}
    return {k: int(f.get(k, 0)) for k in FINDINGS}


def greedy_birads(row):
    v = (row.get("greedy") or {}).get("breast_birads")
    return int(v) if v is not None else None


# ---------------- bootstrap ----------------

def group_by_study(ids):
    groups = {}
    for i, sid in enumerate(ids):
        groups.setdefault(study_of(sid), []).append(i)
    return {k: np.asarray(v) for k, v in groups.items()}


def boot_ci(groups, stat, n_boot=1000, seed=42):
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
    keep = []
    for sid, r in rows.items():
        if r["ground_truth"]["breast_birads"] < 3:
            continue
        gb = greedy_birads(r)
        if gb is None or gb < 3:
            continue
        gf, pf = gt_findings(r), greedy_findings(r)
        if any(gf[k] == 1 and pf[k] == 1 for k in FINDINGS):
            keep.append(sid)
    return sorted(keep)


def subset2_ids(rows):
    keep = []
    for sid, r in rows.items():
        if r["ground_truth"]["breast_birads"] not in (1, 2):
            continue
        if greedy_birads(r) not in (1, 2):
            continue
        if any(v != 0 for v in gt_findings(r).values()):
            continue
        if any(v != 0 for v in greedy_findings(r).values()):
            continue
        keep.append(sid)
    return sorted(keep)


def analyse(ids, rows, mode, kind, empty_as_zero=True):
    """kind='down': 全 0 比例 + 各 P(1) 下降;  kind='up': >=1 个 finding 比例 + 各 P(1) 上升.

    empty_as_zero: 模型在 `"findings": {` 之后直接吐 `}}` 时, 视作 findings 为空对象
    (= 4 个字段全 0), 而不是解析失败丢弃 —— 这是模型"不列任何发现"的真实输出,
    "}}" 形态占全部不完整样本的 100%。置 False 则退回严格解析 (仅保留 4 字段齐全的行)。
    """
    if not ids:
        return None
    m_raw = np.array([f_values(rows[s][mode]) if mode in rows[s]
                      else np.full(len(FINDINGS), np.nan) for s in ids])
    mg_raw = np.array([f_values(rows[s]) for s in ids])
    p_m = np.array([f_p1(rows[s][mode]) if mode in rows[s] else np.full(len(FINDINGS), np.nan)
                    for s in ids])
    p_g = np.array([f_p1(rows[s]) for s in ids])

    empty_m = np.isnan(m_raw).all(axis=1)      # 该条件直接吐 "}}" 的行
    empty_g = np.isnan(mg_raw).all(axis=1)

    m, mg = m_raw.copy(), mg_raw.copy()
    if empty_as_zero:
        m[empty_m] = 0.0
        mg[empty_g] = 0.0

    ok = (~np.isnan(m).any(axis=1)) & (~np.isnan(mg).any(axis=1))
    n_bad = {"greedy": int((~empty_g & np.isnan(mg).any(axis=1)).sum()),
             mode: int((~empty_m & np.isnan(m).any(axis=1)).sum()),
             "both": int((~ok).sum())}
    sub = [s for s, k in zip(ids, ok) if k]
    m, mg, p_m, p_g = m[ok], mg[ok], p_m[ok], p_g[ok]
    groups = group_by_study(sub)

    out = {"n": len(sub), "n_studies": len(groups), "n_sel": len(ids), "n_bad": n_bad,
           "n_empty_mode": int(empty_m.sum()), "n_empty_greedy": int(empty_g.sum())}
    if kind == "down":
        event = (m.sum(axis=1) == 0)                      # down 后 4 个 findings 全 0
    else:
        event = (m.sum(axis=1) >= 1)                      # up 后至少一个 finding = 1
    out["event"] = boot_ci(groups, lambda ix: event[ix].mean())
    out["any1_rate"] = boot_ci(groups, lambda ix: (m[ix].sum(axis=1) >= 1).mean())
    out["all0_rate"] = boot_ci(groups, lambda ix: (m[ix].sum(axis=1) == 0).mean())
    out["empty_rate_mode"] = boot_ci(groups, lambda ix: empty_m[ok][ix].mean())
    out["empty_rate_greedy"] = boot_ci(groups, lambda ix: empty_g[ok][ix].mean())

    per = {}
    for j, k in enumerate(FINDINGS):
        d = p_m[:, j] - p_g[:, j]
        v = ~np.isnan(d)
        if v.sum() == 0:
            per[k] = None
            continue
        # 只保留该 finding 有 P(1) 的行, 并把下标数组重排到 v 过滤后的空间
        p_gj, p_mj, dj = p_g[v, j], p_m[v, j], d[v]
        mgj, mj = mg[v, j], m[v, j]
        sub_groups = group_by_study([s for s, keep in zip(sub, v) if keep])
        per[k] = {
            "p1_greedy": boot_ci(sub_groups, lambda ix, a=p_gj: a[ix].mean()),
            "p1_mode": boot_ci(sub_groups, lambda ix, a=p_mj: a[ix].mean()),
            "delta": boot_ci(sub_groups, lambda ix, a=dj: a[ix].mean()),
            "rate1_greedy": boot_ci(sub_groups, lambda ix, a=mgj: (a[ix] == 1).mean()),
            "rate1_mode": boot_ci(sub_groups, lambda ix, a=mj: (a[ix] == 1).mean()),
            "n": int(v.sum()),
        }
    out["per_finding"] = per
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=str(DEFAULT_OUT))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--strict-parse", action="store_true",
                    help="只保留 4 个 findings 全部解析出的行 (默认把 `}}` 空对象记为全 0)")
    args = ap.parse_args()
    eaz = not args.strict_parse

    out_root = Path(args.out_root)
    data = {m: load_mode(out_root, m) for m in MODES}
    missing = [m for m in MODES if not data[m]]
    if missing:
        print(f"[error] 缺少 {missing} 的 jsonl (目录 {out_root})")
        return 1

    ids = sorted(set(data["greedy"]) & set(data["down"]) & set(data["up"]))
    print(f"[data] greedy={len(data['greedy'])} down={len(data['down'])} "
          f"up={len(data['up'])} 共同 id={len(ids)}")

    merged = {}
    for sid in ids:
        r = dict(data["greedy"][sid])
        r["down"] = data["down"][sid]
        r["up"] = data["up"][sid]
        merged[sid] = r

    s1, s2 = subset1_ids(merged), subset2_ids(merged)
    print(f"[subset1] GT>=3 且 贪心>=3 且 finding 真阳性: {len(s1)} 条 / "
          f"{len(set(study_of(s) for s in s1))} studies")
    print(f"[subset2] GT 1-2 且 findings 全 0 且 贪心 1-2: {len(s2)} 条 / "
          f"{len(set(study_of(s) for s in s2))} studies")

    r1 = analyse(s1, merged, "down", "down", eaz) if s1 else None
    r2 = analyse(s2, merged, "up", "up", eaz) if s2 else None

    L = []
    L.append("# 反方向前缀干预分析 (B 顺序 · density → BI-RADS → findings)")
    L.append("")
    L.append(f"- 数据目录: `{out_root}`")
    L.append(f"- 样本: greedy {len(data['greedy'])} / down {len(data['down'])} / "
             f"up {len(data['up'])}, 三条件共同 id **{len(ids)}**")
    L.append(f"- 模型: 双顺序 LoRA + `prompt_order_b.txt`；贪心预测取自 "
             f"`predictions_qwen3.5-4b_dualorder_b_2000.jsonl`")
    L.append(f"- bootstrap: 按 study 配对重采样 **{args.n_boot}** 次, seed={args.seed}, "
             f"95% CI 取百分位 2.5/97.5")
    L.append("- 干预方式: 只把 `breast_birads` 改成 `down=1` / `up=4`, density 保持贪心; "
             "前缀截断到 `\"findings\": {` 之后, **findings 由模型自己贪心续写**")
    L.append("- P(1) = 生成该 finding 数字那一步的 softmax 概率; 比例类指标按样本计数")
    L.append("- 口径: 模型在 `\"findings\": {` 之后直接吐 `}}` 时, 视作 `findings` 为空对象"
             "(= 4 字段全 0), 而不是丢弃该样本; P(1) 表格只统计 4 字段齐全的行"
             + ("" if eaz else " (本次为严格解析口径)") + "")
    L.append("")

    # ---------------- 子集 1 ----------------
    L.append("## 子集 1 · BI-RADS 改为 1 (down) → findings 是否跟着塌成 0")
    L.append("")
    L.append("筛选: `GT BI-RADS >= 3` 且 `贪心 BI-RADS >= 3` 且 至少一个 finding 真阳性"
             "(真值=1 且贪心=1)")
    L.append("")
    if r1 is None:
        L.append("样本为空。")
    else:
        b = r1["n_bad"]
        L.append(f"- n = **{r1['n']}** 条, {r1['n_studies']} 个 study "
                 f"(筛选命中 {r1['n_sel']} 条; down 直接吐 `}}}}` 空对象 "
                 f"**{r1['n_empty_mode']}** 条, greedy 空对象 {r1['n_empty_greedy']} 条"
                 + ("" if eaz else f"; 严格解析剔除 {b['both']} 条") + ")")
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
                     f"{fmt_ci(*v['delta'])} | {fmt(v['rate1_greedy'][0])} → {fmt(v['rate1_mode'][0])} |")
        L.append("")
        ns = [v["n"] for v in r1["per_finding"].values() if v]
        if ns:
            rng_n = f"{min(ns)}" if min(ns) == max(ns) else f"{min(ns)}–{max(ns)}"
            L.append(f"- P(1) 三列只基于 4 字段齐全的 **{rng_n}** 行, 其余行该 finding 的 P(1) 记为缺失")
    L.append("")

    # ---------------- 子集 2 ----------------
    L.append("## 子集 2 · BI-RADS 改为 4 (up) → findings 是否被凭空生成")
    L.append("")
    L.append("筛选: `GT BI-RADS 1–2` 且 GT 与贪心的 findings 全 0 且 `贪心 BI-RADS 1–2`")
    L.append("")
    if r2 is None:
        L.append("样本为空。")
    else:
        b = r2["n_bad"]
        L.append(f"- n = **{r2['n']}** 条, {r2['n_studies']} 个 study "
                 f"(筛选命中 {r2['n_sel']} 条; up 直接吐 `}}}}` 空对象 "
                 f"**{r2['n_empty_mode']}** 条, greedy 空对象 {r2['n_empty_greedy']} 条"
                 + ("" if eaz else f"; 严格解析剔除 {b['both']} 条") + ")")
        L.append("")
        L.append("| 指标 | 值 (95% CI) |")
        L.append("|---|---|")
        L.append(f"| up 后出现 **>=1 个 finding=1** 的比例 | {fmt_ci(*r2['event'])} |")
        L.append(f"| up 后 4 个 findings 仍全 0 的比例 | {fmt_ci(*r2['all0_rate'])} |")
        L.append(f"| 其中 up 直接输出空 `findings: {{}}` | {fmt_ci(*r2['empty_rate_mode'])} |")
        L.append("")
        L.append("| finding | 平均 P(1) greedy | 平均 P(1) up | Δ | 生成=1 比例 greedy → up |")
        L.append("|---|---|---|---|---|")
        for k in FINDINGS:
            v = r2["per_finding"].get(k)
            if not v:
                L.append(f"| {k} | - | - | - | - |")
                continue
            L.append(f"| {k} | {fmt_ci(*v['p1_greedy'])} | {fmt_ci(*v['p1_mode'])} | "
                     f"{fmt_ci(*v['delta'])} | {fmt(v['rate1_greedy'][0])} → {fmt(v['rate1_mode'][0])} |")
        L.append("")
        ns = [v["n"] for v in r2["per_finding"].values() if v]
        if ns:
            rng_n = f"{min(ns)}" if min(ns) == max(ns) else f"{min(ns)}–{max(ns)}"
            L.append(f"- P(1) 三列只有 {rng_n} 行 (greedy 吐 `}}}}` 的行无 P(1))")
    L.append("")

    md = out_root / "prefix_b_report.md"
    md.write_text("\n".join(L) + "\n", encoding="utf-8")
    print()
    print("\n".join(L))
    print(f"[report] -> {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
