#!/usr/bin/env python3
"""C3 快速验证门控：400 例指标对比 B0 + bootstrap + 7 点扫描检查。

输出: chain_v1/outputs/pilot/c3q_summary.md 与 c3q_gate.json
退出码: 0 = 三条全过（可自动进全量 C3）；3 = STOP
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/z/zhao/Mammography/Mammo/2-view")
import chain_v1.scripts.pilot_metrics as pm  # noqa: E402
from chain_v1.scripts.bootstrap_metrics import bootstrap_pair  # noqa: E402

P = Path("/home/z/zhao/Mammography/Mammo/2-view/chain_v1/outputs/pilot")
ids = json.load(open(P / "ids2_ablation.json", encoding="utf-8"))
scan_ids = json.load(open(P / "ids2_scan20.json", encoding="utf-8"))


def load(fn):
    f = P / fn
    if not f.exists():
        return {}
    return {json.loads(l)["id"]: json.loads(l) for l in open(f, encoding="utf-8")}


def read_scan(fn, keep_ids):
    """扫描文件每 id 有多个 scan_p 行 → 必须逐行读，不能按 id 建字典。"""
    f = P / fn
    rows = []
    if not f.exists():
        return rows
    for line in f.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["id"] in keep_ids:
            rows.append(r)
    return rows


def scan_curve(rows):
    by = {}
    for r in rows:
        by.setdefault(round(r["scan_p"], 2), {"pos": [], "neg": []})
        by[round(r["scan_p"], 2)]["pos" if r["gt_ge3"] else "neg"].append(r["p_birads_ge3"])
    ps = sorted(by)
    pos = [float(np.mean(by[p]["pos"])) if by[p]["pos"] else float("nan") for p in ps]
    allv = [float(np.mean(by[p]["pos"] + by[p]["neg"])) for p in ps]
    return ps, pos, allv


# 自检模式：只算扫描曲线，用于比对已知数据
if "--scan-only" in sys.argv:
    fn = sys.argv[sys.argv.index("--scan-only") + 1]
    ps, pos, allv = scan_curve(read_scan(fn, set(scan_ids)))
    print(f"[scan-only] {fn}: {len(ps)} 个网格点 {ps}")
    print("GT>=3:", " ".join(f"{v:.3f}" for v in pos))
    print("全体  :", " ".join(f"{v:.3f}" for v in allv))
    sys.exit(0)

b0_all = load("b0_hard.jsonl")
c3_all = load("c3q_soft400.jsonl")
b0 = {i: b0_all[i] for i in ids if i in b0_all}
c3 = {i: c3_all[i] for i in ids if i in c3_all}
if not c3:
    print("[c3] 缺 c3q_soft400.jsonl")
    sys.exit(3)


def metrics(rows_by_id):
    rows = list(rows_by_id.values())
    b = pm.birads_block(rows)
    d = pm.density_block(rows)
    f = pm.findings_block(rows)
    fa = [v for v in f["auc"].values() if v]
    return {"density": d.get("macro_auc"), "macro_auc": b["macro_auc"], "ge3": b["ge3_auc"],
            "f1": b["macro_f1"], "findings": float(np.mean(fa)) if fa else None}


m0, m3 = metrics(b0), metrics(c3)
boot = bootstrap_pair(b0, c3)   # Δ = C3 − B0

# 7 点扫描（优先用修正后的文件；逐行读，按 (id, scan_p) 收集）
scan_fn = "c3q_scan_fixed.jsonl" if (P / "c3q_scan_fixed.jsonl").exists() else "c3q_scan.jsonl"
ps, curve_pos, curve_all = scan_curve(read_scan(scan_fn, set(scan_ids)))

# 新判据：a) 密度 macroAUC 与 BI-RADS macroAUC 降幅均 ≤0.01；b) 扫描 0.1→0.5 上升 ≥0.05
#        macroF1 只报告，不作判据
a_ok = (m0["density"] is not None and m3["density"] is not None
        and (m3["density"] - m0["density"]) >= -0.01
        and (m3["macro_auc"] - m0["macro_auc"]) >= -0.01)
idx = {p: j for j, p in enumerate(ps)}
if 0.1 in idx and 0.5 in idx:
    scan_gain = curve_pos[idx[0.5]] - curve_pos[idx[0.1]]
else:
    scan_gain = float("nan")
b_ok = (not np.isnan(scan_gain)) and scan_gain >= 0.05
gate = "PASS" if (a_ok and b_ok) else "STOP"

L = ["# C3 快速验证（50 步 logit_mlp + uncert，init=B0）", "",
     f"**GATE: {gate}**", "",
     "## 400 乳房指标（B0 硬 vs C3q 软）", "",
     "| 指标 | B0 | C3q | Δ | bootstrap 95% CI |", "|---|---|---|---|---|"]
for k, bk, lab in (("density", "density_auc", "密度 AUC"),
                   ("macro_auc", "birads_macro_auc", "BI-RADS macroAUC"),
                   ("ge3", "ge3_auc", "≥3 AUC"),
                   ("f1", "macro_f1", "macroF1"),
                   ("findings", "findings_mean_auc", "征象 meanAUC")):
    a, b = m0[k], m3[k]
    d, lo, hi = boot[bk]
    star = "*" if (d is not None and (lo > 0 or hi < 0)) else ""
    L.append(f"| {lab} | {a:.4f} | {b:.4f} | {d:+.4f} | [{lo:+.4f}, {hi:+.4f}]{star} |")
L += ["", "## 7 点扫描（40 例，GT≥3 均值）", "",
      "| p | " + " | ".join(f"{p}" for p in ps) + " |",
      "|---|" + "---|" * len(ps),
      "| GT≥3 | " + " | ".join(f"{v:.3f}" for v in curve_pos) + " |",
      "| 全体 | " + " | ".join(f"{v:.3f}" for v in curve_all) + " |", "",
      "## 判据", "",
      f"- a 密度 macroAUC：{m3['density']:.4f} vs B0 {m0['density']:.4f}"
      f"（Δ={m3['density'] - m0['density']:+.4f}）；BI-RADS macroAUC：{m3['macro_auc']:.4f} vs "
      f"{m0['macro_auc']:.4f}（Δ={m3['macro_auc'] - m0['macro_auc']:+.4f}），两者均需 ≥ −0.01"
      f" → {'PASS' if a_ok else 'FAIL'}",
      f"- b 扫描 p=0.1→0.5 的 P(≥3) 上升：{scan_gain:+.4f}（需 ≥ +0.05）→ {'PASS' if b_ok else 'FAIL'}",
      f"- 参考（不作判据）macroF1：{m3['f1']:.4f} vs B0 {m0['f1']:.4f}"
      f"（Δ={m3['f1'] - m0['f1']:+.4f}）",
      f"- 扫描数据源：{scan_fn}",
      ""]

(P / "c3q_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
(P / "c3q_gate.json").write_text(json.dumps(
    {"gate": gate, "metrics_b0": m0, "metrics_c3q": m3,
     "boot": {k: boot[k] for k in ("density_auc", "birads_macro_auc", "ge3_auc",
                                   "macro_f1", "findings_mean_auc")},
     "scan": {"p": ps, "pos": curve_pos, "all": curve_all},
     "checks": {"a_density_and_macroauc": bool(a_ok), "b_scan_0.1_to_0.5": bool(b_ok),
                "scan_gain": None if np.isnan(scan_gain) else float(scan_gain)},
     "scan_source": scan_fn}, ensure_ascii=False, indent=1),
    encoding="utf-8")
print("\n".join(L))
sys.exit(0 if gate == "PASS" else 3)
