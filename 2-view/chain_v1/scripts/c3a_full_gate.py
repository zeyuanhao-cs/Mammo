#!/usr/bin/env python3
"""C3a 400 例门控：密度与 BI-RADS macroAUC 相对 B0 的降幅均需 ≤0.01（CPU）。"""
import json
import sys

import numpy as np

sys.path.insert(0, "/home/z/zhao/Mammography/Mammo/2-view")
import chain_v1.scripts.pilot_metrics as pm  # noqa: E402
from chain_v1.scripts.bootstrap_metrics import bootstrap_pair  # noqa: E402

P = pm.P
ids = json.load(open(P / "ids2_ablation.json", encoding="utf-8"))


def load(fn):
    f = P / fn
    return {json.loads(l)["id"]: json.loads(l) for l in open(f, encoding="utf-8")} if f.exists() else {}


b0_all, c3a = load("b0_hard.jsonl"), load("c3a_soft400.jsonl")
b0 = {i: b0_all[i] for i in ids if i in b0_all}
c3a = {i: c3a[i] for i in ids if i in c3a}


def m(rows):
    r = list(rows.values())
    b = pm.birads_block(r)
    d = pm.density_block(r)
    f = pm.findings_block(r)
    fa = [v for v in f["auc"].values() if v]
    return {"density": d.get("macro_auc"), "macro_auc": b["macro_auc"], "ge3": b["ge3_auc"],
            "f1": b["macro_f1"], "findings": float(np.mean(fa)) if fa else None}


m0, m3 = m(b0), m(c3a)
boot = bootstrap_pair(b0, c3a)
d_dens = m3["density"] - m0["density"]
d_auc = m3["macro_auc"] - m0["macro_auc"]
ok = d_dens >= -0.01 and d_auc >= -0.01

print(f"n={len(c3a)}")
print("| 指标 | B0 | C3a | Δ | 95% CI |")
print("|---|---|---|---|---|")
for k, lab, bk in (("density", "密度 AUC", "density_auc"), ("macro_auc", "BI-RADS macroAUC", "birads_macro_auc"),
                   ("ge3", "≥3 AUC", "ge3_auc"), ("f1", "macroF1（参考）", "macro_f1"),
                   ("findings", "征象 meanAUC", "findings_mean_auc")):
    d, lo, hi = boot[bk]
    print(f"| {lab} | {m0[k]:.4f} | {m3[k]:.4f} | {d:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
print(f"\n判据：密度 Δ={d_dens:+.4f}、macroAUC Δ={d_auc:+.4f}（均需 ≥ −0.01）→ {'PASS' if ok else 'FAIL'}")
json.dump({"b0": m0, "c3a": m3, "pass": bool(ok)}, open("/tmp/c3a_full_gate.json", "w"),
          ensure_ascii=False, indent=1)
sys.exit(0 if ok else 3)
