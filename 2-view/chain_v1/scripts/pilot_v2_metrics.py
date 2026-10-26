#!/usr/bin/env python3
"""pilot_v2 报告：B0 vs C3（全量）+ bootstrap + 7 点扫描（C3 vs C1）。"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/z/zhao/Mammography/Mammo/2-view")
import chain_v1.scripts.pilot_metrics as pm  # noqa: E402
from chain_v1.scripts.bootstrap_metrics import bootstrap_pair  # noqa: E402

P = Path("/home/z/zhao/Mammography/Mammo/2-view/chain_v1/outputs/pilot")
OUT = Path("/home/z/zhao/Mammography/Mammo/2-view/chain_v1/outputs/pilot_v2_report.md")
FIND = pm.FIND


def load(fn):
    f = P / fn
    if not f.exists():
        return {}
    return {json.loads(l)["id"]: json.loads(l) for l in open(f, encoding="utf-8")}


def metrics(rows_by_id):
    if not rows_by_id:
        return None
    rows = list(rows_by_id.values())
    b = pm.birads_block(rows)
    d = pm.density_block(rows)
    f = pm.findings_block(rows)
    fa = [v for v in f["auc"].values() if v]
    return {"n": b["n"], "macro_auc": b["macro_auc"], "ge3": b["ge3_auc"], "f1": b["macro_f1"],
            "acc": b["acc"], "bacc": b["bacc"], "density": d.get("macro_auc"),
            "findings": float(np.mean(fa)) if fa else None,
            "recall": b["per_class_recall"], "sens": b["ge3_sens_at_spec"]}


b0 = load("b0_hard.jsonl")
c1 = load("c1_soft.jsonl")
c2 = load("c2_soft.jsonl")
c3 = load("c3_soft1600.jsonl")
M = {k: metrics(v) for k, v in (("B0", b0), ("C1", c1), ("C2", c2), ("C3", c3))}

L = ["# 链条 v1 试验报告 v2（阶段二：C3）", "",
     "## 口径", "",
     "- **C3**：从基础模型从头 1 epoch（439 步，seed 42），`--soft-mode logit_mlp --mixer-init small "
     "--soft-gate uncert`，alpha=0.05、lam=0.05、rho_max=0.5、rho_warmup_frac=0.2、--no-reuse-embeds。",
     "- C3 快速验证（C3q：以 B0 为 init 的 50 步）三条判据全部通过后才启动本全量训练。",
     "- 评测统一 `eval_soft`、batch=1；B0 为硬模式，C1/C2/C3 为软模式。",
     "- 指标已修正每类召回计算 bug；bootstrap 为指标级、study 级 1000 次、seed 42。", ""]

L += ["## 0. 结论表（验证集 1600）", "",
      "| 组 | macroAUC | ≥3 AUC | macroF1 | acc | bACC | 密度 AUC | 征象 meanAUC |",
      "|---|---|---|---|---|---|---|---|"]
for g in ("B0", "C1", "C2", "C3"):
    m = M[g]
    if not m:
        L.append(f"| {g} | — | — | — | — | — | — | — |")
        continue
    L.append(f"| {g} | {m['macro_auc']:.3f} | {m['ge3']:.3f} | {m['f1']:.3f} | {m['acc']:.3f} | "
             f"{m['bacc']:.3f} | {m['density']:.3f} | {m['findings']:.3f} |")
L.append("")

# bootstrap
pairs = []
if c3 and b0:
    pairs.append(("C3-B0", b0, c3))
if c3 and c1:
    pairs.append(("C3-C1", c1, c3))
if c3 and c2:
    pairs.append(("C3-C2", c2, c3))
boots = {t: bootstrap_pair(a, b) for t, a, b in pairs}

if boots:
    L += ["## 差值 bootstrap（study 级 1000 次，seed 42）", "",
          "Δ = 后者 − 前者；`*` = 95% CI 不含 0。", "",
          "| 指标 | " + " | ".join(boots) + " |", "|---|" + "---|" * len(boots)]
    for key, lab in (("birads_macro_auc", "BI-RADS macroAUC"), ("ge3_auc", "≥3 AUC"),
                     ("macro_f1", "macroF1"), ("sens_spec90", "灵敏度@90%"),
                     ("sens_spec95", "灵敏度@95%"), ("density_auc", "密度 AUC"),
                     ("findings_mean_auc", "征象 meanAUC")):
        cells = []
        for t, r in boots.items():
            d, lo, hi = r.get(key, (None, None, None))
            cells.append("—" if d is None else
                         f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]" + ("*" if (lo > 0 or hi < 0) else ""))
        L.append(f"| {lab} | " + " | ".join(cells) + " |")
    L.append("")

# 扫描（每 id 多行 → 必须逐行读，不能按 id 建字典）
SCAN_IDS = set(json.load(open(P / "ids2_scan20.json", encoding="utf-8"))) \
    if (P / "ids2_scan20.json").exists() else set()


def read_scan(fn):
    f = P / fn
    if not f.exists():
        return []
    out = []
    for line in f.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if not SCAN_IDS or r["id"] in SCAN_IDS:
            out.append(r)
    return out


def curve(rows):
    by = {}
    for r in rows:
        by.setdefault(round(r["scan_p"], 2), {"pos": [], "neg": []})
        by[round(r["scan_p"], 2)]["pos" if r["gt_ge3"] else "neg"].append(r["p_birads_ge3"])
    ps = sorted(by)
    return ps, [float(np.mean(by[p]["pos"])) if by[p]["pos"] else float("nan") for p in ps], \
        [float(np.mean(by[p]["neg"])) if by[p]["neg"] else float("nan") for p in ps]


s3 = read_scan("c3_scan.jsonl")
s1 = read_scan("c1_scan.jsonl")


if s3:
    ps3, p3, n3 = curve(s3)
    ps1, p1, n1 = curve(s1) if s1 else ([], [], [])
    L += ["## 7 点概率扫描（40 例：GT≥3 / GT≤2 各 20）", "",
          "| p | " + " | ".join(str(p) for p in ps3) + " |",
          "|---|" + "---|" * len(ps3),
          "| C3 GT≥3 | " + " | ".join(f"{v:.3f}" for v in p3) + " |",
          "| C3 GT≤2 | " + " | ".join(f"{v:.3f}" for v in n3) + " |"]
    if ps1:
        L.append("| C1 GT≥3 | " + " | ".join(f"{v:.3f}" for v in p1) + " |")
        L.append("| C1 GT≤2 | " + " | ".join(f"{v:.3f}" for v in n1) + " |")
    L.append("")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.plot(ps3, p3, "o-", label="C3 GT>=3")
        ax.plot(ps3, n3, "s--", label="C3 GT<=2")
        if ps1:
            ax.plot(ps1, p1, "o:", alpha=0.6, label="C1 GT>=3")
            ax.plot(ps1, n1, "s:", alpha=0.6, label="C1 GT<=2")
        ax.set_xlabel("forced P(mass='1')")
        ax.set_ylabel("mean P(BI-RADS>=3)")
        ax.set_title("C3 vs C1 probability scan (7 points)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(P / "stage2_c3_scan7.png", dpi=140)
        plt.close(fig)
        L += ["![c3_scan7](pilot/stage2_c3_scan7.png)", ""]
    except Exception as e:  # noqa: BLE001
        print("[warn] 画图失败", e)

L += ["## 每类召回（B0 / C3）", "", "| 类 | B0 | C3 |", "|---|---|---|"]
for c in ("1", "2", "3", "4", "5"):
    a = M["B0"]["recall"][c] if M.get("B0") else None
    b = M["C3"]["recall"][c] if M.get("C3") else None
    L.append(f"| {c} | {'—' if a is None else f'{a:.3f}'} | {'—' if b is None else f'{b:.3f}'} |")
L.append("")

OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
json.dump({"metrics": M, "bootstrap": boots}, open(P / "pilot_v2_metrics.json", "w"),
          ensure_ascii=False, indent=1)
print("[ok]", OUT)
