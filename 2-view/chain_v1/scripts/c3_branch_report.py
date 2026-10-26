#!/usr/bin/env python3
"""C3 分支报告：B0（对照）vs C3full(axis 软) vs C3full(硬)。

顺序：C3full soft → B0 hard → 报告(第一版，soft−B0) → C3full hard → 报告(更新，含 hard−B0) → 扫描
子集一律用 B0 划分（B0 已先评测）。
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/z/zhao/Mammography/Mammo/2-view")
import chain_v1.scripts.pilot_metrics as pm  # noqa: E402
from chain_v1.scripts.bootstrap_metrics import bootstrap_pair  # noqa: E402

P = Path("/home/z/zhao/Mammography/Mammo/2-view/chain_v1/outputs/pilot")
OUT = Path("/home/z/zhao/Mammography/Mammo/2-view/chain_v1/outputs/c3_report.md")
FIND = pm.FIND
BINS = [(0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 1.01)]
SCAN_PTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.7, 1.0]


def load(fn):
    """按 id 去重（保留首条），并核对行数。"""
    f = P / fn
    if not f.exists():
        return {}, 0
    out, raw = {}, 0
    for line in f.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw += 1
        r = json.loads(line)
        out.setdefault(r["id"], r)
    if raw != len(out):
        print(f"[dedup] {fn}: {raw} 行 → {len(out)} 个唯一 id")
    return out, raw


def pred(r):
    return int(r["pred_birads"])


def fmt(v, nd=4):
    return "—" if v is None else f"{v:.{nd}f}"


def metrics(rows_by_id):
    if not rows_by_id:
        return None
    rows = list(rows_by_id.values())
    b = pm.birads_block(rows)
    d = pm.density_block(rows)
    f = pm.findings_block(rows)
    fa = [v for v in f["auc"].values() if v]
    return {"n": b["n"], "bi_auc": b["macro_auc"], "ge3": b["ge3_auc"], "f1": b["macro_f1"],
            "bacc": b["bacc"], "acc": b["acc"], "dens": d.get("macro_auc"),
            "find_auc": float(np.mean(fa)) if fa else None,
            "sens90": b["ge3_sens_at_spec"]["spec90"]["sensitivity"], "f": f}


def own_subset(rows):
    ids = [i for i, r in rows.items() if pred(r) <= 2 and (r.get("max_sign_p") or 0) > 0.1]
    v = [rows[i]["p_birads_ge3"] for i in ids]
    return (float(np.median(v)) if v else None), len(v)


def b0_subset(rows, b0):
    ids = [i for i, r in b0.items() if pred(r) <= 2 and (r.get("max_sign_p") or 0) > 0.1]
    v = [rows[i]["p_birads_ge3"] for i in ids if i in rows]
    return (float(np.median(v)) if v else None), len(v)


def stratified(rows, b0):
    out = []
    for lo, hi in BINS:
        ids = [i for i in rows if i in b0 and lo <= (b0[i].get("max_sign_p") or 0) < hi]
        if not ids:
            out.append((lo, hi, 0, None, None))
            continue
        y = np.array([1 if int(rows[i]["gt_birads"]) >= 3 else 0 for i in ids])
        p = np.array([1 if pred(rows[i]) >= 3 else 0 for i in ids])
        sens = float(p[y == 1].mean()) if (y == 1).any() else None
        spec = float(1 - p[y == 0].mean()) if (y == 0).any() else None
        out.append((lo, hi, len(ids), sens, spec))
    return out


def confusion_prf(rows_by_id):
    """BI-RADS 5×5 混淆矩阵（行=GT，列=预测）与逐类 P/R/F1。"""
    cm = np.zeros((5, 5), dtype=int)
    for r in rows_by_id.values():
        gt, pr = int(r["gt_birads"]), pred(r)
        if 1 <= gt <= 5 and 1 <= pr <= 5:
            cm[gt - 1, pr - 1] += 1
    prf = {}
    for c in range(5):
        tp = cm[c, c]
        col = cm[:, c].sum()
        row = cm[c, :].sum()
        p = tp / col if col else None
        rec = tp / row if row else None
        f1 = (2 * p * rec / (p + rec)) if (p and rec) else (0.0 if (col or row) else None)
        prf[c + 1] = {"prec": p, "recall": rec, "f1": f1, "support": int(row)}
    return cm, prf


def scan_table():
    f = P / "c3full_scan.jsonl"
    if not f.exists():
        return None
    ids = set(json.load(open(P / "ids2_scan20.json", encoding="utf-8")))
    by = {}
    for line in f.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["id"] in ids:
            by.setdefault(round(r["scan_p"], 2), {"pos": [], "neg": []})
            by[round(r["scan_p"], 2)]["pos" if r["gt_ge3"] else "neg"].append(r["p_birads_ge3"])
    return {p: (float(np.mean(v["pos"])) if v["pos"] else None,
                float(np.mean(v["neg"])) if v["neg"] else None) for p, v in by.items()}


def main():
    b0, n_b0 = load("b0_test_hard.jsonl")
    c3, n_c3 = load("c3full_test_soft.jsonl")
    c3h, n_c3h = load("c3full_test_hard.jsonl")
    c3m, n_c3m = load("c3m_test_hard.jsonl")
    c3ms, n_c3ms = load("c3m_test_soft.jsonl")
    c3mt, n_c3mt = load("c3mt_test_hard.jsonl")
    c3mts, n_c3mts = load("c3mt_test_soft.jsonl")
    M = {"B0（对照）": metrics(b0), "C3full(soft)": metrics(c3),
         "C3full(hard)": metrics(c3h), "C3m(soft)": metrics(c3ms),
         "C3m(hard)": metrics(c3m), "C3m+T(soft)": metrics(c3mts),
         "C3m+T(hard)": metrics(c3mt)}

    L = ["# C3 分支报告（axis 全量，对照=B0）", "",
         f"> 数据：B0 {n_b0} 行、C3full(soft) {n_c3} 行"
         + (f"、C3full(hard) {n_c3h} 行" if n_c3h else "、C3full(hard) 待评测") + "（测试集 2000）", "",
         "> 注入模式：**C3full 用 axis**（mixer 沿证据轴）；**C3m / C3m+T 用 linear**（s=p，"
         "视角 A 为线性软证据，「未知」向量单独可训练）。", "",
         "> 口径：端点核对结论为「**同协议下 axis ≡ 硬 token**（差 0.0004）」；"
         "扫描曲线**共用 prefill**，与逐例评估存在约 **0.02 的固定偏移**，只用于看形状，"
         "不与 soft/hard 的逐例指标直接比较。", "",
         "## 0. 结论表", "",
         "| 组 | 密度 macroAUC | BI-RADS macroAUC | 征象 meanAUC | ≥3 AUC | sens@spec90 | macroF1 | bACC | acc |",
         "|---|---|---|---|---|---|---|---|---|"]
    for g in ("B0（对照）", "C3full(soft)", "C3full(hard)", "C3m(soft)", "C3m(hard)", "C3m+T(soft)", "C3m+T(hard)"):
        m = M[g]
        if not m:
            L.append("| " + g + " | — | — | — | — | — | — | — | — |")
            continue
        cells = [g, fmt(m["dens"]), fmt(m["bi_auc"]), fmt(m["find_auc"]), fmt(m["ge3"]),
                 fmt(m["sens90"]), fmt(m["f1"]), fmt(m["bacc"]), fmt(m["acc"])]
        L.append("| " + " | ".join(cells) + " |")
    L.append("")

    L += ["## 1. 软一致性与硬矛盾", "",
          "软一致性（各自子集）：该模型 pred≤2 且该模型 max P(1)>0.1。", "",
          "| 组 | 自身子集 n | P(≥3) 中位 | 征象全 0 但 pred≥3（其中 GT≥3） | 硬矛盾 A（pred≥3 且 maxSign<0.1） | B（pred≤2 且 maxSign≥0.5） |",
          "|---|---|---|---|---|---|"]
    rowsmap = {"B0（对照）": b0, "C3full(soft)": c3, "C3full(hard)": c3h,
               "C3m(soft)": c3ms, "C3m(hard)": c3m,
               "C3m+T(soft)": c3mts, "C3m+T(hard)": c3mt}
    for g in ("B0（对照）", "C3full(soft)", "C3full(hard)", "C3m(soft)", "C3m(hard)", "C3m+T(soft)", "C3m+T(hard)"):
        rows = rowsmap[g]
        if not rows:
            continue
        med, n = own_subset(rows)
        a = sum(1 for r in rows.values() if pred(r) >= 3 and (r.get("max_sign_p") or 0) < 0.1)
        b = sum(1 for r in rows.values() if pred(r) <= 2 and (r.get("max_sign_p") or 0) >= 0.5)
        zero_rows = [r for r in rows.values()
                     if (r.get("p_findings") or {}) and pred(r) >= 3
                     and max((r.get("p_findings") or {"x": 0.0}).values()) < 0.5]
        zero = len(zero_rows)
        zero3 = sum(1 for r in zero_rows if int(r["gt_birads"]) >= 3)
        L.append(f"| {g} | {n} | {fmt(med)} | {zero}（其中 GT≥3: {zero3}） | {a} | {b} |")
    L.append("")

    if b0:
        L += ["**共同子集（以 B0 划分）**：B0 pred≤2 且 B0 max P(1)>0.1。", "",
              "| 组 | n | P(≥3) 中位 |", "|---|---|---|"]
        for g in ("B0（对照）", "C3full(soft)", "C3full(hard)", "C3m(soft)", "C3m(hard)", "C3m+T(soft)", "C3m+T(hard)"):
            rows = rowsmap[g]
            if not rows:
                continue
            med, n = b0_subset(rows, b0)
            L.append(f"| {g} | {n} | {fmt(med)} |")
        L.append("")

        L += ["## 1b. 按 B0 的 max P(1) 分段：BI-RADS≥3 灵敏度 / 特异度", "",
              "| B0 maxP(1) 段 | n | B0 灵敏度 | B0 特异度 | C3full(soft) 灵敏度 | C3full(soft) 特异度 | C3full(hard) 灵敏度 | C3full(hard) 特异度 |",
              "|---|---|---|---|---|---|---|---|"]
        sc = {g: stratified(rows, b0) for g, rows in
              (("B0", b0), ("soft", c3), ("hard", c3h)) if rows}
        for k, (lo, hi) in enumerate(BINS):
            n = sc["B0"][k][2] if "B0" in sc else 0
            cells = [f"[{lo}, {hi})" if hi <= 1 else f">={lo}", str(n)]
            for g in ("B0", "soft", "hard"):
                cells += [fmt(sc[g][k][3]), fmt(sc[g][k][4])] if g in sc else ["—", "—"]
            L.append("| " + " | ".join(cells) + " |")
        L.append("")

        ids = [i for i, r in b0.items()
               if int(r["gt_birads"]) >= 3 and pred(r) <= 2 and (r.get("max_sign_p") or 0) > 0.1]
        L += ["## 3.「有怀疑但漏判」子集（GT≥3 且 B0 pred≤2 且 B0 maxSign>0.1）", "",
              "| 组 | n | 征象 meanAUC | mass 召回 | calc 召回 | asym 召回 | arch 召回 |",
              "|---|---|---|---|---|---|---|"]
        for g, rows in (("B0（对照）", b0), ("C3full(soft)", c3), ("C3full(hard)", c3h)):
            sub = [rows[i] for i in ids if i in rows]
            if not sub:
                L.append(f"| {g} | 0 | — | — | — | — | — |")
                continue
            f = pm.findings_block(sub)
            favg = float(np.mean([v for v in f["auc"].values() if v]))
            cells = [g, str(len(sub)), f"{favg:.4f}"] + [fmt(f["recall"][k]) for k in FIND]
            L.append("| " + " | ".join(cells) + " |")
        L.append("")

    # ---- 1c. BI-RADS 5×5 混淆矩阵 + 逐类 P/R/F1 ----
    sets = [("B0（对照）", b0), ("C3full(soft)", c3)]
    if c3h:
        sets.append(("C3full(hard)", c3h))
    if c3ms:
        sets.append(("C3m(soft)", c3ms))
    if c3m:
        sets.append(("C3m(hard)", c3m))
    if c3mts:
        sets.append(("C3m+T(soft)", c3mts))
    if c3mt:
        sets.append(("C3m+T(hard)", c3mt))
    L += ["## 1c. BI-RADS 5×5 混淆矩阵（行=GT，列=预测）", ""]
    for g, rows in sets:
        if not rows:
            continue
        cm, _ = confusion_prf(rows)
        L += [f"**{g}**（n={int(cm.sum())}）", "",
              "| GT \\\\ 预测 | 1 | 2 | 3 | 4 | 5 | 合计 |",
              "|---|---|---|---|---|---|---|"]
        for i in range(5):
            L.append(f"| **{i + 1}** | " + " | ".join(str(int(x)) for x in cm[i])
                     + f" | {int(cm[i].sum())} |")
        L.append("| **合计** | " + " | ".join(str(int(x)) for x in cm.sum(axis=0))
                 + f" | {int(cm.sum())} |")
        L.append("")
    L += ["**逐类精确率 / 召回 / F1**", "",
          "| 组 | 类 | 精确率 | 召回 | F1 | 支持数 |", "|---|---|---|---|---|---|"]
    for g, rows in sets:
        if not rows:
            continue
        _, prf = confusion_prf(rows)
        for c in range(1, 6):
            d = prf[c]
            L.append(f"| {g} | {c} | {fmt(d['prec'], 3)} | {fmt(d['recall'], 3)} | "
                     f"{fmt(d['f1'], 3)} | {d['support']} |")
    L.append("")

    L += ["## 2. 逐征象（召回 / F1 / AUC）", "", "| 组 | 征象 | 召回 | F1 | AUC |", "|---|---|---|---|---|"]
    for g in ("B0（对照）", "C3full(soft)", "C3full(hard)", "C3m(soft)", "C3m(hard)", "C3m+T(soft)", "C3m+T(hard)"):
        m = M[g]
        if not m:
            continue
        f = m["f"]
        for k in FIND:
            star = " **(重点)**" if k == "mass" else ""
            L.append(f"| {g} | {k}{star} | {fmt(f['recall'][k])} | {fmt(f['f1'][k])} | {fmt(f['auc'][k])} |")
        recs = [f["recall"][k] for k in FIND if f["recall"][k] is not None]
        f1s = [f["f1"][k] for k in FIND if f["f1"][k] is not None]
        L.append(f"| {g} | **macro（4 征象均值）** | "
                 f"{fmt(float(np.mean(recs)) if recs else None)} | "
                 f"{fmt(float(np.mean(f1s)) if f1s else None)} | — |")
    L.append("")

    sc = scan_table()
    if sc:
        pts = [p for p in SCAN_PTS if p in sc]
        L += ["## 4. C3full 概率扫描（9 点，40 例：GT≥3 / GT≤2 各 20）", "",
              "| p | " + " | ".join(f"{p}" for p in pts) + " |",
              "|---|" + "---|" * len(pts),
              "| GT≥3 | " + " | ".join(fmt(sc[p][0], 3) for p in pts) + " |",
              "| GT≤2 | " + " | ".join(fmt(sc[p][1], 3) for p in pts) + " |", ""]
        if 0.1 in sc and 0.5 in sc and sc[0.5][0] is not None:
            L += [f"p=0.1→0.5 上升 {sc[0.5][0] - sc[0.1][0]:+.3f}；若为台阶形态，"
                  "标注「阈值前移，非渐变响应」。", ""]

    pairs = [("C3full(soft)-B0", b0, c3)]
    if c3h:
        pairs.append(("C3full(hard)-B0", b0, c3h))
        if c3:
            pairs.append(("C3full(soft)-C3full(hard)", c3h, c3))
    if c3m:
        pairs.append(("C3m-B0", b0, c3m))
        if c3h:
            pairs.append(("C3m-C3full(hard)", c3h, c3m))
    if c3mt:
        pairs.append(("C3m+T-B0", b0, c3mt))
        if c3h:
            pairs.append(("C3m+T-C3full(hard)", c3h, c3mt))
        if c3m:
            pairs.append(("C3m+T-C3m", c3m, c3mt))
    # C3m 的 soft 三组对比（用户指定）
    if c3ms:
        pairs.append(("C3m(soft)-B0", b0, c3ms))
        if c3:
            pairs.append(("C3m(soft)-C3full(soft)", c3, c3ms))
        if c3m:
            pairs.append(("C3m(soft)-C3m(hard)", c3m, c3ms))
    if c3mts:
        pairs.append(("C3m+T(soft)-B0", b0, c3mts))
        if c3:
            pairs.append(("C3m+T(soft)-C3full(soft)", c3, c3mts))
        if c3mt:
            pairs.append(("C3m+T(soft)-C3m+T(hard)", c3mt, c3mts))
    boots = {}
    for tag, rb, rc in pairs:
        if rb and rc:
            boots[tag] = bootstrap_pair(rb, rc)
    if boots:
        L += ["## 5. study 级配对 bootstrap（1000 次，seed 42）", "",
              "Δ = 后者 − 前者；`*` = 95% CI 不含 0。", "",
              "| 指标 | " + " | ".join(boots) + " |", "|---|" + "---|" * len(boots)]
        for key, lab in (("birads_macro_auc", "BI-RADS macroAUC"), ("ge3_auc", "≥3 AUC"),
                         ("macro_f1", "macroF1"), ("bacc", "bACC"), ("acc", "acc"),
                         ("recall_c3", "3 级召回"), ("sens_spec90", "sens@spec90"),
                         ("density_auc", "密度 AUC"), ("findings_mean_auc", "征象 meanAUC"),
                         ("findings_macro_f1", "征象 macroF1（阈值 0.5）")):
            cells = []
            for _t, r in boots.items():
                d, lo, hi = r.get(key, (None, None, None))
                if d is None:
                    cells.append("—")
                else:
                    star = "*" if (lo > 0 or hi < 0) else ""
                    cells.append(f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]{star}")
            L.append(f"| {lab} | " + " | ".join(cells) + " |")
        L.append("")

    rsf = Path("/tmp/c3full_rs.json")
    if rsf.exists():
        d = json.load(open(rsf))
        pts, vals = d["points"], d["r"]
        L += ["## 6. C3full 的 w(p) 与 s−p", "",
              "| p | " + " | ".join(str(p) for p in pts) + " |",
              "|---|" + "---|" * len(pts),
              "| w(p)=s | " + " | ".join(f"{v:+.3f}" for v in vals) + " |",
              "| s−p | " + " | ".join(f"{v - p:+.3f}" for v, p in zip(vals, pts)) + " |", ""]


    # ---- 7. C3m 训练全程：A→M / M→A / 不蒸馏 比例 ----
    logf = Path("/home/z/zhao/Mammography/Mammo/2-view/chain_v1/logs/train_C3m.log")
    if logf.exists():
        import ast as _ast
        import re as _re
        recs = []
        for chunk in logf.read_text(errors="ignore").replace("\r", "\n").split("\n"):
            if "'n_am'" not in chunk:
                continue
            m = _re.search(r"\{'rho'.*\}", chunk)
            if not m:
                continue
            try:
                recs.append(_ast.literal_eval(m.group(0)))
            except Exception:
                pass
        if recs:
            def _f(d, k):
                try:
                    return float(d.get(k))
                except (TypeError, ValueError):
                    return 0.0
            blocks = [(i, min(i + 49, len(recs))) for i in range(0, len(recs), 50)]
            L += ["## 7. C3m 训练全程：老师选择比例（每 50 步）", "",
                  "| 步区间 | A→M（老师 A） | M→A（老师 M） | 不蒸馏 |", "|---|---|---|---|"]
            curve = []
            for lo, hi in blocks:
                seg = recs[lo:hi]
                tot = max(1.0, sum(_f(d, "n_micro") for d in seg))
                a_am = sum(_f(d, "n_am") for d in seg) / tot
                a_ma = sum(_f(d, "n_ma") for d in seg) / tot
                a_no = sum(_f(d, "n_none") for d in seg) / tot
                curve.append((lo + 1, a_am, a_ma, a_no))
                L.append(f"| {lo + 1}-{hi} | {a_am:.3f} | {a_ma:.3f} | {a_no:.3f} |")
            L.append("")
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(6.6, 3.6))
                xs = [c[0] for c in curve]
                ax.plot(xs, [c[1] for c in curve], "o-", label="A->M (teacher A)")
                ax.plot(xs, [c[2] for c in curve], "s-", label="M->A (teacher M)")
                ax.plot(xs, [c[3] for c in curve], "^--", label="no KD")
                ax.set_xlabel("step"); ax.set_ylabel("fraction")
                ax.set_title("C3m teacher selection over training")
                ax.grid(alpha=0.3); ax.legend(fontsize=8)
                fig.tight_layout(); fig.savefig(P / "c3m_teacher_curve.png", dpi=140)
                plt.close(fig)
                L += ["![c3m_teacher_curve](pilot/c3m_teacher_curve.png)", ""]
            except Exception as e:  # noqa: BLE001
                print("[warn] 画 C3m 曲线失败", e)


    # ---- 8. 双视角信号量（dualorder A/B，测试集）----
    fa_ = Path("/home/z/zhao/Mammography/Mammo/2-view/outputs/scoring/dualorder_a_test_with_probs.jsonl")
    fb_ = Path("/home/z/zhao/Mammography/Mammo/2-view/outputs/scoring/dualorder_b_test_with_probs.jsonl")
    if fa_.exists() and fb_.exists():
        A_, B_ = {}, {}
        for line in fa_.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r_ = json.loads(line); A_[r_["id"]] = r_
        for line in fb_.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r_ = json.loads(line); B_[r_["id"]] = r_
        ids_ = sorted(set(A_) & set(B_))
        pa3, pb3, hi_a, hi_b, tie = [], [], 0, 0, 0
        for i in ids_:
            gt_ = int(A_[i]["ground_truth"]["breast_birads"])
            ba = A_[i]["probs_norm"]["breast_birads"]; bb = B_[i]["probs_norm"]["breast_birads"]
            pa3.append(1.0 - float(ba.get("1", 0)) - float(ba.get("2", 0)))
            pb3.append(1.0 - float(bb.get("1", 0)) - float(bb.get("2", 0)))
            pa_g = float(ba.get(str(gt_), 0.0)); pb_g = float(bb.get(str(gt_), 0.0))
            if pa_g - pb_g >= 0.05:
                hi_a += 1
            elif pb_g - pa_g >= 0.05:
                hi_b += 1
            else:
                tie += 1
        n_ = len(ids_)
        L += ["## 8. 双视角（顺序 A：征象→分级 / 顺序 B：分级在前）", "",
              "| 指标 | 顺序 A | 顺序 B |", "|---|---|---|",
              f"| P(≥3) 中位 | {float(np.median(pa3)):.4f} | {float(np.median(pb3)):.4f} |",
              f"| **P(≥3) < 0.01 的比例** | **{np.mean(np.array(pa3) < 0.01):.1%}** | "
              f"{np.mean(np.array(pb3) < 0.01):.1%} |",
              f"| P(≥3) < 0.1 的比例 | {np.mean(np.array(pa3) < 0.1):.1%} | "
              f"{np.mean(np.array(pb3) < 0.1):.1%} |", "",
              f"GT 分级概率差 ≥0.05：A 更高 {hi_a}/{n_}（{hi_a / n_:.1%}），"
              f"B 更高 {hi_b}/{n_}（{hi_b / n_:.1%}），<0.05 {tie}/{n_}（{tie / n_:.1%}）。", ""]

    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    json.dump(M, open(P / "c3_metrics.json", "w"), ensure_ascii=False, indent=1)
    print("[ok]", OUT)


if __name__ == "__main__":
    main()
