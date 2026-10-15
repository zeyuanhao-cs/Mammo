#!/usr/bin/env python3
"""
字段级概率评估: raw LoRA vs Repeat-OS LoRA。

输入: score_2view.py 的两个输出 jsonl (含 probs_norm)
输出: metrics_probs_raw_vs_os.json

指标:
  - BI-RADS 5 类: acc / macro-F1 / 每类 P/R/F1 / 混淆矩阵; 二分类 (>=4): sens/spec/precision/F1
  - Density 4 类: acc / macro-F1 / 每类 P/R/F1
  - Findings 4 个二分类: AUROC / PR-AUC / 阈值 t=0.1..0.5 的 P/R/F1
  - mass+calc macro-F1 @0.5; four-finding macro-F1 @0.5; any-finding F1 @各阈值
"""

import argparse
import json
from pathlib import Path

FINDINGS = ["mass", "suspicious_calcification", "asymmetry", "architectural_distortion"]
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5]
BIRADS_CLASSES = ["1", "2", "3", "4", "5"]
DENSITY_CLASSES = ["A", "B", "C", "D"]


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def multiclass_metrics(y_true, y_pred, classes):
    cm = {t: {p: 0 for p in classes} for t in classes}
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    per_class = {}
    f1s = []
    for c in classes:
        tp = cm[c][c]
        fp = sum(cm[t][c] for t in classes if t != c)
        fn = sum(cm[c][p] for p in classes if p != c)
        p, r, f = prf(tp, fp, fn)
        per_class[c] = {"precision": p, "recall": r, "f1": f, "support": tp + fn}
        f1s.append(f)
    acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true) if y_true else 0.0
    return {"accuracy": acc, "macro_f1": sum(f1s) / len(f1s), "per_class": per_class, "confusion_matrix": cm}


def auroc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return None
    pairs = sorted(zip(scores, labels))
    n = len(pairs)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    rpos = sum(r for r, (_, l) in zip(ranks, pairs) if l == 1)
    npos, nneg = len(pos), len(neg)
    u = rpos - npos * (npos + 1) / 2
    return u / (npos * nneg)


def average_precision(scores, labels):
    npos = sum(labels)
    if npos == 0:
        return None
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    tp, ap = 0, 0.0
    for rank, i in enumerate(order, 1):
        if labels[i] == 1:
            tp += 1
            ap += tp / rank
    return ap / npos


def eval_model(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("probs_norm"):
                rows.append(d)
    n = len(rows)
    res = {"n": n, "file": str(path)}

    # ---------- BI-RADS ----------
    y_true, y_pred = [], []
    for d in rows:
        gt = str(d["ground_truth"]["breast_birads"])
        pn = d["probs_norm"]["breast_birads"]
        if pn is None:
            continue
        y_true.append(gt)
        y_pred.append(max(BIRADS_CLASSES, key=lambda c: pn[c]))
    res["birads_5class"] = multiclass_metrics(y_true, y_pred, BIRADS_CLASSES)

    bt = [1 if v in ("4", "5") else 0 for v in y_true]
    bp = [1 if v in ("4", "5") else 0 for v in y_pred]
    tp = sum(1 for t, p in zip(bt, bp) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(bt, bp) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(bt, bp) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(bt, bp) if t == 0 and p == 0)
    p, r, f = prf(tp, fp, fn)
    res["birads_binary_ge4"] = {
        "sensitivity": r, "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "precision": p, "f1": f,
    }

    # ---------- Density ----------
    y_true, y_pred = [], []
    for d in rows:
        gt = d["ground_truth"]["breast_density"]
        pn = d["probs_norm"]["breast_density"]
        if pn is None:
            continue
        y_true.append(gt)
        y_pred.append(max(DENSITY_CLASSES, key=lambda c: pn[c]))
    res["density_4class"] = multiclass_metrics(y_true, y_pred, DENSITY_CLASSES)

    # ---------- Findings ----------
    findings = {}
    f1_at_05 = {}
    for k in FINDINGS:
        scores, labels = [], []
        for d in rows:
            gt = d["ground_truth"]["findings"][k]
            pn = d["probs_norm"].get(f"findings.{k}")
            if pn is None:
                continue
            scores.append(pn["1"])
            labels.append(gt)
        entry = {"n_pos": sum(labels), "n_neg": len(labels) - sum(labels),
                 "auroc": auroc(scores, labels), "pr_auc": average_precision(scores, labels)}
        entry["thresholds"] = {}
        for t in THRESHOLDS:
            tp = sum(1 for s, l in zip(scores, labels) if s >= t and l == 1)
            fp = sum(1 for s, l in zip(scores, labels) if s >= t and l == 0)
            fn = sum(1 for s, l in zip(scores, labels) if s < t and l == 1)
            p, r, f = prf(tp, fp, fn)
            entry["thresholds"][str(t)] = {"precision": p, "recall": r, "f1": f}
        findings[k] = entry
        f1_at_05[k] = entry["thresholds"]["0.5"]["f1"]
    res["findings"] = findings
    res["mass_calc_macro_f1@0.5"] = (f1_at_05["mass"] + f1_at_05["suspicious_calcification"]) / 2
    res["four_finding_macro_f1@0.5"] = sum(f1_at_05.values()) / len(f1_at_05)

    # ---------- any finding ----------
    any_res = {}
    for t in THRESHOLDS:
        tp = fp = fn = 0
        for d in rows:
            gt_any = 1 if any(d["ground_truth"]["findings"][k] == 1 for k in FINDINGS) else 0
            pred_any = 0
            for k in FINDINGS:
                pn = d["probs_norm"].get(f"findings.{k}")
                if pn and pn["1"] >= t:
                    pred_any = 1
                    break
            if gt_any == 1 and pred_any == 1:
                tp += 1
            elif gt_any == 0 and pred_any == 1:
                fp += 1
            elif gt_any == 1 and pred_any == 0:
                fn += 1
        p, r, f = prf(tp, fp, fn)
        any_res[str(t)] = {"precision": p, "recall": r, "f1": f}
    res["any_finding"] = any_res
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=str, required=True)
    parser.add_argument("--os", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    raw = eval_model(args.raw)
    os_ = eval_model(args.os)

    out = {
        "raw_lora": raw,
        "repeat_os_lora": os_,
        "comparison": {
            "birads_macro_f1": {"raw": raw["birads_5class"]["macro_f1"], "os": os_["birads_5class"]["macro_f1"]},
            "birads_binary_sensitivity": {
                "raw": raw["birads_binary_ge4"]["sensitivity"], "os": os_["birads_binary_ge4"]["sensitivity"],
            },
            "density_macro_f1": {"raw": raw["density_4class"]["macro_f1"], "os": os_["density_4class"]["macro_f1"]},
            "findings_auroc": {
                k: {"raw": raw["findings"][k]["auroc"], "os": os_["findings"][k]["auroc"]} for k in FINDINGS
            },
            "four_finding_macro_f1@0.5": {
                "raw": raw["four_finding_macro_f1@0.5"], "os": os_["four_finding_macro_f1@0.5"],
            },
        },
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"n: raw={raw['n']} os={os_['n']}")
    print(f"BI-RADS macro-F1 : raw={raw['birads_5class']['macro_f1']:.4f}  os={os_['birads_5class']['macro_f1']:.4f}")
    print(f"BI-RADS>=4 sens  : raw={raw['birads_binary_ge4']['sensitivity']:.4f}  os={os_['birads_binary_ge4']['sensitivity']:.4f}")
    print(f"Density macro-F1 : raw={raw['density_4class']['macro_f1']:.4f}  os={os_['density_4class']['macro_f1']:.4f}")
    for k in FINDINGS:
        print(f"{k:26s} AUROC: raw={raw['findings'][k]['auroc']:.4f}  os={os_['findings'][k]['auroc']:.4f}   "
              f"F1@0.5: raw={raw['findings'][k]['thresholds']['0.5']['f1']:.4f}  os={os_['findings'][k]['thresholds']['0.5']['f1']:.4f}")
    print(f"four-finding macro-F1@0.5: raw={raw['four_finding_macro_f1@0.5']:.4f}  os={os_['four_finding_macro_f1@0.5']:.4f}")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
