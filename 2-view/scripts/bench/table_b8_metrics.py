#!/usr/bin/env python3
"""汇总 batch=8 各实验的 density / BI-RADS / findings 指标：AUC、macro-F1、bACC。"""
import json
import sys

RUNS = [
    ("1-epoch b8", "outputs/eval/metrics_probs_os_sub_vt_test_b8.json"),
    ("e1 (b8)", "outputs/eval/metrics_probs_vt-ep3-e1_test_b8.json"),
    ("e2 (b8)", "outputs/eval/metrics_probs_vt-ep3-e2_test_b8.json"),
    ("e3 (b8)", "outputs/eval/metrics_probs_vt-ep3-e3_test_b8.json"),
]
REF = ("1-epoch orig(ref)", "outputs/eval/metrics_probs_os_sub_vt_test.json")

FINDINGS = ["mass", "suspicious_calcification", "asymmetry", "architectural_distortion"]


def g(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def findings_bacc(d):
    vals = []
    for f in FINDINGS:
        a = g(d, "findings", f, "at_0.5")
        if not a:
            continue
        tp, fp, fn, tn = a.get("tp", 0), a.get("fp", 0), a.get("fn", 0), a.get("tn", 0)
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        vals.append((rec + spec) / 2)
    return sum(vals) / len(vals) if vals else None


def row(d):
    return {
        "density_auc": g(d, "density_4class", "ovr_macro_auc"),
        "density_f1": g(d, "density_4class", "macro_f1"),
        "density_bacc": g(d, "density_4class", "bacc"),
        "birads_auc": g(d, "birads_ovr_macro_auc"),
        "birads_f1": g(d, "birads_5class_probs_argmax", "macro_f1"),
        "birads_bacc": g(d, "birads_5class_probs_argmax", "bacc"),
        "fnd_auc": g(d, "findings_macro_auroc"),
        "fnd_f1": g(d, "findings_macro_f1_at_0.5"),
        "fnd_bacc": findings_bacc(d),
    }


COLS = [
    ("Density", "density_auc", "AUC"), ("Density", "density_f1", "macro-F1"), ("Density", "density_bacc", "bACC"),
    ("BI-RADS", "birads_auc", "AUC"), ("BI-RADS", "birads_f1", "macro-F1"), ("BI-RADS", "birads_bacc", "bACC"),
    ("Findings", "fnd_auc", "AUROC"), ("Findings", "fnd_f1", "macro-F1"), ("Findings", "fnd_bacc", "bACC"),
]


def main():
    rows = []
    for name, path in [REF] + RUNS:
        try:
            d = json.load(open(path))
        except FileNotFoundError:
            rows.append((name, None))
            continue
        rows.append((name, row(d)))

    w = 26
    print(f"{'run':<{w}}" + "".join(f"{grp[:5]+'-'+m:>13}" for grp, _, m in COLS))
    print("-" * (w + 13 * len(COLS)))
    for name, r in rows:
        if r is None:
            print(f"{name:<{w}}  (metrics 缺失)")
            continue
        cells = []
        for _, key, _ in COLS:
            v = r.get(key)
            cells.append(f"{v:>13.4f}" if v is not None else f"{'—':>13}")
        print(f"{name:<{w}}" + "".join(cells))

    # 与参考行的差值
    base = next((r for n, r in rows if n == REF[0]), None)
    if base:
        print("\n相对 1-epoch 原流水线 的差值（百分点）:")
        print(f"{'run':<{w}}" + "".join(f"{grp[:4]}-{m:>9}" for grp, _, m in COLS))
        for name, r in rows:
            if r is None or name == REF[0]:
                continue
            if name == "1-epoch b8":
                pass
            cells = []
            for _, key, _ in COLS:
                a, b = base.get(key), r.get(key)
                cells.append(f"{((b - a) * 100):>+13.2f}" if a is not None and b is not None else f"{'—':>13}")
            print(f"{name:<{w}}" + "".join(cells))


if __name__ == "__main__":
    main()
