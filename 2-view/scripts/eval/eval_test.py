#!/usr/bin/env python3
"""读取 test 概率 jsonl + thresholds.json, 一次性评估:
- findings: default (greedy) / calibrated (t_f1) / spec90 三套, 每个 finding 补 AUC / PR-AUC / 阳性率
- density: default / cal (4类log-bias) / cal_merged (AB/C/D 3类) 三套; 补 OvR macro AUC (只算 test 出现的类) + top-label ECE
- birads: default / cal (5类log-bias); 补 OvR macro AUC + top-label ECE
- birads >=3, >=4 二分类 (calibrated 阈值) 的 sen/spec/P/F1/AUC/PR-AUC
- 2 vs 4/5 AUC 两版: A = P(4)+P(5); B = (P4+P5)/(P2+P4+P5)
- 核心指标带按 study 分组的 bootstrap 95% CI (默认 1000 次, seed 42)
- test 不参与任何参数选择

v2 (相对 backup_v1): 新增 findings AUC/PR-AUC/阳性率, density+BI-RADS OvR macro AUC 与
top-label ECE (带 CI), 2v45 AUC 两版; 阈值与偏置来自 v2 拟合 (参考类固定 + val 实际值扫描).
"""
import argparse
import json
import time
from pathlib import Path

import common as C

N_BOOT = 1000
SEED = 42


def _exp(x):
    return pow(2.718281828459045, x)


def _merge_density(p):
    return {"AB": p["A"] + p["B"], "C": p["C"], "D": p["D"]}


def density_pred(recs, classes, bias=None):
    out = []
    for r in recs:
        pd = _merge_density(r["p_density"]) if classes[0] == "AB" else r["p_density"]
        if bias:
            pd = {c: pd[c] * _exp(bias[c]) for c in classes}
        out.append(max(classes, key=lambda c: pd[c]))
    return out


def density_gt(recs, classes):
    if classes[0] == "AB":
        return ["AB" if r["gt_density"] in ("A", "B") else r["gt_density"] for r in recs]
    return [r["gt_density"] for r in recs]


def density_probs(recs, classes, bias=None):
    """每样本 {class: prob}; bias 时乘性偏置后重归一化 (用于 OvR AUC / top-label ECE)."""
    out = []
    for r in recs:
        pd = _merge_density(r["p_density"]) if classes[0] == "AB" else dict(r["p_density"])
        pd = {c: pd[c] for c in classes}
        if bias:
            pd = {c: pd[c] * _exp(bias[c]) for c in classes}
        z = sum(pd.values())
        if z > 0:
            pd = {c: v / z for c, v in pd.items()}
        else:
            pd = {c: 1.0 / len(classes) for c in classes}
        out.append(pd)
    return out


def birads_pred(recs, bias=None):
    out = []
    for r in recs:
        pb = r["p_birads"]
        if bias:
            pb = {i: pb[str(i)] * _exp(bias[i if i in bias else str(i)]) for i in C.BIRADS_CLASSES}
            out.append(max(C.BIRADS_CLASSES, key=lambda i: pb[i]))
        else:
            best, bs = None, -1.0
            for i in C.BIRADS_CLASSES:
                if pb[str(i)] > bs:
                    bs, best = pb[str(i)], i
            out.append(best)
    return out


def birads_probs(recs, bias=None):
    """每样本 {1..5: prob} (int key), bias 时乘性偏置后重归一化."""
    out = []
    for r in recs:
        pb = {i: r["p_birads"][str(i)] for i in C.BIRADS_CLASSES}
        if bias:
            pb = {i: pb[i] * _exp(bias[i if i in bias else str(i)]) for i in C.BIRADS_CLASSES}
        z = sum(pb.values())
        if z > 0:
            pb = {i: v / z for i, v in pb.items()}
        else:
            pb = {i: 1.0 / len(C.BIRADS_CLASSES) for i in C.BIRADS_CLASSES}
        out.append(pb)
    return out


def eval_density(recs, tag, classes, pred_fn, gt_fn, full=True):
    y = gt_fn(recs)
    yhat = pred_fn(recs)
    m = C.multiclass_metrics(y, yhat, classes)
    res = {"acc": m["acc"], "bacc": m["bacc"], "macro_f1": m["macro_f1"],
           "wkappa": C.quadratic_weighted_kappa(y, yhat, classes)}
    print("  [%s] acc=%.4f  bACC=%.4f  macroF1=%.4f  wkappa=%.4f"
          % (tag, res["acc"], res["bacc"], res["macro_f1"], res["wkappa"]))
    if full:
        for c in classes:
            pc = m["per_class"][c]
            print("    %-3s P=%.4f R=%.4f F1=%.4f (n=%d)" % (c, pc["precision"], pc["recall"], pc["f1"], pc["support"]))
    return res


def add_ovr_ece(recs, results, key, classes, gt_fn, probs_fn, n_boot):
    """给 results[key] 补 OvR macro AUC + top-label ECE (含 bootstrap CI)."""
    a = C.ovr_macro_auc(gt_fn(recs), probs_fn(recs), classes)
    e = C.top_label_ece(gt_fn(recs), probs_fn(recs), classes)
    lo, hi = C.bootstrap_ci(recs, lambda rs: C.ovr_macro_auc(gt_fn(rs), probs_fn(rs), classes), n_boot, SEED)
    results[key]["ovr_macro_auc"] = a
    results[key]["ci_ovr_macro_auc"] = [lo, hi]
    lo, hi = C.bootstrap_ci(recs, lambda rs: C.top_label_ece(gt_fn(rs), probs_fn(rs), classes), n_boot, SEED)
    results[key]["top_label_ece"] = e
    results[key]["ci_top_label_ece"] = [lo, hi]
    print("         OvR macroAUC %s | topLabelECE %s"
          % (C.fmt_ci(a, results[key]["ci_ovr_macro_auc"]),
             C.fmt_ci(e, results[key]["ci_top_label_ece"])))


def _bin_score(recs, level):
    y = [1 if r["gt_birads"] >= level else 0 for r in recs]
    s = [sum(r["p_birads"][str(i)] for i in range(level, 6)) for r in recs]
    return y, s


def _2v45_score(recs, mode):
    """mode='A': P4+P5; mode='B': (P4+P5)/(P2+P4+P5). 子集 = gt in {2,4,5}, 正例 gt>=4."""
    sub = [r for r in recs if r["gt_birads"] in (2, 4, 5)]
    y = [1 if r["gt_birads"] >= 4 else 0 for r in sub]
    s = []
    for r in sub:
        p45 = r["p_birads"]["4"] + r["p_birads"]["5"]
        if mode == "A":
            s.append(p45)
        else:
            d = r["p_birads"]["2"] + p45
            s.append(p45 / d if d > 0 else 0.5)
    return y, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--thresholds", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()

    t0 = time.time()
    recs = C.load_probs_jsonl(args.test)
    th = json.load(open(args.thresholds, encoding="utf-8"))
    print("[test] %d records | tag=%s" % (len(recs), args.tag))
    results = {"tag": args.tag, "n_test": len(recs)}

    # ================= density =================
    print("\n== density (4类 A/B/C/D) ==")
    bias_d = th["density"]["logbias"]
    results["density_default"] = eval_density(
        recs, "default", C.DENSITY_CLASSES,
        lambda rs: [r["greedy_density"] for r in rs],
        lambda rs: density_gt(rs, C.DENSITY_CLASSES))
    results["density_cal"] = eval_density(
        recs, "cal", C.DENSITY_CLASSES,
        lambda rs: density_pred(rs, C.DENSITY_CLASSES, bias_d),
        lambda rs: density_gt(rs, C.DENSITY_CLASSES))

    print("\n== density merged (AB/C/D 3类) ==")
    bias_m = th["density_merged"]["logbias"]
    results["density_cal_merged"] = eval_density(
        recs, "cal_merged", C.DENSITY_MERGED_CLASSES,
        lambda rs: density_pred(rs, C.DENSITY_MERGED_CLASSES, bias_m),
        lambda rs: density_gt(rs, C.DENSITY_MERGED_CLASSES))

    # density bootstrap (acc/bacc) + OvR AUC + top-label ECE
    for key, pred_fn, gt_fn, cl, bias in [
        ("density_default", lambda rs: [r["greedy_density"] for r in rs],
         lambda rs: density_gt(rs, C.DENSITY_CLASSES), C.DENSITY_CLASSES, None),
        ("density_cal", lambda rs: density_pred(rs, C.DENSITY_CLASSES, bias_d),
         lambda rs: density_gt(rs, C.DENSITY_CLASSES), C.DENSITY_CLASSES, bias_d),
        ("density_cal_merged", lambda rs: density_pred(rs, C.DENSITY_MERGED_CLASSES, bias_m),
         lambda rs: density_gt(rs, C.DENSITY_MERGED_CLASSES), C.DENSITY_MERGED_CLASSES, bias_m),
    ]:
        for met in ("acc", "bacc"):
            def mf(rs, met=met, cl=cl, pf=pred_fn, gf=gt_fn):
                return C.multiclass_metrics(gf(rs), pf(rs), cl)[met]
            lo, hi = C.bootstrap_ci(recs, mf, args.n_boot, SEED)
            results[key]["ci_" + met] = [lo, hi]
        print("  [%s] acc %s | bACC %s" % (key,
              C.fmt_ci(results[key]["acc"], results[key]["ci_acc"]),
              C.fmt_ci(results[key]["bacc"], results[key]["ci_bacc"])))
        add_ovr_ece(recs, results, key, cl, gt_fn,
                    lambda rs, cl=cl, b=bias: density_probs(rs, cl, b), args.n_boot)

    # ================= BI-RADS =================
    print("\n== BI-RADS (5类) ==")
    bias_b = th["birads"]["logbias"]
    for key, pf in [("birads_default", lambda rs: [r["greedy_birads"] for r in rs]),
                    ("birads_cal", lambda rs: birads_pred(rs, bias_b))]:
        y, yhat = [r["gt_birads"] for r in recs], pf(recs)
        m = C.multiclass_metrics(y, yhat, C.BIRADS_CLASSES)
        wk = C.quadratic_weighted_kappa(y, yhat, C.BIRADS_CLASSES)
        results[key] = {"acc": m["acc"], "bacc": m["bacc"], "macro_f1": m["macro_f1"], "wkappa": wk}
        print("  [%s] acc=%.4f  bACC=%.4f  macroF1=%.4f  wkappa=%.4f" % (key, m["acc"], m["bacc"], m["macro_f1"], wk))
        for met in ("bacc", "macro_f1"):
            def mf(rs, met=met, pf=pf):
                mm = C.multiclass_metrics([r["gt_birads"] for r in rs], pf(rs), C.BIRADS_CLASSES)
                return mm[met]
            lo, hi = C.bootstrap_ci(recs, mf, args.n_boot, SEED)
            results[key]["ci_" + met] = [lo, hi]
        print("         bACC %s | macroF1 %s" % (C.fmt_ci(results[key]["bacc"], results[key]["ci_bacc"]),
                                                 C.fmt_ci(results[key]["macro_f1"], results[key]["ci_macro_f1"])))
        add_ovr_ece(recs, results, key, C.BIRADS_CLASSES,
                    lambda rs: [r["gt_birads"] for r in rs],
                    lambda rs, b=None if key == "birads_default" else bias_b: birads_probs(rs, b),
                    args.n_boot)

    # birads 二分类: >=3 / >=4 (用 val 拟合的阈值)
    for level in (3, 4):
        t = th["birads"]["t_ge3"] if level == 3 else th["birads"]["t_ge4"]
        y, s = _bin_score(recs, level)
        yhat = [1 if p >= t else 0 for p in s]
        m = C.binary_metrics(y, yhat)
        a = C.auc(y, s)
        pa = C.pr_auc(y, s)
        key = "birads_ge%d" % level
        results[key] = {"threshold": t, "recall": m["recall"], "specificity": m["specificity"],
                        "precision": m["precision"], "f1": m["f1"], "auc": a, "pr_auc": pa}
        print("  [>=%d] t=%.4f  sen=%.4f  spec=%.4f  P=%.4f  F1=%.4f  AUC=%s  PR-AUC=%s (n_pos=%d)"
              % (level, t, m["recall"], m["specificity"], m["precision"], m["f1"],
                 C.fmt_ci(a, None), C.fmt_ci(pa, None), m["n_pos"]))
        lo, hi = C.bootstrap_ci(recs, lambda rs, lv=level, tt=t: _bin_f1(rs, lv, tt), args.n_boot, SEED)
        results[key]["ci_f1"] = [lo, hi]
        lo, hi = C.bootstrap_ci(recs, lambda rs, lv=level: _bin_auc(rs, lv), args.n_boot, SEED)
        results[key]["ci_auc"] = [lo, hi]
        print("         F1 %s | AUC %s" % (C.fmt_ci(results[key]["f1"], results[key]["ci_f1"]),
                                           C.fmt_ci(results[key]["auc"], results[key]["ci_auc"])))

    # 2 vs 4/5 AUC: A = P4+P5, B = (P4+P5)/(P2+P4+P5)
    for mode, desc in [("A", "P4+P5"), ("B", "(P4+P5)/(P2+P4+P5)")]:
        y2, s2 = _2v45_score(recs, mode)
        a2 = C.auc(y2, s2)
        key = "birads_2v45_auc_" + mode.lower()
        results[key] = {"auc": a2, "score": desc, "n": len(y2), "n_pos": sum(y2)}
        lo, hi = C.bootstrap_ci(recs, lambda rs, md=mode: C.auc(*_2v45_score(rs, md)), args.n_boot, SEED)
        results[key]["ci_auc"] = [lo, hi]
        print("  [2 vs 4/5 %s] score=%s  AUC=%s  (n=%d, pos=%d)"
              % (mode, desc, C.fmt_ci(a2, results[key]["ci_auc"]), len(y2), sum(y2)))

    # ================= findings =================
    print("\n== findings (4 个二分类) ==")
    findings_res = {}
    f1s_d, f1s_c, f1s_s = [], [], []
    for name in C.FINDINGS:
        fth = th["findings"][name]
        row = {}
        for tag_name, t in [("default", None), ("cal", fth["t_f1"]), ("spec90", fth["t_spec90"])]:
            if t is None:  # default: greedy_prediction
                y = [r["gt_findings"][name] for r in recs]
                yhat = [r["greedy_findings"][name] for r in recs]
                m = C.binary_metrics(y, yhat)
            else:
                y = [r["gt_findings"][name] for r in recs]
                yhat = [1 if r["p_findings"][name] >= t else 0 for r in recs]
                m = C.binary_metrics(y, yhat)
            row[tag_name] = {"threshold": t, "precision": m["precision"], "recall": m["recall"],
                             "specificity": m["specificity"], "f1": m["f1"],
                             "pos_rate": m["tp"] / m["n"] if m["n"] else 0.0}
        # 排序指标 (概率本身, 与阈值无关)
        y = [r["gt_findings"][name] for r in recs]
        s = [r["p_findings"][name] for r in recs]
        row["auc"] = C.auc(y, s)
        row["pr_auc"] = C.pr_auc(y, s)
        # ECE (calibrated 概率)
        row["ece_cal"] = C.ece(y, s)
        # bootstrap CI: cal F1 / AUC / PR-AUC
        lo, hi = C.bootstrap_ci(
            recs,
            lambda rs, nm=name, tt=fth["t_f1"]: C.binary_metrics(
                [r["gt_findings"][nm] for r in rs],
                [1 if r["p_findings"][nm] >= tt else 0 for r in rs])["f1"],
            args.n_boot, SEED)
        row["ci_f1_cal"] = [lo, hi]
        lo, hi = C.bootstrap_ci(
            recs, lambda rs, nm=name: C.auc([r["gt_findings"][nm] for r in rs],
                                            [r["p_findings"][nm] for r in rs]),
            args.n_boot, SEED)
        row["ci_auc"] = [lo, hi]
        lo, hi = C.bootstrap_ci(
            recs, lambda rs, nm=name: C.pr_auc([r["gt_findings"][nm] for r in rs],
                                               [r["p_findings"][nm] for r in rs]),
            args.n_boot, SEED)
        row["ci_pr_auc"] = [lo, hi]
        findings_res[name] = row
        f1s_d.append(row["default"]["f1"]); f1s_c.append(row["cal"]["f1"]); f1s_s.append(row["spec90"]["f1"])
        print("  [%s] n_pos=%d flag=%s t_f1=%.4f t_spec90=%.4f" % (name, fth["val_n_pos"], fth["flag"], fth["t_f1"], fth["t_spec90"]))
        print("    default: P=%.4f R=%.4f F1=%.4f 阳性率=%.4f" % (
            row["default"]["precision"], row["default"]["recall"], row["default"]["f1"], row["default"]["pos_rate"]))
        print("    cal    : P=%.4f R=%.4f spec=%.4f F1=%s ECE=%.4f 阳性率=%.4f" % (
            row["cal"]["precision"], row["cal"]["recall"], row["cal"]["specificity"],
            C.fmt_ci(row["cal"]["f1"], row["ci_f1_cal"]), row["ece_cal"], row["cal"]["pos_rate"]))
        print("    spec90 : P=%.4f R=%.4f spec=%.4f F1=%.4f 阳性率=%.4f" % (
            row["spec90"]["precision"], row["spec90"]["recall"], row["spec90"]["specificity"],
            row["spec90"]["f1"], row["spec90"]["pos_rate"]))
        print("    AUC=%s  PR-AUC=%s" % (C.fmt_ci(row["auc"], row["ci_auc"]),
                                         C.fmt_ci(row["pr_auc"], row["ci_pr_auc"])))
    results["findings"] = findings_res
    results["findings_macro_f1"] = {"default": sum(f1s_d) / 4, "cal": sum(f1s_c) / 4, "spec90": sum(f1s_s) / 4}
    print("  macro-F1: default=%.4f  cal=%.4f  spec90=%.4f" % (
        results["findings_macro_f1"]["default"], results["findings_macro_f1"]["cal"],
        results["findings_macro_f1"]["spec90"]))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print("\n[out] %s  (%.1fs, bootstrap x%d)" % (args.out, time.time() - t0, args.n_boot))


def _bin_f1(recs, level, t):
    y, s = _bin_score(recs, level)
    return C.binary_metrics(y, [1 if p >= t else 0 for p in s])["f1"]


def _bin_auc(recs, level):
    y, s = _bin_score(recs, level)
    return C.auc(y, s)


if __name__ == "__main__":
    main()
