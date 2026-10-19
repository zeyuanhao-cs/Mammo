#!/usr/bin/env python3
"""只用概率、不拟合任何阈值/偏置的评测 (读取 score_2view.py 的输出 jsonl)。

口径与 eval_test.py 完全一致 —— 复用同一个 common.py 的指标函数, 只是去掉所有
依赖阈值/log-bias 的部分 (那是 fit_thresholds.py 在 val 上拟合出来的)。

包含:
  birads  5 类: greedy / probs-argmax 两版 acc-bACC-macroF1-wkappa-每类PRF1-混淆矩阵
          + OvR macro AUC + top-label ECE
          + >=3 / >=4 二分类 AUC、PR-AUC (score = sum_{i>=level} P(i), 无阈值)
          + 2 vs 4/5 AUC 两版: A = P4+P5; B = (P4+P5)/(P2+P4+P5)
  density 4 类 / 合并 AB C D 3 类: 同上 (greedy / argmax / 混淆矩阵 / OvR macro AUC / ECE)
  findings 4 个二分类: AUROC / PR-AUC / ECE / P-R-F1@0.5 / 阳性率 + macro-F1@0.5
          + any-finding@0.5
  核心指标带按 study 分组的 bootstrap 95% CI (可用 --n-boot 0 关闭)

用法:
  python3 summarize_probs.py --probs outputs/scoring/os_sub_vt_test_with_probs.jsonl \
                             --out outputs/eval/metrics_probs_os_sub_vt_test.json \
                             --tag os_sub_vt
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

N_BOOT = 1000
SEED = 42
FIXED_T = 0.5  # 仅用于给一个"默认阈值"下的 P/R/F1 参考, 非拟合值


# ------------------------------ helpers ------------------------------

def density_argmax(rs, merged=False):
    out = []
    for r in rs:
        pd = r["p_density"]
        if merged:
            pd = {"AB": pd["A"] + pd["B"], "C": pd["C"], "D": pd["D"]}
            out.append(max(C.DENSITY_MERGED_CLASSES, key=lambda c: pd[c]))
        else:
            out.append(max(C.DENSITY_CLASSES, key=lambda c: pd[c]))
    return out


def density_gt(rs, merged=False):
    if merged:
        return ["AB" if r["gt_density"] in ("A", "B") else r["gt_density"] for r in rs]
    return [r["gt_density"] for r in rs]


def density_probs(rs, merged=False):
    if not merged:
        return [dict(r["p_density"]) for r in rs]
    return [{"AB": r["p_density"]["A"] + r["p_density"]["B"],
             "C": r["p_density"]["C"], "D": r["p_density"]["D"]} for r in rs]


def birads_argmax(rs):
    out = []
    for r in rs:
        pb = r["p_birads"]
        out.append(max(C.BIRADS_CLASSES, key=lambda i: pb[str(i)]))
    return out


def birads_gt(rs):
    return [r["gt_birads"] for r in rs]


def birads_probs(rs):
    return [{i: r["p_birads"][str(i)] for i in C.BIRADS_CLASSES} for r in rs]


def sum_score(rs, level):
    """P(level)+...+P(5)."""
    return [sum(r["p_birads"][str(i)] for i in range(level, 6)) for r in rs]


def bin_labels(rs, level):
    return [1 if r["gt_birads"] >= level else 0 for r in rs]


def v45_pairs(rs, mode):
    """子集 gt in {2,4,5}, 正例 gt>=4; score: A=P4+P5, B=(P4+P5)/(P2+P4+P5)."""
    sub = [r for r in rs if r["gt_birads"] in (2, 4, 5)]
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


def ovr_auc_avg(y, probs_list, classes):
    """各类 one-vs-rest AUC 的简单平均; 无法算 AUC 的类 (无正例或无负例) 跳过。
    返回 (平均 AUC, 参与平均的类数)。"""
    aucs = []
    for c in classes:
        yc = [1 if a == c else 0 for a in y]
        if sum(yc) == 0 or sum(yc) == len(yc):
            continue
        s = [p[c if c in p else str(c)] for p in probs_list]
        a = C.auc(yc, s)
        if a is not None:
            aucs.append(a)
    return (sum(aucs) / len(aucs) if aucs else None), len(aucs)


def mc_block(rs, gt_fn, pred_fn, classes):
    y, yhat = gt_fn(rs), pred_fn(rs)
    m = C.multiclass_metrics(y, yhat, classes)
    cm = C.confusion(y, yhat, classes)
    f1_all = sum(m["per_class"][c]["f1"] for c in classes) / len(classes)
    return {
        "n": m["n"], "acc": m["acc"], "bacc": m["bacc"], "macro_f1": m["macro_f1"],
        # 本次口径: 各类 F1 的简单平均, 类别固定为 classes 全量 (含 support=0 的类)
        "macro_f1_all_classes": f1_all,
        "n_classes": len(classes),
        "n_classes_with_support": sum(1 for c in classes if m["per_class"][c]["support"] > 0),
        "wkappa": C.quadratic_weighted_kappa(y, yhat, classes),
        "per_class": {str(c): m["per_class"][c] for c in classes},
        "confusion_matrix": {str(t): {str(p): cm[t][p] for p in classes} for t in classes},
    }


def find_p(rs, name):
    return [r["p_findings"][name] for r in rs]


def find_y(rs, name):
    return [r["gt_findings"][name] for r in rs]


def fmt(v):
    return "  n/a" if v is None else "%.4f" % v


def fmt_ci(v, ci):
    if v is None:
        return "  n/a"
    if not ci or ci[0] is None:
        return "%.4f" % v
    return "%.4f [%.4f, %.4f]" % (v, ci[0], ci[1])


def boot(recs, fn, n_boot):
    if n_boot <= 0:
        return None, None
    return C.bootstrap_ci(recs, fn, n_boot, SEED)


# ------------------------------ main ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", required=True, help="score_2view.py 输出 jsonl")
    ap.add_argument("--out", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--n-boot", type=int, default=N_BOOT, help="bootstrap 次数, 0=跳过")
    args = ap.parse_args()

    t0 = time.time()
    recs = C.load_probs_jsonl(args.probs)
    print("[probs] %d records from %s" % (len(recs), args.probs))
    print("[probs] 只用概率, 未拟合任何阈值; findings 的 0.5/0.5 仅为参考固定阈值\n")

    res = {"tag": args.tag, "file": args.probs, "n": len(recs),
           "note": "probs-only, 无阈值拟合; 0.5 为参考固定阈值"}

    # ---------- BI-RADS ----------
    print("== BI-RADS 5 类 ==")
    for key, pf, gtf in [("greedy", lambda rs: [r["greedy_birads"] for r in rs], birads_gt),
                         ("probs_argmax", birads_argmax, birads_gt)]:
        b = mc_block(recs, gtf, pf, C.BIRADS_CLASSES)
        res["birads_5class_" + key] = b
        print("  [%-12s] acc=%s bACC=%s macroF1=%s (全类别 %s) wkappa=%s"
              % (key, fmt(b["acc"]), fmt(b["bacc"]), fmt(b["macro_f1"]),
                 fmt(b["macro_f1_all_classes"]), fmt(b["wkappa"])))
        for c in C.BIRADS_CLASSES:
            pc = b["per_class"][str(c)]
            print("      %-2s P=%s R=%s F1=%s (n=%d)"
                  % (c, fmt(pc["precision"]), fmt(pc["recall"]), fmt(pc["f1"]), pc["support"]))
    b = res["birads_5class_probs_argmax"]
    cm = b["confusion_matrix"]
    print("  混淆矩阵 (行=GT, 列=预测):")
    print("        " + "".join("%8s" % ("p" + str(c)) for c in C.BIRADS_CLASSES) + "%8s" % "|rec")
    for t in C.BIRADS_CLASSES:
        row = cm[str(t)]
        print("    %-4s" % t + "".join("%8d" % row[str(p)] for p in C.BIRADS_CLASSES) + "%8d" % sum(row.values()))

    auc, n_auc = ovr_auc_avg(birads_gt(recs), birads_probs(recs), C.BIRADS_CLASSES)
    ece = C.top_label_ece(birads_gt(recs), birads_probs(recs), C.BIRADS_CLASSES)
    res["birads_ovr_macro_auc"] = auc
    res["birads_ovr_macro_auc_n_classes"] = n_auc
    res["birads_top_label_ece"] = ece
    res["ci_birads_ovr_macro_auc"] = list(boot(
        recs, lambda rs: C.ovr_macro_auc(birads_gt(rs), birads_probs(rs), C.BIRADS_CLASSES), args.n_boot))
    res["ci_birads_macro_f1"] = list(boot(
        recs, lambda rs: C.multiclass_metrics(birads_gt(rs), birads_argmax(rs), C.BIRADS_CLASSES)["macro_f1"],
        args.n_boot))
    print("  OvR macroAUC %s | topLabelECE %s" % (fmt_ci(auc, res["ci_birads_ovr_macro_auc"]), fmt(ece)))

    for lvl in (3, 4):
        y, s = bin_labels(recs, lvl), sum_score(recs, lvl)
        a, pa = C.auc(y, s), C.pr_auc(y, s)
        key = "birads_ge%d" % lvl
        res[key] = {"auc": a, "pr_auc": pa, "n_pos": sum(y), "score": "sum_{i>=%d} P(i)" % lvl}
        res["ci_" + key + "_auc"] = list(boot(
            recs, lambda rs, l=lvl: C.auc(bin_labels(rs, l), sum_score(rs, l)), args.n_boot))
        res["ci_" + key + "_pr_auc"] = list(boot(
            recs, lambda rs, l=lvl: C.pr_auc(bin_labels(rs, l), sum_score(rs, l)), args.n_boot))
        print("  [>=%d] AUC=%s  PR-AUC=%s  (n_pos=%d)"
              % (lvl, fmt_ci(a, res["ci_" + key + "_auc"]), fmt_ci(pa, res["ci_" + key + "_pr_auc"]), sum(y)))

    for mode, desc in [("A", "P4+P5"), ("B", "(P4+P5)/(P2+P4+P5)")]:
        y, s = v45_pairs(recs, mode)
        a = C.auc(y, s)
        key = "birads_2v45_auc_" + mode.lower()
        res[key] = {"auc": a, "score": desc, "n": len(y), "n_pos": sum(y)}
        res["ci_" + key] = list(boot(recs, lambda rs, m=mode: C.auc(*v45_pairs(rs, m)), args.n_boot))
        print("  [2 vs 4/5 %s] AUC=%s  (n=%d, n_pos=%d)"
              % (mode, fmt_ci(a, res["ci_" + key]), len(y), sum(y)))

    # ---------- Density ----------
    print("\n== Density ==")
    for key, merged, cl in [("4class", False, C.DENSITY_CLASSES),
                            ("merged_ab", True, C.DENSITY_MERGED_CLASSES)]:
        g = mc_block(recs, lambda rs, m=merged: density_gt(rs, m),
                     lambda rs, m=merged: density_argmax(rs, m), cl)
        res["density_" + key] = g
        print("  [%-9s] acc=%s bACC=%s macroF1=%s (全类别 %s) wkappa=%s"
              % (key, fmt(g["acc"]), fmt(g["bacc"]), fmt(g["macro_f1"]),
                 fmt(g["macro_f1_all_classes"]), fmt(g["wkappa"])))
        for c in cl:
            pc = g["per_class"][str(c)]
            print("      %-3s P=%s R=%s F1=%s (n=%d)"
                  % (c, fmt(pc["precision"]), fmt(pc["recall"]), fmt(pc["f1"]), pc["support"]))
        cmr = g["confusion_matrix"]
        print("      混淆矩阵 (行=GT, 列=预测):")
        print("        " + "".join("%8s" % ("p" + str(c)) for c in cl) + "%8s" % "|rec")
        for t in cl:
            row = cmr[str(t)]
            print("    %-4s" % t + "".join("%8d" % row[str(p)] for p in cl) + "%8d" % sum(row.values()))
        a, n_a = ovr_auc_avg(density_gt(recs, merged), density_probs(recs, merged), cl)
        e = C.top_label_ece(density_gt(recs, merged), density_probs(recs, merged), cl)
        res["density_" + key]["ovr_macro_auc"] = a
        res["density_" + key]["ovr_macro_auc_n_classes"] = n_a
        res["density_" + key]["top_label_ece"] = e
        res["ci_density_%s_ovr_macro_auc" % key] = list(boot(
            recs, lambda rs, m=merged, c=cl: C.ovr_macro_auc(density_gt(rs, m), density_probs(rs, m), c),
            args.n_boot))
        print("      OvR macroAUC %s | topLabelECE %s"
              % (fmt_ci(a, res["ci_density_%s_ovr_macro_auc" % key]), fmt(e)))

    # ---------- findings ----------
    print("\n== findings (4 个二分类) ==")
    print("  %-26s %6s %6s %8s %8s %8s %8s %8s" % ("finding", "n_pos", "pos%", "AUROC", "PR-AUC", "P@0.5", "R@0.5", "F1@0.5"))
    per = {}
    f1s = []
    for name in C.FINDINGS:
        y, s = find_y(recs, name), find_p(recs, name)
        m = C.binary_metrics(y, [1 if p >= FIXED_T else 0 for p in s])
        entry = {
            "n_pos": sum(y), "n_neg": len(y) - sum(y),
            "auroc": C.auc(y, s), "pr_auc": C.pr_auc(y, s), "ece": C.ece(y, s),
            "at_0.5": {"precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
                       "pred_pos_rate": (m["tp"] + m["fp"]) / m["n"] if m["n"] else 0.0,
                       "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tn": m["tn"]},
        }
        res["ci_findings_%s_auroc" % name] = list(boot(
            recs, lambda rs, nm=name: C.auc(find_y(rs, nm), find_p(rs, nm)), args.n_boot))
        res["ci_findings_%s_pr_auc" % name] = list(boot(
            recs, lambda rs, nm=name: C.pr_auc(find_y(rs, nm), find_p(rs, nm)), args.n_boot))
        per[name] = entry
        f1s.append(entry["at_0.5"]["f1"])
        print("  %-26s %6d %6.4f %8s %8s %8.4f %8.4f %8.4f"
              % (name, entry["n_pos"], entry["n_pos"] / len(recs),
                 fmt_ci(entry["auroc"], res["ci_findings_%s_auroc" % name]),
                 fmt_ci(entry["pr_auc"], res["ci_findings_%s_pr_auc" % name]),
                 entry["at_0.5"]["precision"], entry["at_0.5"]["recall"], entry["at_0.5"]["f1"]))
    res["findings"] = per
    res["findings_macro_f1_at_0.5"] = sum(f1s) / len(f1s)
    _aucs = [per[n]["auroc"] for n in C.FINDINGS if per[n]["auroc"] is not None]
    res["findings_macro_auroc"] = sum(_aucs) / len(_aucs) if _aucs else None
    print("  macro-F1@0.5=%.4f   macro-AUROC=%s"
          % (res["findings_macro_f1_at_0.5"], fmt(res["findings_macro_auroc"])))

    y = [1 if any(r["gt_findings"][k] for k in C.FINDINGS) else 0 for r in recs]
    yhat = [1 if any(r["p_findings"][k] >= FIXED_T for k in C.FINDINGS) else 0 for r in recs]
    m = C.binary_metrics(y, yhat)
    res["any_finding_at_0.5"] = {"auroc": C.auc(y, [max(r["p_findings"][k] for k in C.FINDINGS) for r in recs]),
                                 "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
                                 "n_pos": m["n_pos"], "pred_pos_rate": (m["tp"] + m["fp"]) / m["n"]}
    print("  any-finding@0.5: P=%.4f R=%.4f F1=%.4f (n_pos=%d, 阳性率=%.4f)"
          % (m["precision"], m["recall"], m["f1"], m["n_pos"], res["any_finding_at_0.5"]["pred_pos_rate"]))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print("\n[out] %s  (%.1fs, bootstrap x%d)" % (args.out, time.time() - t0, args.n_boot))


if __name__ == "__main__":
    main()
