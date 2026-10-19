#!/usr/bin/env python3
"""对 infer_2view.py / infer_2view_local.py 的贪心预测 jsonl 做整体统计。

输入 jsonl 每行 (与 infer_2view_local.py 输出一致):
  {id, index, prediction_raw, prediction_json, ground_truth, seconds, [error]}

输出:
  - 控制台可读摘要
  - --out 指向的 json: accuracy / balanced-acc / macro-F1 / 每类 P/R/F1 /
    混淆矩阵 / 二次加权 kappa / 二分类 sens-spec / 预测阳性率 / 一致性违规数

纯标准库 (可用 /usr/bin/python3 直接跑); 指标函数复用同目录 common.py。
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

BIRADS = C.BIRADS_CLASSES                 # 1..5
DENSITY = C.DENSITY_CLASSES               # A B C D
DENSITY_MERGED = C.DENSITY_MERGED_CLASSES  # AB C D
FINDINGS = C.FINDINGS


# ------------------------------ 解析 ------------------------------

def _as_birads(v):
    try:
        i = int(v)
    except (TypeError, ValueError):
        return None
    return i if i in BIRADS else None


def _as_density(v):
    s = str(v).strip().upper() if v is not None else ""
    return s if s in DENSITY else None


def _as_finding(d, name):
    if not isinstance(d, dict) or name not in d:
        return None
    try:
        return 1 if int(d[name]) else 0
    except (TypeError, ValueError):
        return None


def load_predictions(path):
    """读预测 jsonl, 返回 (records, meta)."""
    recs = []
    n_total = n_parse_fail = n_error = 0
    secs = []
    seen = set()
    dup = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            n_total += 1
            rid = d.get("id")
            if rid in seen:
                dup += 1
            seen.add(rid)

            if d.get("error"):
                n_error += 1
            pj = d.get("prediction_json")
            if pj is None:
                n_parse_fail += 1
                pj = {}
            if isinstance(d.get("seconds"), (int, float)):
                secs.append(d["seconds"])

            gt = d.get("ground_truth") or {}
            gtf = gt.get("findings") or {}
            recs.append({
                "id": rid,
                "study": C.study_id_of(rid),
                "gt_birads": _as_birads(gt.get("breast_birads")),
                "gt_density": _as_density(gt.get("breast_density")),
                "gt_findings": {k: _as_finding(gtf, k) for k in FINDINGS},
                "pr_birads": _as_birads(pj.get("breast_birads")),
                "pr_density": _as_density(pj.get("breast_density")),
                "pr_findings": {k: _as_finding(pj.get("findings"), k) for k in FINDINGS},
            })

    meta = {
        "file": str(path),
        "n_lines": n_total,
        "n_dup_ids": dup,
        "n_parse_fail": n_parse_fail,
        "n_error": n_error,
        "sec_total": round(sum(secs), 1),
        "sec_mean": round(sum(secs) / len(secs), 2) if secs else None,
    }
    return recs, meta


# ------------------------------ 指标 ------------------------------

def multiclass_block(recs, gt_key, pr_key, classes, merge_ab=False):
    """多分类: acc / bACC / macro-F1 / 每类 P-R-F1 / 混淆矩阵 / wkappa."""
    y, yhat = [], []
    for r in recs:
        g, p = r[gt_key], r[pr_key]
        if g is None or p is None:
            continue
        if merge_ab:
            g = "AB" if g in ("A", "B") else g
            p = "AB" if p in ("A", "B") else p
        y.append(g)
        yhat.append(p)

    m = C.multiclass_metrics(y, yhat, classes)
    cm = C.confusion(y, yhat, classes)
    return {
        "n": len(y),
        "n_covered": len(y),
        "acc": m["acc"],
        "bacc": m["bacc"],
        "macro_f1": m["macro_f1"],
        "wkappa": C.quadratic_weighted_kappa(y, yhat, classes),
        "per_class": {str(c): m["per_class"][c] for c in classes},
        "confusion_matrix": {str(t): {str(p): cm[t][p] for p in classes} for t in classes},
    }


def binary_block(y, yhat, name="", extra=None):
    m = C.binary_metrics(y, yhat)
    out = {
        "n": m["n"], "n_pos": m["n_pos"], "n_neg": m["n"] - m["n_pos"],
        "accuracy": m["acc"], "sensitivity": m["recall"], "specificity": m["specificity"],
        "precision": m["precision"], "f1": m["f1"],
        "pred_pos_rate": (m["tp"] + m["fp"]) / m["n"] if m["n"] else 0.0,
        "confusion_matrix": {"tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tn": m["tn"]},
    }
    if extra:
        out.update(extra)
    return out


def birads_block(recs):
    res = {"multiclass": multiclass_block(recs, "gt_birads", "pr_birads", BIRADS)}
    for lvl in (3, 4):
        y, yhat = [], []
        for r in recs:
            if r["gt_birads"] is None or r["pr_birads"] is None:
                continue
            y.append(1 if r["gt_birads"] >= lvl else 0)
            yhat.append(1 if r["pr_birads"] >= lvl else 0)
        res["ge%d" % lvl] = binary_block(y, yhat, extra={"level": lvl})
    return res


def density_block(recs):
    return {
        "four_class": multiclass_block(recs, "gt_density", "pr_density", DENSITY),
        "merged_ab": multiclass_block(recs, "gt_density", "pr_density", DENSITY_MERGED, merge_ab=True),
    }


def findings_block(recs):
    per = {}
    f1s, accs = [], []
    for k in FINDINGS:
        y, yhat = [], []
        for r in recs:
            g, p = r["gt_findings"][k], r["pr_findings"][k]
            if g is None or p is None:
                continue
            y.append(g)
            yhat.append(p)
        b = binary_block(y, yhat)
        per[k] = b
        f1s.append(b["f1"])
        accs.append(b["accuracy"])

    # any-finding (4 个发现里任一为 1)
    y, yhat = [], []
    for r in recs:
        g = [r["gt_findings"][k] for k in FINDINGS]
        p = [r["pr_findings"][k] for k in FINDINGS]
        if any(v is None for v in g + p):
            continue
        y.append(1 if any(g) else 0)
        yhat.append(1 if any(p) else 0)
    any_f = binary_block(y, yhat)

    return {
        "per_finding": per,
        "macro_f1": sum(f1s) / len(f1s),
        "macro_accuracy": sum(accs) / len(accs),
        "any_finding": any_f,
    }


def consistency_block(recs):
    """prompt 里的一致性约束: findings 全 0 -> birads<=2; 有 finding -> birads>=3."""
    n_gt_viol = n_pred_viol = n_cov = 0
    pred_examples = []
    for r in recs:
        gb, pb = r["gt_birads"], r["pr_birads"]
        gf = [r["gt_findings"][k] for k in FINDINGS]
        pf = [r["pr_findings"][k] for k in FINDINGS]
        if gb is not None and all(v is not None for v in gf):
            n_cov += 1
            if any(gf) != (gb >= 3):
                n_gt_viol += 1
        if pb is not None and all(v is not None for v in pf):
            if any(pf) != (pb >= 3):
                n_pred_viol += 1
                if len(pred_examples) < 5:
                    pred_examples.append({"id": r["id"], "birads": pb,
                                          "findings": {k: r["pr_findings"][k] for k in FINDINGS}})
    return {
        "rule": "any(finding)==1 <=> birads>=3",
        "n_covered": n_cov,
        "gt_violations": n_gt_viol,
        "pred_violations": n_pred_viol,
        "pred_violation_rate": n_pred_viol / n_cov if n_cov else 0.0,
        "pred_examples": pred_examples,
    }


# ------------------------------ 打印 ------------------------------

def pct(x):
    return "n/a" if x is None else "%.4f" % x


def cm_lines(title, cm, classes):
    out = ["  %s" % title,
           "       " + "".join("%9s" % ("pred " + str(c)) for c in classes) + "%9s" % "| rec"]
    for t in classes:
        row = cm[str(t)]
        rec = sum(row.values())
        out.append("    %4s" % t + "".join("%9d" % row[str(p)] for p in classes) + "%9d" % rec)
    col = [sum(cm[str(t)][str(p)] for t in classes) for p in classes]
    out.append("    %4s" % "sup" + "".join("%9d" % c for c in col) + "%9d" % sum(col))
    return "\n".join(out)


def print_report(res, meta):
    print("=" * 78)
    print("预测文件: %s" % meta["file"])
    print("总行数=%d  解析失败=%d  异常=%d  重复id=%d  平均耗时=%ss/条  总耗时=%.1f min"
          % (meta["n_lines"], meta["n_parse_fail"], meta["n_error"], meta["n_dup_ids"],
             meta["sec_mean"], meta["sec_total"] / 60))
    print("=" * 78)

    b = res["birads"]["multiclass"]
    print("\n== BI-RADS 5 类 (n=%d) ==" % b["n"])
    print("  acc=%s  bACC=%s  macroF1=%s  wkappa=%s"
          % (pct(b["acc"]), pct(b["bacc"]), pct(b["macro_f1"]), pct(b["wkappa"])))
    for c in BIRADS:
        pc = b["per_class"][str(c)]
        print("    %-3s P=%s R=%s F1=%s (support=%d)"
              % (c, pct(pc["precision"]), pct(pc["recall"]), pct(pc["f1"]), pc["support"]))
    print(cm_lines("confusion matrix (行=GT, 列=预测):", b["confusion_matrix"], BIRADS))
    for lvl in (3, 4):
        x = res["birads"]["ge%d" % lvl]
        print("  [>=%d] acc=%s  sens=%s spec=%s  P=%s F1=%s  阳性率 pred=%s (GT n_pos=%d, n=%d)"
              % (lvl, pct(x["accuracy"]), pct(x["sensitivity"]), pct(x["specificity"]),
                 pct(x["precision"]), pct(x["f1"]), pct(x["pred_pos_rate"]), x["n_pos"], x["n"]))
        print("        TP=%d FP=%d FN=%d TN=%d"
              % (x["confusion_matrix"]["tp"], x["confusion_matrix"]["fp"],
                 x["confusion_matrix"]["fn"], x["confusion_matrix"]["tn"]))

    for key, title in (("four_class", "Density 4 类 A/B/C/D"), ("merged_ab", "Density 合并 3 类 AB/C/D")):
        d = res["density"][key]
        cl = DENSITY if key == "four_class" else DENSITY_MERGED
        print("\n== %s (n=%d) ==" % (title, d["n"]))
        print("  acc=%s  bACC=%s  macroF1=%s  wkappa=%s"
              % (pct(d["acc"]), pct(d["bacc"]), pct(d["macro_f1"]), pct(d["wkappa"])))
        for c in cl:
            pc = d["per_class"][str(c)]
            print("    %-3s P=%s R=%s F1=%s (support=%d)"
                  % (c, pct(pc["precision"]), pct(pc["recall"]), pct(pc["f1"]), pc["support"]))
        print(cm_lines("confusion matrix (行=GT, 列=预测):", d["confusion_matrix"], cl))

    f = res["findings"]
    print("\n== Findings 4 个二分类 ==")
    print("  %-26s %6s %6s %8s %8s %8s %8s %8s" % ("finding", "n_pos", "acc", "sens", "spec", "P", "F1", "阳性率"))
    for k in FINDINGS:
        x = f["per_finding"][k]
        print("  %-26s %6d %6s %8s %8s %8s %8s %8s"
              % (k, x["n_pos"], pct(x["accuracy"]), pct(x["sensitivity"]), pct(x["specificity"]),
                 pct(x["precision"]), pct(x["f1"]), pct(x["pred_pos_rate"])))
        cm = x["confusion_matrix"]
        print("      混淆矩阵: TP=%d FP=%d FN=%d TN=%d" % (cm["tp"], cm["fp"], cm["fn"], cm["tn"]))
    print("  macro-F1=%s  macro-accuracy=%s" % (pct(f["macro_f1"]), pct(f["macro_accuracy"])))
    a = f["any_finding"]
    print("  any-finding: acc=%s sens=%s spec=%s P=%s F1=%s 阳性率=%s (n_pos=%d)"
          % (pct(a["accuracy"]), pct(a["sensitivity"]), pct(a["specificity"]),
             pct(a["precision"]), pct(a["f1"]), pct(a["pred_pos_rate"]), a["n_pos"]))

    c = res["consistency"]
    print("\n== 一致性约束 (%s) ==" % c["rule"])
    print("  GT 违规=%d / %d   预测违规=%d / %d (%s)"
          % (c["gt_violations"], c["n_covered"], c["pred_violations"], c["n_covered"],
             pct(c["pred_violation_rate"])))


def main():
    ap = argparse.ArgumentParser(description="贪心预测 jsonl 整体统计")
    ap.add_argument("--pred", required=True, help="predictions_*.jsonl")
    ap.add_argument("--out", default="", help="指标 json 输出路径")
    ap.add_argument("--tag", default="", help="结果里的标识")
    args = ap.parse_args()

    recs, meta = load_predictions(args.pred)
    res = {
        "tag": args.tag,
        "meta": meta,
        "birads": birads_block(recs),
        "density": density_block(recs),
        "findings": findings_block(recs),
        "consistency": consistency_block(recs),
    }
    print_report(res, meta)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print("\n[out] %s" % args.out)


if __name__ == "__main__":
    main()
