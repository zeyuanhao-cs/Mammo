#!/usr/bin/env python3
"""四张表（纯文本 CSV）: VLM b8 (unfrozen 1ep / 3ep) + EfficientNet-B2。

表1 整体指标；表2 每小类；表3 AUC 配对 bootstrap 显著性（3ep vs 1ep / B2）；表4 病灶检出计数。
"""
import random
import sys
from collections import defaultdict
from pathlib import Path

THIS = Path(__file__).resolve().parent
PROJ = THIS.parent.parent
sys.path.insert(0, str(PROJ / "scripts" / "eval"))
import common as C  # noqa: E402

MODELS = [
    ("qwen_unfrozen_1ep", PROJ / "outputs" / "scoring" / "os_sub_vt_test_b8_with_probs.jsonl"),
    ("qwen_unfrozen_3ep", PROJ / "outputs" / "scoring" / "vt-ep3-e3_test_b8_with_probs.jsonl"),
    ("efficientnet_b2", PROJ / "outputs" / "vision_baseline" / "repeat_os" / "test_probs.jsonl"),
]
TARGET = "qwen_unfrozen_3ep"
OTHERS = ["qwen_unfrozen_1ep", "efficientnet_b2"]
N_BOOT, SEED, T = 1000, 42, 0.5


# ---------------- 指标 ----------------
def mc_block(recs, gt_key, pred_key, classes):
    y = [r[gt_key] for r in recs]
    yh = [r[pred_key] for r in recs]
    m = C.multiclass_metrics(y, yh, classes)
    present = [c for c in classes if m["per_class"][c]["support"] > 0]
    mp = sum(m["per_class"][c]["precision"] for c in present) / len(present)
    mr = sum(m["per_class"][c]["recall"] for c in present) / len(present)
    mf = sum(m["per_class"][c]["f1"] for c in present) / len(present)
    return {"precision": mp, "recall": mr, "f1": mf, "bacc": m["bacc"]}


def birads_probs(recs):
    return [{i: r["p_birads"][str(i)] for i in C.BIRADS_CLASSES} for r in recs]


def find_rates(recs):
    ps, rs_, fs, aucs, baccs = [], [], [], [], []
    for k in C.FINDINGS:
        y = [r["gt_findings"][k] for r in recs]
        s = [r["p_findings"][k] for r in recs]
        m = C.binary_metrics(y, [1 if v >= T else 0 for v in s])
        ps.append(m["precision"])
        rs_.append(m["recall"])
        fs.append(m["f1"])
        a = C.auc(y, s)
        if a is not None:
            aucs.append(a)
        baccs.append((m["recall"] + m["specificity"]) / 2)
    n = len(C.FINDINGS)
    return {"precision": sum(ps) / n, "recall": sum(rs_) / n, "f1": sum(fs) / n,
            "auc": sum(aucs) / len(aucs) if aucs else None, "bacc": sum(baccs) / n}


def overall(recs):
    b = mc_block(recs, "gt_birads", "greedy_birads", C.BIRADS_CLASSES)
    b["auc"] = C.ovr_macro_auc([r["gt_birads"] for r in recs], birads_probs(recs), C.BIRADS_CLASSES)
    d = mc_block(recs, "gt_density", "greedy_density", C.DENSITY_CLASSES)
    d["auc"] = C.ovr_macro_auc([r["gt_density"] for r in recs],
                               [r["p_density"] for r in recs], C.DENSITY_CLASSES)
    return {"BI-RADS": b, "Density": d, "Findings_mean": find_rates(recs)}


def per_class(recs):
    out = {}
    y_b = [r["gt_birads"] for r in recs]
    yh_b = [r["greedy_birads"] for r in recs]
    pb = birads_probs(recs)
    for c in C.BIRADS_CLASSES:
        yc = [1 if a == c else 0 for a in y_b]
        s = [p[c] for p in pb]
        tp = sum(1 for a, b in zip(y_b, yh_b) if a == c and b == c)
        fp = sum(1 for a, b in zip(y_b, yh_b) if a != c and b == c)
        fn = sum(1 for a, b in zip(y_b, yh_b) if a == c and b != c)
        p_, r_, f_ = C.prf(tp, fp, fn)
        out[f"BI-RADS:{c}"] = {"support": tp + fn, "n_pred": tp + fp, "TP": tp, "FP": fp, "FN": fn,
                               "TN": None, "precision": p_, "recall": r_, "f1": f_,
                               "auc": C.auc(yc, s), "bacc": None}
    y_d = [r["gt_density"] for r in recs]
    yh_d = [r["greedy_density"] for r in recs]
    for c in C.DENSITY_CLASSES:
        yc = [1 if a == c else 0 for a in y_d]
        s = [r["p_density"][c] for r in recs]
        tp = sum(1 for a, b in zip(y_d, yh_d) if a == c and b == c)
        fp = sum(1 for a, b in zip(y_d, yh_d) if a != c and b == c)
        fn = sum(1 for a, b in zip(y_d, yh_d) if a == c and b != c)
        p_, r_, f_ = C.prf(tp, fp, fn)
        out[f"Density:{c}"] = {"support": tp + fn, "n_pred": tp + fp, "TP": tp, "FP": fp, "FN": fn,
                               "TN": None, "precision": p_, "recall": r_, "f1": f_,
                               "auc": C.auc(yc, s), "bacc": None}
    for k in C.FINDINGS:
        y = [r["gt_findings"][k] for r in recs]
        s = [r["p_findings"][k] for r in recs]
        m = C.binary_metrics(y, [1 if v >= T else 0 for v in s])
        out[f"Findings:{k}"] = {"support": m["n_pos"], "n_pred": m["tp"] + m["fp"], "TP": m["tp"],
                                "FP": m["fp"], "FN": m["fn"], "TN": m["tn"],
                                "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
                                "auc": C.auc(y, s),
                                "bacc": (m["recall"] + m["specificity"]) / 2}
    return out


def lesion_detect(recs):
    gt = sum(1 for r in recs if any(r["gt_findings"][k] for k in C.FINDINGS))
    det = sum(1 for r in recs if any(r["gt_findings"][k] for k in C.FINDINGS)
              and any(r["greedy_findings"].get(k, 0) == 1 for k in C.FINDINGS))
    return gt, det


# ---------------- 显著性用的指标函数 ----------------
AUC_FNS = {
    "BI-RADS:macro": lambda rs: C.ovr_macro_auc([r["gt_birads"] for r in rs], birads_probs(rs), C.BIRADS_CLASSES),
    "Density:macro": lambda rs: C.ovr_macro_auc([r["gt_density"] for r in rs],
                                                [r["p_density"] for r in rs], C.DENSITY_CLASSES),
    "Findings:mean": lambda rs: find_rates(rs)["auc"],
}
for _c in C.BIRADS_CLASSES:
    AUC_FNS[f"BI-RADS:{_c}"] = (lambda c: lambda rs: C.auc(
        [1 if r["gt_birads"] == c else 0 for r in rs], [r["p_birads"][str(c)] for r in rs]))(_c)
for _c in C.DENSITY_CLASSES:
    AUC_FNS[f"Density:{_c}"] = (lambda c: lambda rs: C.auc(
        [1 if r["gt_density"] == c else 0 for r in rs], [r["p_density"][c] for r in rs]))(_c)
for _k in C.FINDINGS:
    AUC_FNS[f"Findings:{_k}"] = (lambda k: lambda rs: C.auc(
        [r["gt_findings"][k] for r in rs], [r["p_findings"][k] for r in rs]))(_k)


def paired_boot(recs_a, recs_b, keys, n_boot=N_BOOT, seed=SEED):
    A = {r["id"]: r for r in recs_a}
    B = {r["id"]: r for r in recs_b}
    ids = [i for i in A if i in B]
    groups = defaultdict(list)
    for i in ids:
        groups[A[i]["study"]].append(i)
    studies = sorted(groups)
    acc = {k: [] for k in keys}
    rng = random.Random(seed)
    for _ in range(n_boot):
        sid = []
        for _ in range(len(studies)):
            sid.extend(groups[studies[rng.randrange(len(studies))]])
        ra = [A[i] for i in sid]
        rb = [B[i] for i in sid]
        for k in keys:
            fn = AUC_FNS[k]
            va, vb = fn(ra), fn(rb)
            if va is not None and vb is not None:
                acc[k].append(vb - va)
    out = {}
    for k, s in acc.items():
        if len(s) < 100:
            out[k] = (None, None)
            continue
        s.sort()
        out[k] = (s[int(0.025 * len(s))], s[min(int(0.975 * len(s)), len(s) - 1)])
    return out


def f4(v):
    return "" if v is None else f"{v:.4f}"


def main():
    recs = {}
    for name, path in MODELS:
        recs[name] = C.load_probs_jsonl(str(path))
        print(f"# loaded {name}: n={len(recs[name])}", file=sys.stderr)

    # ---- 表 1 ----
    print("== TABLE1 ==")
    print("model,field,precision,recall,F1,AUC,bACC")
    for name, _ in MODELS:
        ov = overall(recs[name])
        for f in ("BI-RADS", "Density", "Findings_mean"):
            v = ov[f]
            print(f"{name},{f},{f4(v['precision'])},{f4(v['recall'])},{f4(v['f1'])},{f4(v['auc'])},{f4(v['bacc'])}")

    # ---- 表 2 ----
    print("\n== TABLE2 ==")
    print("model,field,class,support,n_pred,TP,FP,FN,TN,precision,recall,F1,AUC_ovr,bACC")
    for name, _ in MODELS:
        pc = per_class(recs[name])
        for key, v in pc.items():
            field, cls = key.split(":", 1)
            print(f"{name},{field},{cls},{v['support']},{v['n_pred']},{v['TP']},{v['FP']},{v['FN']},"
                  f"{'' if v['TN'] is None else v['TN']},{f4(v['precision'])},{f4(v['recall'])},"
                  f"{f4(v['f1'])},{f4(v['auc'])},{f4(v['bacc'])}")

    # ---- 表 3 ----
    print("\n== TABLE3 ==")
    print("model_vs,field,class,delta_AUC,CI_low,CI_high")
    keys = list(AUC_FNS.keys())
    for other in OTHERS:
        ci = paired_boot(recs[other], recs[TARGET], keys)
        for k in keys:
            field, cls = k.split(":", 1)
            va, vb = AUC_FNS[k](recs[TARGET]), AUC_FNS[k](recs[other])
            if va is None or vb is None:
                continue
            lo, hi = ci[k]
            d = va - vb
            star = " *" if (lo is not None and (lo > 0 or hi < 0)) else ""
            print(f"{TARGET} vs {other},{field},{cls},{d:+.4f}{star},{f4(lo)},{f4(hi)}")

    # ---- 表 4 ----
    print("\n== TABLE4 ==")
    print("model,lesion_gt,detected")
    for name, _ in MODELS:
        gt, det = lesion_detect(recs[name])
        print(f"{name},{gt},{det}")


if __name__ == "__main__":
    main()
