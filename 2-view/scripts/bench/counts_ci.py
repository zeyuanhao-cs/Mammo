#!/usr/bin/env python3
"""os_sub_vt: 原流水线 vs batch=8
 1) mass / calcification 在 0.5 概率阈值下的 TP/FP/FN/TN
 2) BI-RADS 贪心输出每类 TP/FP/FN
 3) 原流水线这些 F1 的 study 级 bootstrap 95% CI 与宽度（1000 次, seed=42）
"""
import random
import sys
from collections import defaultdict
from pathlib import Path

THIS = Path(__file__).resolve().parent
PROJ = THIS.parent.parent
sys.path.insert(0, str(PROJ / "scripts" / "eval"))
import common as C  # noqa: E402

OLD = PROJ / "outputs" / "scoring" / "os_sub_vt_test_with_probs.jsonl"
NEW = PROJ / "outputs" / "scoring" / "os_sub_vt_test_b8_with_probs.jsonl"
N_BOOT, SEED, T = 1000, 42, 0.5


def bin_counts(recs, name):
    tp = fp = fn = tn = 0
    for r in recs:
        y = r["gt_findings"][name]
        yhat = 1 if r["p_findings"][name] >= T else 0
        if y == 1 and yhat == 1:
            tp += 1
        elif y == 0 and yhat == 1:
            fp += 1
        elif y == 1 and yhat == 0:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def mc_counts(recs, cls):
    """贪心输出下某一 BI-RADS 类的 TP/FP/FN."""
    tp = fp = fn = 0
    for r in recs:
        y, yhat = r["gt_birads"], r["greedy_birads"]
        if y == cls and yhat == cls:
            tp += 1
        elif y != cls and yhat == cls:
            fp += 1
        elif y == cls and yhat != cls:
            fn += 1
    return tp, fp, fn


def boot_stats(recs, n_boot=N_BOOT, seed=SEED):
    """一次重采样同时算: mass F1, calc F1, birads 每类 F1, birads macro-F1."""
    groups = defaultdict(list)
    for r in recs:
        groups[r["study"]].append(r)
    studies = sorted(groups)
    rng = random.Random(seed)

    keys = ["mass", "calc"] + [f"birads_{c}" for c in C.BIRADS_CLASSES] + ["birads_macro_f1"]
    stats = {k: [] for k in keys}

    for _ in range(n_boot):
        sample = []
        for _ in range(len(studies)):
            sample.extend(groups[studies[rng.randrange(len(studies))]])
        for name, k in (("mass", "mass"), ("suspicious_calcification", "calc")):
            tp, fp, fn, _ = bin_counts(sample, name)
            _, _, f = C.prf(tp, fp, fn)
            stats[k].append(f)
        f1s = []
        for c in C.BIRADS_CLASSES:
            tp, fp, fn = mc_counts(sample, c)
            _, _, f = C.prf(tp, fp, fn)
            stats[f"birads_{c}"].append(f)
            f1s.append(f)
        stats["birads_macro_f1"].append(sum(f1s) / len(f1s))

    out = {}
    for k, s in stats.items():
        s.sort()
        lo = s[int(0.025 * len(s))]
        hi = s[min(int(0.975 * len(s)), len(s) - 1)]
        out[k] = (lo, hi)
    return out


def main():
    old = C.load_probs_jsonl(str(OLD))
    new = C.load_probs_jsonl(str(NEW))
    print(f"原流水线 n={len(old)}   batch=8 n={len(new)}")

    print("\n================ 1) findings @0.5 (概率阈值) ================")
    print(f"{'模型':<14}{'finding':<26}{'TP':>6}{'FP':>6}{'FN':>6}{'TN':>6}{'F1':>9}{'P':>9}{'R':>9}")
    for tag, recs in (("原流水线", old), ("batch=8", new)):
        for name, lab in (("suspicious_calcification", "calcification"), ("mass", "mass")):
            tp, fp, fn, tn = bin_counts(recs, name)
            p, r, f = C.prf(tp, fp, fn)
            print(f"{tag:<14}{lab:<26}{tp:>6}{fp:>6}{fn:>6}{tn:>6}{f:>9.4f}{p:>9.4f}{r:>9.4f}")

    print("\n================ 2) BI-RADS 每类 TP/FP/FN（贪心输出） ================")
    print(f"{'模型':<14}{'class':<8}{'TP':>6}{'FP':>6}{'FN':>6}{'F1':>9}{'P':>9}{'R':>9}")
    for tag, recs in (("原流水线", old), ("batch=8", new)):
        for c in C.BIRADS_CLASSES:
            tp, fp, fn = mc_counts(recs, c)
            p, r, f = C.prf(tp, fp, fn)
            print(f"{tag:<14}{c:<8}{tp:>6}{fp:>6}{fn:>6}{f:>9.4f}{p:>9.4f}{r:>9.4f}")

    print(f"\n================ 3) 原流水线 F1 的 study 级 bootstrap CI "
          f"({N_BOOT} 次, seed={SEED}) ================")
    ci = boot_stats(old)
    point = {}
    for name, k in (("mass", "mass"), ("suspicious_calcification", "calc")):
        tp, fp, fn, _ = bin_counts(old, name)
        point[k] = C.prf(tp, fp, fn)[2]
    f1s = []
    for c in C.BIRADS_CLASSES:
        tp, fp, fn = mc_counts(old, c)
        f = C.prf(tp, fp, fn)[2]
        point[f"birads_{c}"] = f
        f1s.append(f)
    point["birads_macro_f1"] = sum(f1s) / len(f1s)

    print(f"{'metric':<24}{'F1':>9}{'95% CI':>20}{'CI 宽度':>10}")
    for k in ["mass", "calc"] + [f"birads_{c}" for c in C.BIRADS_CLASSES] + ["birads_macro_f1"]:
        lo, hi = ci[k]
        print(f"{k:<24}{point[k]:>9.4f}{f'[{lo:.4f}, {hi:.4f}]':>20}{hi - lo:>10.4f}")


if __name__ == "__main__":
    main()
