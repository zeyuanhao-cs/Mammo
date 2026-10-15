#!/usr/bin/env python3
"""在 val 上拟合阈值/偏置, 输出 thresholds.json. 只读 val, 不碰 test.

- 4 findings: F1 最大阈值 + 特异度约 0.90 参考阈值; 正例 <10 unstable, <20 caution
- density: 4 类 log-bias 坐标上升最大化 bACC, 范围 [-1.5, 1.5] 步长 0.25
- density_merged: A+B 合并为 AB, 3 类同法拟合
- birads: 5 类 log-bias 同法; P(3+4+5) 的 >=3 阈值 / P(4+5) 的 >=4 阈值 (各最大化 F1)
"""
import argparse
import json
import time
from pathlib import Path

import common as C

GRID = [round(-1.5 + 0.25 * i, 2) for i in range(13)]  # [-1.5, 1.5] step 0.25
TH_GRID = [round(0.005 * i, 4) for i in range(201)]    # 阈值扫描 0..1 step 0.005


def apply_logbias(probs_list, classes, bias):
    """p'_c ∝ p_c * exp(b_c), 返回 argmax 类别列表."""
    out = []
    for pdict in probs_list:
        best_c, best_s = None, -1e18
        for c in classes:
            k = c if c in pdict else str(c)  # birads 概率 key 是字符串
            s = pdict[k] * bias[c]
            if s > best_s:
                best_s, best_c = s, c
        out.append(best_c)
    return out


def fit_logbias(recs, key, classes, obj="bacc"):
    """坐标上升拟合每类 log-bias (在 exp 域上乘性偏置)."""
    probs_list = [r[key] for r in recs]
    y = None
    if key == "p_density":
        y = [r["gt_density"] for r in recs]
    elif key == "p_birads":
        y = [r["gt_birads"] for r in recs]
    else:
        raise ValueError(key)

    def score(bias):
        pred = apply_logbias(probs_list, classes, {c: _exp(bias[c]) for c in classes})
        m = C.multiclass_metrics(y, pred, classes)
        return m["bacc"] if obj == "bacc" else m["acc"]

    bias = {c: 0.0 for c in classes}
    best = score(bias)
    for _ in range(20):  # 最多 20 轮坐标上升
        improved = False
        for c in classes:
            cur = bias[c]
            for v in GRID:
                if v == cur:
                    continue
                bias[c] = v
                s = score(bias)
                if s > best + 1e-12:
                    best, cur = s, v
                    improved = True
            bias[c] = cur
        if not improved:
            break
    return bias, best


def _exp(x):
    return pow(2.718281828459045, x)


def fit_finding_threshold(recs, name):
    y = [r["gt_findings"][name] for r in recs]
    s = [r["p_findings"][name] for r in recs]
    best_t, best_f1 = 0.5, -1.0
    for t in TH_GRID:
        yhat = [1 if p >= t else 0 for p in s]
        m = C.binary_metrics(y, yhat)
        if m["f1"] > best_f1 + 1e-12 or (abs(m["f1"] - best_f1) <= 1e-12 and t > best_t and best_f1 >= 0):
            best_f1, best_t = m["f1"], t
    # 特异度约 0.90 的参考阈值: |spec-0.9| 最小, 平手取更小阈值 (更敏感)
    ref_t, ref_d = 0.5, 1e9
    for t in TH_GRID:
        yhat = [1 if p >= t else 0 for p in s]
        m = C.binary_metrics(y, yhat)
        d = abs(m["specificity"] - 0.90)
        if d < ref_d - 1e-12 or (abs(d - ref_d) <= 1e-12 and t < ref_t):
            ref_d, ref_t = d, t
    n_pos = sum(y)
    flag = "ok"
    if n_pos < 10:
        flag = "unstable"
    elif n_pos < 20:
        flag = "caution"
    return {"t_f1": best_t, "val_f1": best_f1, "t_spec90": ref_t,
            "val_n_pos": n_pos, "flag": flag}


def fit_birads_binary(recs, level):
    """level=3: score=P(3)+P(4)+P(5), 正例 gt>=3; level=4: score=P(4)+P(5), 正例 gt>=4."""
    y = [1 if r["gt_birads"] >= level else 0 for r in recs]
    s = [sum(r["p_birads"][str(i)] for i in range(level, 6)) for r in recs]
    best_t, best_f1 = 0.5, -1.0
    for t in TH_GRID:
        yhat = [1 if p >= t else 0 for p in s]
        m = C.binary_metrics(y, yhat)
        if m["f1"] > best_f1 + 1e-12:
            best_f1, best_t = m["f1"], t
    return {"t": best_t, "val_f1": best_f1, "val_n_pos": sum(y)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True, help="val 概率 jsonl")
    ap.add_argument("--out", required=True, help="输出 thresholds.json")
    args = ap.parse_args()

    t0 = time.time()
    recs = C.load_probs_jsonl(args.val)
    print("[val] %d records from %s" % (len(recs), args.val))

    # ---- density 4 类 ----
    bias_d, bacc_d = fit_logbias(recs, "p_density", C.DENSITY_CLASSES)
    m_default = C.multiclass_metrics([r["gt_density"] for r in recs],
                                     [r["greedy_density"] for r in recs], C.DENSITY_CLASSES)
    print("[density] default bacc=%.4f -> cal bacc=%.4f  logbias=%s"
          % (m_default["bacc"], bacc_d, {c: round(bias_d[c], 2) for c in C.DENSITY_CLASSES}))

    # ---- density merged (A+B -> AB) ----
    recs_m = []
    for r in recs:
        recs_m.append({
            "p_density": {"AB": r["p_density"]["A"] + r["p_density"]["B"],
                          "C": r["p_density"]["C"], "D": r["p_density"]["D"]},
            "gt_density": "AB" if r["gt_density"] in ("A", "B") else r["gt_density"],
        })
    bias_m, bacc_m = fit_logbias(recs_m, "p_density", C.DENSITY_MERGED_CLASSES)
    print("[density_merged] cal bacc=%.4f  logbias=%s"
          % (bacc_m, {c: round(bias_m[c], 2) for c in C.DENSITY_MERGED_CLASSES}))

    # ---- birads 5 类 ----
    bias_b, bacc_b = fit_logbias(recs, "p_birads", C.BIRADS_CLASSES)
    m_default_b = C.multiclass_metrics([r["gt_birads"] for r in recs],
                                       [r["greedy_birads"] for r in recs], C.BIRADS_CLASSES)
    print("[birads] default bacc=%.4f -> cal bacc=%.4f  logbias=%s"
          % (m_default_b["bacc"], bacc_b, {i: round(bias_b[i], 2) for i in C.BIRADS_CLASSES}))

    # ---- birads 二分类阈值 ----
    ge3 = fit_birads_binary(recs, 3)
    ge4 = fit_birads_binary(recs, 4)
    print("[birads] >=3: t=%.3f val_f1=%.4f (n_pos=%d) | >=4: t=%.3f val_f1=%.4f (n_pos=%d)"
          % (ge3["t"], ge3["val_f1"], ge3["val_n_pos"], ge4["t"], ge4["val_f1"], ge4["val_n_pos"]))

    # ---- findings ----
    findings = {}
    for name in C.FINDINGS:
        findings[name] = fit_finding_threshold(recs, name)
        f = findings[name]
        print("[finding] %-24s t_f1=%.3f val_f1=%.4f | t_spec90=%.3f | n_pos=%d (%s)"
              % (name, f["t_f1"], f["val_f1"], f["t_spec90"], f["val_n_pos"], f["flag"]))

    out = {
        "meta": {"val_file": str(args.val), "n_val": len(recs),
                 "grid": {"logbias": [-1.5, 1.5, 0.25], "threshold": [0, 1, 0.005]},
                 "val_default_density_bacc": m_default["bacc"],
                 "val_cal_density_bacc": bacc_d,
                 "val_cal_density_merged_bacc": bacc_m,
                 "val_default_birads_bacc": m_default_b["bacc"],
                 "val_cal_birads_bacc": bacc_b},
        "findings": findings,
        "density": {"logbias": {c: bias_d[c] for c in C.DENSITY_CLASSES}},
        "density_merged": {"logbias": {c: bias_m[c] for c in C.DENSITY_MERGED_CLASSES}},
        "birads": {"logbias": {i: bias_b[i] for i in C.BIRADS_CLASSES},
                   "t_ge3": ge3["t"], "t_ge4": ge4["t"]},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("[out] %s  (%.1fs)" % (args.out, time.time() - t0))

    # 边界检查
    hits = [c for c in C.DENSITY_CLASSES if abs(bias_d[c]) >= 1.5]
    hits += [c for c in C.DENSITY_MERGED_CLASSES if abs(bias_m[c]) >= 1.5]
    hits += [i for i in C.BIRADS_CLASSES if abs(bias_b[i]) >= 1.5]
    print("[boundary] log-bias 撞到 ±1.5 边界的类别: %s" % (hits if hits else "无"))


if __name__ == "__main__":
    main()
