#!/usr/bin/env python3
"""在 val 上拟合阈值/偏置, 输出 thresholds.json. 只读 val, 不碰 test.

v2 (相对 backup_v1) 修改:
  - 偏置参考类固定为 0, 不参与搜索: density 四类/合并版固定 C, BI-RADS 固定 1;
    其余类相对参考类在 [-1.5, 1.5] 步长 0.25 内搜索; 边界检查只看非参考类
  - 阈值扫描 (findings F1 最优 / spec90 / BI-RADS >=3 / >=4) 不再用固定步长网格,
    改为在 val 的实际概率值 (去重后) 里搜索, 平手时选更接近 0.5 的
"""
import argparse
import json
import time
from pathlib import Path

import common as C

GRID = [round(-1.5 + 0.25 * i, 2) for i in range(13)]  # [-1.5, 1.5] step 0.25, 仅非参考类
REF_DENSITY = "C"
REF_DENSITY_MERGED = "C"
REF_BIRADS = 1


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


def fit_logbias(recs, key, classes, ref, obj="bacc"):
    """坐标上升拟合每类 log-bias; ref 参考类偏置固定 0, 不参与搜索."""
    probs_list = [r[key] for r in recs]
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
            if c == ref:
                continue  # 参考类固定 0
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
    bias[ref] = 0.0
    return bias, best


def _exp(x):
    return pow(2.718281828459045, x)


def _scan_thresholds(y, s):
    """在 val 实际概率值 (去重) 里扫描. 返回 (t_f1, f1, t_spec90, spec90).
    平手 (F1 或 |spec-0.9| 相同) 时选更接近 0.5 的阈值."""
    cands = sorted(set(s))
    best_t, best_f1 = None, -1.0
    for t in cands:
        m = C.binary_metrics(y, [1 if p >= t else 0 for p in s])
        if (m["f1"] > best_f1 + 1e-12
                or (abs(m["f1"] - best_f1) <= 1e-12 and best_t is not None
                    and abs(t - 0.5) < abs(best_t - 0.5))):
            best_f1, best_t = m["f1"], t
    ref_t, ref_d, ref_spec = None, 1e9, None
    for t in cands:
        m = C.binary_metrics(y, [1 if p >= t else 0 for p in s])
        d = abs(m["specificity"] - 0.90)
        if (d < ref_d - 1e-12
                or (abs(d - ref_d) <= 1e-12 and ref_t is not None
                    and abs(t - 0.5) < abs(ref_t - 0.5))):
            ref_d, ref_t, ref_spec = d, t, m["specificity"]
    return best_t, best_f1, ref_t, ref_spec


def fit_finding_threshold(recs, name):
    y = [r["gt_findings"][name] for r in recs]
    s = [r["p_findings"][name] for r in recs]
    t_f1, f1, t_spec90, spec90 = _scan_thresholds(y, s)
    n_pos = sum(y)
    flag = "ok"
    if n_pos < 10:
        flag = "unstable"
    elif n_pos < 20:
        flag = "caution"
    return {"t_f1": t_f1, "val_f1": f1, "t_spec90": t_spec90, "val_spec90": spec90,
            "val_n_pos": n_pos, "flag": flag}


def fit_birads_binary(recs, level):
    """level=3: score=P(3)+P(4)+P(5), 正例 gt>=3; level=4: score=P(4)+P(5), 正例 gt>=4."""
    y = [1 if r["gt_birads"] >= level else 0 for r in recs]
    s = [sum(r["p_birads"][str(i)] for i in range(level, 6)) for r in recs]
    t, f1, _, _ = _scan_thresholds(y, s)
    return {"t": t, "val_f1": f1, "val_n_pos": sum(y)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True, help="val 概率 jsonl")
    ap.add_argument("--out", required=True, help="输出 thresholds.json")
    args = ap.parse_args()

    t0 = time.time()
    recs = C.load_probs_jsonl(args.val)
    print("[val] %d records from %s" % (len(recs), args.val))

    # ---- density 4 类 (参考类 C 固定 0) ----
    bias_d, bacc_d = fit_logbias(recs, "p_density", C.DENSITY_CLASSES, REF_DENSITY)
    m_default = C.multiclass_metrics([r["gt_density"] for r in recs],
                                     [r["greedy_density"] for r in recs], C.DENSITY_CLASSES)
    print("[density] default bacc=%.4f -> cal bacc=%.4f  logbias=%s (ref: %s=0)"
          % (m_default["bacc"], bacc_d,
             {c: round(bias_d[c], 2) for c in C.DENSITY_CLASSES}, REF_DENSITY))

    # ---- density merged (A+B -> AB, 参考类 C 固定 0) ----
    recs_m = []
    for r in recs:
        recs_m.append({
            "p_density": {"AB": r["p_density"]["A"] + r["p_density"]["B"],
                          "C": r["p_density"]["C"], "D": r["p_density"]["D"]},
            "gt_density": "AB" if r["gt_density"] in ("A", "B") else r["gt_density"],
        })
    bias_m, bacc_m = fit_logbias(recs_m, "p_density", C.DENSITY_MERGED_CLASSES, REF_DENSITY_MERGED)
    print("[density_merged] cal bacc=%.4f  logbias=%s (ref: %s=0)"
          % (bacc_m, {c: round(bias_m[c], 2) for c in C.DENSITY_MERGED_CLASSES}, REF_DENSITY_MERGED))

    # ---- birads 5 类 (参考类 1 固定 0) ----
    bias_b, bacc_b = fit_logbias(recs, "p_birads", C.BIRADS_CLASSES, REF_BIRADS)
    m_default_b = C.multiclass_metrics([r["gt_birads"] for r in recs],
                                       [r["greedy_birads"] for r in recs], C.BIRADS_CLASSES)
    print("[birads] default bacc=%.4f -> cal bacc=%.4f  logbias=%s (ref: %s=0)"
          % (m_default_b["bacc"], bacc_b, {i: round(bias_b[i], 2) for i in C.BIRADS_CLASSES}, REF_BIRADS))

    # ---- birads 二分类阈值 ----
    ge3 = fit_birads_binary(recs, 3)
    ge4 = fit_birads_binary(recs, 4)
    print("[birads] >=3: t=%.4f val_f1=%.4f (n_pos=%d) | >=4: t=%.4f val_f1=%.4f (n_pos=%d)"
          % (ge3["t"], ge3["val_f1"], ge3["val_n_pos"], ge4["t"], ge4["val_f1"], ge4["val_n_pos"]))

    # ---- findings ----
    findings = {}
    for name in C.FINDINGS:
        findings[name] = fit_finding_threshold(recs, name)
        f = findings[name]
        print("[finding] %-24s t_f1=%.4f val_f1=%.4f | t_spec90=%.4f (val spec=%.4f) | n_pos=%d (%s)"
              % (name, f["t_f1"], f["val_f1"], f["t_spec90"], f["val_spec90"], f["val_n_pos"], f["flag"]))

    out = {
        "meta": {"val_file": str(args.val), "n_val": len(recs),
                 "grid": {"logbias": [-1.5, 1.5, 0.25], "threshold": "unique val probability values"},
                 "threshold_search": "val 实际概率值去重搜索, 平手取更接近 0.5",
                 "ref_class": {"density": REF_DENSITY, "density_merged": REF_DENSITY_MERGED,
                               "birads": REF_BIRADS},
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

    # 边界检查: 只看非参考类
    hits = [c for c in C.DENSITY_CLASSES if c != REF_DENSITY and abs(bias_d[c]) >= 1.5]
    hits += [c for c in C.DENSITY_MERGED_CLASSES if c != REF_DENSITY_MERGED and abs(bias_m[c]) >= 1.5]
    hits += [i for i in C.BIRADS_CLASSES if i != REF_BIRADS and abs(bias_b[i]) >= 1.5]
    print("[boundary] 非参考类 log-bias 撞到 ±1.5 边界: %s" % (hits if hits else "无"))
    print("[boundary] 参考类 (固定 0): density=%s, density_merged=%s, birads=%s"
          % (REF_DENSITY, REF_DENSITY_MERGED, REF_BIRADS))


if __name__ == "__main__":
    main()
