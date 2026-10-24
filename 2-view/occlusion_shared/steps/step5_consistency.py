#!/usr/bin/env python3
"""第 5 步收尾: 把本次 orig 重跑结果与已有 2000 条测试集概率逐一对比 (192 个乳房)

对比项:
  - 贪心预测 BI-RADS 一致的比例
  - max 征象 P(=1) 的 |Δ| 最大值 / 中位数
  - P(BI-RADS>=3) 的 |Δ| 最大值 / 中位数
判据: 不一致比例 > 2% 或 中位绝对差 > 0.01 -> consistent = false
输出: reports/step5_consistency.json
"""
import json
from pathlib import Path

import numpy as np

OCC = Path("/nesi/project/massey04900/Mammography/occlusion")
ORIG = OCC / "occlusion_probs" / "orig.jsonl"
EXISTING = Path("/home/zhao/Mammography/Mammo/2-view/outputs/scoring/os_sub_test_with_probs.jsonl")
FINDINGS = ["mass", "suspicious_calcification", "asymmetry", "architectural_distortion"]
THRESH_DISAGREE = 0.02
THRESH_MEDIAN = 0.01


def pred_of(pn):
    bi = {k: float(pn["breast_birads"][k]) for k in ("1", "2", "3", "4", "5")}
    return int(max(bi.items(), key=lambda kv: kv[1])[0])


def margin_of(pn):
    """top1 - top2 的 BI-RADS 概率间隔, 用来判断预测是否处于边界."""
    bi = sorted((float(pn["breast_birads"][k]) for k in ("1", "2", "3", "4", "5")), reverse=True)
    return bi[0] - bi[1]


def p45_of(pn):
    return float(pn["breast_birads"]["4"]) + float(pn["breast_birads"]["5"])


def max_sign_of(pn):
    return max(float(pn[f"findings.{k}"]["1"]) for k in FINDINGS)


def ge3_of(pn):
    return 1.0 - float(pn["breast_birads"]["1"]) - float(pn["breast_birads"]["2"])


def main():
    old = {}
    for line in open(EXISTING, encoding="utf-8"):
        r = json.loads(line)
        old[r["id"]] = r["probs_norm"]

    new = {}
    for line in open(ORIG, encoding="utf-8"):
        r = json.loads(line)
        new[r["id"]] = r["probs_norm"]

    ids = sorted(set(new) & set(old))
    assert len(ids) == len(new), f"orig 有 {len(new)} 条, 与已有结果交集 {len(ids)}"

    same = 0
    d_sign, d_ge3 = [], []
    margins_new, margins_old, dis = [], [], []
    dmax_gt02 = d45_gt01 = 0
    d45_max = 0.0
    for bid in ids:
        a, b = new[bid], old[bid]
        pa, pb = pred_of(a), pred_of(b)
        same += int(pa == pb)
        ds = abs(max_sign_of(a) - max_sign_of(b))
        d_sign.append(ds)
        d_ge3.append(abs(ge3_of(a) - ge3_of(b)))
        ma, mb = margin_of(a), margin_of(b)
        margins_new.append(ma)
        margins_old.append(mb)
        d45 = p45_of(a) - p45_of(b)
        if ds > 0.2:
            dmax_gt02 += 1
        if abs(d45) > 0.1:
            d45_gt01 += 1
        d45_max = max(d45_max, abs(d45))
        if pa != pb:
            dis.append({"id": bid, "old_pred": pb, "new_pred": pa,
                        "margin_old": round(mb, 4), "margin_new": round(ma, 4),
                        "abs_diff_max_sign_p": round(ds, 4),
                        "diff_p_birads_45": round(d45, 4)})

    disagree = 1.0 - same / len(ids)
    med_sign = float(np.median(d_sign))
    med_ge3 = float(np.median(d_ge3))
    consistent = bool(disagree <= THRESH_DISAGREE and max(med_sign, med_ge3) <= THRESH_MEDIAN)

    out = {
        "n_compared": len(ids),
        "pred_birads_agreement": round(same / len(ids), 6),
        "pred_birads_disagreement": round(disagree, 6),
        "max_sign_p_max_abs_diff": round(float(max(d_sign)), 8),
        "max_sign_p_median_abs_diff": round(med_sign, 8),
        "p_birads_ge3_max_abs_diff": round(float(max(d_ge3)), 8),
        "p_birads_ge3_median_abs_diff": round(med_ge3, 8),
        "thresholds": {"disagreement": THRESH_DISAGREE, "median_abs_diff": THRESH_MEDIAN},
        "consistent": consistent,
        "diagnosis": {
            "n_disagree": len(dis),
            "disagreements": sorted(dis, key=lambda d: -d["abs_diff_max_sign_p"]),
            "median_margin_all": round(float(np.median(margins_new)), 4),
            "median_margin_disagree": round(float(np.median([d["margin_new"] for d in dis])), 4)
                                       if dis else None,
            "n_abs_dmaxsign_gt_0_2": dmax_gt02,
            "n_abs_diff_p45_gt_0_1": d45_gt01,
            "max_abs_diff_p_birads_45": round(d45_max, 4),
        },
        "baseline_file": str(EXISTING),
        "note": "Δ 基准仍用本次重跑的 orig; 本文件只用于流程一致性核对",
    }
    (OCC / "reports").mkdir(parents=True, exist_ok=True)
    (OCC / "reports" / "step5_consistency.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    print("== 第 5 步一致性: 本次 orig vs 既有 2000 条 ==")
    print(f"  n={len(ids)}  预测 BI-RADS 一致 {same}/{len(ids)} = {1-disagree:.4%} "
          f"(不一致 {disagree:.2%}, 阈值 {THRESH_DISAGREE:.0%})")
    print(f"  max 征象 P(=1): max|Δ|={max(d_sign):.2e}  median|Δ|={med_sign:.2e}")
    print(f"  P(BI-RADS≥3)  : max|Δ|={max(d_ge3):.2e}  median|Δ|={med_ge3:.2e}")
    print(f"  判定: {'一致' if consistent else '不一致 -> report.md 开头会写明'}")
    dg = out["diagnosis"]
    print(f"  边界诊断: 不一致 {dg['n_disagree']} 例, margin 中位 新/旧 = "
          f"{dg['median_margin_disagree']}/{dg['median_margin_all']} (全样本中位 {dg['median_margin_all']})")
    print(f"            |Δmax_sign|>0.2 的 {dg['n_abs_dmaxsign_gt_0_2']} 例, "
          f"|ΔP(4)+P(5)|>0.1 的 {dg['n_abs_diff_p45_gt_0_1']} 例 "
          f"(max {dg['max_abs_diff_p_birads_45']})")
    print(f"[ok] -> {OCC / 'reports' / 'step5_consistency.json'}")


if __name__ == "__main__":
    main()
