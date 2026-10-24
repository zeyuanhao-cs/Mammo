#!/usr/bin/env python3
"""第 4 步 (CPU): 用模型已有测试集概率把 192 个 GT BI-RADS>=3 乳房分成
detected / missed_sign / missed_nosign, 并算"最大征象概率"区分漏报与 GT=1&预测<=2 的 AUC.

输入 (已存在, 与本次推理同一 adapter / prompt / 分辨率):
  /home/zhao/Mammography/Mammo/2-view/outputs/scoring/os_sub_test_with_probs.jsonl  (2000 条)
分类:
  detected      : 预测 BI-RADS >= 3                    (argmax over P(1..5))
  missed_sign   : 预测 <= 2 且 (预测 == 2 或 4 个征象 max P(=1) > 0.1)
  missed_nosign : 其余 (预测 <= 2 且 预测 == 1 且 max P(=1) <= 0.1)
输出:
  reports/step4.json
"""
import json
from pathlib import Path

BASE = Path("/nesi/project/massey04900/Mammography")
OCC = BASE / "occlusion"
PROBS = Path("/home/zhao/Mammography/Mammo/2-view/outputs/scoring/os_sub_test_with_probs.jsonl")
TEST_JSON = BASE / "test.json"
FINDINGS = ["mass", "suspicious_calcification", "asymmetry", "architectural_distortion"]
SIGN_THRESH = 0.1


def auc(y, s):
    pairs = sorted(zip(s, y))
    n_pos = sum(y)
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank = 0.0
    i = 0
    acc = 0.0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg = (i + j + 1) / 2.0  # 1-based 平均秩
        for k in range(i, j):
            if pairs[k][1]:
                acc += avg
        i = j
    return (acc - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def main():
    test = {r["id"]: r for r in json.load(open(TEST_JSON, encoding="utf-8"))}
    recs = {}
    for line in open(PROBS, encoding="utf-8"):
        r = json.loads(line)
        p = r["probs_norm"]        # 与仓库既有分析 (scripts/eval/common.py) 口径一致
        bi = {k: float(v) for k, v in p["breast_birads"].items()}
        sign = {k: (float(p[f"findings.{k}"]["1"]) if p.get(f"findings.{k}") else None) for k in FINDINGS}
        recs[r["id"]] = {
            "birads_probs": bi,
            "pred_birads": int(max(bi.items(), key=lambda kv: kv[1])[0]),
            "sign_p1": sign,
            "max_sign_p": max(v for v in sign.values() if v is not None),
        }

    ge3 = [bid for bid, r in test.items() if int(r["output"]["breast_birads"]) >= 3]
    assert len(ge3) == 192, len(ge3)

    groups = {"detected": [], "missed_sign": [], "missed_nosign": []}
    detail = {}
    for bid in ge3:
        r = recs[bid]
        if r["pred_birads"] >= 3:
            g = "detected"
        elif r["pred_birads"] == 2 or r["max_sign_p"] > SIGN_THRESH:
            g = "missed_sign"
        else:
            g = "missed_nosign"
        groups[g].append(bid)
        gt_cls = [k for k, v in test[bid]["output"]["findings"].items() if v]
        detail[bid] = {
            "group": g,
            "gt_birads": int(test[bid]["output"]["breast_birads"]),
            "gt_classes": gt_cls,
            "pred_birads": r["pred_birads"],
            "max_sign_p": round(r["max_sign_p"], 6),
            "p_birads": {k: round(v, 6) for k, v in r["birads_probs"].items()},
        }

    # --- AUC: 最大征象概率区分 "漏报病灶" vs "GT=1 且预测<=2"
    missed_all = [b for b in ge3 if recs[b]["pred_birads"] <= 2]
    neg = [b for b, r in test.items()
           if int(r["output"]["breast_birads"]) == 1 and recs[b]["pred_birads"] <= 2]
    auc_all = auc([1] * len(missed_all) + [0] * len(neg),
                  [recs[b]["max_sign_p"] for b in missed_all] + [recs[b]["max_sign_p"] for b in neg])
    ms = groups["missed_sign"]
    auc_ms = auc([1] * len(ms) + [0] * len(neg),
                 [recs[b]["max_sign_p"] for b in ms] + [recs[b]["max_sign_p"] for b in neg])
    mn = groups["missed_nosign"]
    auc_ms_vs_mn = auc([1] * len(ms) + [0] * len(mn),
                       [recs[b]["max_sign_p"] for b in ms] + [recs[b]["max_sign_p"] for b in mn])

    out = {
        "probs_source": str(PROBS),
        "n_test_breasts": len(test),
        "n_ge3": len(ge3),
        "counts": {k: len(v) for k, v in groups.items()},
        "groups": groups,
        "auc": {
            "primary__missed(GT>=3,pred<=2) vs (GT=1,pred<=2)": round(auc_all, 4),
            "missed_sign vs (GT=1,pred<=2)": round(auc_ms, 4),
            "missed_sign vs missed_nosign": round(auc_ms_vs_mn, 4),
            "n_pos_primary": len(missed_all), "n_neg_primary": len(neg),
            "n_missed_sign": len(ms), "n_missed_nosign": len(mn),
        },
        "detail": detail,
    }
    (OCC / "reports").mkdir(parents=True, exist_ok=True)
    (OCC / "reports" / "step4.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                               encoding="utf-8")

    print("== 第 4 步: 192 个 GT BI-RADS>=3 乳房的分类 ==")
    for k, v in groups.items():
        print(f"  {k:14s} {len(v):4d}")
    print(f"  GT>=3 但 pred<=2 (漏报病灶)        {len(missed_all):4d}")
    print(f"  GT=1 且 pred<=2 (对照负样本)       {len(neg):4d}")
    print("== AUC (最大征象概率) ==")
    print(f"  漏报病灶 vs GT=1&pred<=2   AUC = {auc_all:.4f}   (n+={len(missed_all)}, n-={len(neg)})")
    print(f"  missed_sign vs GT=1&pred<=2 AUC = {auc_ms:.4f}")
    print(f"  missed_sign vs missed_nosign AUC = {auc_ms_vs_mn:.4f}")
    print(f"[ok] -> {OCC / 'reports' / 'step4.json'}")


if __name__ == "__main__":
    main()
