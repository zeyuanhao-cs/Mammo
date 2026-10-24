#!/usr/bin/env python3
"""对齐检查: eval_soft hard 模式 vs b8 管线在同批乳房上的输出一致性.

判据: BI-RADS argmax 全一致 且 各概率字段最大绝对差 < 1e-3。
输出: outputs/pilot/align_check.json；不一致 exit 3（总控停下等用户）。
"""
import json
import sys
from pathlib import Path

import numpy as np

P = Path("/home/z/zhao/Mammography/Mammo/2-view/chain_v1/outputs/pilot")
FIND = ["mass", "suspicious_calcification", "asymmetry", "architectural_distortion"]
TOL = 1e-3


def b8map(fn):
    out = {}
    for l in open(fn, encoding="utf-8"):
        r = json.loads(l)
        pn = r.get("probs_norm") or {}
        bi = pn.get("breast_birads") or {}
        if len(bi) != 5:
            continue
        out[r["id"]] = {
            "pred": int(max(bi.items(), key=lambda kv: kv[1])[0]),
            "bi": {k: float(v) for k, v in bi.items()},
            "find": {k: float((pn.get(f"findings.{k}") or {}).get("1", 0.0)) for k in FIND},
        }
    return out


def main():
    soft_rows = {}
    for l in open(P / "b0_hard_align200.jsonl", encoding="utf-8"):
        r = json.loads(l)
        soft_rows[r["id"]] = r
    hard = b8map(P / "b0_b8_align200.jsonl")
    ids = sorted(set(soft_rows) & set(hard))
    assert ids, "无交集"
    disagree = []
    max_diff = 0.0
    for i in ids:
        a, b = soft_rows[i], hard[i]
        if a["pred_birads"] != b["pred"]:
            disagree.append({"id": i, "soft": a["pred_birads"], "b8": b["pred"]})
        for k in range(1, 6):
            max_diff = max(max_diff, abs(a["p_birads"][str(k)] - b["bi"][str(k)]))
        for k in FIND:
            max_diff = max(max_diff, abs(a["p_findings"][k] - b["find"][k]))
    agree = 1.0 - len(disagree) / len(ids)
    ok = (not disagree) and max_diff < TOL
    out = {"n": len(ids), "argmax_agreement": agree,
           "max_abs_prob_diff": max_diff, "tol": TOL, "consistent": ok,
           "disagreements": disagree[:10]}
    (P / "align_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    sys.exit(0 if ok else 3)


if __name__ == "__main__":
    main()
