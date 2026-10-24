#!/usr/bin/env python3
"""从某模型验证集输出挑批次 id（兼容 eval_soft jsonl 与 b8 jsonl 两种格式）:
  - <prefix>_soft.json  : pred<=2 且 max_sign_p>0.1 的前 N 个（软一致性批次，所有组共用）
  - <prefix>_scan.json  : GT>=3 前 N1 个 + GT<=2 前 N2 个（概率扫描批次）
  - <prefix>_ablation.json : 按 study 抽样 N 个乳房（seed 42；teacher/swap/软换硬共用）
"""
import argparse
import json
import random
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--hard-jsonl", required=True,
                help="eval_soft 输出（有 pred_birads/max_sign_p）或 b8 jsonl（有 probs_norm）")
ap.add_argument("--out-prefix", required=True)
ap.add_argument("--n-soft", type=int, default=200)
ap.add_argument("--n-scan-pos", type=int, default=50)
ap.add_argument("--n-scan-neg", type=int, default=50)
ap.add_argument("--n-ablation", type=int, default=0, help=">0 时按 study 抽样该数目")
args = ap.parse_args()


def norm(r):
    """统一成 {id, pred_birads, max_sign_p, gt_birads}。"""
    if "pred_birads" in r and "max_sign_p" in r:
        return {"id": r["id"], "pred": r["pred_birads"], "sign": r["max_sign_p"],
                "gt": r.get("gt_birads")}
    pn = r["probs_norm"]
    bi = pn["breast_birads"]
    pred = int(max(bi.items(), key=lambda kv: kv[1])[0])
    sign = max(float(pn[f"findings.{k}"]["1"]) for k in
               ("mass", "suspicious_calcification", "asymmetry", "architectural_distortion"))
    # b8 管线（infer_2view_batched）的 GT 在 ground_truth.breast_birads
    gt = r.get("gt_birads")
    if gt is None and r.get("ground_truth"):
        gt = r["ground_truth"].get("breast_birads")
    return {"id": r["id"], "pred": pred, "sign": sign, "gt": gt}


rows = [norm(json.loads(l)) for l in open(args.hard_jsonl, encoding="utf-8")]
pre = Path(args.out_prefix)

soft = [r["id"] for r in rows if r["pred"] <= 2 and r["sign"] > 0.1][: args.n_soft]
pos = [r["id"] for r in rows if r["gt"] is not None and r["gt"] >= 3][: args.n_scan_pos]
neg = [r["id"] for r in rows if r["gt"] is not None and r["gt"] <= 2][: args.n_scan_neg]
pre.with_name(pre.name + "_soft.json").write_text(json.dumps(soft), encoding="utf-8")
pre.with_name(pre.name + "_scan.json").write_text(json.dumps(pos + neg), encoding="utf-8")

n_ab = 0
if args.n_ablation > 0:
    by_study = {}
    for r in rows:
        by_study.setdefault(r["id"].rsplit("_", 1)[0], []).append(r["id"])
    studies = sorted(by_study)
    random.Random(42).shuffle(studies)
    picked = []
    for s in studies:
        picked += by_study[s]
        if len(picked) >= args.n_ablation:
            break
    picked = picked[: args.n_ablation]
    picked.sort()
    pre.with_name(pre.name + "_ablation.json").write_text(json.dumps(picked), encoding="utf-8")
    n_ab = len(picked)

print(f"[ids] soft={len(soft)} scan={len(pos)}+{len(neg)} ablation={n_ab} -> {pre.name}_*.json")
