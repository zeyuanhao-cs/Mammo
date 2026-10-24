#!/usr/bin/env python3
"""挑"确定样本"：GT BI-RADS=1 且起点模型 4 个征象 p 全 < 0.05。

用途：阶段一"确定样本上软注入的扰动"（QC1 软 vs 软换硬）。
输入：b8 管线全量输出（含 probs_norm 与 ground_truth）或 eval_soft 输出。
输出：ids json（按 id 排序，取前 N，确定性）。
"""
import argparse
import json

FIND = ["mass", "suspicious_calcification", "asymmetry", "architectural_distortion"]

ap = argparse.ArgumentParser()
ap.add_argument("--hard-jsonl", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--p-max", type=float, default=0.05)
args = ap.parse_args()

rows = []
for line in open(args.hard_jsonl, encoding="utf-8"):
    r = json.loads(line)
    gt = r.get("gt_birads")
    if gt is None and r.get("ground_truth"):
        gt = r["ground_truth"].get("breast_birads")
    if gt != 1:
        continue
    pn = r.get("probs_norm") or {}
    ps = [float((pn.get(f"findings.{k}") or {}).get("1", 1.0)) for k in FIND]
    if ps and max(ps) < args.p_max:
        rows.append(r["id"])

rows.sort()
picked = rows[: args.n]
json.dump(picked, open(args.out, "w", encoding="utf-8"))
print(f"[cert] 候选 {len(rows)} 例，取 {len(picked)} -> {args.out}")
