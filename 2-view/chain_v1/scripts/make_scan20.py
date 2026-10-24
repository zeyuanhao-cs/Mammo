#!/usr/bin/env python3
"""小规模扫描批次：从原 100 个（ids_scan.json）里按 id 排序取 GT>=3 前 20 + GT<=2 前 20。"""
import json
from pathlib import Path

P = Path("/home/z/zhao/Mammography/Mammo/2-view/chain_v1/outputs/pilot")
ids = json.load(open(P / "ids_scan.json", encoding="utf-8"))

# GT 从 qc1_scan.jsonl（含 gt_ge3）或 start_hard.jsonl 读取
gt = {}
f = P / "qc1_scan.jsonl"
if f.exists():
    for line in f.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            gt[r["id"]] = bool(r["gt_ge3"])
else:
    for line in (P / "start_hard.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            g = (r.get("ground_truth") or {}).get("breast_birads")
            gt[r["id"]] = g is not None and g >= 3

pos = sorted(i for i in ids if gt.get(i))[:20]
neg = sorted(i for i in ids if not gt.get(i))[:20]
out = pos + neg
(P / "ids2_scan20.json").write_text(json.dumps(out), encoding="utf-8")
print(f"[scan20] GT>=3 {len(pos)} + GT<=2 {len(neg)} = {len(out)} -> ids2_scan20.json")
