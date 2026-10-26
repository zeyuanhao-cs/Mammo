#!/usr/bin/env python3
"""C3a 扫描门控 + r(p)/s(p) 导出（CPU）。退出码 0 = 上升 ≥0.05，可进 400 例评测。"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

P = Path("/home/z/zhao/Mammography/Mammo/2-view/chain_v1/outputs/pilot")
IDS = set(json.load(open(P / "ids2_scan20.json", encoding="utf-8")))
POINTS = [0.0, 0.1, 0.3, 0.5, 1.0]

# ---- 扫描曲线 ----
rows = []
f = P / "c3a_scan.jsonl"
if f.exists():
    for line in f.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r["id"] in IDS:
                rows.append(r)
by = {}
for r in rows:
    by.setdefault(round(r["scan_p"], 2), {"pos": [], "neg": []})
    by[round(r["scan_p"], 2)]["pos" if r["gt_ge3"] else "neg"].append(r["p_birads_ge3"])
ps = sorted(by)
pos = {p: float(np.mean(by[p]["pos"])) for p in ps}
neg = {p: float(np.mean(by[p]["neg"])) for p in ps}
gain = pos.get(0.5, float("nan")) - pos.get(0.1, float("nan"))
ok = (not np.isnan(gain)) and gain >= 0.05

print(f"扫描行数 {len(rows)}（{len({r['id'] for r in rows})} 例 × {len(ps)} 点）")
print("| p | " + " | ".join(f"{p}" for p in ps) + " |")
print("|---|" + "---|" * len(ps))
print("| GT>=3 | " + " | ".join(f"{pos[p]:.3f}" for p in ps) + " |")
print("| GT<=2 | " + " | ".join(f"{neg[p]:.3f}" for p in ps) + " |")
print(f"\np=0.1→0.5 上升 = {gain:+.4f}（阈值 +0.05）→ {'PASS' if ok else 'FAIL'}")

# ---- r(p) / s(p) ----
try:
    sys.path.insert(0, "/home/z/zhao/Mammography/Mammo/2-view")
    from safetensors import safe_open

    from chain_v1.train.soft_mixer import load_soft_mixer, soft_vector

    BASE = Path("/home/z/zhao/Mammography/model/Qwen3.5-4B")
    idx = json.loads((BASE / "model.safetensors.index.json").read_text())["weight_map"]
    key = next(k for k in idx if k.endswith("embed_tokens.weight"))
    with safe_open(str(BASE / idx[key]), framework="pt", device="cpu") as fh:
        emb = fh.get_tensor(key)
    m = load_soft_mixer("/home/z/zhao/Mammography/model/chain_v1_C3a/soft_mixer.pt")
    E0, E1 = emb[15].float(), emb[16].float()
    axis = E1 - E0
    print(f"\nmixer soft_mode={getattr(m, 'soft_mode', '?')} gate={getattr(m, 'gate', '?')}")
    print("\n| p | w(p)=s | s-p |")
    print("|---|---|---|")
    rs = []
    for p in POINTS:
        v = soft_vector(m, emb, torch.tensor(p), 0, 15, mode=getattr(m, "soft_mode", "axis")).float()
        d = v - E0
        sv = float((d * axis).sum() / (axis * axis).sum())
        rs.append(sv)
        print(f"| {p} | {sv:+.4f} | {sv - p:+.4f} |")
    json.dump({"points": POINTS, "r": rs}, open("/tmp/c3a_rs.json", "w"))
except Exception as e:  # noqa: BLE001
    print("[warn] r/s 导出失败:", e)

json.dump({"ps": ps, "pos": [pos[p] for p in ps], "neg": [neg[p] for p in ps], "gain": gain},
          open("/tmp/c3a_scan_gate.json", "w"), ensure_ascii=False, indent=1)
sys.exit(0 if ok else 3)
