#!/usr/bin/env python3
"""从扫描 jsonl 汇总曲线，并从 mixer 导出 w(p)=s 与 s−p（axis 模式）。"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

PROJ = Path("/home/z/zhao/Mammography/Mammo/2-view")
DEFAULT_POINTS = "0,0.1,0.2,0.3,0.4,0.45,0.5,0.7,1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mixer", required=True)
    ap.add_argument("--scan-jsonl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids", default=str(PROJ / "chain_v1/outputs/pilot/ids2_scan20.json"))
    ap.add_argument("--points", default=DEFAULT_POINTS)
    a = ap.parse_args()
    POINTS = [float(x) for x in a.points.split(",")]

    ids = set(json.load(open(a.ids, encoding="utf-8")))
    by = {}
    for line in open(a.scan_jsonl, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r["id"] in ids and r["gt_ge3"]:
            by.setdefault(round(r["scan_p"], 2), []).append(r["p_birads_ge3"])
    curve = {p: float(np.mean(v)) for p, v in by.items() if v}
    print("[curve] " + "  ".join(f"p={p}:{curve[p]:.3f}" for p in sorted(curve)))

    import sys
    sys.path.insert(0, str(PROJ))
    from chain_v1.train.soft_mixer import load_soft_mixer, soft_vector

    base = Path("/home/z/zhao/Mammography/model/Qwen3.5-4B")
    idx = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    key = next(k for k in idx if k.endswith("embed_tokens.weight"))
    with safe_open(str(base / idx[key]), framework="pt", device="cpu") as fh:
        emb = fh.get_tensor(key)
    m = load_soft_mixer(a.mixer)
    E0, E1 = emb[15].float(), emb[16].float()
    axis = E1 - E0
    rows = []
    for p in POINTS:
        v = soft_vector(m, emb, torch.tensor(p), 0, 15,
                        mode=getattr(m, "soft_mode", "axis")).float()
        d = v - E0
        s = float((d * axis).sum() / (axis * axis).sum())
        rows.append({"p": p, "s": s, "s_minus_p": s - p})
        print(f"[w] p={p}  w(p)={s:+.4f}  s-p={s - p:+.4f}")
    json.dump({"points": POINTS, "r": [r["s"] for r in rows], "curve": curve},
              open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"[done] {a.out}")


if __name__ == "__main__":
    main()
