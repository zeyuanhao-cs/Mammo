#!/usr/bin/env python3
"""把 outputs/eval/occlusion_3ep 的 occlusion_manifest 转成 eval_soft 可用的 jsonl（按 condition 分文件）。

每条: {id: "<breast>__<cond>", images: [CC图, MLO图], output: {breast_birads, breast_density, findings}}
orig 用原图 path；lesion/ctrl1/ctrl2 用 occluded/<cond>/{bid}__{view}.png。
输出: outputs/pilot/occl_{orig,lesion,ctrl1,ctrl2}.jsonl（每个 192 条）
"""
import json
from pathlib import Path

PROJ = Path("/home/z/zhao/Mammography/Mammo/2-view")
OCC3 = PROJ / "outputs" / "eval" / "occlusion_3ep"
OUTP = PROJ / "chain_v1" / "outputs" / "pilot"

man = json.load(open(OCC3 / "occlusion_manifest.json", encoding="utf-8"))
OUTP.mkdir(parents=True, exist_ok=True)

conds = ("orig", "lesion", "ctrl1", "ctrl2")
rows = {c: [] for c in conds}
missing = []
for bid, e in man["breasts"].items():
    gt = {
        "breast_birads": int(e["gt_birads"]),
        "breast_density": None,      # manifest 不含密度，GT 密度从 test_local 补
        "findings": {k: int(k in (e.get("gt_classes") or [])) for k in
                     ("mass", "suspicious_calcification", "asymmetry", "architectural_distortion")},
    }
    imgs = {}
    for view in ("CC", "MLO"):
        v = e["views"].get(view)
        if v is None:
            continue
        for c in conds[1:]:
            p = OCC3 / "occluded" / c / f"{bid}__{view}.png"
            if not p.exists():
                missing.append(str(p))
    if len(imgs) != 2:
        continue
    for c in conds:
        if c == "orig":
            images = [e["views"]["CC"]["path"], e["views"]["MLO"]["path"]]
        else:
            images = [str(OCC3 / "occluded" / c / f"{bid}__{view}.png") for view in ("CC", "MLO")]
        rows[c].append({"id": f"{bid}__{c}", "images": images, "output": gt})

# 补 GT 密度（从 test_local.json 找同一乳房）
test = json.load(open(PROJ / "data" / "splits" / "test_local.json", encoding="utf-8"))
tmap = {}
for r in test:
    pid, lat = r["id"].rsplit("_", 1)
    tmap[f"{pid}_{lat}"] = r["output"]["breast_density"]
n_dens = 0
for c in conds:
    for r in rows[c]:
        bid = r["id"].rsplit("__", 1)[0]
        if bid in tmap:
            r["output"]["breast_density"] = tmap[bid]
            n_dens += 1

if missing:
    raise SystemExit(f"缺遮挡图 {len(missing)} 个，例: {missing[:3]}")
for c in conds:
    p = OUTP / f"occl_{c}.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for r in rows[c]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[occl] {p.name}: {len(rows[c])} 条 (密度补全 {n_dens // 4})")
