#!/usr/bin/env python3
"""人工核对: 把每个被抽中乳房的 lesion 框区域放大 2x, orig vs lesion 并排."""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

OCC = Path("/nesi/project/massey04900/Mammography/occlusion")
sys.path.insert(0, str(OCC))
from apply_occlusion import occlude_view  # noqa: E402

OUT = OCC / "_diag"
OUT.mkdir(exist_ok=True)


def main():
    man = json.load(open(OCC / "occlusion_manifest.json", encoding="utf-8"))
    tiles = []
    picks = ["14a63118d94643a3d65b32039a20b351_L", "cf399797f9211774fdb1470bf4ba5600_R",
             "8dd986e8174fc84c2a984e66a416ff9f_R"]
    for bid in picks:
        e = man["breasts"][bid]
        for view in ("CC", "MLO"):
            v = e["views"].get(view)
            if not v or not v["lesion_boxes"]:
                continue
            orig = np.asarray(Image.open(v["path"]).convert("L"))
            les = occlude_view(v, "lesion")
            x0, y0, x1, y1 = [int(round(t)) for t in v["conditions"]["lesion"][0]["rect"]]
            pad = 60
            cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
            cx1, cy1 = min(912, x1 + pad), min(1520, y1 + pad)
            a = Image.fromarray(orig[cy0:cy1, cx0:cx1]).convert("RGB").resize(
                ((cx1 - cx0) * 2, (cy1 - cy0) * 2), Image.NEAREST)
            b = Image.fromarray(les[cy0:cy1, cx0:cx1]).convert("RGB").resize(
                ((cx1 - cx0) * 2, (cy1 - cy0) * 2), Image.NEAREST)
            d = ImageDraw.Draw(a)
            d.rectangle([(x0 - cx0) * 2, (y0 - cy0) * 2, (x1 - cx0) * 2, (y1 - cy0) * 2],
                        outline=(255, 0, 0), width=2)
            d.text((4, 4), f"{bid} {view} orig", fill=(255, 255, 0))
            d = ImageDraw.Draw(b)
            d.rectangle([(x0 - cx0) * 2, (y0 - cy0) * 2, (x1 - cx0) * 2, (y1 - cy0) * 2],
                        outline=(255, 0, 0), width=2)
            d.text((4, 4), f"lesion-masked (fill={v['conditions']['lesion'][0]['fill']})",
                   fill=(255, 255, 0))
            pair = Image.new("RGB", (a.width + b.width + 6, max(a.height, b.height)), (20, 20, 20))
            pair.paste(a, (0, 0))
            pair.paste(b, (a.width + 6, 0))
            tiles.append(pair)
            print(f"[zoom] {bid} {view}  rect=({x0},{y0},{x1},{y1})")
    W = max(t.width for t in tiles)
    H = sum(t.height for t in tiles) + 6 * (len(tiles) - 1)
    canvas = Image.new("RGB", (W, H), (20, 20, 20))
    y = 0
    for t in tiles:
        canvas.paste(t, (0, y))
        y += t.height + 6
    canvas.save(OUT / "zoom_lesion.png")
    print("[zoom]", OUT / "zoom_lesion.png", canvas.size)


if __name__ == "__main__":
    main()
