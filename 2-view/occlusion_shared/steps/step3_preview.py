#!/usr/bin/env python3
"""第 3 步 (续): 抽 6 个乳房 (按病灶面积分位选, 覆盖不同大小), 把
orig / lesion / ctrl1 / ctrl2 四种情况的 CC、MLO 拼成图, 供人工核对.

输出:
  occlusion_check/{breast_id}.png   每个乳房 4 行 (情况) x 2 列 (视角)
  occlusion_check/_montage.png      6 个乳房横向拼图
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

OCC = Path("/nesi/project/massey04900/Mammography/occlusion")
sys.path.insert(0, str(OCC))
from apply_occlusion import occlude_view  # noqa: E402

MANIFEST = OCC / "occlusion_manifest.json"
OUTDIR = OCC / "occlusion_check"
CONDS = ("orig", "lesion", "ctrl1", "ctrl2")
SCALE = 0.25
TARGETS = (0.05, 0.2, 0.4, 0.6, 0.8, 0.95)


def pick_breasts(man, k=6):
    rows = []
    for bid, e in man["breasts"].items():
        cc = e["views"].get("CC", {}).get("lesion_boxes", [])
        mlo = e["views"].get("MLO", {}).get("lesion_boxes", [])
        if not cc or not mlo:
            continue
        area = sum((b[2] - b[0]) * (b[3] - b[1]) for b in cc + mlo) / (912.0 * 1520.0)
        rows.append((area, bid))
    rows.sort()
    picked, used = [], set()
    n = len(rows)
    for t in TARGETS:
        idx = min(n - 1, max(0, int(round(t * (n - 1)))))
        for d in range(n):
            for j in (idx - d, idx + d):
                if 0 <= j < n and j not in used:
                    used.add(j)
                    picked.append(rows[j])
                    break
            else:
                continue
            break
    return picked[:k]


def render(man, bid, entry):
    panels = []
    for cond in CONDS:
        cols = []
        for view in ("CC", "MLO"):
            v = entry["views"].get(view)
            if v is None:
                cols.append(None)
                continue
            if cond == "orig":
                arr = np.asarray(Image.open(v["path"]).convert("L"))
            else:
                arr = occlude_view(v, cond)
            im = Image.fromarray(arr).convert("RGB")
            w = int(im.width * SCALE)
            h = int(im.height * SCALE)
            im = im.resize((w, h), Image.BILINEAR)
            d = ImageDraw.Draw(im)
            items = v["conditions"].get(cond) or []
            if cond != "orig":
                for it in items:
                    x0, y0, x1, y1 = [int(round(t * SCALE)) for t in it["rect"]]
                    d.rectangle([x0, y0, x1, y1], outline=(255, 40, 40), width=1)
            elif v["lesion_boxes"]:
                for b in v["lesion_boxes"]:
                    x0, y0, x1, y1 = [int(round(t * SCALE)) for t in b]
                    d.rectangle([x0, y0, x1, y1], outline=(60, 220, 60), width=1)
            d.text((3, 3), f"{view} {cond} rects={len(items)}", fill=(255, 255, 0))
            cols.append(im)
        panels.append(cols)
    return panels


def main():
    man = json.load(open(MANIFEST, encoding="utf-8"))
    OUTDIR.mkdir(parents=True, exist_ok=True)
    picked = pick_breasts(man)
    canvas_rows = []
    for area, bid in picked:
        entry = man["breasts"][bid]
        panels = render(man, bid, entry)
        pw = max(c.width for row in panels for c in row if c)
        ph = max(c.height for row in panels for c in row if c)
        tile = Image.new("RGB", (pw * 2 + 6, ph * len(CONDS) + 20 * len(CONDS)), (25, 25, 25))
        d = ImageDraw.Draw(tile)
        for r, (cond, row) in enumerate(zip(CONDS, panels)):
            d.text((3, r * (ph + 20)), f"{bid}  {cond}   area={area*100:.2f}%", fill=(255, 255, 255))
            for c, im in enumerate(row):
                if im is None:
                    continue
                tile.paste(im, (c * (pw + 6), r * (ph + 20) + 18))
        p = OUTDIR / f"{bid}.png"
        tile.save(p)
        print(f"[check] {p}  ({tile.width}x{tile.height}) area={area*100:.2f}%")
        canvas_rows.append((bid, area, tile))

    W = sum(t.width for _, _, t in canvas_rows) + 8 * (len(canvas_rows) - 1)
    H = max(t.height for _, _, t in canvas_rows) + 26
    mont = Image.new("RGB", (W, H), (15, 15, 15))
    x = 0
    dm = ImageDraw.Draw(mont)
    for bid, area, t in canvas_rows:
        mont.paste(t, (x, 26))
        dm.text((x + 3, 8), f"{bid[:-2]}…{bid[-1]}  {area*100:.2f}%", fill=(255, 255, 255))
        x += t.width + 8
    mp = OUTDIR / "_montage.png"
    mont.save(mp)
    print(f"[check] {mp}  ({mont.width}x{mont.height})")
    print("[图例] 每块 4 行 = orig / lesion / ctrl1 / ctrl2, 2 列 = CC / MLO; "
          "红框 = 该情况遮挡的矩形, 绿框 = orig 上的原始病灶框")


if __name__ == "__main__":
    main()
