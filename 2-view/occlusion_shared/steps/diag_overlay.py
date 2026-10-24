#!/usr/bin/env python3
"""诊断 3: 把两种候选坐标画到同一张图上, 人工核对哪个框落在病灶上.

红 = resized_*  绿 = orig * (912/W, 1520/H)
"""
import csv
import random

from PIL import Image, ImageDraw

CSV = "/nesi/project/massey04900/Mammography/vindr_detection_v1_folds.csv"
ROOT = "/nesi/project/massey04900/Mammography/images_png"
OUT = "/nesi/project/massey04900/Mammography/occlusion/_diag"
IMG_W, IMG_H = 912, 1520


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    with open(CSV, newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if r["split"] == "test" and r["Mass"] == "1"
                and r["finding_birads"].strip() in ("BI-RADS 4", "BI-RADS 5", "BI-RADS 3")]
    random.seed(7)
    samp = random.sample(rows, 4)
    for i, r in enumerate(samp):
        path = "%s/%s/%s" % (ROOT, r["patient_id"], r["image_id"])
        im = Image.open(path).convert("L").resize((IMG_W // 2, IMG_H // 2))
        im = im.convert("RGB")
        d = ImageDraw.Draw(im)
        s = 0.5
        a = [float(r["resized_xmin"]) * s, float(r["resized_ymin"]) * s,
             float(r["resized_xmax"]) * s, float(r["resized_ymax"]) * s]
        b = [float(r["xmin"]) * IMG_W / float(r["width"]) * s,
             float(r["ymin"]) * IMG_H / float(r["height"]) * s,
             float(r["xmax"]) * IMG_W / float(r["width"]) * s,
             float(r["ymax"]) * IMG_H / float(r["height"]) * s]
        d.rectangle(a, outline=(255, 0, 0), width=3)
        d.rectangle(b, outline=(0, 255, 0), width=3)
        dst = "%s/ov%d_%s.png" % (OUT, i, r["view"])
        im.save(dst)
        print(dst, "view", r["view"], "resized_box", [round(v) for v in a], "scaled_box", [round(v) for v in b])


if __name__ == "__main__":
    main()
