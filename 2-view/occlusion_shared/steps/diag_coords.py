#!/usr/bin/env python3
"""诊断: vindr_detection_v1_folds.csv 的 resized_* 坐标是否已经是 912x1520 PNG 像素空间.

做法: 随机抽 test split 带 finding_birads 的若干框, 比较两种候选坐标下框内"亮像素"(乳腺组织)
占比, 以及框是否越界; 正确的坐标映射应当落在乳腺组织内(亮占比高).
"""
import csv
import random

import numpy as np
from PIL import Image

CSV = "/nesi/project/massey04900/Mammography/vindr_detection_v1_folds.csv"
ROOT = "/nesi/project/massey04900/Mammography/images_png"
IMG_W, IMG_H = 912, 1520


def bright_frac(arr, box, thr=20):
    h, w = arr.shape
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0 = max(0, min(x0, w - 1)); x1 = max(x0 + 1, min(x1, w))
    y0 = max(0, min(y0, h - 1)); y1 = max(y0 + 1, min(y1, h))
    sub = arr[y0:y1, x0:x1]
    return float((sub > thr).mean())


def main():
    with open(CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    splits = {}
    for r in rows:
        splits[r["split"]] = splits.get(r["split"], 0) + 1
    print("csv rows", len(rows), "splits", splits)

    pool = [r for r in rows if r["split"] == "test" and r["finding_birads"].strip()]
    random.seed(0)
    samp = random.sample(pool, min(150, len(pool)))

    fa, fb, ious = [], [], []
    inside_a = inside_b = 0
    for r in samp:
        path = "%s/%s/%s" % (ROOT, r["patient_id"], r["image_id"])
        arr = np.asarray(Image.open(path).convert("L"))
        w0, h0 = float(r["width"]), float(r["height"])
        box_a = (float(r["resized_xmin"]), float(r["resized_ymin"]),
                 float(r["resized_xmax"]), float(r["resized_ymax"]))
        box_b = (float(r["xmin"]) * IMG_W / w0, float(r["ymin"]) * IMG_H / h0,
                 float(r["xmax"]) * IMG_W / w0, float(r["ymax"]) * IMG_H / h0)
        fa.append(bright_frac(arr, box_a))
        fb.append(bright_frac(arr, box_b))
        inside_a += int(box_a[0] >= 0 and box_a[2] <= IMG_W and box_a[1] >= 0 and box_a[3] <= IMG_H)
        inside_b += int(box_b[0] >= 0 and box_b[2] <= IMG_W and box_b[1] >= 0 and box_b[3] <= IMG_H)
        ix0, iy0 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
        ix1, iy1 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
        inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        aa = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        ab = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        ious.append(inter / (aa + ab - inter) if aa + ab - inter > 0 else 0.0)

    print("resized_* 原样 : 框内亮占比 mean %.3f median %.3f, 全在界内 %d/%d"
          % (np.mean(fa), np.median(fa), inside_a, len(samp)))
    print("原图坐标缩放到 PNG: 框内亮占比 mean %.3f median %.3f, 全在界内 %d/%d"
          % (np.mean(fb), np.median(fb), inside_b, len(samp)))
    print("两候选框 IoU 中位数 %.3f" % np.median(ious))

    r = samp[0]
    print("样例 width,height=%s,%s view=%s" % (r["width"], r["height"], r["view"]))
    print("  orig   ", (r["xmin"], r["ymin"], r["xmax"], r["ymax"]))
    print("  resized", (r["resized_xmin"], r["resized_ymin"], r["resized_xmax"], r["resized_ymax"]))
    print("  resized 框宽高 = %.1f x %.1f" % (float(r["resized_xmax"]) - float(r["resized_xmin"]),
                                              float(r["resized_ymax"]) - float(r["resized_ymin"])))
    print("  orig 框宽高   = %.1f x %.1f" % (float(r["xmax"]) - float(r["xmin"]),
                                             float(r["ymax"]) - float(r["ymin"])))


if __name__ == "__main__":
    main()
