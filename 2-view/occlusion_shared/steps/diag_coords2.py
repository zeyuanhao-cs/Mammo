#!/usr/bin/env python3
"""诊断 2: resized_* 与 orig (xmin..ymax) 之间是否存在线性关系."""
import csv

import numpy as np

CSV = "/nesi/project/massey04900/Mammography/vindr_detection_v1_folds.csv"


def main():
    with open(CSV, newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if r["split"] == "test" and r["finding_birads"].strip()
                and float(r["xmax"]) > float(r["xmin"])
                and float(r["resized_xmax"]) > float(r["resized_xmin"])]
    print("rows", len(rows))

    def arr(col):
        return np.array([float(r[col]) for r in rows])

    ox0, oy0, ox1, oy1 = arr("xmin"), arr("ymin"), arr("xmax"), arr("ymax")
    rx0, ry0, rx1, ry1 = arr("resized_xmin"), arr("resized_ymin"), arr("resized_xmax"), arr("resized_ymax")
    W, H = arr("width"), arr("height")

    for name, o, r in (("x0", ox0, rx0), ("y0", oy0, ry0), ("x1", ox1, rx1), ("y1", oy1, ry1)):
        corr = np.corrcoef(o, r)[0, 1]
        print("%s: corr(orig, resized)=%.3f  resized range [%.0f, %.0f]"
              % (name, corr, r.min(), r.max()))

    # 全局常数缩放假设
    sx = np.median((rx1 - rx0) / (ox1 - ox0))
    sy = np.median((ry1 - ry0) / (oy1 - oy0))
    print("median 宽比 %.4f  高比 %.4f   (期望若为 912x1520 非等比缩放: %.4f / %.4f)"
          % (sx, sy, 912 / np.median(W), 1520 / np.median(H)))
    print("宽比 IQR", np.percentile((rx1 - rx0) / (ox1 - ox0), [25, 50, 75]).round(3))
    print("高比 IQR", np.percentile((ry1 - ry0) / (oy1 - oy0), [25, 50, 75]).round(3))
    print("width/height 唯一值数", len(np.unique(W)), len(np.unique(H)))

    # resized 是否在本图范围内
    print("resized 越界比例 x:%.4f y:%.4f"
          % (np.mean((rx0 < 0) | (rx1 > 912)), np.mean((ry0 < 0) | (ry1 > 1520))))
    print("resized 框宽高 中位数 %.1f x %.1f" % (np.median(rx1 - rx0), np.median(ry1 - ry0)))
    print("orig    框宽高 中位数 %.1f x %.1f" % (np.median(ox1 - ox0), np.median(oy1 - oy0)))

    # 用 912/width, 1520/height 缩放 orig 后, 与 resized 的中心距离
    cx_o = (ox0 + ox1) / 2 * 912 / W
    cy_o = (oy0 + oy1) / 2 * 1520 / H
    cx_r = (rx0 + rx1) / 2
    cy_r = (ry0 + ry1) / 2
    d = np.hypot(cx_o - cx_r, cy_o - cy_r)
    print("中心距离(1024 尺度的对角归一): 中位 %.1f px, p90 %.1f" % (np.median(d), np.percentile(d, 90)))


if __name__ == "__main__":
    main()
