#!/usr/bin/env python3
"""第 2 步 (CPU, seed 42, 与模型无关): 生成遮挡区域.

规则
- 病灶: 每个 finding_birads>=3 框, 每边外扩 20% (宽高变 1.4 倍, 以框中心为基准), 裁到图像边界内.
- 对照: 每个对照条件 (ctrl1 / ctrl2) 与病灶框**一一对应、同样大小**; 位置在乳腺组织内随机,
  要求 (a) 框面积 >=80% 落在乳腺区域内 (Otsu 阈值), (b) 不与任何外扩后的病灶框重叠,
  (c) 尽量不与其他对照框重叠. 每个框独立采样, RNG 由 (breast, view, condition, 框序号, seed) 派生,
  因此结果与遍历顺序无关、可复现.
- 填充值: 框外 16 像素宽环的中位灰度 (在**原图**上计算).
- CC / MLO 两视角分别处理; 该视角没有 finding_birads>=3 框时该视角无遮挡 (后续单独统计).

输出: coords/occlusion_regions.json
"""
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

OCC = Path("/nesi/project/massey04900/Mammography/occlusion")
COORDS = OCC / "coords" / "breast_boxes.json"

SEED = 42
EXPAND = 0.20          # 每边外扩 20%
RING = 16              # 填充环宽度
IN_MASK_MIN = 0.80     # 对照框至少 80% 在乳腺内
MAX_TRY = 20000
IMG_W, IMG_H = 912, 1520


# ---------------------------------------------------------------- 工具
def otsu_threshold(gray):
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    omega = np.cumsum(hist)
    mu = np.cumsum(hist * np.arange(256))
    mu_t = mu[-1]
    w0 = omega
    w1 = total - omega
    valid = (w0 > 0) & (w1 > 0)
    between = np.zeros(256)
    between[valid] = ((mu_t * w0[valid] / total - mu[valid]) ** 2) / (w0[valid] * w1[valid])
    return int(np.argmax(between))


def integral(mask):
    return np.pad(mask.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))


def area_sum(ii, x0, y0, x1, y1):
    return int(ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0])


def rect_iou_free(a, b):
    inter_w = min(a[2], b[2]) - max(a[0], b[0])
    inter_h = min(a[3], b[3]) - max(a[1], b[1])
    return inter_w <= 0 or inter_h <= 0


def expand(box, img_w, img_h, frac=EXPAND):
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    w, h = (x1 - x0) * (1.0 + 2 * frac), (y1 - y0) * (1.0 + 2 * frac)
    nx0, ny0 = cx - w / 2.0, cy - h / 2.0
    nx1, ny1 = cx + w / 2.0, cy + h / 2.0
    nx0, ny0 = max(0.0, nx0), max(0.0, ny0)
    nx1, ny1 = min(float(img_w), nx1), min(float(img_h), ny1)
    return [int(round(nx0)), int(round(ny0)), int(round(nx1)), int(round(ny1))]


def ring_median(gray, rect, ring=RING):
    h, w = gray.shape
    x0, y0, x1, y1 = rect
    rx0, ry0 = max(0, x0 - ring), max(0, y0 - ring)
    rx1, ry1 = min(w, x1 + ring), min(h, y1 + ring)
    sub = gray[ry0:ry1, rx0:rx1]
    mask = np.ones_like(sub, dtype=bool)
    mask[(y0 - ry0):(y1 - ry0), (x0 - rx0):(x1 - rx0)] = False
    vals = sub[mask]
    if vals.size == 0:
        return int(np.median(gray))
    return int(np.median(vals))


def sample_control(rng, rect, ii, keep_out, others):
    """在乳腺内随机放一个与 rect 同尺寸的框."""
    w = rect[2] - rect[0]
    h = rect[3] - rect[1]
    W, H = IMG_W, IMG_H
    if w >= W or h >= H:
        return [max(0, (W - w) // 2), max(0, (H - h) // 2),
                min(W, max(0, (W - w) // 2) + w), min(H, max(0, (H - h) // 2) + h)], False
    best, best_score = None, -1.0
    for t in range(MAX_TRY):
        x0 = rng.randint(0, W - w)
        y0 = rng.randint(0, H - h)
        cand = [x0, y0, x0 + w, y0 + h]
        frac_in = area_sum(ii, x0, y0, x0 + w, y0 + h) / float(w * h)
        if frac_in < IN_MASK_MIN:
            continue
        if not all(rect_iou_free(cand, l) for l in keep_out):
            continue
        if not all(rect_iou_free(cand, c) for c in others):
            # 先记为候选, 继续找完全干净的
            if frac_in > best_score:
                best, best_score = cand, frac_in - 0.5
            continue
        return cand, True
    return (best if best is not None else rect), False


def main():
    data = json.load(open(COORDS, encoding="utf-8"))
    breasts = data["breasts"]
    out = {"meta": {"seed": SEED, "expand": EXPAND, "ring_px": RING,
                    "in_mask_min": IN_MASK_MIN, "img_w": IMG_W, "img_h": IMG_H,
                    "coord_space": "原图 PNG 像素 (912x1520)",
                    "ctrl_rule": "每个对照条件与被遮挡的病灶框一一对应、同样大小; n 个病灶框 -> 每个对照条件 n 个框",
                    "fill": "框外 16px 环的中位灰度"},
           "breasts": {}}
    stats = {"n_views": 0, "n_lesion_rects": 0, "n_ctrl_rects": 0,
             "ctrl_fail": [], "otsu": {}, "view_no_lesion": []}

    for bid, e in breasts.items():
        oute = {k: e[k] for k in ("patient_id", "laterality", "gt_birads", "gt_classes")}
        oute["views"] = {}
        for view in ("CC", "MLO"):
            v = e["views"].get(view)
            if v is None:
                continue
            stats["n_views"] += 1
            gray = np.asarray(Image.open(v["image"]).convert("L"))
            assert gray.shape == (IMG_H, IMG_W)
            thr = otsu_threshold(gray)
            stats["otsu"][f"{bid}_{view}"] = thr
            ii = integral((gray > thr).astype(np.int64))

            lesion = [expand(b["box"], IMG_W, IMG_H) for b in v["boxes"]]
            conds = {"orig": [], "lesion": lesion, "ctrl1": [], "ctrl2": []}
            fills = {"orig": [], "lesion": [], "ctrl1": [], "ctrl2": []}

            if not lesion:
                stats["view_no_lesion"].append(f"{bid}_{view}")
            for k in ("ctrl1", "ctrl2"):
                placed = []
                for i, l in enumerate(lesion):
                    rng = random.Random(f"{SEED}|{bid}|{view}|{k}|{i}")
                    cand, ok = sample_control(rng, l, ii, lesion, placed)
                    if not ok:
                        stats["ctrl_fail"].append(f"{bid}_{view}_{k}_{i}")
                    placed.append(cand)
                conds[k] = placed

            for k in ("lesion", "ctrl1", "ctrl2"):
                fills[k] = [ring_median(gray, r) for r in conds[k]]
            stats["n_lesion_rects"] += len(lesion)
            stats["n_ctrl_rects"] += len(conds["ctrl1"]) + len(conds["ctrl2"])

            oute["views"][view] = {
                "image": v["image"], "file": v["file"], "image_id": v["image_id"],
                "width": IMG_W, "height": IMG_H,
                "otsu_threshold": thr,
                "lesion_boxes": [b["box"] for b in v["boxes"]],
                "lesion_classes": [b["classes"] for b in v["boxes"]],
                "conditions": {k: [{"rect": r, "fill": f}
                                   for r, f in zip(conds[k], fills[k])]
                               for k in ("orig", "lesion", "ctrl1", "ctrl2")},
            }
        out["breasts"][bid] = oute

    (OCC / "coords" / "occlusion_regions.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    n_breast = len(out["breasts"])
    print(f"[ok] {n_breast} 乳房 / {stats['n_views']} 视角")
    print(f"     病灶矩形 {stats['n_lesion_rects']} 个, 对照矩形 {stats['n_ctrl_rects']} 个")
    print(f"     无病灶框视角 {len(stats['view_no_lesion'])} 个")
    print(f"     对照未满足全部约束(退化为最佳候选) {len(stats['ctrl_fail'])} 个")
    if stats["ctrl_fail"][:10]:
        print("     例:", stats["ctrl_fail"][:10])
    thr = list(stats["otsu"].values())
    print(f"     Otsu 阈值 min/median/max = {min(thr)}/{int(np.median(thr))}/{max(thr)}")


if __name__ == "__main__":
    main()
