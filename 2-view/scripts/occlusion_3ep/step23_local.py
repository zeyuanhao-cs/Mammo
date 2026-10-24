#!/usr/bin/env python3
"""本地版 step2+step3 (CPU, seed 42): 遮挡区域生成 + occlusion_manifest.json.

与 NeSI step2_regions.py / step3_manifest.py 逻辑完全一致, 仅路径改本地:
  - coords/breast_boxes.json 来自 step1_local.py
  - PNG 用本地 Kaggle 版 (与 NeSI md5 不同但像素应等价, 用 Otsu/fill 对比验证)
  - manifest 里 path 指向本地原图; 另存 md5_match 报告

输出 (outputs/eval/occlusion_3ep/):
  - coords/occlusion_regions.json
  - occlusion_manifest.json
  - reports/step2.txt (与 NeSI manifest 的差异统计)
"""
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

PROJ = Path("/home/z/zhao/Mammography/Mammo/2-view")
OCC = PROJ / "outputs" / "eval" / "occlusion_3ep"
COORDS = OCC / "coords" / "breast_boxes.json"
NESI_MAN = PROJ / "occlusion_shared" / "occlusion_manifest.json"
OUT_MAN = OCC / "occlusion_manifest.json"

SEED = 42
EXPAND = 0.20
RING = 16
IN_MASK_MIN = 0.80
MAX_TRY = 20000
IMG_W, IMG_H = 912, 1520


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
            if frac_in > best_score:
                best, best_score = cand, frac_in - 0.5
            continue
        return cand, True
    return (best if best is not None else rect), False


def md5_of(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    data = json.load(open(COORDS, encoding="utf-8"))
    breasts = data["breasts"]
    out = {"meta": {"seed": SEED, "expand": EXPAND, "ring_px": RING,
                    "in_mask_min": IN_MASK_MIN, "img_w": IMG_W, "img_h": IMG_H,
                    "coord_space": "原图 PNG 像素 (912x1520)",
                    "ctrl_rule": "每个对照条件与被遮挡的病灶框一一对应、同样大小; "
                                 "n 个病灶框 -> 每个对照条件 n 个框",
                    "fill": "框外 16px 环的中位灰度"},
           "breasts": {}}
    stats = {"n_views": 0, "n_lesion_rects": 0, "n_ctrl_rects": 0,
             "ctrl_fail": [], "view_no_lesion": []}

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

    # ---- step3: 生成 manifest (本地 path + md5) ----
    man = {
        "meta": {"purpose": "遮挡实验(本地 3ep 复现): 检验模型对漏报病灶的'怀疑'是否来自病灶本身",
                 "model": "/home/z/zhao/Mammography/model/Qwen3.5-4B-lora-2view-os-sub-vt-ep3",
                 "base_model": "/home/z/zhao/Mammography/model/Qwen3.5-4B",
                 "views": ["CC", "MLO"],
                 "conditions": ["orig", "lesion", "ctrl1", "ctrl2"],
                 "rect_order": "[x0, y0, x1, y1] 整数像素, 左上开区间, "
                               "图像坐标 = 原图 912x1520 PNG 像素",
                 "img_w": IMG_W, "img_h": IMG_H,
                 "n_breasts": len(out["breasts"]),
                 **out["meta"]},
        "breasts": {},
    }
    errs = []
    n_views = n_rects = 0
    for bid, e in out["breasts"].items():
        oute = {k: e[k] for k in ("patient_id", "laterality", "gt_birads", "gt_classes")}
        oute["views"] = {}
        for view, v in e["views"].items():
            p = Path(v["image"])
            conds = v["conditions"]
            n_lesion = len(conds["lesion"])
            for k in ("ctrl1", "ctrl2"):
                if len(conds[k]) != n_lesion:
                    errs.append(f"{bid}_{view}: {k} 框数 {len(conds[k])} != lesion {n_lesion}")
            for k, items in conds.items():
                for it in items:
                    x0, y0, x1, y1 = it["rect"]
                    if not (0 <= x0 < x1 <= IMG_W and 0 <= y0 < y1 <= IMG_H):
                        errs.append(f"{bid}_{view}:{k} 矩形越界 {it['rect']}")
                    if not (0 <= it["fill"] <= 255):
                        errs.append(f"{bid}_{view}:{k} 填充值异常 {it['fill']}")
            n_views += 1
            n_rects += n_lesion + len(conds["ctrl1"]) + len(conds["ctrl2"])
            oute["views"][view] = {
                "file": v["file"],
                "path": str(p),
                "image_id": v["image_id"],
                "size": [v["width"], v["height"]],
                "md5": md5_of(p),
                "bytes": p.stat().st_size,
                "otsu_threshold": v["otsu_threshold"],
                "lesion_boxes": v["lesion_boxes"],
                "conditions": conds,
            }
        man["breasts"][bid] = oute
    if errs:
        raise SystemExit("清单校验失败:\n" + "\n".join(errs[:20]))
    OUT_MAN.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- 与 NeSI manifest 对比: rect / fill / otsu ----
    nesi = json.load(open(NESI_MAN, encoding="utf-8"))["breasts"]
    d_otsu = d_rect = d_fill = n_cmp = 0
    for bid in nesi:
        for vw in ("CC", "MLO"):
            a = man["breasts"][bid]["views"].get(vw)
            b = nesi[bid]["views"].get(vw)
            if not a or not b:
                continue
            n_cmp += 1
            if a["otsu_threshold"] != b["otsu_threshold"]:
                d_otsu += 1
            for k in ("lesion", "ctrl1", "ctrl2"):
                ra = [it["rect"] for it in a["conditions"][k]]
                rb = [it["rect"] for it in b["conditions"][k]]
                fa = [it["fill"] for it in a["conditions"][k]]
                fb = [it["fill"] for it in b["conditions"][k]]
                if ra != rb:
                    d_rect += 1
                if fa != fb:
                    d_fill += 1
    lines = [
        f"# 第 2+3 步(本地): 遮挡区域与 manifest", "",
        f"- 乳房 {len(man['breasts'])} 个, 视角 {n_views} 个, 遮挡矩形 {n_rects} 个",
        f"- 无病灶框视角 {len(stats['view_no_lesion'])} 个; "
        f"对照未满足全部约束 {len(stats['ctrl_fail'])} 个",
        f"- 与 NeSI manifest 对比 ({n_cmp} 视角):",
        f"  - lesion/ctrl rect 不一致视角: {d_rect}",
        f"  - fill 不一致视角: {d_fill}",
        f"  - Otsu 阈值不一致视角: {d_otsu}",
    ]
    (OCC / "reports" / "step2.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"[ok] {OUT_MAN}")


if __name__ == "__main__":
    main()
