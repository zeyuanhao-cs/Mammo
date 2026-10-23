#!/usr/bin/env python3
"""构造 patch-zoom 局部块数据（只做数据 + 统计，不训练）.

输入:
  - vindr_detection_v1_folds.csv (resized_* 坐标已是 912x1520 PNG 空间)
  - /nesi/.../images_png/{patient_id}/{image_id}.png
  - split_study_ids.json: train_studies(3200) / val_studies(800)
  - breath_level_vindr.json: birads / density / split

规则（与任务约定一致）:
  阳性块: 每个合法病灶框 1 块, 224x224, 框中心加随机偏移(seed 42, 偏移后仍包含框中心);
          框宽或高 > 224 的记为 excluded_big, 不作阳性。
          目标类别 multi-hot: mass / suspicious_calcification /
          asymmetry(Asymmetry|Focal_Asymmetry|Global_Asymmetry) / architectural_distortion。
  阴性块: 数量 = 2 x 阳性。一半来自 BI-RADS 1 正常乳房乳腺区域, 一半来自有病灶乳房中
          与所有病灶框都不重叠的区域。像素均值 > MEAN_THRESH 视为乳腺区域。
  划分: 训练块只来自 train_studies(不含 val), 测试块只来自官方 test。

输出 (OUT_DIR):
  images/{train,test}/{p,n}*.png
  meta_train.jsonl / meta_test.jsonl
  stats.json
  lf/train_llamafactory_patch.json + lf/dataset_info.json
"""
import argparse
import csv
import json
import math
import os
import random
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

PNG_ROOT = Path("/nesi/project/massey04900/Mammography/images_png")
CSV_PATH = Path("/nesi/project/massey04900/Mammography/vindr_detection_v1_folds.csv")
BREAST_JSON = Path("/nesi/project/massey04900/Mammography/breast_level_vindr.json")
SPLIT_JSON = Path("/home/zhao/Mammography/Mammo/2-view/data/splits/split_study_ids.json")
OUT_DIR = Path("/nesi/project/massey04900/Mammography/patch_zoom")

PATCH = 224
MEAN_THRESH = 80.0
SEED = 42
IMG_W, IMG_H = 912, 1520
CLASSES = ["mass", "suspicious_calcification", "asymmetry", "architectural_distortion"]


def png_path(pid, image_id):
    """CSV 的 image_id 自带 .png, breast_level json 的路径也带 .png."""
    name = image_id if str(image_id).endswith(".png") else f"{image_id}.png"
    return PNG_ROOT / pid / name


def row_classes(r):
    return {
        "mass": r["Mass"] == "1",
        "suspicious_calcification": r["Suspicious_Calcification"] == "1",
        "asymmetry": (r["Asymmetry"] == "1" or r["Focal_Asymmetry"] == "1"
                      or r["Global_Asymmetry"] == "1"),
        "architectural_distortion": r["Architectural_Distortion"] == "1",
    }


def box_of(r):
    return (float(r["resized_xmin"]), float(r["resized_ymin"]),
            float(r["resized_xmax"]), float(r["resized_ymax"]))


def intersects(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def load_gray(path):
    return np.asarray(Image.open(path).convert("L"))


def patch_mean(arr, x0, y0):
    return float(arr[y0:y0 + PATCH, x0:x0 + PATCH].mean())


def clamp_crop(cx, cy):
    x0 = int(min(max(round(cx - PATCH / 2), 0), IMG_W - PATCH))
    y0 = int(min(max(round(cy - PATCH / 2), 0), IMG_H - PATCH))
    return x0, y0


def save_patch(arr, x0, y0, dst):
    Image.fromarray(arr[y0:y0 + PATCH, x0:x0 + PATCH]).save(dst)


# ------------------------------------------------------------------ positive
def plan_positives(rows, studies, split_name, rng):
    """返回 (tasks, stats)。tasks: dict(需在子进程里裁图的全部信息)。"""
    tasks, excluded_big, excluded_other = [], 0, 0
    big_by_class = Counter()
    for r in rows:
        if r["split"] != ("test" if split_name == "test" else "training"):
            continue
        pid = r["patient_id"]
        if split_name != "test" and pid not in studies:
            continue
        cls = row_classes(r)
        if not any(cls.values()):
            excluded_other += 1
            continue
        xmin, ymin, xmax, ymax = box_of(r)
        w, h = xmax - xmin, ymax - ymin
        if w > PATCH or h > PATCH:
            excluded_big += 1
            for k, v in cls.items():
                if v:
                    big_by_class[k] += 1
            continue
        cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
        # 偏移后仍包含框中心: |dx|,|dy| <= 112
        dx = rng.uniform(-(PATCH / 2 - 1), PATCH / 2 - 1)
        dy = rng.uniform(-(PATCH / 2 - 1), PATCH / 2 - 1)
        x0, y0 = clamp_crop(cx + dx, cy + dy)
        tasks.append({
            "kind": "pos",
            "split": split_name,
            "src": str(png_path(pid, r["image_id"])),
            "breast": f"{pid}_{r['laterality']}",
            "patient": pid,
            "view": r["view"],
            "box": [xmin, ymin, xmax, ymax],
            "crop": [x0, y0],
            "labels": {k: int(cls[k]) for k in CLASSES},
            "area_ratio": (w * h) / float(IMG_W * IMG_H),
        })
    stats = {"n_pos": len(tasks), "excluded_big": excluded_big,
             "excluded_other_findings": excluded_other,
             "excluded_big_by_class": dict(big_by_class)}
    return tasks, stats


# ------------------------------------------------------------------ negative
def plan_negatives(rows, studies, split_name, n_neg, rng):
    """一半来自 BI-RADS 1 正常乳房, 一半来自有病灶乳房的无病灶区域."""
    # image -> boxes (所有病灶框, 含 >224 的, 用于非重叠判定)
    img_boxes = defaultdict(list)
    lesion_breasts = set()
    for r in rows:
        pid = r["patient_id"]
        if split_name != "test" and pid not in studies:
            continue
        if not any(row_classes(r).values()):
            continue
        lesion_breasts.add(f"{pid}_{r['laterality']}")

    # breast -> 该乳房所有图 (来自 breast_level_vindr.json, 与 split 过滤一致)
    breast_imgs, normal_breasts = {}, []
    for rec in json.load(open(BREAST_JSON, encoding="utf-8")):
        pid = rec["patient_id"]
        if split_name == "test":
            if rec.get("split") != "test":
                continue
        else:
            if pid not in studies:
                continue
        b = f"{pid}_{rec['laterality']}"
        imgs = [(pid, Path(p).stem) for p in rec["images"].values() if p]
        if not imgs:
            continue
        breast_imgs[b] = imgs
        if rec.get("breast_birads") == "BI-RADS 1":
            normal_breasts.append((b, imgs))
    lesion_breasts = sorted(b for b in lesion_breasts if b in breast_imgs)

    n_norm = n_neg // 2
    n_les = n_neg - n_norm
    tasks = []
    fails = Counter()

    def sample_from(breast_imgs_pool, other_boxes_fn, source, need):
        out = []
        tries = 0
        while len(out) < need and tries < need * 400:
            tries += 1
            b, imgs = breast_imgs_pool[rng.randrange(len(breast_imgs_pool))]
            pid, iid = imgs[rng.randrange(len(imgs))]
            arr = load_gray(png_path(pid, iid))
            x0 = rng.randrange(0, IMG_W - PATCH + 1)
            y0 = rng.randrange(0, IMG_H - PATCH + 1)
            crop = (x0, y0, x0 + PATCH, y0 + PATCH)
            if any(intersects(crop, bx) for bx in other_boxes_fn(iid)):
                continue
            if patch_mean(arr, x0, y0) <= MEAN_THRESH:
                continue
            out.append({
                "kind": "neg", "split": split_name,
                "src": str(png_path(pid, iid)),
                "breast": b,
                "patient": pid, "view": _view_of(iid, rows),
                "box": None, "crop": [x0, y0],
                "labels": {k: 0 for k in CLASSES},
                "area_ratio": None, "neg_source": source,
            })
        fails[source] += tries - len(out)
        return out

    tasks += sample_from(normal_breasts, lambda iid: [], "normal_birads1", n_norm)
    tasks += sample_from(
        [(b, breast_imgs[b]) for b in lesion_breasts],
        lambda iid: img_boxes.get(iid, []), "lesion_breast_nonoverlap", n_les)
    return tasks, {k: int(v) for k, v in fails.items()}


_VIEW = {}


def _view_of(image_id, rows):
    if not _VIEW:
        for r in rows:
            _VIEW[r["image_id"]] = r["view"]
    return _VIEW.get(image_id, "")


# ------------------------------------------------------------------ worker
def run_task(t):
    dst = Path(t["dst"])
    x0, y0 = t["crop"]
    arr = load_gray(t["src"])
    save_patch(arr, x0, y0, dst)
    t["patch_mean"] = round(patch_mean(arr, x0, y0), 3)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="smoke: 每类最多取 N 块")
    args = ap.parse_args()

    out = Path(args.out_dir)
    rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8")))
    sp = json.load(open(SPLIT_JSON, encoding="utf-8"))
    train_studies = set(sp["train_studies"])
    val_studies = set(sp["val_studies"])
    assert not (train_studies & val_studies)

    all_stats, all_tasks = {}, {}
    for split_name, studies in (("train", train_studies), ("test", set())):
        rng = random.Random(SEED)
        pos, st = plan_positives(rows, studies, split_name, rng)
        n_neg = 2 * len(pos)
        neg, fails = plan_negatives(rows, studies, split_name, n_neg, rng)
        st.update({"n_neg": len(neg), "n_neg_planned": n_neg,
                   "neg_sample_fails": fails})
        tasks = pos + neg
        if args.limit > 0:  # smoke: 阳/阴各取 N 块
            tasks = ([t for t in pos if t["kind"] == "pos"][: args.limit]
                     + [t for t in neg if t["kind"] == "neg"][: args.limit])
        all_stats[split_name] = st
        all_tasks[split_name] = tasks

    # 分配输出文件名
    for split_name, tasks in all_tasks.items():
        (out / "images" / split_name).mkdir(parents=True, exist_ok=True)
        for i, t in enumerate(tasks):
            tag = "p" if t["kind"] == "pos" else "n"
            pfx = "tr" if split_name == "train" else "te"
            t["id"] = f"{pfx}_{tag}{i:06d}"
            t["dst"] = str(out / "images" / split_name / f"{t['id']}.png")

    order = []
    for split_name, tasks in all_tasks.items():
        order.extend(tasks)
    with Pool(args.workers) as pool:
        done = list(pool.imap_unordered(run_task, order, chunksize=32))
    by_id = {t["id"]: t for t in done}

    for split_name, tasks in all_tasks.items():
        tasks = sorted((by_id[t["id"]] for t in tasks), key=lambda t: t["id"])
        with open(out / f"meta_{split_name}.jsonl", "w", encoding="utf-8") as f:
            for t in tasks:
                f.write(json.dumps({k: t[k] for k in (
                    "id", "kind", "breast", "patient", "view", "box", "crop",
                    "labels", "area_ratio", "neg_source", "patch_mean", "src"
                ) if k in t}, ensure_ascii=False) + "\n")
        all_tasks[split_name] = tasks

    # ---------------- stats ----------------
    def cls_counts(tasks):
        c = Counter()
        for t in tasks:
            if t["kind"] != "pos":
                continue
            for k, v in t["labels"].items():
                c[k] += v
        return dict(c)

    for split_name in all_stats:
        tasks = all_tasks[split_name]
        st = all_stats[split_name]
        pos = [t for t in tasks if t["kind"] == "pos"]
        neg = [t for t in tasks if t["kind"] == "neg"]
        st["n_pos_kept"] = len(pos)
        st["n_neg_kept"] = len(neg)
        st["pos_by_class"] = cls_counts(pos)
        st["neg_by_source"] = dict(Counter(t["neg_source"] for t in neg))
        st["n_breasts_pos"] = len(set(t["breast"] for t in pos))
        st["n_breasts_neg"] = len(set(t["breast"] for t in neg))
        areas = sorted(t["area_ratio"] for t in pos)
        if areas:
            st["area_ratio_quantiles"] = {
                "p25": float(np.quantile(areas, 0.25)),
                "p50": float(np.quantile(areas, 0.50)),
                "p75": float(np.quantile(areas, 0.75)),
            }

    # 面积三分位来自 train 阳性块
    tr_areas = [t["area_ratio"] for t in all_tasks["train"] if t["kind"] == "pos"]
    te_areas = [t["area_ratio"] for t in all_tasks["test"] if t["kind"] == "pos"]
    q1, q2 = (float(np.quantile(tr_areas, 1 / 3)), float(np.quantile(tr_areas, 2 / 3)))
    tert = {"q1": q1, "q2": q2}
    for name, arr in (("train", tr_areas), ("test", te_areas)):
        tert[f"{name}_counts"] = [
            int(sum(1 for a in arr if a <= q1)),
            int(sum(1 for a in arr if q1 < a <= q2)),
            int(sum(1 for a in arr if a > q2)),
        ]
    all_stats["area_tertiles"] = tert

    # 交叉检查: 同一乳房不跨集合
    tr_b = set(t["breast"] for t in all_tasks["train"])
    te_b = set(t["breast"] for t in all_tasks["test"])
    all_stats["breast_overlap_train_test"] = len(tr_b & te_b)
    all_stats["n_train_patches"] = len(all_tasks["train"])
    all_stats["n_test_patches"] = len(all_tasks["test"])

    with open(out / "stats.json", "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)

    # ---------------- LLaMA-Factory 训练文件 ----------------
    lf = out / "lf"
    lf.mkdir(parents=True, exist_ok=True)
    instruction = PATCH_INSTRUCTION
    recs = []
    for t in all_tasks["train"]:
        recs.append({
            "instruction": instruction,
            "input": "",
            "output": json.dumps(t["labels"], ensure_ascii=False),
            "images": [str(out / "images" / "train" / f"{t['id']}.png")],
        })
    with open(lf / "train_llamafactory_patch.json", "w", encoding="utf-8") as f:
        json.dump(recs, f, ensure_ascii=False, indent=1)
    with open(lf / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump({
            "patch_zoom_train": {
                "file_name": "train_llamafactory_patch.json",
                "formatting": "alpaca",
                "columns": {"prompt": "instruction", "query": "input",
                            "response": "output", "images": "images"},
            }
        }, f, ensure_ascii=False, indent=2)

    print(json.dumps(all_stats, ensure_ascii=False, indent=2))
    print("output:", out)


PATCH_INSTRUCTION = """<image>
You are an expert breast imaging assistant.

Below is ONE 224x224 local patch cropped from a mammogram. The patch may
contain a lesion, or may show normal breast tissue.

Mark 1 only for suspicious findings that are visible in this patch and
warrant follow-up; do not mark clearly benign findings.

Return ONLY this JSON, with integer 0/1 values and no other text:
{"mass": <0|1>, "suspicious_calcification": <0|1>,
 "asymmetry": <0|1>, "architectural_distortion": <0|1>}"""


if __name__ == "__main__":
    main()
