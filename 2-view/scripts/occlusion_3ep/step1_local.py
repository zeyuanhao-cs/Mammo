#!/usr/bin/env python3
"""本地 A40 版 step1: 192 个 GT BI-RADS>=3 测试乳房的 finding_birads>=3 框,
换算坐标用 vindr_detection_v1_folds.csv 的 resized_* (已是 912x1520 PNG 像素).

输入:
  - Mammo/2-view/data/splits/test_local.json           (2-view 乳房级, 2000 条)
  - Mammo/2-view/occlusion_shared/vindr_detection_v1_folds.csv
  - 本地 PNG: dataset/vindr-mammo/vindr_mammo_processed/images_png/Test/BI-RADS_*/{pid}/{iid}.png
    (路径直接取 test_local.json 的 images, 不重新拼)

输出 (outputs/eval/occlusion_3ep/):
  - coords/breast_boxes.json
  - size_check/ep3_test_ge3_categories.csv
  - reports/step1.txt
  - check/ov_*.png  (6 张叠加验证图: 红=resized_*, 绿=orig*宽高比)
"""
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

PROJ = Path("/home/z/zhao/Mammography/Mammo/2-view")
TEST_JSON = PROJ / "data" / "splits" / "test_local.json"
DET_CSV = PROJ / "occlusion_shared" / "vindr_detection_v1_folds.csv"
OCC = PROJ / "outputs" / "eval" / "occlusion_3ep"

IMG_W, IMG_H = 912, 1520
BIRADS_RE = re.compile(r"(\d+)")


def parse_birads(s):
    if s is None:
        return None
    m = BIRADS_RE.search(str(s))
    return int(m.group(1)) if m else None


def row_classes(r):
    out = set()
    if r["Mass"] == "1":
        out.add("mass")
    if r["Suspicious_Calcification"] == "1":
        out.add("suspicious_calcification")
    if r["Asymmetry"] == "1" or r["Focal_Asymmetry"] == "1" or r["Global_Asymmetry"] == "1":
        out.add("asymmetry")
    if r["Architectural_Distortion"] == "1":
        out.add("architectural_distortion")
    return out


def pick_overlay_samples(boxes_by_breast):
    """选 6 个 (breast, view) 画框: 覆盖多框、四个类别、CC/MLO."""
    cand = []
    for bid, e in boxes_by_breast.items():
        for view, v in e["views"].items():
            if not v["boxes"]:
                continue
            cls = set()
            for b in v["boxes"]:
                cls.update(b["classes"])
            cand.append((bid, view, len(v["boxes"]), cls))
    picked, used = [], set()
    def take(pred, n=1):
        for c in cand:
            if len(picked) >= 6:
                break
            if (c[0], c[1]) in used or not pred(c):
                continue
            picked.append(c); used.add((c[0], c[1]))
    take(lambda c: c[2] >= 2, 1)                                   # 多框
    for cl in ("suspicious_calcification", "asymmetry",
               "architectural_distortion", "mass"):
        take(lambda c, cl=cl: cl in c[3], 1)                       # 每类一张
    take(lambda c: c[1] == "MLO", 1)                               # 补 MLO
    take(lambda c: True)                                           # 任意补齐
    return picked[:6]


def main():
    for sub in ("coords", "size_check", "reports", "check"):
        (OCC / sub).mkdir(parents=True, exist_ok=True)

    test = json.load(open(TEST_JSON, encoding="utf-8"))
    ge3 = [r for r in test if int(r["output"]["breast_birads"]) >= 3]
    assert len(ge3) == 192, f"GT BI-RADS>=3 期望 192, 实际 {len(ge3)}"

    det = defaultdict(list)
    birads_vals = Counter()
    with open(DET_CSV, newline="") as f:
        for r in csv.DictReader(f):
            if r["split"] != "test":
                continue
            birads_vals[r["finding_birads"]] += 1
            det[(r["patient_id"], r["image_id"])].append(r)

    boxes_by_breast = {}
    only_one_view = []
    n_box_total = 0
    n_clipped = 0
    n_missing_png = 0
    per_view_boxes = Counter()

    for rec in ge3:
        bid = rec["id"]
        pid, lat = bid.rsplit("_", 1)   # id 形如 {study_id}_{L|R}
        entry = {
            "id": bid,
            "patient_id": pid,
            "laterality": lat,
            "gt_birads": int(rec["output"]["breast_birads"]),
            "gt_density": rec["output"]["breast_density"],
            "views": {},
            "gt_classes": sorted(k for k, v in rec["output"]["findings"].items() if v),
        }
        for path in rec["images"]:
            p = Path(path)
            image_id = p.name
            rows = det.get((pid, image_id), [])
            view = rows[0]["view"] if rows else None
            assert view is not None, f"CSV 缺 view: {pid}/{image_id}"
            if not p.is_file():
                n_missing_png += 1
            with Image.open(p) as im:
                img_w, img_h = im.size
            assert (img_w, img_h) == (IMG_W, IMG_H), f"PNG 尺寸异常 {p}: {img_w}x{img_h}"

            boxes = []
            for r in rows:
                fb = parse_birads(r["finding_birads"])
                if fb is None or fb < 3:
                    continue
                x0, y0 = float(r["resized_xmin"]), float(r["resized_ymin"])
                x1, y1 = float(r["resized_xmax"]), float(r["resized_ymax"])
                cx0, cy0 = max(0.0, min(x0, IMG_W)), max(0.0, min(y0, IMG_H))
                cx1, cy1 = max(0.0, min(x1, IMG_W)), max(0.0, min(y1, IMG_H))
                if (cx0, cy0, cx1, cy1) != (x0, y0, x1, y1):
                    n_clipped += 1
                if cx1 - cx0 < 2 or cy1 - cy0 < 2:
                    continue
                boxes.append({
                    "box": [round(v, 2) for v in (cx0, cy0, cx1, cy1)],
                    "box_raw": [round(v, 2) for v in (x0, y0, x1, y1)],
                    "finding_birads": fb,
                    "classes": sorted(row_classes(r)),
                    "image_id": image_id,
                })
            boxes.sort(key=lambda b: (b["box"][1], b["box"][0]))
            entry["views"][view] = {
                "image": str(p),
                "image_id": image_id,
                "file": f"{pid}/{image_id}.png",
                "width": img_w,
                "height": img_h,
                "boxes": boxes,
            }
            per_view_boxes[view] += len(boxes)
            n_box_total += len(boxes)

        cls = set()
        for v in entry["views"].values():
            for b in v["boxes"]:
                cls.update(b["classes"])
        entry["gt_classes"] = sorted(cls) if cls else entry["gt_classes"]

        n_cc = len(entry["views"].get("CC", {}).get("boxes", []))
        n_mlo = len(entry["views"].get("MLO", {}).get("boxes", []))
        if (n_cc == 0) != (n_mlo == 0):
            only_one_view.append({"id": bid, "CC": n_cc, "MLO": n_mlo})
        boxes_by_breast[bid] = entry

    (OCC / "coords" / "breast_boxes.json").write_text(
        json.dumps({"meta": {"img_w": IMG_W, "img_h": IMG_H,
                             "coord_space": "912x1520 PNG (resized_*)",
                             "n_breasts": len(boxes_by_breast)},
                    "breasts": boxes_by_breast}, ensure_ascii=False, indent=1),
        encoding="utf-8")

    with open(OCC / "size_check" / "ep3_test_ge3_categories.csv", "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "study_id", "laterality", "gt_birads", "gt_density",
                    "gt_categories", "n_boxes_cc", "n_boxes_mlo"])
        for bid, e in boxes_by_breast.items():
            w.writerow([bid, e["patient_id"], e["laterality"], e["gt_birads"],
                        e["gt_density"], "|".join(e["gt_classes"]),
                        len(e["views"].get("CC", {}).get("boxes", [])),
                        len(e["views"].get("MLO", {}).get("boxes", []))])

    # ---- 叠加验证图: 红 = resized_*(正确), 绿 = orig*宽高比(预期偏离) ----
    for i, (bid, view, nb, cls) in enumerate(pick_overlay_samples(boxes_by_breast)):
        v = boxes_by_breast[bid]["views"][view]
        rows = det[(boxes_by_breast[bid]["patient_id"], v["image_id"])]
        im = Image.open(v["image"]).convert("L").resize((IMG_W // 2, IMG_H // 2))
        im = im.convert("RGB")
        d = ImageDraw.Draw(im)
        s = 0.5
        wx, wh = IMG_W / float(rows[0]["width"]), IMG_H / float(rows[0]["height"])
        for r in rows:
            fb = parse_birads(r["finding_birads"])
            if fb is None or fb < 3:
                continue
            d.rectangle([float(r["resized_xmin"]) * s, float(r["resized_ymin"]) * s,
                         float(r["resized_xmax"]) * s, float(r["resized_ymax"]) * s],
                        outline=(255, 0, 0), width=3)
            d.rectangle([float(r["xmin"]) * wx * s, float(r["ymin"]) * wh * s,
                         float(r["xmax"]) * wx * s, float(r["ymax"]) * wh * s],
                        outline=(0, 255, 0), width=3)
        dst = OCC / "check" / f"ov{i}_{bid}_{view}.png"
        im.save(dst)
        print(f"[overlay] {dst.name}  n_boxes={nb} classes={','.join(sorted(cls))}")

    n_no_box = sum(1 for e in boxes_by_breast.values()
                   if all(len(v["boxes"]) == 0 for v in e["views"].values()))
    lines = []
    add = lines.append
    add("# 第 1 步(本地): 坐标与框清单")
    add("")
    add(f"- 坐标空间: `vindr_detection_v1_folds.csv` 的 `resized_*` 已是 {IMG_W}x{IMG_H} PNG 像素坐标")
    add(f"- 192 个 GT BI-RADS>=3 乳房, 共 {n_box_total} 个 finding_birads>=3 框 "
        f"(CC {per_view_boxes['CC']} / MLO {per_view_boxes['MLO']})")
    add(f"- 框越界被裁剪: {n_clipped} 个; 缺 PNG: {n_missing_png} 个")
    add(f"- finding_birads 取值分布 (test split): {dict(birads_vals)}")
    add(f"- 两个视角都没有框的乳房: {n_no_box}")
    add(f"- **只有一个视角有框的乳房: {len(only_one_view)}**")
    add("")
    if only_one_view:
        add("| id | CC 框数 | MLO 框数 |")
        add("|---|---|---|")
        for x in only_one_view:
            add(f"| {x['id']} | {x['CC']} | {x['MLO']} |")
        add("")
    add("## 每个乳房的框明细 (仅列有框的视角)")
    add("")
    add("| id | GT BI-RADS | view | 框 (x0,y0,x1,y1) | 框内 finding_birads | classes |")
    add("|---|---|---|---|---|---|")
    for bid, e in boxes_by_breast.items():
        for view in ("CC", "MLO"):
            v = e["views"].get(view)
            if not v:
                continue
            for b in v["boxes"]:
                add(f"| {bid} | {e['gt_birads']} | {view} | "
                    f"({b['box'][0]:.0f},{b['box'][1]:.0f},{b['box'][2]:.0f},{b['box'][3]:.0f}) | "
                    f"{b['finding_birads']} | {','.join(b['classes'])} |")
    (OCC / "reports" / "step1.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:10]))
    print(f"[ok] {len(boxes_by_breast)} 乳房, {n_box_total} 框, 单视角 {len(only_one_view)} 例, "
          f"无框 {n_no_box} 例 -> reports/step1.txt")


if __name__ == "__main__":
    main()
