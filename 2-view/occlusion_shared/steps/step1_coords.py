#!/usr/bin/env python3
"""第 1 步 (CPU): 确定 192 个 GT BI-RADS>=3 测试乳房的病灶框, 并把 finding_birads>=3 的框
换算到模型实际读取的 PNG 像素坐标 (912x1520).

输入:
  - /nesi/.../Mammography/test.json                      (2-view 乳房级, 2000 条, images 顺序 = CC, MLO)
  - /nesi/.../Mammography/vindr_detection_v1_folds.csv   (resized_* = 912x1520 PNG 像素坐标)
  - /nesi/.../Mammography/images_png/{patient_id}/{image_id}.png

输出:
  - size_check/ep3_test_ge3_categories.csv   (重建的 192 个 id 清单, 只有 id 与 GT 元信息)
  - coords/breast_boxes.json                 (每乳房每视角的框 + GT 类别)
  - reports/step1.txt                        (统计报告)
"""
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

BASE = Path("/nesi/project/massey04900/Mammography")
OCC = BASE / "occlusion"
TEST_JSON = BASE / "test.json"
DET_CSV = BASE / "vindr_detection_v1_folds.csv"
PNG_ROOT = BASE / "images_png"

IMG_W, IMG_H = 912, 1520          # PNG 尺寸 (已用 PIL 校验, mode=L)
CLASSES = ["mass", "suspicious_calcification", "asymmetry", "architectural_distortion"]
COL2CLS = {
    "mass": "mass",
    "suspicious_calcification": "suspicious_calcification",
    "asymmetry": "asymmetry",            # Asymmetry | Focal_Asymmetry | Global_Asymmetry
    "architectural_distortion": "architectural_distortion",
}
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


def main():
    (OCC / "size_check").mkdir(parents=True, exist_ok=True)
    (OCC / "coords").mkdir(parents=True, exist_ok=True)
    (OCC / "reports").mkdir(parents=True, exist_ok=True)

    test = json.load(open(TEST_JSON, encoding="utf-8"))
    ge3 = [r for r in test if int(r["output"]["breast_birads"]) >= 3]
    assert len(ge3) == 192, f"GT BI-RADS>=3 期望 192, 实际 {len(ge3)}"

    # --- 检测 CSV: 只保留 test split; 按 (patient_id, image_id) 建索引
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
        pid = rec["patient_id"]
        lat = rec["laterality"]
        entry = {
            "id": bid,
            "patient_id": pid,
            "laterality": lat,
            "gt_birads": int(rec["output"]["breast_birads"]),
            "gt_density": rec["output"]["breast_density"],
            "views": {},
            "gt_classes": sorted({c for k, v in rec["output"]["findings"].items() if v for c in [k]}),
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

        # GT 类别: 优先取病灶框上的 finding 类别, 没有则回落到乳房级 findings
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

    # --- 输出
    (OCC / "coords" / "breast_boxes.json").write_text(
        json.dumps({"meta": {"img_w": IMG_W, "img_h": IMG_H, "coord_space": "912x1520 PNG (resized_*)",
                             "n_breasts": len(boxes_by_breast)},
                    "breasts": boxes_by_breast}, ensure_ascii=False, indent=1), encoding="utf-8")

    with open(OCC / "size_check" / "ep3_test_ge3_categories.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "study_id", "laterality", "gt_birads", "gt_density",
                    "gt_categories", "n_boxes_cc", "n_boxes_mlo"])
        for bid, e in boxes_by_breast.items():
            w.writerow([bid, e["patient_id"], e["laterality"], e["gt_birads"], e["gt_density"],
                        "|".join(e["gt_classes"]),
                        len(e["views"].get("CC", {}).get("boxes", [])),
                        len(e["views"].get("MLO", {}).get("boxes", []))])

    n_no_box = sum(1 for e in boxes_by_breast.values()
                   if all(len(v["boxes"]) == 0 for v in e["views"].values()))
    lines = []
    add = lines.append
    add("# 第 1 步: 坐标与框清单")
    add("")
    add(f"- 输入图像: `images_png/{{patient_id}}/{{image_id}}.png`, mode=L, **{IMG_W}x{IMG_H}** (PIL 校验)")
    add("- 坐标空间: `vindr_detection_v1_folds.csv` 的 `resized_*` **已经是 912x1520 PNG 像素坐标**")
    add("  (人工叠加图核对: 红色 resized 框落在病灶上, 绿色 原图坐标×缩放 偏离; 见 `_diag/`)")
    add(f"- 192 个 GT BI-RADS>=3 乳房, 共 {n_box_total} 个 finding_birads>=3 框 "
        f"(CC {per_view_boxes['CC']} / MLO {per_view_boxes['MLO']})")
    add(f"- 框越界被裁剪: {n_clipped} 个; 缺 PNG: {n_missing_png} 个")
    add(f"- finding_birads 原始取值分布 (test split): {dict(birads_vals)}")
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
    print("\n".join(lines[:20]))
    print("...")
    print(f"[ok] {len(boxes_by_breast)} 乳房, {n_box_total} 框, 单视角 {len(only_one_view)} 例, "
          f"无框 {n_no_box} 例 -> reports/step1.txt")


if __name__ == "__main__":
    main()
