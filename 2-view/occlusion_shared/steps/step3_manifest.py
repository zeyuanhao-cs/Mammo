#!/usr/bin/env python3
"""第 3 步 (CPU): 由 coords/occlusion_regions.json 生成 occlusion_manifest.json.

清单内容 (每个乳房 / 每个视角):
  - 原图文件名、绝对路径、尺寸、md5、字节数
  - lesion_boxes (原始病灶框) 与四种情况 orig / lesion / ctrl1 / ctrl2 下要遮的矩形 + 填充灰度
"""
import hashlib
import json
from pathlib import Path

OCC = Path("/nesi/project/massey04900/Mammography/occlusion")
REGIONS = OCC / "coords" / "occlusion_regions.json"
OUT = OCC / "occlusion_manifest.json"
IMG_W, IMG_H = 912, 1520


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
    data = json.load(open(REGIONS, encoding="utf-8"))
    man = {
        "meta": {
            "purpose": "遮挡实验: 检验模型对漏报病灶的'怀疑'是否来自病灶本身",
            "model": "/nesi/project/massey04900/Mammography/model/Qwen3.5-4B-lora-2view-os-sub",
            "base_model": "/nesi/project/massey04900/Mammography/model/Qwen3.5-4B",
            "views": ["CC", "MLO"],
            "conditions": ["orig", "lesion", "ctrl1", "ctrl2"],
            "rect_order": "[x0, y0, x1, y1] 整数像素, 左上开区间, 图像坐标 = 原图 912x1520 PNG 像素",
            "img_w": IMG_W, "img_h": IMG_H,
            "n_breasts": len(data["breasts"]),
            **data["meta"],
        },
        "breasts": {},
    }
    n_views = n_rects = 0
    errs = []
    for bid, e in data["breasts"].items():
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

    OUT.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")

    single = [bid for bid, e in man["breasts"].items()
              if (len(e["views"].get("CC", {}).get("lesion_boxes", [])) == 0)
              != (len(e["views"].get("MLO", {}).get("lesion_boxes", [])) == 0)]
    lines = [
        "# 第 3 步: occlusion_manifest.json",
        "",
        f"- 文件: `{OUT.name}` ({OUT.stat().st_size / 1e6:.2f} MB)",
        f"- 乳房 {len(man['breasts'])} 个, 视角 {n_views} 个, 遮挡矩形 {n_rects} 个 (lesion+ctrl1+ctrl2)",
        f"- 填充: 框外 16px 环中位灰度; 外扩 20%/边; seed {man['meta']['seed']}",
        f"- 只有一个视角有病灶框的乳房: {len(single)} 个",
        "",
        "| 情况 | 每视角矩形数 |",
        "|---|---|",
        "| orig | 0 (不遮) |",
        "| lesion | = 该视角 finding_birads>=3 框数 |",
        "| ctrl1 / ctrl2 | 与 lesion 一一对应同尺寸 |",
        "",
        "## 只有一个视角有框的乳房",
        "",
        "| id | CC 框数 | MLO 框数 |",
        "|---|---|---|",
    ]
    for bid in single:
        e = man["breasts"][bid]
        lines.append(f"| {bid} | {len(e['views'].get('CC', {}).get('lesion_boxes', []))} "
                     f"| {len(e['views'].get('MLO', {}).get('lesion_boxes', []))} |")
    (OCC / "reports" / "step3.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] {OUT} ({OUT.stat().st_size/1e6:.2f} MB), 乳房 {len(man['breasts'])}, "
          f"视角 {n_views}, 矩形 {n_rects}, 单视角乳房 {len(single)}")


if __name__ == "__main__":
    main()
