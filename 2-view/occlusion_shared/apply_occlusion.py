#!/usr/bin/env python3
"""apply_occlusion.py — 按 occlusion_manifest.json 把遮挡画到原图上.

只依赖 numpy + PIL, 结果完全确定 (占位矩形与填充灰度都写在清单里, 与本脚本的时间/环境无关).

用法
  python apply_occlusion.py --manifest occlusion_manifest.json \
      --condition lesion --out-dir occluded/lesion

  # 指定乳房 / 视角
  python apply_occlusion.py --manifest occlusion_manifest.json \
      --condition ctrl1 --out-dir occluded/ctrl1 \
      --ids 02dba952e4ef2cf35613761a64bbc751_R 043febde473009753d404fe89a41c395_R

  # orig 只拷贝不修改 (对照"没遮挡"的基线); --skip-orig 可跳过
  python apply_occlusion.py --manifest occlusion_manifest.json \
      --condition orig --out-dir occluded/orig

输出
  {out_dir}/{breast_id}__{view}.png   与原图同尺寸、同灰度 (8-bit L), 无附加元数据.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

CONDITIONS = ("orig", "lesion", "ctrl1", "ctrl2")


def load_manifest(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def apply_rects(gray, rects, fills=None):
    """把 rects 区域填成 fills (缺省为 0). 返回新数组, 不改原数组."""
    out = np.array(gray, copy=True)
    h, w = out.shape[:2]
    for i, r in enumerate(rects):
        x0, y0, x1, y1 = [int(round(v)) for v in r]
        x0, y0 = max(0, min(x0, w)), max(0, min(y0, h))
        x1, y1 = max(0, min(x1, w)), max(0, min(y1, h))
        if x1 <= x0 or y1 <= y0:
            continue
        val = 0 if fills is None else int(fills[i])
        out[y0:y1, x0:x1] = val
    return out


def image_path(view_entry):
    """清单里图像路径的键名: manifest 用 'path', 中间产物 regions 用 'image'."""
    return view_entry.get("path") or view_entry["image"]


def occlude_view(view_entry, condition):
    """返回遮挡后的 uint8 数组 (L)."""
    with Image.open(image_path(view_entry)) as im:
        gray = np.asarray(im.convert("L"))
    items = view_entry["conditions"].get(condition) or []
    rects = [it["rect"] for it in items]
    fills = [it["fill"] for it in items]
    return apply_rects(gray, rects, fills)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--condition", required=True, choices=CONDITIONS)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ids", nargs="*", default=None, help="只处理这些乳房 id (默认全部)")
    ap.add_argument("--views", nargs="*", default=["CC", "MLO"])
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    man = load_manifest(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = set(args.ids) if args.ids else None

    n = 0
    skipped = 0
    for bid, breast in man["breasts"].items():
        if keep is not None and bid not in keep:
            continue
        for view in args.views:
            v = breast["views"].get(view)
            if v is None:
                continue
            dst = out_dir / f"{bid}__{view}.png"
            if dst.exists() and not args.overwrite:
                skipped += 1
                continue
            if args.condition == "orig":
                shutil.copyfile(v["image"], dst)   # 基线: 原图逐字节拷贝
            else:
                arr = occlude_view(v, args.condition)
                Image.fromarray(arr, mode="L").save(dst, format="PNG")
            n += 1
    print(f"[ok] condition={args.condition} written={n} skipped={skipped} -> {out_dir}")


if __name__ == "__main__":
    main()
