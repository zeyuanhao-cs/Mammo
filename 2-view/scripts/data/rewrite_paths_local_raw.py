#!/usr/bin/env python3
"""把 2-view/raw 划分文件里引用的 NeSI 图片路径重写为本地 VinDr-Mammo 路径。

与 rewrite_paths_local.py (重写 alpaca 训练文件) 同一套映射逻辑:
  本地: dataset/vindr-mammo/vindr_mammo_processed/images_png/<Train|Test>/BI-RADS_x/<study_id>/<image_id>.png
  原引用: /nesi/.../images_png/<study_id>/<image_id>.png   (按 image_id 文件名全局索引)

输出 (不改动原文件):
  data/splits/train_sub_raw_local.json
  data/splits/train_sub_repeat_os_local.json
  data/splits/val_raw_local.json
"""
import glob
import json
import os
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
RAW_DIR = THIS_DIR.parent.parent / "data" / "splits"
IMG_ROOT = Path("/home/z/zhao/Mammography/dataset/vindr-mammo/vindr_mammo_processed/images_png")

DATA_DIR = THIS_DIR.parent.parent / "data"
# (源文件, 目标文件); 全部落在 data/splits/
SRCS = [
    (RAW_DIR / "train_sub_raw.json", RAW_DIR / "train_sub_raw_local.json"),
    (RAW_DIR / "train_sub_repeat_os.json", RAW_DIR / "train_sub_repeat_os_local.json"),
    (RAW_DIR / "val_raw.json", RAW_DIR / "val_raw_local.json"),
]


def build_index(root: Path) -> dict:
    """image_id (hex32) -> 本地实际 png 路径"""
    idx = {}
    for p in glob.glob(str(root / "*" / "*" / "*" / "*.png")):
        name = os.path.basename(p)[:-4]
        if name not in idx:  # 同一 image 可能出现在多个 BI-RADS 类目录, 取第一个即可
            idx[name] = p
    return idx


def main():
    idx = build_index(IMG_ROOT)
    print(f"本地图片索引: {len(idx)} 个唯一 image_id")

    for src, dst in SRCS:
        data = json.loads(src.read_text(encoding="utf-8"))

        out, missing = [], []
        for s in data:
            imgs = []
            for p in s["images"]:
                iid = os.path.basename(p)[:-4]
                if iid in idx:
                    imgs.append(idx[iid])
                else:
                    missing.append(iid)
                    imgs.append(p)
            out.append({**s, "images": imgs})

        print(f"[{src.name}] n={len(data)}")
        if missing:
            print(f"  !! 缺失 image_id: {len(missing)} 个, 样例 {missing[:5]}")
            raise SystemExit("存在缺失, 终止")
        for s in out[:2]:
            for p in s["images"]:
                assert os.path.isfile(p), p
        dst.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        print(f"  -> {dst.name} ({len(out)} 样本, 路径已全部本地化, 抽查 2 条通过)")


if __name__ == "__main__":
    main()
