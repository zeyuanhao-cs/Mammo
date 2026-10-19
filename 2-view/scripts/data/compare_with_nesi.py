#!/usr/bin/env python3
"""
跨机图片一致性对拍: 按 image_id(文件名 stem) 确定性抽样, 输出 md5 + 尺寸。
在 NeSI 与本地分别运行, 用 diff 比较输出即可判断是否为同一批预处理结果。

用法:
  python3 compare_with_nesi.py <images_png_root>

  NeSI:  python3 compare_with_nesi.py /nesi/project/massey04900/Mammography/images_png
  本地:  python3 compare_with_nesi.py /home/z/zhao/Mammography/dataset/vindr-mammo/vindr_mammo_processed/images_png

输出: 固定 10 个 image_id (按 id 排序 + seed 抽样) 的
      image_id, size(WxH), mode, md5
"""
import sys
import os
import glob
import hashlib
import random
from PIL import Image

N = 10


def collect(root: str) -> dict:
    """image_id -> 路径 (递归任意层数)"""
    idx = {}
    for p in glob.glob(os.path.join(root, "**", "*.png"), recursive=True):
        iid = os.path.basename(p)[:-4]
        idx.setdefault(iid, p)
    return idx


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    root = sys.argv[1]
    idx = collect(root)
    if not idx:
        print(f"!! {root} 下未找到 png")
        sys.exit(2)

    ids = sorted(idx)
    rng = random.Random(42)
    sample = rng.sample(ids, min(N, len(ids)))

    print(f"# root   : {root}")
    print(f"# total  : {len(ids)} unique image_id")
    print(f"# image_id                                 size           mode  md5")
    for iid in sample:
        p = idx[iid]
        h = hashlib.md5(open(p, "rb").read()).hexdigest()
        im = Image.open(p)
        print(f"{iid}  {str(im.size):>12}  {im.mode:>4}  {h}")


if __name__ == "__main__":
    main()
