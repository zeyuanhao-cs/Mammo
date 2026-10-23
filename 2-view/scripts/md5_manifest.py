#!/usr/bin/env python3
"""对 images_png 下所有 PNG 计算 md5 与尺寸, 输出 CSV(image_id,md5,width,height,mode).

image_id = 文件名去掉 .png (文件本身是内容哈希命名, 与 study 目录无关).
8 进程并行; 逐行落盘, 支持断点续跑 (已完成的 image_id 跳过).
"""
import csv
import hashlib
import os
import sys
from multiprocessing import Pool

ROOT = "/nesi/project/massey04900/Mammography/images_png"
OUT = "/home/zhao/Mammography/Mammo/2-view/outputs/images_manifest/images_manifest.csv"
NPROC = 8


def one(path):
    """返回 (image_id, md5, width, height, mode) 或 (image_id, None, err)."""
    image_id = os.path.basename(path)[:-4]
    try:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        from PIL import Image  # 仅读头部取尺寸, 不解码全图
        with Image.open(path) as im:
            w, ht = im.size
            mode = im.mode
        return (image_id, h.hexdigest(), w, ht, mode)
    except Exception as e:  # noqa: BLE001
        return (image_id, None, f"{type(e).__name__}: {e}")


def main():
    paths = []
    for dirpath, _dirnames, filenames in os.walk(ROOT):
        for fn in filenames:
            if fn.endswith(".png"):
                paths.append(os.path.join(dirpath, fn))
    paths.sort()
    print(f"共发现 {len(paths)} 个 PNG", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    done = set()
    if os.path.exists(OUT):  # 断点续跑
        with open(OUT, newline="") as f:
            for row in csv.reader(f):
                if row and row[0] != "image_id":
                    done.add(row[0])
        print(f"已完成 {len(done)}, 跳过", flush=True)

    todo = [p for p in paths if os.path.basename(p)[:-4] not in done]
    if not todo:
        print("全部完成", flush=True)
        return

    n_err = 0
    with open(OUT, "a", newline="") as f:
        w = csv.writer(f)
        if not done:
            w.writerow(["image_id", "md5", "width", "height", "mode"])
        with Pool(NPROC) as pool:
            for i, (image_id, md5, a, b, c) in enumerate(pool.imap_unordered(one, todo, chunksize=16)):
                if md5 is None:
                    n_err += 1
                    print(f"ERROR {image_id}: {a}", file=sys.stderr, flush=True)
                    continue
                w.writerow([image_id, md5, a, b, c])
                if (i + 1) % 1000 == 0:
                    f.flush()
                    print(f"{i + 1}/{len(todo)}", flush=True)
    print(f"完成: {len(todo) - n_err} 成功, {n_err} 失败 -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
