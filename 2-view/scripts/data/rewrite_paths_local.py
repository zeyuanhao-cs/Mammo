#!/usr/bin/env python3
"""
把 train_llamafactory_os_sub.json 中的 NESI 图片路径重写为本地 VinDr-Mammo
(Kaggle: mdsafanurislam/vindr-mammo-official-organized-by-class) 的实际路径。

本地结构: dataset/vindr-mammo/vindr_mammo_processed/images_png/<Train|Test>/BI-RADS_x/<study_id>/<image_id>.png
原引用  : /nesi/.../images_png/<study_id>/<image_id>.png  (按 image_id 文件名全局索引)

输出:
  - data/llamafactory/train_llamafactory_os_sub_local.json (alpaca, 本地路径)
  - data/dataset_info.json 注册 vindr_2view_train_os_sub_local
"""
import json
import os
import glob
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
SRC = DATA_DIR / "llamafactory" / "train_llamafactory_os_sub.json"
DST = DATA_DIR / "llamafactory" / "train_llamafactory_os_sub_local.json"
INFO = DATA_DIR / "dataset_info.json"
IMG_ROOT = Path("/home/z/zhao/Mammography/dataset/vindr-mammo/vindr_mammo_processed/images_png")


def build_index(root: Path) -> dict:
    """image_id (hex32) -> 本地实际 png 路径"""
    idx = {}
    for p in glob.glob(str(root / "*" / "*" / "*" / "*.png")):
        name = os.path.basename(p)[:-4]
        if name not in idx:  # 同一 image 可能在多个 BI-RADS 类目录, 取第一个即可
            idx[name] = p
    return idx


def main():
    idx = build_index(IMG_ROOT)
    print(f"本地图片索引: {len(idx)} 个唯一 image_id")

    data = json.loads(SRC.read_text(encoding="utf-8"))
    print(f"训练样本: {len(data)}")

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

    if missing:
        print(f"!! 缺失 image_id: {len(missing)} 个, 样例: {missing[:5]}")
        raise SystemExit("存在缺失, 终止 (不应发生)")

    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"输出: {DST} ({len(out)} 样本, 路径全部重写为本地)")

    # 注册新数据集 (不动原有注册)
    info = json.loads(INFO.read_text(encoding="utf-8"))
    info["vindr_2view_train_os_sub_local"] = {
        "file_name": "llamafactory/train_llamafactory_os_sub_local.json",
        "formatting": "alpaca",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
            "images": "images",
        },
    }
    INFO.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print("dataset_info.json += vindr_2view_train_os_sub_local")

    # 抽查
    chk = out[0]["images"]
    for p in chk:
        assert os.path.isfile(p), p
    print("抽查通过: 首 sample 图片均存在 ->", chk)


if __name__ == "__main__":
    main()
