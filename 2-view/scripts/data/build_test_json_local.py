#!/usr/bin/env python3
"""
从本地 VinDr-Mammo (Kaggle) 标注构建 2-view 推理用 test.json / (可选) train_raw.json。

复刻 conf_py/build_breast_level.py 的规则:
  - 按 (study_id, laterality) 分组, 合并 CC + MLO
  - images 顺序 = [CC, MLO]   (与 val_raw.json / infer_2view.py 一致)
  - breast_birads = 两视图取 max (按 BI-RADS 1..5 排序)
  - breast_density = 一致则取之; 冲突时取 CC (复刻 Counter.most_common 平票取先插入)
  - findings: 任一视图/任一标注命中即为 1
      mass                    <- 'Mass'
      suspicious_calcification<- 'Suspicious Calcification'
      asymmetry               <- 'Asymmetry' | 'Focal Asymmetry' | 'Global Asymmetry'
      architectural_distortion<- 'Architectural Distortion'
  - split 取组内值 (通常一致)

输出: 2-view/data/splits/test_local.json   (schema 与 val_raw.json 完全一致)

校验: --validate 时把本地推导标签与 val_raw.json(NeSI 同源, 1600 条) 逐条比对。
"""
import argparse
import ast
import glob
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

CSV = "/home/z/zhao/Mammography/dataset/vindr-mammo/finding_annotations.csv"
IMG_ROOT = "/home/z/zhao/Mammography/dataset/vindr-mammo/vindr_mammo_processed/images_png"
THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR.parent.parent / "data"

BIRADS_ORDER = ["BI-RADS 1", "BI-RADS 2", "BI-RADS 3", "BI-RADS 4", "BI-RADS 5"]
birads_rank = {b: i + 1 for i, b in enumerate(BIRADS_ORDER)}
FINDINGS = ["mass", "suspicious_calcification", "asymmetry", "architectural_distortion"]
ASYM = {"Asymmetry", "Focal Asymmetry", "Global Asymmetry"}


def build_image_index() -> dict:
    idx = {}
    for p in glob.glob(os.path.join(IMG_ROOT, "*", "*", "*", "*.png")):
        idx.setdefault(os.path.basename(p)[:-4], p)
    return idx


def cats(s: str) -> set:
    try:
        return set(ast.literal_eval(s))
    except Exception:
        return set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true", help="与 val_raw.json 比对标签一致性")
    ap.add_argument("--emit-train", action="store_true", help="同时输出 train_raw_local.json")
    args = ap.parse_args()

    idx = build_image_index()
    print(f"[img] 本地 image_id 索引: {len(idx)}")
    df = pd.read_csv(CSV, low_memory=False)
    df["_cats"] = df["finding_categories"].map(cats)

    samples, miss, conflicts = [], [], defaultdict(int)
    for (pid, lat), g in df.groupby(["study_id", "laterality"]):
        img_rows = g.drop_duplicates(subset="image_id")
        cc = img_rows[img_rows["view_position"] == "CC"]
        mlo = img_rows[img_rows["view_position"] == "MLO"]
        cc_row = cc.iloc[0] if len(cc) else None
        mlo_row = mlo.iloc[0] if len(mlo) else None

        imgs = []
        for r in (cc_row, mlo_row):
            if r is None:
                imgs.append(None)
            else:
                p = idx.get(r["image_id"])
                imgs.append(p)
        if imgs[0] is None or imgs[1] is None:
            miss.append((pid, lat, imgs[0] is None, imgs[1] is None))

        bvals = [r["breast_birads"] for r in (cc_row, mlo_row) if r is not None]
        if len(bvals) > 1 and bvals[0] != bvals[1]:
            conflicts["birads"] += 1
        breast_birads = max(bvals, key=lambda b: birads_rank.get(b, 0))

        dvals = [r["breast_density"] for r in (cc_row, mlo_row) if r is not None]
        if len(dvals) > 1 and dvals[0] != dvals[1]:
            conflicts["density"] += 1
            breast_density = Counter(dvals).most_common(1)[0][0]
        else:
            breast_density = dvals[0]

        ac = set().union(*g["_cats"].tolist())
        findings = {
            "mass": int("Mass" in ac),
            "suspicious_calcification": int("Suspicious Calcification" in ac),
            "asymmetry": int(bool(ac & ASYM)),
            "architectural_distortion": int("Architectural Distortion" in ac),
        }

        splits = g["split"].unique().tolist()
        if len(splits) > 1:
            conflicts["split"] += 1
        split = splits[0]

        samples.append({
            "id": f"{pid}_{lat}",
            "patient_id": pid,
            "laterality": lat,
            "images": imgs,
            "output": {
                "breast_birads": int(birads_rank.get(breast_birads, 0)),
                "breast_density": breast_density.split()[-1],
                "findings": findings,
            },
            "split": split,
        })

    n = len(samples)
    n_ok = sum(1 for s in samples if s["images"][0] and s["images"][1])
    print(f"[build] 乳房级样本: {n}  (CC+MLO 齐全: {n_ok}, 缺视图: {len(miss)})")
    print(f"[build] 冲突统计: {dict(conflicts)}")
    for sp in ("training", "test"):
        k = [s for s in samples if s["split"] == sp]
        print(f"  split={sp}: {len(k)}")

    # ---- 校验: 与 val_raw.json 比标签 ----
    if args.validate:
        val = json.loads((DATA_DIR / "splits" / "val_raw.json").read_text(encoding="utf-8"))
        mine = {s["id"]: s for s in samples}
        diff = defaultdict(int)
        n_cmp = 0
        for v in val:
            m = mine.get(v["id"])
            if m is None:
                diff["id_not_found"] += 1
                continue
            n_cmp += 1
            if m["output"]["breast_birads"] != v["output"]["breast_birads"]:
                diff["birads"] += 1
            if m["output"]["breast_density"] != v["output"]["breast_density"]:
                diff["density"] += 1
            for f in FINDINGS:
                if m["output"]["findings"][f] != int(v["output"]["findings"][f]):
                    diff[f"finding:{f}"] += 1
        print(f"\n[validate] 与 val_raw.json 比对 {n_cmp}/{len(val)} 条")
        if not diff:
            print("  ✅ 标签完全一致 (birads / density / 4 findings 全对)")
        else:
            print(f"  !! 差异: {dict(diff)}")

    # ---- 写 test ----
    test = [s for s in samples if s["split"] == "test" and s["images"][0] and s["images"][1]]
    test = [{"id": s["id"], "images": s["images"], "output": s["output"]} for s in test]
    out = DATA_DIR / "splits" / "test_local.json"
    out.write_text(json.dumps(test, ensure_ascii=False), encoding="utf-8")
    print(f"\n[out] {out}  ({len(test)} 条)")

    if args.emit_train:
        tr = [s for s in samples if s["split"] == "training" and s["images"][0] and s["images"][1]]
        tpath = DATA_DIR / "raw" / "train_raw_local.json"
        tpath.write_text(json.dumps(tr, ensure_ascii=False), encoding="utf-8")
        print(f"[out] {tpath}  ({len(tr)} 条)")


if __name__ == "__main__":
    main()
