#!/usr/bin/env python3
"""作业开头硬校验: occluded/{lesion,ctrl1,ctrl2} 各 384 张, 共 1152 张,
且每张都能被 PIL 打开、尺寸 912x1520. 不满足 -> exit 1.
"""
import json
import sys
from pathlib import Path

from PIL import Image

OCC = Path("/nesi/project/massey04900/Mammography/occlusion")
MANIFEST = OCC / "occlusion_manifest.json"
CONDS = ("lesion", "ctrl1", "ctrl2")
IMG_W, IMG_H = 912, 1520
EXPECT_PER_COND = 384
EXPECT_TOTAL = 1152


def main():
    man = json.load(open(MANIFEST, encoding="utf-8"))
    expect = set()
    for bid, e in man["breasts"].items():
        for view in e["views"]:
            expect.add(f"{bid}__{view}.png")
    print(f"[check] manifest 期望 {len(expect)} 张 ({len(man['breasts'])} 乳房 x 2 视角)")

    problems = []
    total = 0
    for cond in CONDS:
        d = OCC / "occluded" / cond
        if not d.is_dir():
            problems.append(f"缺目录 {d}")
            continue
        files = sorted(p for p in d.iterdir() if p.suffix.lower() == ".png")
        print(f"[check] {cond}: {len(files)} 张")
        if len(files) != EXPECT_PER_COND:
            problems.append(f"{cond} 张数 {len(files)} != {EXPECT_PER_COND}")
        names = {p.name for p in files}
        missing = expect - names
        if missing:
            problems.append(f"{cond} 缺 {len(missing)} 张, 例 {sorted(missing)[:3]}")
        for p in files:
            try:
                with Image.open(p) as im:
                    im.load()
                    if im.size != (IMG_W, IMG_H):
                        problems.append(f"{p.name} 尺寸 {im.size}")
            except Exception as ex:
                problems.append(f"{p.name} 打不开: {type(ex).__name__}")
            total += 1

    print(f"[check] 实际检查 {total} 张 (期望 {EXPECT_TOTAL})")
    if total != EXPECT_TOTAL:
        problems.append(f"总张数 {total} != {EXPECT_TOTAL}")
    if problems:
        print("[check] 失败:")
        for p in problems[:20]:
            print("   -", p)
        sys.exit(1)
    print("[check] PASS: 1152 张全部可读且为 912x1520")


if __name__ == "__main__":
    main()
