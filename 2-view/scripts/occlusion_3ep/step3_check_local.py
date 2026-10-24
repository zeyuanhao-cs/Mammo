#!/usr/bin/env python3
"""检查遮挡图数量与尺寸, 并抽 6 个乳房生成 orig/lesion/ctrl1/ctrl2 拼图 -> check/.

输出:
  - reports/step3_check.txt
  - check/preview_{bid}_{view}.png (每个乳房一行 4 列: orig/lesion/ctrl1/ctrl2)
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJ = Path("/home/z/zhao/Mammography/Mammo/2-view")
OCC = PROJ / "outputs" / "eval" / "occlusion_3ep"
MAN = json.load(open(OCC / "occlusion_manifest.json", encoding="utf-8"))
CONDS = ("orig", "lesion", "ctrl1", "ctrl2")
CELL = (304, 507)          # 912x1520 / 3
N_PREVIEW = 6


def check_counts():
    lines = ["# 第 3 步检查: 遮挡图数量与尺寸", ""]
    ok = True
    for c in CONDS:
        d = OCC / "occluded" / c
        if c == "orig":
            # orig 不生成: 推理直接用原图 path; 这里只验证源图齐全
            n_src = sum(1 for bid, e in MAN["breasts"].items()
                        for vw in ("CC", "MLO")
                        if e["views"].get(vw) and Path(e["views"][vw]["path"]).is_file())
            n_exp = sum(1 for bid, e in MAN["breasts"].items()
                        for vw in ("CC", "MLO") if e["views"].get(vw))
            lines.append(f"- {c}: 不生成遮挡图, 原图存在 {n_src}/{n_exp}")
            ok &= (n_src == n_exp)
            continue
        if not d.exists():
            lines.append(f"- {c}: 目录不存在")
            ok = False
            continue
        files = sorted(d.glob("*.png"))
        n_expect = sum(1 for bid, e in MAN["breasts"].items()
                       for vw in ("CC", "MLO") if e["views"].get(vw))
        bad = []
        for p in files:
            with Image.open(p) as im:
                if im.size != (912, 1520) or im.mode != "L":
                    bad.append((p.name, im.size, im.mode))
        lines.append(f"- {c}: {len(files)} 张 (期望 {n_expect}), 尺寸/模式异常 {len(bad)} 个")
        if len(files) != n_expect or bad:
            ok = False
            for b in bad[:5]:
                lines.append(f"  - 异常: {b}")
    # 每个乳房 4 种情况齐全性
    missing = []
    for bid, e in MAN["breasts"].items():
        for vw in ("CC", "MLO"):
            if not e["views"].get(vw):
                continue
            for c in CONDS:
                p = OCC / "occluded" / c / f"{bid}__{vw}.png"
                if c != "orig" and not p.exists():
                    missing.append(f"{c}/{bid}__{vw}")
    lines.append(f"- 缺失文件: {len(missing)} 个")
    if missing:
        ok = False
        lines += [f"  - {m}" for m in missing[:10]]
    lines.append("")
    lines.append(f"**判定: {'PASS' if ok else 'FAIL'}**")
    return lines, ok


def previews():
    # 选 6 个乳房: 覆盖多框 / 不同 GT 类别 / 单视角有框
    cands = []
    for bid, e in MAN["breasts"].items():
        n_boxes = sum(len(v["lesion_boxes"]) for v in e["views"].values())
        single = (len(e["views"].get("CC", {}).get("lesion_boxes", [])) == 0) != \
                 (len(e["views"].get("MLO", {}).get("lesion_boxes", [])) == 0)
        cands.append((bid, n_boxes, single, tuple(e.get("gt_classes") or ())))
    picked, used = [], set()

    def take(pred):
        for c in cands:
            if len(picked) >= N_PREVIEW:
                return
            if c[0] in used or not pred(c):
                continue
            picked.append(c)
            used.add(c[0])

    take(lambda c: c[1] >= 2)                                   # 多框
    take(lambda c: "suspicious_calcification" in c[3])
    take(lambda c: "asymmetry" in c[3] or "architectural_distortion" in c[3])
    take(lambda c: c[2])                                        # 单视角有框
    take(lambda c: True)
    take(lambda c: True)

    for bid, nb, single, cls in picked[:N_PREVIEW]:
        e = MAN["breasts"][bid]
        # 选框最多的视角
        view = max(("CC", "MLO"), key=lambda vw: len(e["views"].get(vw, {}).get("lesion_boxes", [])))
        row = []
        for c in CONDS:
            if c == "orig":
                p = e["views"][view]["path"]
            else:
                p = OCC / "occluded" / c / f"{bid}__{view}.png"
            im = Image.open(p).convert("L").resize(CELL)
            row.append(np.asarray(im.convert("RGB")))
        grid = np.concatenate(row, axis=1)
        im = Image.fromarray(grid)
        d = ImageDraw.Draw(im)
        # 在 lesion 列画上遮挡矩形位置 (黄框)
        s = CELL[0] / 912.0, CELL[1] / 1520.0
        for it in e["views"][view]["conditions"]["lesion"]:
            r = it["rect"]
            d.rectangle([r[0] * s[0] + CELL[0], r[1] * s[1],
                         r[2] * s[0] + CELL[0], r[3] * s[1]], outline=(255, 200, 0), width=2)
        dst = OCC / "check" / f"preview_{bid}_{view}.png"
        im.save(dst)
        print(f"[preview] {dst.name}  n_boxes={nb} single_view={single} classes={','.join(cls)}")


def main():
    (OCC / "reports").mkdir(parents=True, exist_ok=True)
    (OCC / "check").mkdir(parents=True, exist_ok=True)
    lines, ok = check_counts()
    previews()
    (OCC / "reports" / "step3_check.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
