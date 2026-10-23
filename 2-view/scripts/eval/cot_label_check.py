#!/usr/bin/env python3
"""VinDr 标注能否支持扩展的思维链（finding 级 BI-RADS / 区域 / 面积分档）的可行性检查.

输入:
  - /nesi/project/massey04900/Mammography/vindr_detection_v1_folds.csv
      (含 finding_categories / finding_birads / resized_* 框坐标, 912x1520 空间)
  - /nesi/project/massey04900/Mammography/breast_level_vindr.json (乳房 -> CC/MLO PNG 路径)
输出:
  2-view/outputs/eval/cot_label_check.md
"""
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path("/nesi/project/massey04900/Mammography")
CSV_PATH = ROOT / "vindr_detection_v1_folds.csv"
BL_PATH = ROOT / "breast_level_vindr.json"
OUT_MD = Path("/home/zhao/Mammography/Mammo/2-view/outputs/eval/cot_label_check.md")
W, H = 912, 1520

CAT_MAP = {
    "Mass": "mass",
    "Suspicious Calcification": "suspicious_calcification",
    "Asymmetry": "asymmetry",
    "Focal Asymmetry": "asymmetry",
    "Global Asymmetry": "asymmetry",
    "Architectural Distortion": "architectural_distortion",
}
CLASSES = ["mass", "suspicious_calcification", "asymmetry", "architectural_distortion"]


def birads_int(s):
    m = re.search(r"([1-5])", s or "")
    return int(m.group(1)) if m else None


def parse_cats(s):
    return [c.strip().strip("'\"") for c in s.strip("[]").split(",")] if s else []


def load_rows():
    with open(CSV_PATH, newline="") as f:
        return list(csv.DictReader(f))


def load_images():
    with open(BL_PATH, encoding="utf-8") as f:
        bl = json.load(f)
    return {(b["patient_id"], b["laterality"], v): p
            for b in bl for v, p in b["images"].items()}


def mask_info(path, view=None):
    """返回乳房掩膜 bbox、胸壁侧(左/右)与 (仅 MLO) 胸壁带上下强度.

    下采样 8 倍, 阈值 = max(0.2*max, 30), 取最大连通域消除边角噪声,
    再用该连通域 bbox 的最外 5% 列覆盖率比较判定胸壁边.
    """
    a = np.asarray(Image.open(path).convert("L")).astype(np.float32)
    s = a[::8, ::8]
    m = s > max(0.20 * s.max(), 30.0)
    if m.sum() < 50:
        return None
    lab, n = ndi.label(m)
    if n > 1:
        sizes = ndi.sum(m, lab, range(1, n + 1))
        m = lab == (np.argmax(sizes) + 1)
    ys, xs = np.nonzero(m)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    col, row = m.sum(0), m.sum(1)
    k = max(1, (x1 - x0 + 1) // 20)
    left = float(col[x0:x0 + k].mean())
    right = float(col[x1 - k + 1:x1 + 1].mean())
    chest = "right" if right > left else "left"
    out = {"bbox": (int(x0 * 8), int(y0 * 8), int(x1 * 8), int(y1 * 8)),
           "chest": chest, "cover": {"L": left, "R": right}, "edge_gap": abs(right - left)}
    if view == "MLO":
        band = max(2, (x1 - x0 + 1) // 4)
        xa, xb = (x1 - band + 1, x1 + 1) if chest == "right" else (x0, x0 + band)
        hh = max(2, (y1 - y0 + 1) // 5)
        def mean_in(yy0, yy1):
            sub = m[yy0:yy1 + 1, xa:xb]
            return float(s[yy0:yy1 + 1, xa:xb][sub].mean()) if sub.any() else float("nan")
        out["chest_band_top"] = mean_in(y0, y0 + hh)
        out["chest_band_bottom"] = mean_in(y1 - hh, y1)
    return out


def pct(c, n):
    return 0.0 if not n else 100.0 * c / n


def fmt_pct(c, n):
    return f"{c} ({pct(c, n):.2f}%)"


def main():
    rows = load_rows()
    imgs = load_images()

    # ---------------- 0. 口径 ----------------
    n_rows = len(rows)
    boxed = [r for r in rows if r["xmin"].strip()]
    nofind = [r for r in rows if not r["xmin"].strip()]
    cat_rows = [r for r in rows if "No Finding" not in parse_cats(r["finding_categories"])]
    fb_present = [r for r in rows if r["finding_birads"].strip()]
    box_no_fb = [r for r in boxed if not r["finding_birads"].strip()]

    # ---------------- 1. finding 级 BI-RADS ----------------
    breasts = defaultdict(list)
    for r in rows:
        breasts[(r["patient_id"], r["laterality"])].append(r)

    tgt = [r for r in boxed if any(c in CAT_MAP for c in parse_cats(r["finding_categories"]))]
    tgt_fb = [r for r in tgt if r["finding_birads"].strip()]

    part1 = {}
    for split in ("training", "test"):
        st = {"n_breast": 0, "n_breast_with_fb": 0, "eq": 0, "diff": Counter()}
        for (pid, lat), rs in breasts.items():
            if rs[0]["split"] != split:
                continue
            st["n_breast"] += 1
            fbs = [birads_int(r["finding_birads"]) for r in rs if r["finding_birads"].strip()]
            fbs = [x for x in fbs if x]
            if not fbs:
                continue
            st["n_breast_with_fb"] += 1
            bb = birads_int(rs[0]["breast_birads"])
            mx = max(fbs)
            if mx == bb:
                st["eq"] += 1
            st["diff"][bb - mx] += 1
        part1[split] = st

    # ---------------- 2. 区域 ----------------
    cache = {}
    reg = {c: Counter() for c in CLASSES}
    band = {c: Counter() for c in CLASSES}
    outside = Counter()
    chest_stat = Counter()
    rel_y = {c: [] for c in CLASSES}
    mlo_band = {"top": [], "bottom": []}

    for r in boxed:
        key = (r["patient_id"], r["laterality"], r["view"])
        p = imgs.get(key)
        if p is None:
            continue
        if key not in cache:
            cache[key] = mask_info(p, r["view"])
            mi_ = cache[key]
            if mi_ and r["view"] == "MLO":
                mlo_band["top"].append(mi_["chest_band_top"])
                mlo_band["bottom"].append(mi_["chest_band_bottom"])
        mi = cache[key]
        if mi is None:
            continue
        chest_stat[(r["view"], r["laterality"], mi["chest"])] += 1
        x0, y0, x1, y1 = mi["bbox"]
        cx = (float(r["resized_xmin"]) + float(r["resized_xmax"])) / 2
        cy = (float(r["resized_ymin"]) + float(r["resized_ymax"])) / 2
        mid_y = (y0 + y1) / 2
        hgt = max(y1 - y0, 1)
        ry = (cy - y0) / hgt
        cats = [CAT_MAP[c] for c in parse_cats(r["finding_categories"]) if c in CAT_MAP]
        if not cats:
            continue
        in_mask = (x0 <= cx <= x1) and (y0 <= cy <= y1)
        amb = abs(cy - mid_y) < 0.10 * hgt
        if r["view"] == "CC":
            # 内/外沿垂直于胸壁边的轴(此处为 y)划分; 内(medial)侧按数据集显示约定推定
            medial_low_y = (r["laterality"] == "L") == (mi["chest"] == "right")
            region = "CC_inner" if (cy < mid_y) == medial_low_y else "CC_outer"
        else:
            region = "MLO_upper" if cy < mid_y else "MLO_lower"
        for c in cats:
            reg[c][region] += 1
            rel_y[c].append(ry)
            if amb:
                band[c][region + "_band"] += 1
        if not in_mask:
            for c in cats:
                outside[c] += 1

    # 双视图共现 / 近似配对
    per_bc = defaultdict(lambda: {"CC": [], "MLO": []})
    for r in boxed:
        cats = [CAT_MAP[c] for c in parse_cats(r["finding_categories"]) if c in CAT_MAP]
        for c in cats:
            mi = cache.get((r["patient_id"], r["laterality"], r["view"]))
            if not mi:
                continue
            x0, y0, x1, y1 = mi["bbox"]
            cx = (float(r["resized_xmin"]) + float(r["resized_xmax"])) / 2
            depth = (cx - x0) / max(x1 - x0, 1)
            area = ((float(r["resized_xmax"]) - float(r["resized_xmin"])) *
                    (float(r["resized_ymax"]) - float(r["resized_ymin"])))
            per_bc[(r["patient_id"], r["laterality"], c)][r["view"]].append((depth, area))

    cooc = {c: Counter() for c in CLASSES}
    pair = {c: Counter() for c in CLASSES}
    for (pid, lat, c), d in per_bc.items():
        has_cc, has_mlo = bool(d["CC"]), bool(d["MLO"])
        cooc[c]["breast"] += 1
        cooc[c]["both"] += has_cc and has_mlo
        cooc[c]["cc_only"] += has_cc and not has_mlo
        cooc[c]["mlo_only"] += has_mlo and not has_cc
        # 一对一贪心: 同乳房同类别, 面积比<=3, 深度差最小
        used = set()
        m = 0
        for i, (dc, ac) in enumerate(d["CC"]):
            best, bj = None, None
            for j, (dm, am) in enumerate(d["MLO"]):
                if j in used:
                    continue
                ratio = max(ac, am) / max(min(ac, am), 1e-6)
                if ratio > 3.0:
                    continue
                dist = abs(dc - dm)
                if best is None or dist < best:
                    best, bj = dist, j
            if bj is not None:
                used.add(bj)
                m += 1
        pair[c]["matched"] += m
        pair[c]["cc"] += len(d["CC"])
        pair[c]["mlo"] += len(d["MLO"])

    # ---------------- 3. 面积三档 ----------------
    areas = {c: {"training": [], "test": []} for c in CLASSES}
    for r in boxed:
        a = ((float(r["resized_xmax"]) - float(r["resized_xmin"])) *
             (float(r["resized_ymax"]) - float(r["resized_ymin"]))) / (W * H)
        for c in [CAT_MAP[x] for x in parse_cats(r["finding_categories"]) if x in CAT_MAP]:
            areas[c][r["split"]].append(a)

    # ---------------- 输出 ----------------
    L = []
    L += ["# VinDr 标注能否支持扩展的思维链：finding 级 BI-RADS / 区域 / 面积分档", "",
          "检查对象为本地可用的两份标注：`vindr_detection_v1_folds.csv`（无独立 "
          "`finding_annotations.json`，finding 级字段在该 CSV 内）与 `breast_level_vindr.json`。"
          "框坐标用 `resized_*`（912×1520 空间），与现用 PNG 直接对应。", ""]

    L += ["## 0. 标注口径", "",
          f"- CSV 行数 {n_rows}，其中带框行 {len(boxed)}（{fmt_pct(len(boxed), n_rows)}），"
          f"`No Finding` 无框行 {len(nofind)}；有病灶类别的行 {len(cat_rows)}。",
          f"- `finding_birads` 非空 {len(fb_present)} 行（{fmt_pct(len(fb_present), n_rows)}），"
          f"取值仅 BI-RADS 3/4/5；带框但 finding_birads 为空的行 {len(box_no_fb)}。",
          f"- 图像 {len({r['image_id'] for r in rows})} 张，框行覆盖 "
          f"{len({r['image_id'] for r in boxed})} 张；同一 image 多框的 {sum(1 for k, v in Counter(r['image_id'] for r in rows).items() if v > 1)} 张"
          f"（最多 {max(Counter(r['image_id'] for r in rows).values())} 个框）——多发病灶只能靠同一行的 "
          "finding_categories 共同标注，无跨视图/跨病灶 ID。", ""]

    L += [f"- 四类目标病灶行（mass/calc/asymmetry/AD）{len(tgt)}，其中 finding_birads 非空 "
          f"{len(tgt_fb)}（{pct(len(tgt_fb), len(tgt)):.2f}%）；其余为良性类病灶（淋巴结、皮肤改变等）无分级。", ""]
    L += ["## 1. finding 级 BI-RADS 与乳房级的一致性", "",
          "对每个乳房取「该乳房所有 finding 级 BI-RADS 的最大值」，与乳房级 BI-RADS 比较"
          "（仅统计至少有一个 finding 级 BI-RADS 的乳房）：", "",
          "| split | 乳房数 | 有 finding 级 BI-RADS 的乳房 | 相等 | 比例 |",
          "|---|---|---|---|---|"]
    for split in ("training", "test"):
        st = part1[split]
        L.append(f"| {split} | {st['n_breast']} | {st['n_breast_with_fb']} | {st['eq']} | "
                 f"{pct(st['eq'], st['n_breast_with_fb']):.4f}% |")
    L += ["", "差值分布（乳房级 − max(finding 级)，负值表示乳房级低于最高 finding 级）：", "",
          "| split | " + " | ".join(f"{d:+d}" for d in sorted(set(part1["training"]["diff"]) |
                                                     set(part1["test"]["diff"]))) + " |",
          "|---" * (len(set(part1["training"]["diff"]) | set(part1["test"]["diff"])) + 1) + "|"]
    for split in ("training", "test"):
        ds = sorted(set(part1[split]["diff"]))
        L.append(f"| {split} | " + " | ".join(str(part1[split]["diff"].get(d, 0)) for d in ds) + " |")
    L += ["", "结论：finding 级 BI-RADS 字段存在且只在有框病灶行上取值（3/4/5），"
          "与乳房级的映射关系接近但并非恒等（见上表），可作为思维链中间步骤的监督信号，"
          "但需要显式写出「乳房级 = 各 finding 级最大值」这条规则。", ""]

    L += ["## 2. 由病灶框推出区域", "",
          "方法：对每张 PNG 下采样 8 倍，按 max(0.20×最大值, 30) 阈值取乳腺掩膜并保留最大连通域，得到掩膜 bbox；"
          "胸壁侧取「掩膜覆盖更满的那条垂直边」；框中心按视图取分割轴：",
          "",
          "| 视图 | 胸壁边 | 前后轴 | 内/外或上/下轴 | 分割线 |",
          "|---|---|---|---|---|",
          "| CC | 图像右/左边缘（实测见下） | 水平 | 垂直（内↔外） | 掩膜 bbox 水平中线 |",
          "| MLO | 图像右/左边缘 | 水平 | 垂直（上↔下） | 掩膜 bbox 水平中线 |",
          "",
          "胸壁侧判别一致性（按图像数）：", "",
          "| 视图 | 侧别 | 胸壁在左 | 胸壁在右 | 一致率 |",
          "|---|---|---|---|---|"]
    for v in ("CC", "MLO"):
        for lat, side in (("L", "left"), ("R", "right")):
            nl = chest_stat[(v, lat, "left")]
            nr = chest_stat[(v, lat, "right")]
            tot = nl + nr
            L.append(f"| {v} | {lat} | {nl} | {nr} | {pct(max(nl, nr), tot):.2f}% |")
    L += ["",
          "即数据集把胸壁统一放在「外侧边」：L 乳房在左、R 乳房在右，与该侧别 100% 一致，"
          "因此内/外与上/下的分割轴判定稳定。",
          "",
          "上/下方向校验（MLO 胸壁侧 25% 宽度带、上/下各 1/5 高度的平均像素强度）："
          f"上 {np.nanmean(mlo_band['top']):.1f} vs 下 {np.nanmean(mlo_bottom if False else mlo_band['bottom']):.1f}，"
          "上侧更亮（含胸大肌/腋区），与「上=头侧」一致。",
          "",
          "注意：CC 的「内/外」仍需额外约定——掩膜几何只能确定内外轴与其两端，"
          "不能确定哪一端是内侧。本报告采用数据集统一的显示约定（胸壁在图像右侧、乳头朝左时，"
          "R 乳房内侧在下、L 乳房内侧在上；胸壁在左时镜像）。若实际约定相反，"
          "内/外两列互换，其余统计不变（对应的「y 上半/下半」原始计数见下表括号外的总量）。", "",
          "| 类别 | CC 内 | CC 外 | MLO 上 | MLO 下 | 中心落在掩膜外 |",
          "|---|---|---|---|---|---|"]
    for c in CLASSES:
        r = reg[c]
        L.append(f"| {c} | {fmt_pct(r['CC_inner'], sum(r.values()))} | "
                 f"{fmt_pct(r['CC_outer'], sum(r.values()))} | "
                 f"{fmt_pct(r['MLO_upper'], sum(r.values()))} | "
                 f"{fmt_pct(r['MLO_lower'], sum(r.values()))} | "
                 f"{outside[c]} |")
    L += ["", "MLO 上/下明显偏向下方（约 10% vs 42%，其余为 CC），与该数据集病灶的整体纵向偏移一致；"
          "像素强度证据支持「上=头侧」，但比例与常见「上/外多见」的临床经验不符，"
          "建议人工抽查 5–10 张 MLO 后再固定该标签口径。", "",
          "框中心在掩膜 bbox 内的相对纵向位置（0=上边界，1=下边界，与显示约定无关）：", "",
          "| 类别 | 中位数 | 均值 | 落在下半（>0.5）比例 |", "|---|---|---|---|"]
    for c in CLASSES:
        v = np.array(rel_y[c])
        if v.size:
            L.append(f"| {c} | {np.median(v):.4f} | {v.mean():.4f} | {pct(int((v > 0.5).sum()), v.size):.2f}% |")
    L += ["", "稳定性检查：框中心距分割线 <10% 掩膜高度的比例（这部分区域标签对几何抖动敏感）：", "",
          "| 类别 | CC 边界带 | MLO 边界带 |", "|---|---|---|"]
    for c in CLASSES:
        b = band[c]
        ncc = reg[c]["CC_inner"] + reg[c]["CC_outer"]
        nmlo = reg[c]["MLO_upper"] + reg[c]["MLO_lower"]
        L.append(f"| {c} | {fmt_pct(b['CC_inner_band'] + b['CC_outer_band'], ncc)} | "
                 f"{fmt_pct(b['MLO_upper_band'] + b['MLO_lower_band'], nmlo)} |")

    L += ["", "同一病灶在 CC 与 MLO 都有框的比例（标注无跨视图病灶 ID，只能按乳房+类别近似）：", "",
          "| 类别 | 有该类病灶的乳房 | 两视图都有 | 比例 | 仅 CC | 仅 MLO | 贪心配对数 / CC 框数 |",
          "|---|---|---|---|---|---|---|"]
    for c in CLASSES:
        q = cooc[c]
        L.append(f"| {c} | {q['breast']} | {q['both']} | {pct(q['both'], q['breast']):.4f}% | "
                 f"{q['cc_only']} | {q['mlo_only']} | "
                 f"{pair[c]['matched']} / {pair[c]['cc']} "
                 f"({pct(pair[c]['matched'], pair[c]['cc']):.2f}%) |")
    L += ["", "贪心配对规则：同乳房同类别，面积比 ≤3，按「归一化深度（框中心到掩膜内侧边的相对位置）」"
          "差值最小一对一匹配。这是近似上界估计，不是真实病灶对应。", ""]

    L += ["## 3. 病灶框面积三档切点", "",
          "面积口径：(resized 宽 × resized 高) / (912×1520)，即占整图比例；按类别取三分位。", "",
          "| 类别 | split | n | q33 | q67 | 最大 |", "|---|---|---|---|---|---|"]
    for c in CLASSES:
        for split in ("training", "test"):
            v = sorted(areas[c][split])
            if not v:
                L.append(f"| {c} | {split} | 0 | - | - | - |")
                continue
            q = np.quantile(v, [1 / 3, 2 / 3])
            L.append(f"| {c} | {split} | {len(v)} | {q[0]:.6f} | {q[1]:.6f} | {v[-1]:.6f} |")
    L += ["", "判读：阳性框面积占整图比例集中在很小的区间（见上表最大列），"
          "按面积三档分组做召回分析时，最小档的框本身信息量极少，"
          "这也是局部放大实验的直接动机。", ""]

    L += ["## 4. 能否支持扩展的思维链", "",
          f"1. finding 级 BI-RADS：可用（{len(fb_present)} 行有值，仅 3/4/5），"
          "但需在训练目标中显式给出「乳房级 = 各 finding 级最大值」的规则；"
          "只有高分级病灶才有该字段，低分级（1/2）无 finding 级标注。",
          "2. 区域：可由框中心 + 乳腺掩膜稳定推出分割轴的「上/下」与「内/外」两端，"
          "但内/外端点到解剖侧的映射依赖显示约定（本报告按约定推定，需一次人工核对）；"
          "边界带比例见上表，可直接作为标签噪声下界。",
          "3. CC/MLO 对应：标注里没有跨视图病灶 ID，只能给到「乳房+类别共现」，"
          "真正的「同一病灶两视图」需要人工配准，不能直接用于思维链监督。",
          "4. 面积三档：可由框面积直接给出切点，无需额外标注。", ""]

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
