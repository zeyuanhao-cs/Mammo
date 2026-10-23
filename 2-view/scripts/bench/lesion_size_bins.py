#!/usr/bin/env python3
"""病灶大小分档（可复现、与推理解耦）。

口径：
  1) 取测试集 GT BI-RADS ≥ 3 的乳房（breast 级 = study_id + laterality）；
  2) 在该乳房的所有标注框里，只保留 finding_birads ≥ 3 的框；
  3) 面积比 = 框面积 / 整张图面积（图尺寸取 finding_annotations.csv 的 height/width）；
  4) 每例取这些框里面积比的最大值；没有合格框的记 0；
  5) 对 192 个乳房的该值取三分位（33.3% / 66.7%）切成 小 / 中 / 大。

输出（供报告脚本读取，换阈值只需重跑本脚本 + 报告，不用重新推理）：
  outputs/eval/size_bins.json   —— 阈值、每档样本数、每例的面积比与档位
  outputs/eval/size_bins.md     —— 人读的说明
"""
import csv
import json
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent.parent      # scripts/bench -> 2-view
ANN = Path("/home/z/zhao/Mammography/dataset/vindr-mammo/finding_annotations.csv")
TEST = PROJ / "data" / "splits" / "test_local.json"
OUT_JSON = PROJ / "outputs" / "eval" / "size_bins.json"
OUT_MD = PROJ / "outputs" / "eval" / "size_bins.md"


def birads_num(s):
    s = (s or "").strip()
    for ch in s:
        if ch.isdigit():
            return int(ch)
    return None


def tertiles(vals):
    s = sorted(vals)
    n = len(s)

    def q(p):
        k = (n - 1) * p
        lo, hi = int(k), min(int(k) + 1, n - 1)
        return s[lo] + (s[hi] - s[lo]) * (k - lo)

    return q(1 / 3), q(2 / 3)


def main():
    test = json.load(open(TEST, encoding="utf-8"))
    gt3 = {}
    for r in test:
        b = int(r["output"]["breast_birads"])
        if b >= 3:
            sid, lat = r["id"].rsplit("_", 1)
            gt3[r["id"]] = (sid, lat)
    print(f"测试集 GT≥3 乳房：{len(gt3)}")

    ratios = {k: 0.0 for k in gt3}
    with open(ANN, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("split") and row["split"] != "test":
                continue
            key = f"{row['study_id']}_{row['laterality']}"
            if key not in gt3:
                continue
            fb = birads_num(row.get("finding_birads"))
            if fb is None or fb < 3:
                continue
            try:
                h, w = float(row["height"]), float(row["width"])
                x0, y0 = float(row["xmin"]), float(row["ymin"])
                x1, y1 = float(row["xmax"]), float(row["ymax"])
            except (KeyError, ValueError):
                continue
            if h * w <= 0:
                continue
            area_ratio = max(0.0, (x1 - x0)) * max(0.0, (y1 - y0)) / (h * w)
            ratios[key] = max(ratios[key], area_ratio)

    vals = list(ratios.values())
    t1, t2 = tertiles(vals)

    def bucket(v):
        return "small" if v <= t1 else ("medium" if v <= t2 else "large")

    per = {k: {"ratio": v, "bin": bucket(v)} for k, v in ratios.items()}
    counts = {"small": 0, "medium": 0, "large": 0}
    for v in per.values():
        counts[v["bin"]] += 1

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "metric": "max(lesion_bbox_area) / image_area  (finding_birads>=3 boxes only)",
        "thresholds": {"t1": t1, "t2": t2},
        "thresholds_pct": {"t1": 100 * t1, "t2": 100 * t2},
        "counts": counts,
        "n": len(per),
        "zero_ratio_cases": sum(1 for v in vals if v == 0.0),
        "per_breast": per,
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    OUT_MD.write_text("\n".join([
        "# 病灶大小分档（与推理解耦）", "",
        "口径：GT BI-RADS ≥3 的乳房，取 `finding_birads ≥ 3` 的标注框，",
        "面积比 = 框面积 / 整图面积，每例取最大框；对 192 例取三分位。", "",
        f"- **小**：面积比 ≤ **{t1:.6f}（{100*t1:.4f}%）** → {counts['small']} 例",
        f"- **中**：{t1:.6f} < 面积比 ≤ **{t2:.6f}（{100*t2:.4f}%）** → {counts['medium']} 例",
        f"- **大**：面积比 > {t2:.6f} → {counts['large']} 例",
        f"- 无合格框（面积比记 0）的例数：{sum(1 for v in vals if v == 0.0)}",
        "", f"数据文件：`{OUT_JSON.name}`（含每例面积比与档位，报告脚本直接读它）",
    ]) + "\n", encoding="utf-8")

    print(f"阈值 t1={t1:.6f} ({100*t1:.4f}%), t2={t2:.6f} ({100*t2:.4f}%)")
    print(f"每档样本数：{counts}；零面积比 {sum(1 for v in vals if v == 0.0)} 例")


if __name__ == "__main__":
    main()
