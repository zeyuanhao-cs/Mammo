#!/usr/bin/env python3
"""过拟合检查汇总表（6 位小数）: train1000 vs val 的 AUC / 检出率 / 取值 NLL + val ECE。

取值 NLL 口径: 直接读 batch=8 输出的 probs_norm（生成时 6 个字段候选值上的归一化概率），
对每条样本取 6 个字段的 -log P(GT 值) 再平均；分字段列亦然。因其条件是"模型自生成的前缀 + 候选值集合"，
与训练时的 teacher-forcing CE 不完全等价。
"""
import json
import math
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
PROJ = THIS.parent.parent
sys.path.insert(0, str(PROJ / "scripts" / "eval"))
import common as C  # noqa: E402

TAGS = ["e1", "e2", "e3"]
OUT_MD = PROJ / "outputs" / "eval" / "overfit_report.md"
FIELDS = ["breast_density", "breast_birads"] + [f"findings.{k}" for k in C.FINDINGS]
SHORT = {"breast_density": "density", "breast_birads": "BI-RADS",
         "findings.mass": "mass", "findings.suspicious_calcification": "calcification",
         "findings.asymmetry": "asymmetry", "findings.architectural_distortion": "arch.distortion"}


def metric(split, tag):
    p = PROJ / "outputs" / "eval" / f"metrics_overfit_{split}_{tag}_b8.json"
    return json.load(open(p, encoding="utf-8"))


def detect(split, tag):
    p = PROJ / "outputs" / "scoring" / f"overfit_{split}_{tag}_b8_with_probs.jsonl"
    recs = C.load_probs_jsonl(str(p))
    gt = sum(1 for r in recs if any(r["gt_findings"][k] for k in C.FINDINGS))
    det = sum(1 for r in recs if any(r["gt_findings"][k] for k in C.FINDINGS)
              and any(r["greedy_findings"].get(k, 0) == 1 for k in C.FINDINGS))
    return det, gt


def nll(split, tag):
    """每样本 6 个字段 -log P(GT) 的平均，以及分字段平均。"""
    p = PROJ / "outputs" / "scoring" / f"overfit_{split}_{tag}_b8_with_probs.jsonl"
    per = {f: [] for f in FIELDS}
    means = []
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            pn, gt = d["probs_norm"], d["ground_truth"]
            prob = {
                "breast_density": pn["breast_density"][gt["breast_density"]],
                "breast_birads": pn["breast_birads"][str(gt["breast_birads"])],
            }
            for k in C.FINDINGS:
                prob[f"findings.{k}"] = pn[f"findings.{k}"][str(int(gt["findings"][k]))]
            vals = {f: -math.log(max(v, 1e-12)) for f, v in prob.items()}
            for f, v in vals.items():
                per[f].append(v)
            means.append(sum(vals.values()) / len(vals))
    return (sum(means) / len(means) if means else None,
            {f: (sum(v) / len(v) if v else None) for f, v in per.items()}, len(means))


def main():
    lines = ["# vt-ep3 过拟合检查（batch=8, val 1600 / train1000 seed42）", ""]
    lines.append("| model | train_density_auc | val_density_auc | diff | train_birads_auc | val_birads_auc | diff | "
                 "train_findings_auc | val_findings_auc | diff | train_detect | val_detect | "
                 "train_value_nll(b8) | val_value_nll(b8) | val_birads_ece |")
    lines.append("|" + "---|" * 15)
    for tag in TAGS:
        mt, mv = metric("train", tag), metric("val", tag)
        da, dv = mt["density_4class"]["ovr_macro_auc"], mv["density_4class"]["ovr_macro_auc"]
        ba, bv = mt["birads_ovr_macro_auc"], mv["birads_ovr_macro_auc"]
        fa, fv = mt["findings_macro_auroc"], mv["findings_macro_auroc"]
        dt, gt_t = detect("train", tag)
        dvv, gt_v = detect("val", tag)
        nt, _, _ = nll("train", tag)
        nv, _, _ = nll("val", tag)
        ece = mv["birads_top_label_ece"]
        lines.append(
            f"| {tag} | {da:.6f} | {dv:.6f} | {dv - da:+.6f} | {ba:.6f} | {bv:.6f} | {bv - ba:+.6f} | "
            f"{fa:.6f} | {fv:.6f} | {fv - fa:+.6f} | {dt / gt_t:.6f} ({dt}/{gt_t}) | "
            f"{dvv / gt_v:.6f} ({dvv}/{gt_v}) | {nt:.6f} | {nv:.6f} | {ece:.6f} |")

    lines += ["", "## 取值 NLL 分字段（train / val）", "",
              "| model | field | train_nll | val_nll | diff |", "|---|---|---|---|---|"]
    for tag in TAGS:
        _, pt, _ = nll("train", tag)
        _, pv, _ = nll("val", tag)
        for f in FIELDS:
            if pt[f] is not None and pv[f] is not None:
                lines.append(f"| {tag} | {SHORT[f]} | {pt[f]:.6f} | {pv[f]:.6f} | {pv[f] - pt[f]:+.6f} |")

    lines += ["", "### 口径说明", "",
              "- 取值 NLL: 读 batch=8 输出的 `probs_norm`（每个字段只在其候选取值上归一化），"
              "单样本 NLL = 6 个字段 `-log P(GT 值)` 的平均；分字段列为该字段的平均。",
              "- 条件是「模型自生成的前缀 + 候选值集合」，因此与训练时的 teacher-forcing CE "
              "（GT 前缀、全词表 softmax）不完全等价，仅作 train/val 相对比较。",
              "- detect = 在 GT 有病灶的乳房中，预测任一 finding 为阳性的比例（贪心输出）。",
              "- AUC: density / BI-RADS 为 OvR macro AUC，findings 为 4 个二分类 AUROC 的均值；"
              "val_birads_ece 为 BI-RADS top-label ECE。",
              f""]

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
