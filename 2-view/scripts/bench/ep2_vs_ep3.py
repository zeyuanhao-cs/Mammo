#!/usr/bin/env python3
"""ep2(2 epoch) 与 ep3(3 epoch) 在 val/test 上的 b8 指标对比 -> outputs/eval/ep2_vs_ep3.md"""
import json
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
PROJ = THIS.parent.parent
sys.path.insert(0, str(PROJ / "scripts" / "eval"))
import common as C  # noqa: E402

OUT = PROJ / "outputs" / "eval" / "ep2_vs_ep3.md"
E = PROJ / "outputs" / "eval"
S = PROJ / "outputs" / "scoring"

RUNS = [
    ("ep2-e1 (ckpt439)", E / "metrics_probs_vt-ep2-e1_val_b8.json", E / "metrics_probs_vt-ep2-e1_test_b8.json",
     S / "vt-ep2-e1_val_b8_with_probs.jsonl", S / "vt-ep2-e1_test_b8_with_probs.jsonl"),
    ("ep2-e2 (ckpt878)", E / "metrics_probs_vt-ep2-e2_val_b8.json", E / "metrics_probs_vt-ep2-e2_test_b8.json",
     S / "vt-ep2-e2_val_b8_with_probs.jsonl", S / "vt-ep2-e2_test_b8_with_probs.jsonl"),
    ("ep3-e1 (ckpt439)", E / "metrics_overfit_val_e1_b8.json", E / "metrics_probs_vt-ep3-e1_test_b8.json",
     S / "overfit_val_e1_b8_with_probs.jsonl", S / "vt-ep3-e1_test_b8_with_probs.jsonl"),
    ("ep3-e2 (ckpt878)", E / "metrics_overfit_val_e2_b8.json", E / "metrics_probs_vt-ep3-e2_test_b8.json",
     S / "overfit_val_e2_b8_with_probs.jsonl", S / "vt-ep3-e2_test_b8_with_probs.jsonl"),
    ("ep3-e3 (ckpt1317)", E / "metrics_overfit_val_e3_b8.json", E / "metrics_probs_vt-ep3-e3_test_b8.json",
     S / "overfit_val_e3_b8_with_probs.jsonl", S / "vt-ep3-e3_test_b8_with_probs.jsonl"),
]


def detect(path):
    if not Path(path).exists():
        return None, None
    recs = C.load_probs_jsonl(str(path))
    gt = sum(1 for r in recs if any(r["gt_findings"][k] for k in C.FINDINGS))
    det = sum(1 for r in recs if any(r["gt_findings"][k] for k in C.FINDINGS)
              and any(r["greedy_findings"].get(k, 0) == 1 for k in C.FINDINGS))
    return det, gt


def row(m, probs_path):
    f = lambda v: "—" if v is None else f"{v:.4f}"
    d, g = detect(probs_path)
    return [f(m["density_4class"]["ovr_macro_auc"]), f(m["birads_ovr_macro_auc"]), f(m["findings_macro_auroc"]),
            f(m["density_4class"]["macro_f1"]), f(m["birads_5class_probs_argmax"]["macro_f1"]),
            f(m["findings_macro_f1_at_0.5"]), f(m["density_4class"]["bacc"]),
            f(m["birads_5class_probs_argmax"]["bacc"]),
            f(m["birads_top_label_ece"]),
            "—" if d is None else f"{d / g:.4f} ({d}/{g})"]


HDR = ("| run | density AUC | BI-RADS AUC | findings AUROC | density F1 | BI-RADS F1 | findings F1 | "
       "density bACC | BI-RADS bACC | BI-RADS ECE | 检出率 |")
SEP = "|" + "---|" * 11


def main():
    lines = ["# ep2 (2 epochs) vs ep3 (3 epochs) — b8 流水线", ""]
    for title, idx in (("验证集 val (1600)", 1), ("测试集 test (2000)", 2)):
        lines += [f"## {title}", "", HDR, SEP]
        for name, vm, tm, vp, tp in RUNS:
            mp = vm if idx == 1 else tm
            pp = vp if idx == 1 else tp
            if not Path(mp).exists():
                lines.append(f"| {name} | (metrics 缺失) " + "| " * 9 + "|")
                continue
            m = json.load(open(mp, encoding="utf-8"))
            lines.append(f"| {name} | " + " | ".join(row(m, pp)) + " |")
        lines.append("")
    lines += ["## 说明", "",
              "- 全部为 batch=8 流水线（`infer_2view_batched.py` + `summarize_probs.py`）。",
              "- ECE 为 BI-RADS top-label ECE；检出率为 GT 有病灶乳房中预测任一 finding 阳性的比例。",
              "- ep2 与 ep3 训练数据/超参完全相同，仅 num_train_epochs 2.0 vs 3.0。"]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
