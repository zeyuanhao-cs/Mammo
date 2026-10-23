#!/usr/bin/env python3
"""汇总软证据探针结果 -> chain_v1/outputs/soft_evidence_probe.md + 敏感性图。"""
import json
import statistics as st
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

THIS = Path(__file__).resolve().parent
CV1 = THIS.parent                       # chain_v1
PROJ = CV1.parent                       # 2-view
sys.path.insert(0, str(PROJ / "scripts" / "eval"))
import common as C  # noqa: E402

OUT_MD = CV1 / "outputs" / "soft_evidence_probe.md"
OUT_PNG = CV1 / "outputs" / "soft_evidence_sensitivity.png"
SPLITS = ["test", "val"]
FIND = C.FINDINGS


def load(split):
    p = CV1 / "outputs" / f"soft_evidence_{split}.jsonl"
    rows = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def macro_ovr_auc(rows, key):
    y = [r["gt_birads"] for r in rows]
    aucs = []
    for cls in range(1, 6):
        yc = [1 if v == cls else 0 for v in y]
        if sum(yc) in (0, len(yc)):
            continue
        s = [r[key][cls - 1] for r in rows]
        a = C.auc(yc, s)
        if a is not None:
            aucs.append(a)
    return sum(aucs) / len(aucs) if aucs else None


def ge3_auc(rows, key):
    y = [1 if r["gt_birads"] >= 3 else 0 for r in rows]
    s = [sum(r[key][2:]) for r in rows]
    return C.auc(y, s)


def n_all_zero_greedy(rows):
    return [r for r in rows if all(r["greedy_findings"][k] == 0 for k in FIND)]


def is_lesion(r):
    return any(r["gt_findings"][k] for k in FIND)


def med(v):
    return st.median(v) if v else None


def groups_teacher(rows):
    """(e) 组内区分：只 mass / 只 calc / 4 征象全 0。"""
    out = {}

    def only(k):
        return [r for r in rows if r["gt_findings"][k] == 1 and sum(r["gt_findings"].values()) == 1]

    m = only("mass")
    sub = [r for r in m if r["gt_birads"] in (3, 4, 5)]
    out["only_mass"] = {
        "n": len(sub),
        "n3": sum(1 for r in sub if r["gt_birads"] == 3),
        "n45": sum(1 for r in sub if r["gt_birads"] >= 4),
        "auc": C.auc([1 if r["gt_birads"] >= 4 else 0 for r in sub],
                     [sum(r["teacher_birads"][3:]) for r in sub]),
    }
    c = only("suspicious_calcification")
    sub = [r for r in c if r["gt_birads"] in (3, 4, 5)]
    out["only_calc"] = {
        "n": len(sub),
        "n3": sum(1 for r in sub if r["gt_birads"] == 3),
        "n45": sum(1 for r in sub if r["gt_birads"] >= 4),
        "auc": C.auc([1 if r["gt_birads"] >= 4 else 0 for r in sub],
                     [sum(r["teacher_birads"][3:]) for r in sub]),
    }
    z = [r for r in rows if all(r["gt_findings"][k] == 0 for k in FIND)]
    sub = [r for r in z if r["gt_birads"] in (1, 2)]
    out["all_zero"] = {
        "n": len(sub),
        "n1": sum(1 for r in sub if r["gt_birads"] == 1),
        "n2": sum(1 for r in sub if r["gt_birads"] == 2),
        "auc": C.auc([1 if r["gt_birads"] == 2 else 0 for r in sub],
                     [r["teacher_birads"][1] for r in sub]),
    }
    return out


def f6(v):
    return "—" if v is None else f"{v:.6f}"


def main():
    data = {s: load(s) for s in SPLITS}
    lines = ["# 链条侧 v1 · 免训练软证据探针", ""]
    lines.append("模型：`Qwen3.5-4B-lora-2view-os-sub-vt`（1-epoch 解冻版），b8 流水线，"
                 "图像预算 786432（每视角 756 token）。")
    lines.append("")
    lines.append("## 1. 硬 token vs 软证据（BI-RADS 分布）")
    lines.append("")
    lines.append("| split | n | 指标 | hard (b8 token) | soft (嵌入混合) |")
    lines.append("|---|---|---|---|---|")
    for s in SPLITS:
        rows = data[s]
        for name, fn in (("BI-RADS macro OvR AUC", macro_ovr_auc), (">=3 AUC", ge3_auc)):
            lines.append(f"| {s} | {len(rows)} | {name} | {f6(fn(rows, 'hard_birads'))} | "
                         f"{f6(fn(rows, 'soft_birads'))} |")
    lines += ["", "## 2. 贪心 findings 全 0 子集", "",
              "| split | n(全0) | hard >=3 AUC | soft >=3 AUC | hard 中位 P(>=3) 病灶/正常 | soft 中位 P(>=3) 病灶/正常 |",
              "|---|---|---|---|---|---|"]
    for s in SPLITS:
        rows = n_all_zero_greedy(data[s])
        les = [sum(r["hard_birads"][2:]) for r in rows if is_lesion(r)]
        nor = [sum(r["hard_birads"][2:]) for r in rows if not is_lesion(r)]
        les_s = [sum(r["soft_birads"][2:]) for r in rows if is_lesion(r)]
        nor_s = [sum(r["soft_birads"][2:]) for r in rows if not is_lesion(r)]
        lines.append(f"| {s} | {len(rows)} | {f6(ge3_auc(rows, 'hard_birads'))} | "
                     f"{f6(ge3_auc(rows, 'soft_birads'))} | "
                     f"{f6(med(les))} / {f6(med(nor))} | {f6(med(les_s))} / {f6(med(nor_s))} |")
    lines.append("")

    # ---- 敏感性 ----
    sens_path = CV1 / "outputs" / "soft_evidence_sensitivity.json"
    if sens_path.exists():
        d = json.loads(sens_path.read_text(encoding="utf-8"))
        grid, rows = d["grid"], d["rows"]
        fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
        for r in rows:
            ax.plot(grid, r["p_ge3"], color="#888888", lw=0.7, alpha=0.6)
        mean = [sum(r["p_ge3"][i] for r in rows) / len(rows) for i in range(len(grid))]
        ax.plot(grid, mean, color="#c53030", lw=2.0, label="mean over 20 breasts")
        ax.set_xlabel("mass soft p")
        ax.set_ylabel("P(BI-RADS >= 3)")
        ax.set_title("soft-evidence sensitivity (mass)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT_PNG)
        plt.close(fig)
        lines += ["## 3. 敏感性（mass p: 0→1，20 个乳房）", "", f"![sens]({OUT_PNG.name})", "",
                  "| 指标 | 值 |", "|---|---|",
                  f"| 20 例整体 ΔP(>=3)（p=1 减 p=0）中位数 | {f6(med([r['p_ge3'][-1]-r['p_ge3'][0] for r in rows]))} |",
                  f"| 同上最大值 | {f6(max(r['p_ge3'][-1]-r['p_ge3'][0] for r in rows))} |",
                  f"| 同上最小值 | {f6(min(r['p_ge3'][-1]-r['p_ge3'][0] for r in rows))} |", ""]

    # ---- 组内区分 ----
    lines += ["## 4. 老师分布（正确征象作前缀）的组内区分", "",
              "| split | 组 | n | 正/负 | AUC |", "|---|---|---|---|---|"]
    for s in SPLITS:
        g = groups_teacher(data[s])
        for k, v in g.items():
            if k == "all_zero":
                pos, neg = f"{v['n2']}(gt2)", f"{v['n1']}(gt1)"
            else:
                pos, neg = f"{v['n45']}(4/5)", f"{v['n3']}(gt3)"
            lines.append(f"| {s} | {k} | {v['n']} | {pos} / {neg} | {f6(v['auc'])} |")
    g_all = groups_teacher(data["test"] + data["val"])
    for k, v in g_all.items():
        if k == "all_zero":
            pos, neg = f"{v['n2']}(gt2)", f"{v['n1']}(gt1)"
        else:
            pos, neg = f"{v['n45']}(4/5)", f"{v['n3']}(gt3)"
        lines.append(f"| test+val | {k} | {v['n']} | {pos} / {neg} | {f6(v['auc'])} |")
    lines.append("")
    # ---- 自检：hard(本脚本) vs b8 记录 ----
    lines += ["## 5. 自检：本脚本 hard 与 b8 记录的一致性", "",
              "| split | n | argmax 一致率 | 平均绝对差（5 类逐样本平均后再平均） | 单例最大平均绝对差 |",
              "|---|---|---|---|---|"]
    for s_ in SPLITS:
        p_ = CV1 / "outputs" / f"soft_evidence_{s_}_selfcheck.json"
        if p_.exists():
            d = json.loads(p_.read_text(encoding="utf-8"))
            lines.append(f"| {s_} | {d['n']} | {d['argmax_agree']:.6f} | {d['mean_abs_diff']:.6f} | "
                         f"{d['max_sample_abs_diff']:.6f} |")
    lines.append("")

    lines += ["## 6. 口径", "",
              "- 软证据：b8 已记录的每个 finding 的 p=P(1)/(P(0)+P(1))；把生成 JSON 中 4 个 finding 取值 token 的"
              "输入嵌入替换为 (1−p)·E[0]+p·E[1]（在 decoder 第 0 层入口替换 hidden state，等价于替换输入嵌入），"
              "前向一次后取 BI-RADS 取值位置在 '1'..'5' 上的归一化分布。",
              "- hard 为本脚本用同一份自生成 JSON、不替换嵌入时重算的 BI-RADS 概率；soft 的唯一差别是 4 个 finding 取值位置的输入嵌入被换成 (1−p)E[0]+pE[1]。b8 记录的概率仅用于提供 p 与自检。",
              "- 组内区分用 GT 征象构成的前缀（teacher），BI-RADS 位置仍按模型分布读取。",
              ""]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
