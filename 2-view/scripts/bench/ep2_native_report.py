#!/usr/bin/env python3
"""原分辨率 ep2 vs 普通分辨率 ep2 的 b8 测试集对比报告 → outputs/eval/ep2_native_report.md

口径与 ep3_b8_report.md 一致：
- BI-RADS / density：probs argmax 的 macro-F1 / acc / bACC + OvR macro AUC，另给各类 P/R
- BI-RADS ≥3 AUC：score = P3+P4+P5
- findings：AUROC；F1/P/R 用贪心输出（greedy_findings）
- 病灶检出：GT 有病灶乳房中贪心任一 finding=1 的比例；按 size_bins.json 的小/中/大分档
- 三类分析（基于普通分辨率 ep2）：detected / missed_sign / missed_nosign，再看原分辨率在各类里的检出
- 差值显著性：study 级配对 bootstrap（1000 次，seed 42）
"""
import json
import random
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJ / "scripts" / "eval"))
sys.path.insert(0, str(PROJ / "scripts" / "bench"))
import common as C          # noqa: E402
import ep3_b8_report as R   # noqa: E402

E = PROJ / "outputs" / "eval"
S = PROJ / "outputs" / "scoring"
OUT = E / "ep2_native_report.md"
SIZE_BINS = E / "size_bins.json"
FIND = C.FINDINGS
BIR = C.BIRADS_CLASSES
DEN = C.DENSITY_CLASSES
N_BOOT, SEED = 1000, 42

RUNS = [
    ("ep2 (786432)", S / "vt-ep2-e2_test_b8_with_probs.jsonl"),
    ("ep2-native (896x1536)", S / "vt-ep2-native_test_b8_with_probs.jsonl"),
]


def load(path):
    return C.load_probs_jsonl(str(path))


def argmax_probs(r, kind):
    p = r["p_" + kind]
    keys = sorted(p, key=lambda k: int(k) if str(k).isdigit() else k)
    v = max(keys, key=lambda k: p[k])
    return int(v) if str(v).isdigit() else v


def auc_finding(recs, k):
    return C.auc([r["gt_findings"][k] for r in recs], [r["p_findings"][k] for r in recs])


def birads_ge3_score(r):
    return sum(r["p_birads"][str(i)] for i in (3, 4, 5))


def det(recs):
    """GT 有病灶的乳房中，贪心任一 finding=1 的个数。"""
    les = [r for r in recs if any(r["gt_findings"][k] for k in FIND)]
    hit = [r for r in les if any(r["greedy_findings"].get(k, 0) == 1 for k in FIND)]
    return len(hit), len(les)


def f1_greedy(recs, k):
    tp = sum(1 for r in recs if r["gt_findings"][k] == 1 and r["greedy_findings"].get(k, 0) == 1)
    fp = sum(1 for r in recs if r["gt_findings"][k] == 0 and r["greedy_findings"].get(k, 0) == 1)
    fn = sum(1 for r in recs if r["gt_findings"][k] == 1 and r["greedy_findings"].get(k, 0) == 0)
    return C.prf(tp, fp, fn), (tp, fp, fn)


def section_metrics(recs):
    y_b = [r["gt_birads"] for r in recs]
    p_b = [argmax_probs(r, "birads") for r in recs]
    mb = C.multiclass_metrics(y_b, p_b, BIR)
    y_d = [r["gt_density"] for r in recs]
    p_d = [argmax_probs(r, "density") for r in recs]
    md = C.multiclass_metrics(y_d, p_d, DEN)
    return {
        "birads_macro_f1": mb["macro_f1"], "birads_acc": mb["acc"], "birads_bacc": mb["bacc"],
        "birads_auc": C.ovr_macro_auc(y_b, [{i: r["p_birads"][str(i)] for i in BIR} for r in recs], BIR),
        "birads_ge3_auc": C.auc([1 if v >= 3 else 0 for v in y_b], [birads_ge3_score(r) for r in recs]),
        "birads_cls": mb["per_class"],
        "density_macro_f1": md["macro_f1"], "density_bacc": md["bacc"],
        "density_auc": C.ovr_macro_auc(y_d, [{i: r["p_density"][i] for i in DEN} for r in recs], DEN),
        "density_cls": md["per_class"],
        "findings_auc": R.auc_findings(recs),
    }


def paired(recs_a, recs_b, scorers):
    A = {r["id"]: r for r in recs_a}
    B = {r["id"]: r for r in recs_b}
    ids = [i for i in A if i in B]
    groups = {}
    for i in ids:
        groups.setdefault(A[i]["study"], []).append(i)
    studies = sorted(groups)
    rng = random.Random(SEED)
    acc = {k: [] for k in scorers}
    for _ in range(N_BOOT):
        sid = []
        for _ in range(len(studies)):
            sid.extend(groups[studies[rng.randrange(len(studies))]])
        ra = [A[i] for i in sid]
        rb = [B[i] for i in sid]
        for k, fn in scorers.items():
            va, vb = fn(ra), fn(rb)
            if va is not None and vb is not None:
                acc[k].append(vb - va)
    out = {}
    for k, s in acc.items():
        if len(s) < 100:
            out[k] = (None, None)
        else:
            s.sort()
            out[k] = (s[int(0.025 * len(s))], s[min(int(0.975 * len(s)), len(s) - 1)])
    return out


def f6(v):
    return "—" if v is None else f"{v:.6f}"


def main():
    data = {name: load(p) for name, p in RUNS if p.exists()}
    missing = [name for name, p in RUNS if not p.exists()]
    lines = ["# 原分辨率 ep2（896×1536）vs 普通分辨率 ep2（786432）· b8 测试集", ""]
    if missing:
        lines.append(f"**缺失**：{', '.join(missing)}")
    lines += ["| model | BI-RADS AUC | ≥3 AUC | BI-RADS F1 | acc | bACC | Density AUC | Density F1 | Density bACC | findings mean AUROC |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    M = {}
    for name, _ in RUNS:
        if name not in data:
            continue
        m = section_metrics(data[name])
        M[name] = m
        lines.append(f"| {name} | {f6(m['birads_auc'])} | {f6(m['birads_ge3_auc'])} | {f6(m['birads_macro_f1'])} | "
                     f"{f6(m['birads_acc'])} | {f6(m['birads_bacc'])} | {f6(m['density_auc'])} | "
                     f"{f6(m['density_macro_f1'])} | {f6(m['density_bacc'])} | {f6(m['findings_auc'])} |")
    lines.append("")

    # ---- 各类 P/R ----
    lines += ["## 各类 precision / recall（probs argmax / greedy）", "",
              "| model | 类别 | precision | recall | F1 | support |", "|---|---|---|---|---|---|"]
    for name, _ in RUNS:
        if name not in data:
            continue
        for kind, cls in (("BI-RADS", BIR), ("Density", DEN)):
            key = "birads_cls" if kind == "BI-RADS" else "density_cls"
            for c, v in M[name][key].items():
                lines.append(f"| {name} | {kind} {c} | {f6(v['precision'])} | {f6(v['recall'])} | "
                             f"{f6(v['f1'])} | {v['support']} |")
    lines.append("")

    # ---- findings ----
    lines += ["## findings（AUC=AUROC，F1/P/R 用贪心输出）", "",
              "| model | finding | AUC | precision | recall | F1 | TP/FP/FN |", "|---|---|---|---|---|---|---|"]
    for name, _ in RUNS:
        if name not in data:
            continue
        for k in FIND:
            (p, r_, f_), (tp, fp, fn) = f1_greedy(data[name], k)
            lines.append(f"| {name} | {k} | {f6(auc_finding(data[name], k))} | "
                         f"{f6(p)} | {f6(r_)} | {f6(f_)} | {tp}/{fp}/{fn} |")
    lines.append("")

    # ---- 检出（总体 + 大小档）----
    sb = json.loads(SIZE_BINS.read_text(encoding="utf-8")) if SIZE_BINS.exists() else None
    lines += ["## 病灶检出（GT 有病灶乳房，贪心任一 finding=1）", ""]
    if sb:
        lines += [f"大小分档阈值（框面积/整图面积）：小 ≤ {sb['thresholds_pct']['t1']:.4f}%，"
                  f"中 ≤ {sb['thresholds_pct']['t2']:.4f}%，大 > {sb['thresholds_pct']['t2']:.4f}%；"
                  f"样本数 {sb['counts']}", ""]
    lines += ["| model | 总体 | 小 | 中 | 大 |", "|---|---|---|---|---|"]
    for name, _ in RUNS:
        if name not in data:
            continue
        recs = data[name]
        tot_h, tot = det(recs)
        cells = [f"{tot_h}/{tot}"]
        for b in ("small", "medium", "large"):
            sub = [r for r in recs if sb and sb["per_breast"].get(r["id"], {}).get("bin") == b]
            h, n = det(sub)
            cells.append(f"{h}/{n}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")

    # ---- 三类分析（基于普通分辨率 ep2）----
    base_name = "ep2 (786432)"
    if base_name in data:
        base = data[base_name]
        les = [r for r in base if r["gt_birads"] >= 3]
        cat = {"detected": [], "missed_sign": [], "missed_nosign": []}
        for r in les:
            pred = int(argmax_probs(r, "birads"))
            if pred >= 3:
                cat["detected"].append(r["id"])
            else:
                mx = max(r["p_findings"][k] for k in FIND)
                if pred == 2 or mx > 0.1:
                    cat["missed_sign"].append(r["id"])
                else:
                    cat["missed_nosign"].append(r["id"])
        lines += ["## 三类分析（基于普通分辨率 ep2 的 b8 结果，GT BI-RADS ≥3 的 192 个乳房）", "",
                  "detected = 预测 BI-RADS ≥3；missed_sign = 预测 ≤2 且（预测=2 或 4 个 finding 的 probs_norm 最大值 >0.1）；"
                  "missed_nosign = 其余。", "",
                  "| 类别 | 个数 | 原分辨率模型检出数（预测 BI-RADS ≥3） |", "|---|---|---|"]
        for c, ids in cat.items():
            if "ep2-native (896x1536)" in data:
                nmap = {r["id"]: r for r in data["ep2-native (896x1536)"]}
                hit = sum(1 for i in ids if i in nmap and int(argmax_probs(nmap[i], "birads")) >= 3)
            else:
                hit = None
            lines.append(f"| {c} | {len(ids)} | {hit if hit is not None else '—'} |")
        lines.append("")

    # ---- 配对 bootstrap ----
    if "ep2 (786432)" in data and "ep2-native (896x1536)" in data:
        scorers = {
            "BI-RADS macro AUC": lambda rs: C.ovr_macro_auc([r["gt_birads"] for r in rs],
                                                            [{i: r["p_birads"][str(i)] for i in BIR} for r in rs], BIR),
            "BI-RADS ≥3 AUC": lambda rs: C.auc([1 if r["gt_birads"] >= 3 else 0 for r in rs],
                                               [birads_ge3_score(r) for r in rs]),
            "Density macro AUC": lambda rs: C.ovr_macro_auc([r["gt_density"] for r in rs],
                                                            [{i: r["p_density"][i] for i in DEN} for r in rs], DEN),
            "findings mean AUROC": lambda rs: R.auc_findings(rs),
        }
        for k in FIND:
            scorers[f"finding {k}"] = (lambda kk: lambda rs: C.auc([r["gt_findings"][kk] for r in rs],
                                                                  [r["p_findings"][kk] for r in rs]))(k)
        ci = paired(data["ep2 (786432)"], data["ep2-native (896x1536)"], scorers)
        lines += ["## 差值显著性（配对 bootstrap，study 级，1000 次，seed 42；Δ = native − ep2）", "",
                  "| 指标 | ΔAUC | CI_low | CI_high | 是否显著 |", "|---|---|---|---|---|"]
        for k, (lo, hi) in ci.items():
            va = scorers[k](data["ep2-native (896x1536)"])
            vb = scorers[k](data["ep2 (786432)"])
            d = (va - vb) if (va is not None and vb is not None) else None
            sig = "*" if (lo is not None and (lo > 0 or hi < 0)) else ""
            lines.append(f"| {k} | {f6(d)}{sig} | {f6(lo)} | {f6(hi)} | {'是' if sig else '否'} |")
        lines.append("")

    lines += ["## 口径", "",
              "- 全部为 b8 流水线（`infer_2view_batched.py`），测试集 2000 条；原分辨率推理用 `--image-max-pixels 1376256`（896×1536）。",
              "- macro-F1 / acc / bACC 基于 probs argmax；bACC = 各类 recall 的平均。",
              "- findings 的 F1/P/R 用贪心输出（greedy_findings），AUC 用 prob。",
              "- 病灶大小分档脚本：`scripts/bench/lesion_size_bins.py`（面积比三分位，报告只读 `size_bins.json`，换阈值不必重推理）。"]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:12]))


if __name__ == "__main__":
    main()
