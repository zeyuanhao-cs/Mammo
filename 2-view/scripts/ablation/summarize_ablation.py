#!/usr/bin/env python3
"""
消融汇总 (不依赖阈值): 每条件的排序/argmax 指标 + 相对 orig 的差值
+ study 级配对 bootstrap (1000 次, seed=42) 95% CI。

指标定义 (与 eval/common.py 及 eval_test.py 口径一致):
  findings AUC x4          P(finding=1) vs gt
  birads_ovr               OvR macro AUC (只计入 test 中 support>=10 的类)
  birads_ge3               y=gt>=3, score=P(3)+P(4)+P(5)
  birads_ge4               y=gt>=4, score=P(4)+P(5)
  birads_3v4               子集 gt in {3,4}, y=gt==4, score=(P4+P5)/(P3+P4+P5)
  birads_2v4               子集 gt in {2,4,5}, y=gt>=4, score=(P4+P5)/(P2+P4+P5)
                            (与 eval_test.py 的 2v45-B 版一致)
  density_ovr              OvR macro AUC (A/B/C/D, support>=10 的类)
  birads_argmax_f1         argmax(probs_norm) 的 BI-RADS 5 类 macro-F1

样本不足规则: 二分类指标 n_pos<10 或 n_neg<10 标 "样本不足";
OvR 中 support<10 的类被剔除 (全部满足, 见输出)。
bootstrap: 同一 resample (seed=42, 按 study 有放回) 同时用于 orig 与各条件, 取差值。
纯 Python, 无 GPU 依赖。
"""

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent / "eval"))
import common as C  # noqa: E402

FINDINGS = C.FINDINGS
BIRADS = [1, 2, 3, 4, 5]
DENSITY = ["A", "B", "C", "D"]
MIN_CLASS_N = 10
ABLATION_CONDS = ["blank", "cc_only", "mlo_only", "swap_side", "shuffle_patient"]


# ---------------- loading ----------------

def load_cond(path):
    """读取概率 jsonl (兼容 probs=None 行, 计为异常), 返回 (recs, n_lines, n_none)."""
    recs, seen = [], set()
    n_lines, n_none = 0, 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            d = json.loads(line)
            rid = d["id"]
            assert rid not in seen, f"duplicate id {rid} in {path}"
            seen.add(rid)
            if d.get("probs_norm") is None:
                n_none += 1
                continue
            gt, greedy, pn = d["ground_truth"], d.get("greedy_prediction") or {}, d["probs_norm"]
            recs.append({
                "id": rid,
                "study": C.study_id_of(rid),
                "gt_birads": int(gt["breast_birads"]),
                "gt_density": gt["breast_density"],
                "gt_findings": {k: int(gt["findings"][k]) for k in FINDINGS},
                "p_density": {c: float(pn["breast_density"][c]) for c in DENSITY},
                "p_birads": {str(i): float(pn["breast_birads"][str(i)]) for i in BIRADS},
                "p_findings": {k: C._finding_prob(pn, k) for k in FINDINGS},
            })
    return recs, n_lines, n_none


# ---------------- metric 构建与求值 ----------------
# build(recs) -> 逐样本数组; eval(arrays, positions) -> float/None

def build_auc(y, s):
    return {"kind": "auc", "y": y, "s": s}


def build_auc_subset(y, s, mask):
    return {"kind": "auc", "y": y, "s": s, "mask": mask}


def build_ovr(y, s_per_class, classes):
    return {"kind": "ovr", "y": y, "s_per_class": s_per_class, "classes": classes}


def build_macrof1(y, yhat):
    return {"kind": "macrof1", "y": y, "yhat": yhat}


def eval_metric(a, pos):
    if a["kind"] == "auc":
        mask = a.get("mask")
        if mask is not None:
            pos = [i for i in pos if mask[i]]
        if not pos:
            return None
        return C.auc([a["y"][i] for i in pos], [a["s"][i] for i in pos])
    if a["kind"] == "ovr":
        aucs = []
        for c in a["classes"]:
            yc = [1 if a["y"][i] == c else 0 for i in pos]
            if sum(yc) == 0 or sum(yc) == len(yc):
                continue
            v = C.auc(yc, [a["s_per_class"][c][i] for i in pos])
            if v is not None:
                aucs.append(v)
        return sum(aucs) / len(aucs) if aucs else None
    if a["kind"] == "macrof1":
        y = [a["y"][i] for i in pos]
        yh = [a["yhat"][i] for i in pos]
        return C.multiclass_metrics(y, yh, BIRADS)["macro_f1"]
    raise ValueError(a["kind"])


def make_metric_builders():
    """返回 {key: (描述, builder_fn, support_fn)}; builder 输入 recs (需含 p_* 字段)."""
    M = {}

    for k in FINDINGS:
        def b(recs, k=k):
            return build_auc([r["gt_findings"][k] for r in recs],
                             [r["p_findings"][k] for r in recs])
        def sup(recs, k=k):
            n_pos = sum(r["gt_findings"][k] for r in recs)
            return n_pos, len(recs) - n_pos
        M["auc_" + k] = (f"{k} AUC", b, sup)

    def b_bovr(recs):
        classes = [c for c in BIRADS
                   if sum(1 for r in recs if r["gt_birads"] == c) >= MIN_CLASS_N]
        return build_ovr([r["gt_birads"] for r in recs],
                         {c: [r["p_birads"][str(c)] for r in recs] for c in classes}, classes)
    M["birads_ovr"] = ("BI-RADS macro-OvR AUC", b_bovr, lambda recs: (None, None))

    def b_ge3(recs):
        return build_auc([1 if r["gt_birads"] >= 3 else 0 for r in recs],
                         [r["p_birads"]["3"] + r["p_birads"]["4"] + r["p_birads"]["5"]
                          for r in recs])
    def sup_ge3(recs):
        n_pos = sum(1 for r in recs if r["gt_birads"] >= 3)
        return n_pos, len(recs) - n_pos
    M["birads_ge3"] = ("BI-RADS >=3 AUC [P(3)+P(4)+P(5)]", b_ge3, sup_ge3)

    def b_ge4(recs):
        return build_auc([1 if r["gt_birads"] >= 4 else 0 for r in recs],
                         [r["p_birads"]["4"] + r["p_birads"]["5"] for r in recs])
    def sup_ge4(recs):
        n_pos = sum(1 for r in recs if r["gt_birads"] >= 4)
        return n_pos, len(recs) - n_pos
    M["birads_ge4"] = ("BI-RADS >=4 AUC [P(4)+P(5)]", b_ge4, sup_ge4)

    def b_3v4(recs):
        mask = [r["gt_birads"] in (3, 4) for r in recs]
        y, s = [], []
        for r in recs:
            p45 = r["p_birads"]["4"] + r["p_birads"]["5"]
            d = r["p_birads"]["3"] + p45
            y.append(1 if r["gt_birads"] == 4 else 0)
            s.append(p45 / d if d > 0 else 0.5)
        return build_auc_subset(y, s, mask)
    def sup_3v4(recs):
        sub = [r for r in recs if r["gt_birads"] in (3, 4)]
        return sum(1 for r in sub if r["gt_birads"] == 4), sum(1 for r in sub if r["gt_birads"] == 3)
    M["birads_3v4"] = ("BI-RADS 3v4 AUC [(P4+P5)/(P3+P4+P5)]", b_3v4, sup_3v4)

    def b_2v4(recs):
        mask = [r["gt_birads"] in (2, 4, 5) for r in recs]
        y, s = [], []
        for r in recs:
            p45 = r["p_birads"]["4"] + r["p_birads"]["5"]
            d = r["p_birads"]["2"] + p45
            y.append(1 if r["gt_birads"] >= 4 else 0)
            s.append(p45 / d if d > 0 else 0.5)
        return build_auc_subset(y, s, mask)
    def sup_2v4(recs):
        sub = [r for r in recs if r["gt_birads"] in (2, 4, 5)]
        return (sum(1 for r in sub if r["gt_birads"] >= 4),
                sum(1 for r in sub if r["gt_birads"] == 2))
    M["birads_2v4"] = ("BI-RADS 2v4/5 AUC [(P4+P5)/(P2+P4+P5)]", b_2v4, sup_2v4)

    def b_dovr(recs):
        classes = [c for c in DENSITY
                   if sum(1 for r in recs if r["gt_density"] == c) >= MIN_CLASS_N]
        return build_ovr([r["gt_density"] for r in recs],
                         {c: [r["p_density"][c] for r in recs] for c in classes}, classes)
    M["density_ovr"] = ("density macro-OvR AUC", b_dovr, lambda recs: (None, None))

    def b_f1(recs):
        y = [r["gt_birads"] for r in recs]
        yhat = [int(max(r["p_birads"], key=r["p_birads"].get)) for r in recs]
        return build_macrof1(y, yhat)
    def sup_f1(recs):
        return (len(recs), None)
    M["birads_argmax_f1"] = ("argmax BI-RADS Macro-F1", b_f1, sup_f1)

    return M


# ---------------- paired bootstrap ----------------

def make_bootstrap_indices(recs, n_boot, seed):
    """按 study 有放回重采样, 返回每次 bootstrap 的记录下标列表 (一次生成, 全条件共用)."""
    groups = defaultdict(list)
    for i, r in enumerate(recs):
        groups[r["study"]].append(i)
    studies = sorted(groups.keys())
    rng = random.Random(seed)
    out = []
    for _ in range(n_boot):
        idx = []
        for _ in range(len(studies)):
            idx.extend(groups[studies[rng.randrange(len(studies))]])
        out.append(idx)
    return out


def paired_diff_ci(orig_boot_vals, a_cond, boot_idx, n_boot):
    """同一 resample 下 cond-orig 差值的 95% CI; orig 各 resample 的值已预计算缓存."""
    diffs = []
    for v0, pos in zip(orig_boot_vals, boot_idx):
        v1 = eval_metric(a_cond, pos)
        if v1 is not None and v0 is not None:
            diffs.append(v1 - v0)
    if len(diffs) < 0.95 * n_boot:
        return None, None, len(diffs)
    diffs.sort()
    lo = diffs[int(0.025 * len(diffs))]
    hi = diffs[min(int(0.975 * len(diffs)), len(diffs) - 1)]
    return lo, hi, len(diffs)


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent.parent
                                          / "outputs" / "ablation"))
    ap.add_argument("--conditions", default=",".join(ABLATION_CONDS))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.root)
    conds = args.conditions.split(",")
    t0 = time.time()

    orig_recs, n_lines, n_none = load_cond(root / "orig" / "test_with_probs.jsonl")
    print(f"[orig] rows={n_lines} probs_none={n_none} loaded={len(orig_recs)}")

    cond_data = {}
    for c in conds:
        recs, nl, nn = load_cond(root / c / "test_with_probs.jsonl")
        ids_o = [r["id"] for r in orig_recs]
        ids_c = {r["id"] for r in recs}
        missing = sum(1 for i in ids_o if i not in ids_c)
        print(f"[{c}] rows={nl} probs_none={nn} loaded={len(recs)} missing_vs_orig={missing}")
        cond_data[c] = (recs, nl, nn, missing)

    # 对齐: 以 orig 顺序为基准 (gt 相同, id 集合应一致)
    id_pos = {r["id"]: i for i, r in enumerate(orig_recs)}
    M = make_metric_builders()

    # bootstrap resample (一次生成, 全条件/全指标共用 => 严格配对)
    boot_idx = make_bootstrap_indices(orig_recs, args.n_boot, args.seed)

    results = {"n_boot": args.n_boot, "seed": args.seed, "metrics": {}, "anomalies": {}}
    table = []  # (label, orig_val_str, {cond: cell_str})

    for key in M:
        label, builder, sup = M[key]
        # 点估计 + 支持数 (用 orig 的 gt, 各条件相同)
        a_orig = builder(orig_recs)
        n_pos, n_neg = sup(orig_recs)
        if a_orig["kind"] == "ovr":
            # OvR: n_pos=计入的类数(support>=10), 无有效类则样本不足
            n_pos, n_neg = len(a_orig["classes"]), None
            insufficient = n_pos == 0
        else:
            insufficient = (n_pos is not None and n_pos < MIN_CLASS_N) or \
                           (n_neg is not None and n_neg < MIN_CLASS_N)
        v_orig = None if insufficient else eval_metric(a_orig, list(range(len(orig_recs))))
        # orig 各 bootstrap resample 的指标值只算一次, 所有条件共用 (严格配对且省时)
        orig_boot_vals = (None if insufficient else
                          [eval_metric(a_orig, pos) for pos in boot_idx])
        row = {"label": label, "n_pos": n_pos, "n_neg": n_neg,
               "insufficient": insufficient, "orig": v_orig, "conditions": {}}
        cells = {}
        cells["orig"] = ("样本不足" if insufficient else
                         ("n/a" if v_orig is None else f"{v_orig:.4f}"))
        for c in conds:
            recs, nl, nn, missing = cond_data[c]
            assert missing == 0, f"条件 {c} 缺少 {missing} 个 orig id, 终止"
            # 按 orig id 顺序重排条件记录
            pos_of = {r["id"]: i for i, r in enumerate(recs)}
            aligned = [recs[pos_of[r["id"]]] for r in orig_recs]
            if insufficient:
                row["conditions"][c] = {"value": None, "diff": None, "ci": None,
                                        "note": "样本不足"}
                cells[c] = "样本不足"
                continue
            a_cond = builder(aligned)
            v_cond = eval_metric(a_cond, list(range(len(aligned))))
            if v_cond is None or v_orig is None:
                row["conditions"][c] = {"value": v_cond, "diff": None, "ci": None}
                cells[c] = "n/a" if v_cond is None else f"{v_cond:.4f}"
                continue
            lo, hi, n_ok = paired_diff_ci(orig_boot_vals, a_cond, boot_idx, args.n_boot)
            diff = v_cond - v_orig
            row["conditions"][c] = {"value": v_cond, "diff": diff,
                                    "ci": [lo, hi] if lo is not None else None,
                                    "n_boot_valid": n_ok}
            cells[c] = (f"{v_cond:.4f} Δ{diff:+.4f} "
                        f"[{lo:+.4f},{hi:+.4f}]" if lo is not None else
                        f"{v_cond:.4f} Δ{diff:+.4f} [n/a]")
        results["metrics"][key] = row
        if a_orig["kind"] == "ovr":
            sup_str = f" ({n_pos}类 support>=10)"
        else:
            sup_str = f" (n_pos={n_pos}" + (f", n_neg={n_neg}" if n_neg is not None else "") + ")"
        table.append((label + sup_str, cells))
        print(f"[metric] {label}: done ({time.time()-t0:.0f}s)")

    # 异常与耗时
    for c in conds + ["orig"]:
        if c == "orig":
            results["anomalies"][c] = {"probs_none": n_none, "rows": n_lines}
        else:
            recs, nl, nn, missing = cond_data[c]
            results["anomalies"][c] = {"rows": nl, "probs_none": nn, "missing_vs_orig": missing}
    for name in ("smoke_report.json", "full_report.json"):
        p = root / name
        if p.exists():
            results[name] = json.loads(p.read_text(encoding="utf-8"))

    out_json = root / "summary.json"
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # markdown 表
    lines = ["# 图像消融汇总 (Run A Repeat-OS LoRA, 官方 test 2000 乳房, 无阈值指标)",
             "",
             "说明: 所有条件共用 orig 的贪心预测 JSON 作为 teacher-forcing 前缀, "
             "各字段 (尤其 BI-RADS 等位于 JSON 后部的字段) 是前缀固定下的条件概率, "
             "概率差异只来自图像替换。",
             f"配对 bootstrap: {args.n_boot} 次, seed={args.seed}, 按 study 重采样; "
             "Δ=条件-orig, [ ]为Δ的95% CI; 样本不足: n_pos 或 n_neg < 10。",
             ""]
    header = "| 指标 | orig | " + " | ".join(conds) + " |"
    sep = "|---|" + "---|" * (len(conds) + 1)
    lines += [header, sep]
    for label, cells in table:
        lines.append("| " + label + " | " + " | ".join(cells[c] for c in ["orig"] + conds) + " |")
    md = "\n".join(lines) + "\n"
    (root / "summary_table.md").write_text(md, encoding="utf-8")

    print("\n" + md)
    print(f"[done] {time.time()-t0:.0f}s -> {out_json}, {root / 'summary_table.md'}")


if __name__ == "__main__":
    main()
