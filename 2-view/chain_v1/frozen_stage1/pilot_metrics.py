#!/usr/bin/env python3
"""阶段二指标汇总: a-h 指标 + 对齐器基线 + study 级 paired bootstrap + pilot_v1_report.md.

输入: chain_v1/outputs/pilot/{b0,c1,c2}_{hard,soft,swap,teacher,scan,b8}.jsonl
对齐器: numpy/scipy 手写 softmax 回归（特征 = B0 的 BI-RADS 5 类 + 4 征象 log 概率）,
        study 分组 5 折交叉验证（折外预测）。
"""
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr

PROJ = Path("/home/z/zhao/Mammography/Mammo/2-view")
P = PROJ / "chain_v1" / "outputs" / "pilot"
FIND = ["mass", "suspicious_calcification", "asymmetry", "architectural_distortion"]
SEED, N_BOOT = 42, 1000
MTRAIN_STEPS = 2000


def load(fn):
    f = P / fn
    if not f.exists():
        return {}
    return {json.loads(l)["id"]: json.loads(l) for l in open(f, encoding="utf-8")}


def listload(fn):
    f = P / fn
    if not f.exists():
        return []
    return [json.loads(l) for l in open(f, encoding="utf-8")]


# ---------------------------------------------------------------- 指标
def auc(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(np.concatenate([neg, pos]), kind="mergesort")
    ranks = np.empty(len(order))
    allv = np.concatenate([neg, pos])
    i = 0
    ranks_sorted = np.empty(len(allv))
    while i < len(allv):
        j = i
        while j < len(allv) and allv[order[j]] == allv[order[i]]:
            j += 1
        ranks_sorted[i:j] = (i + 1 + j) / 2.0
        i = j
    ranks[order] = ranks_sorted
    n_pos, n_neg = len(pos), len(neg)
    return float((ranks[y == 0].sum() if False else ranks[len(neg):].sum()
                  - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def prf_at05(y, p):
    yhat = [1 if v > 0.5 else 0 for v in p]
    tp = sum(1 for a, b in zip(y, yhat) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(y, yhat) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y, yhat) if a == 1 and b == 0)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def ovr_macro_auc(y, prob_rows, n_cls):
    aucs = []
    for c in range(n_cls):
        a = auc([1 if v == c else 0 for v in y], [row[c] for row in prob_rows])
        if a is not None:
            aucs.append(a)
    return float(np.mean(aucs)) if aucs else None


def birads_block(rows):
    """rows: list of record(dict)。返回 BI-RADS 指标 dict。"""
    rows = [r for r in rows if r.get("p_birads")]
    y = [r["gt_birads"] - 1 for r in rows]                      # 0-based
    pred = [r["pred_birads"] - 1 for r in rows]
    prob = [[r["p_birads"][str(c + 1)] for c in range(5)] for r in rows]
    ge3_score = [1.0 - r["p_birads"].get("1", 0) - r["p_birads"].get("2", 0) for r in rows]
    macro_auc = ovr_macro_auc(y, prob, 5)
    ge3_auc = auc([1 if v >= 2 else 0 for v in y], ge3_score)
    n = len(y)
    acc = float(np.mean([a == b for a, b in zip(y, pred)]))
    recs = {c: (float(np.mean([1 for a, b in zip(y, pred) if a == c and b == c]) /
                   max(1, sum(1 for a in y if a == c)))) for c in range(5)}
    f1s = []
    for c in range(5):
        prec = sum(1 for a, b in zip(y, pred) if a == c and b == c) / max(1, sum(1 for b in pred if b == c))
        f1 = 2 * prec * recs[c] / max(1e-9, prec + recs[c])
        f1s.append(f1)
    macro_f1 = float(np.mean(f1s))
    bacc = float(np.mean(list(recs.values())))
    # >=3 特异度 90/95% 灵敏度（在 GT<=2 上定阈值）
    yge3 = [1 if v >= 2 else 0 for v in y]
    thr = {}
    for spec_t, tag in ((0.90, "spec90"), (0.95, "spec95")):
        neg = sorted([s for s, v in zip(ge3_score, yge3) if v == 0])
        k = max(0, min(len(neg) - 1, int(math.ceil(spec_t * len(neg))) - 1))
        t = neg[k]
        sens = float(np.mean([1 if s > t else 0 for s, v in zip(ge3_score, yge3) if v == 1])) \
            if any(v == 1 for v in yge3) else None
        thr[tag] = {"threshold": float(t), "sensitivity": sens}
    per_cls_prob = {str(c + 1): [row[c] for row in prob] for c in range(5)}
    return {"n": n, "macro_auc": macro_auc, "ge3_auc": ge3_auc, "macro_f1": macro_f1,
            "acc": acc, "bacc": bacc, "per_class_recall": {str(c + 1): recs[c] for c in range(5)},
            "ge3_sens_at_spec": thr, "_y": y, "_ge3_score": ge3_score,
            "_prob": prob, "_pred": pred, "_per_cls_prob": per_cls_prob}


def findings_block(rows):
    out = {"auc": {}, "f1": {}, "precision": {}, "recall": {}}
    for k in FIND:
        key = f"findings.{k}"
        y = [r["gt_findings"][k] for r in rows]
        s = [r["p_findings"][k] for r in rows]
        out["auc"][k] = auc(y, s)
        _, _, f1 = prf_at05(y, s)
        out["f1"][k] = f1
        out["precision"][k], out["recall"][k], _ = prf_at05(y, s)
    return out


def density_block(rows):
    rows = [r for r in rows if r.get("probs_norm", {}).get("breast_density")]
    if not rows:
        return {}
    cls = ["A", "B", "C", "D"]
    y = [cls.index(r["gt_density"]) for r in rows if r["gt_density"] in cls]
    prob = [[r["probs_norm"]["breast_density"].get(c, 0.0) for c in cls]
            for r in rows if r["gt_density"] in cls]
    pred = [int(np.argmax(row)) for row in prob]
    recs = {c: sum(1 for a, b in zip(y, pred) if a == c and b == c) / max(1, sum(1 for a in y if a == c))
            for c in range(4)}
    f1s = []
    for c in range(4):
        prec = sum(1 for a, b in zip(y, pred) if a == c and b == c) / max(1, sum(1 for b in pred if b == c))
        f1s.append(2 * prec * recs[c] / max(1e-9, prec + recs[c]))
    return {"n": len(y), "macro_auc": ovr_macro_auc(y, prob, 4),
            "macro_f1": float(np.mean(f1s)), "bacc": float(np.mean(list(recs.values())))}


def hard_inconsistency(rows):
    """征象与 BI-RADS 矛盾: A = pred>=3 但 argmax 征象全 0; B = pred<=1 但任一 argmax 征象=1."""
    a = b = 0
    for r in rows:
        signs = [1 if r["p_findings"][k] > 0.5 else 0 for k in FIND]
        if r["pred_birads"] >= 3 and sum(signs) == 0:
            a += 1
        if r["pred_birads"] <= 1 and sum(signs) > 0:
            b += 1
    return {"pred_ge3_no_sign": a, "pred_le1_with_sign": b}


def soft_consistency(rows, soft_ids):
    sub = [rows[i] for i in soft_ids if i in rows]
    if not sub:
        return {}
    sp = spearmanr([r["p_birads_ge3"] for r in sub], [r["max_sign_p"] for r in sub]).statistic
    return {"n": len(sub),
            "median_p_ge3": float(np.median([r["p_birads_ge3"] for r in sub])),
            "spearman_ge3_vs_maxsign": float(sp)}


def gt23_auc(rows, soft_ids):
    sub = [rows[i] for i in soft_ids if i in rows]
    y = [1 if r["gt_birads"] >= 3 else 0 for r in sub]
    s = [r["max_sign_p"] for r in sub]
    if len(set(y)) < 2:
        return None
    return auc(y, s)


# ---------------------------------------------------------------- 对齐器
def softmax_reg_fit(X, Y, l2=1e-4):
    """X: [n,d], Y: [n] 0..4 -> (W[d,5], b[5])，L-BFGS 全量优化。"""
    n, d = X.shape
    K = 5
    W0 = np.zeros((d, K))
    b0 = np.zeros(K)

    def unpack(t):
        return t[:d * K].reshape(d, K), t[d * K:]

    def f(t):
        W, b = unpack(t)
        Z = X @ W + b
        Z -= Z.max(axis=1, keepdims=True)
        Pm = np.exp(Z)
        Pm /= Pm.sum(axis=1, keepdims=True)
        ce = -np.log(Pm[np.arange(n), Y] + 1e-12).mean()
        gZ = Pm
        gZ[np.arange(n), Y] -= 1
        gZ /= n
        gW = X.T @ gZ + l2 * W
        gb = gZ.sum(axis=0)
        return ce, np.concatenate([gW.ravel(), gb])

    res = minimize(f, np.concatenate([W0.ravel(), b0]), jac=True, method="L-BFGS-B",
                   options={"maxiter": MTRAIN_STEPS})
    return unpack(res.x)


def aligner_baseline(b0_rows):
    """study 分组 5 折交叉验证的 softmax 对齐器（折外预测）。"""
    ids = [i for i, r in b0_rows.items() if r.get("p_birads")]
    ids.sort()
    X, Y, groups = [], [], []
    for i in ids:
        r = b0_rows[i]
        logb = [math.log(max(1e-9, r["p_birads"][str(c + 1)])) for c in range(5)]
        logf = [math.log(max(1e-9, r["p_findings"][k])) for k in FIND]
        X.append(logb + logf)
        Y.append(r["gt_birads"] - 1)
        groups.append(i.rsplit("_", 1)[0])
    X, Y = np.array(X), np.array(Y)
    studies = sorted(set(groups))
    folds = {s: k % 5 for k, s in enumerate(studies)}
    prob = np.zeros((len(ids), 5))
    for k in range(5):
        te = [j for j, s in enumerate(groups) if folds[s] == k]
        tr = [j for j, s in enumerate(groups) if folds[s] != k]
        if not tr or not te:
            continue
        W, b = softmax_reg_fit(X[tr], Y[tr])
        Z = X[te] @ W + b
        Z -= Z.max(axis=1, keepdims=True)
        Pm = np.exp(Z)
        Pm /= Pm.sum(axis=1, keepdims=True)
        prob[te] = Pm
    y = Y.tolist()
    pred = prob.argmax(axis=1).tolist()
    recs = {c: sum(1 for a, b in zip(y, pred) if a == c and b == c) / max(1, sum(1 for a in y if a == c))
            for c in range(5)}
    f1s = []
    for c in range(5):
        prec = sum(1 for a, b in zip(y, pred) if a == c and b == c) / max(1, sum(1 for b in pred if b == c))
        f1s.append(2 * prec * recs[c] / max(1e-9, prec + recs[c]))
    ge3_score = [1.0 - p[0] - p[1] for p in prob]
    return {
        "n": len(ids), "macro_auc": ovr_macro_auc(y, prob.tolist(), 5),
        "ge3_auc": auc([1 if v >= 2 else 0 for v in y], ge3_score),
        "macro_f1": float(np.mean(f1s)),
        "acc": float(np.mean([a == b for a, b in zip(y, pred)])),
        "bacc": float(np.mean(list(recs.values()))),
        "_prob": prob.tolist(), "_y": y, "_ids": ids,
    }


# ---------------------------------------------------------------- bootstrap
def study_groups(ids):
    g = defaultdict(list)
    for j, i in enumerate(ids):
        g[i.rsplit("_", 1)[0]].append(j)
    return sorted(g), [g[s] for s in sorted(g)]


def paired_bootstrap(ids_a, metric_a, ids_b, metric_b, n_boot=N_BOOT, seed=SEED):
    """metric_*: id -> float。返回 (Δ均值点的CI低, CI高) —— Δ = b - a 逐 study 重采样。"""
    common = sorted(set(ids_a) & set(ids_b))
    va = np.array([metric_a[i] for i in common])
    vb = np.array([metric_b[i] for i in common])
    studies, glist = study_groups(common)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(studies), len(studies))
        idx = [j for gi in pick for j in glist[gi]]
        diffs.append(float(np.mean(vb[idx]) - np.mean(va[idx])))
    diffs.sort()
    lo, hi = diffs[int(0.025 * len(diffs))], diffs[min(int(0.975 * len(diffs)), len(diffs) - 1)]
    return float(np.mean(vb) - np.mean(va)), lo, hi


def birads_ge3_metric(rows_by_id):
    return {i: r["p_birads_ge3"] for i, r in rows_by_id.items() if r.get("p_birads")}


def macro_auc_metric(rows_by_id):
    """对齐器以外：直接用预测 argmax 的 OvR macro AUC 不便逐样本 bootstrap，
    用逐样本 >=3 score 的 AUC 代替做 bootstrap（主表 CI 也给 macro AUC 的点估计）。
    这里返回 ge3 分数。"""
    return birads_ge3_metric(rows_by_id)


# ---------------------------------------------------------------- 主流程
def convert_b8_rows(fn):
    """b8 管线 jsonl → eval_soft 同构记录（B0 主要指标用 b8 全量）。
    b8 字段: id, probs_norm{breast_birads, breast_density, findings.*}, ground_truth。"""
    out = {}
    f = P / fn
    if not f.exists():
        return out
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        pn = r["probs_norm"]
        gt = r["ground_truth"]
        bi = {k: float(v) for k, v in pn["breast_birads"].items()}
        pf = {k: float((pn.get(f"findings.{k}") or {}).get("1", 0.0)) for k in FIND}
        out[r["id"]] = {
            "id": r["id"],
            "gt_birads": int(gt["breast_birads"]),
            "gt_density": gt.get("breast_density"),
            "gt_findings": {k: int(v) for k, v in gt["findings"].items()},
            "p_birads": bi,
            "pred_birads": int(max(bi.items(), key=lambda kv: kv[1])[0]),
            "p_birads_ge3": 1.0 - bi.get("1", 0.0) - bi.get("2", 0.0),
            "p_findings": pf,
            "max_sign_p": max(pf.values()),
        }
    return out


def main():
    b0_hard = load("b0_hard.jsonl")
    c1_soft = load("c1_soft.jsonl")
    c2_soft = load("c2_soft.jsonl")
    # B0 主要指标一律用 eval_soft 硬模式（b0_hard.jsonl），不优先 b8；
    # b8 结果只在文件存在时进附表（见 write_report 末尾，标注"仅供参考"）。
    b0_by = b0_hard
    b0_src = "soft_decode_hard"
    if not b0_by:
        raise SystemExit("缺 B0 输出（b0_hard.jsonl，eval_soft 硬模式全量）")
    b0_b8c = convert_b8_rows("b0_b8_probs.jsonl")  # 仅附表参考用

    soft_ids = json.load(open(P / "ids2_soft.json", encoding="utf-8")) \
        if (P / "ids2_soft.json").exists() else []

    blocks = {}
    for name, rows in (("B0", list(b0_by.values())), ("C1", list(c1_soft.values())),
                       ("C2", list(c2_soft.values()) or None)):
        if rows is None:
            continue
        blocks[name] = {
            "birads": birads_block(rows),
            "findings": findings_block(rows),
            "density": density_block(rows),
            "hard_inconsistency": hard_inconsistency(rows),
        }
    blocks["B0"]["source"] = b0_src
    blocks["B0"]["soft_consistency"] = soft_consistency(b0_by, soft_ids)
    blocks["B0"]["gt23_auc_on_softsubset"] = gt23_auc(b0_by, soft_ids)
    blocks["C1"]["soft_consistency"] = soft_consistency(c1_soft, soft_ids)
    blocks["C1"]["gt23_auc_on_softsubset"] = gt23_auc(c1_soft, soft_ids)
    if c2_soft:
        blocks["C2"]["soft_consistency"] = soft_consistency(c2_soft, soft_ids)
        blocks["C2"]["gt23_auc_on_softsubset"] = gt23_auc(c2_soft, soft_ids)

    # f. 老师模式
    for tag, fn in (("B0", "b0_teacher.jsonl"), ("C1", "c1_teacher.jsonl"), ("C2", "c2_teacher.jsonl")):
        rows = list(load(fn).values())
        if rows:
            blocks[tag]["teacher_birads"] = birads_block(rows)

    # g. 软换硬
    for tag, sfn, wfn in (("C1", "c1_soft.jsonl", "c1_swap.jsonl"),
                          ("C2", "c2_soft.jsonl", "c2_swap.jsonl")):
        swap = load(wfn)
        if not swap:
            continue
        soft = load(sfn)
        blocks[tag]["swap_hard"] = {
            "birads_swap": birads_block(list(swap.values())),
            "median_p_ge3_soft_subset": soft_consistency(soft, soft_ids).get("median_p_ge3"),
            "median_p_ge3_swap_subset": soft_consistency(swap, soft_ids).get("median_p_ge3"),
        }

    # h. 扫描曲线 png（阶段二 C1/C2 + 阶段一起点 vs QC1）
    draw_scan()
    draw_start_qc1_scan()

    # 对齐器
    al = aligner_baseline(b0_by)
    blocks["Aligner"] = al

    # b8 vs soft_decode hard 差别（用对齐检查的 200 条；全量对齐时为 1600）
    align = {}
    if (P / "align_check.json").exists():
        align = json.load(open(P / "align_check.json", encoding="utf-8"))
    blocks["pipeline_check"] = {
        "n": align.get("n", 0),
        "pred_agreement": align.get("argmax_agreement"),
        "median_abs_diff_ge3": align.get("max_abs_prob_diff"),
        "note": "align_check.json（eval_soft hard vs b8，容差 1e-3）",
    }

    # b8 附表（仅供参考，不进主比较；主比较一律 eval_soft）
    if b0_b8c:
        blocks["B0_b8_ref"] = {
            "birads": birads_block(list(b0_b8c.values())),
            "findings": findings_block(list(b0_b8c.values())),
            "density": density_block(list(b0_b8c.values())),
        }

    # bootstrap 差值（>=3 AUC 的分数代理 + macro AUC 点估计差）
    boots = {}
    def ge3map(rows_by_id):
        return {i: (r["p_birads_ge3"] if r.get("p_birads_ge3") is not None else None)
                for i, r in rows_by_id.items()}

    pairs = [("C1-B0", c1_soft, b0_by), ("C2-B0", c2_soft, b0_by), ("C1-C2", c1_soft, c2_soft)]
    for tag, a_rows, b_rows in pairs:
        if not a_rows or not b_rows:
            continue
        ma = {i: r["p_birads_ge3"] for i, r in a_rows.items() if r.get("p_birads_ge3") is not None}
        mb = {i: r["p_birads_ge3"] for i, r in b_rows.items() if r.get("p_birads_ge3") is not None}
        d, lo, hi = paired_bootstrap(ma, ma, mb, mb)
        boots[tag] = {"metric": "P(>=3) 逐样本均值差代理", "mean_diff": d, "ci_low": lo, "ci_high": hi}
    # C1 vs 对齐器（对齐器概率固定，重采样评估）
    al_map = {i: 1.0 - al["_prob"][j][0] - al["_prob"][j][1] for j, i in enumerate(al["_ids"])}
    c1_map = {i: r["p_birads_ge3"] for i, r in c1_soft.items() if r.get("p_birads_ge3") is not None}
    if al_map and c1_map:
        d, lo, hi = paired_bootstrap(c1_map, c1_map, al_map, al_map)
        boots["C1-Aligner"] = {"metric": "P(>=3) 逐样本均值差代理", "mean_diff": d,
                               "ci_low": lo, "ci_high": hi}
    blocks["bootstrap"] = boots

    (P / "pilot_stage2_metrics.json").write_text(
        json.dumps({k: v for k, v in blocks.items()}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    write_report(blocks)
    print("[ok] pilot_stage2_metrics.json + pilot_v1_report.md")


def draw_scan():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    ok = False
    for ax, ge3, t in ((axes[0], True, "GT BI-RADS>=3"), (axes[1], False, "GT BI-RADS<=2")):
        for tag, fn, st in (("C1", "c1_scan.jsonl", "o-"), ("C2", "c2_scan.jsonl", "^--")):
            rows = listload(fn)
            if not rows:
                continue
            by_p = defaultdict(list)
            for r in rows:
                if r["gt_ge3"] == ge3:
                    by_p[r["scan_p"]].append(r["p_birads_ge3"])
            if not by_p:
                continue
            ps = sorted(by_p)
            ax.plot(ps, [float(np.mean(by_p[p])) for p in ps], st, label=tag)
            ok = True
        ax.set_xlabel("forced P(mass=1)")
        ax.set_title(t)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    if ok:
        axes[0].set_ylabel("mean P(BI-RADS>=3)")
        fig.suptitle("阶段二概率扫描")
        fig.tight_layout()
        fig.savefig(P / "stage2_scan.png", dpi=140)
    plt.close(fig)


def draw_start_qc1_scan():
    """阶段一：起点(linear) vs QC1(logit_mlp) 扫描对比（数据在阶段二末补齐）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    q = listload("qc1_scan.jsonl")
    s = listload("start_scan.jsonl")
    if not q or not s:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, ge3, title in ((axes[0], True, "GT BI-RADS>=3"), (axes[1], False, "GT BI-RADS<=2")):
        for tag, rows, st in (("QC1 (logit_mlp)", q, "o-"), ("start (linear)", s, "s--")):
            by_p = defaultdict(list)
            for r in rows:
                if r.get("gt_ge3") == ge3:
                    by_p[r["scan_p"]].append(r["p_birads_ge3"])
            ps = sorted(by_p)
            if not ps:
                continue
            ax.plot(ps, [float(np.mean(by_p[p])) for p in ps], st, label=tag)
        ax.set_xlabel("forced P(mass=1)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("mean P(BI-RADS>=3)")
    fig.suptitle("起点 vs QC1 概率扫描")
    fig.tight_layout()
    fig.savefig(P / "stage1_scan_compare.png", dpi=140)
    plt.close(fig)
    return P / "stage1_scan_compare.png"


def f3(v):
    if v is None:
        return "—"
    if isinstance(v, float) and math.isnan(v):
        return "—"
    return f"{v:.3f}"


def write_report(B):
    L = ["# 链条 v1 全量试验报告（阶段二）", ""]
    L.append("## 0. 结论表（验证集 1600 乳房）")
    L.append("")
    L.append("| 组 | BI-RADS macroAUC | ≥3 AUC | macroF1 | acc | bACC | 密度 macroAUC | "
             "征象 meanAUC | 硬矛盾 A/B | 软一致性 P(≥3) 中位 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for g in ("B0", "C1", "C2", "Aligner"):
        b = B.get(g)
        if not b:
            continue
        bi = b["birads"] if "birads" in b else b
        mean_sign_auc = (float(np.mean([v for v in b["findings"]["auc"].values() if v]))
                         if "findings" in b else None)
        hi = b.get("hard_inconsistency", {})
        sc = b.get("soft_consistency", {})
        L.append(f"| {g} | {f3(bi.get('macro_auc'))} | {f3(bi.get('ge3_auc'))} | "
                 f"{f3(bi.get('macro_f1'))} | {f3(bi.get('acc'))} | {f3(bi.get('bacc'))} | "
                 f"{f3(b['density'].get('macro_auc')) if 'density' in b else '—'} | "
                 f"{f3(mean_sign_auc)} | {hi.get('pred_ge3_no_sign', '—')}/{hi.get('pred_le1_with_sign', '—')} | "
                 f"{f3(sc.get('median_p_ge3')) if sc else '—'} |")
    L.append("")
    pc = B.get("pipeline_check", {})
    pa, md = pc.get("pred_agreement"), pc.get("median_abs_diff_ge3")
    L.append(f"管线对照：B0 的 soft_decode(hard) vs b8 管线（对齐检查 {pc.get('n')} 条，容差 1e-3）："
             f"预测一致率 {'—' if pa is None else f'{pa:.4f}'}，"
             f"最大概率绝对差 {'—' if md is None else f'{md:.2e}'}。"
             f"B0 主要指标来源：{B.get('B0', {}).get('source', '—')}。")
    L.append("")
    for sec, path in (("遮挡机制分析", "pilot_occlusion.md"),):
        f = P / path
        if f.exists():
            L.append(f.read_text(encoding="utf-8"))
            L.append("")
    L.append("## 详细指标")
    L.append("")
    L.append("见 `pilot_stage2_metrics.json`（全部数值：各类召回、特异度 90/95% 灵敏度、"
             "征象 AUC/F1、密度、老师模式、软换硬、bootstrap CI）。")
    L.append("")
    if (P / "stage1_scan_compare.png").exists():
        L += ["## 起点 vs QC1 概率扫描（阶段一）", "",
              "![stage1_scan_compare](pilot/stage1_scan_compare.png)", ""]
    b8ref = B.get("B0_b8_ref")
    if b8ref:
        bi = b8ref["birads"]
        ms = (float(np.mean([v for v in b8ref["findings"]["auc"].values() if v]))
              if "findings" in b8ref else None)
        L += ["## 附表：B0 b8 管线（b8 管线，仅供参考，不进主比较口径）", "",
              "| BI-RADS macroAUC | ≥3 AUC | macroF1 | acc | bACC | 密度 macroAUC | 征象 meanAUC |",
              "|---|---|---|---|---|---|---|",
              f"| {f3(bi.get('macro_auc'))} | {f3(bi.get('ge3_auc'))} | {f3(bi.get('macro_f1'))} | "
              f"{f3(bi.get('acc'))} | {f3(bi.get('bacc'))} | "
              f"{f3(b8ref['density'].get('macro_auc')) if 'density' in b8ref else '—'} | {f3(ms)} |"]
        L.append("")
    (P / "pilot_v1_report_body.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    # 汇总成最终报告
    final = ["# 链条 v1 试验报告（阶段一 + 阶段二）", ""]
    qr = P / "quick_report.md"
    if qr.exists():
        final.append("## 阶段一（最快版）")
        final.append("")
        final.append(qr.read_text(encoding="utf-8"))
        final.append("")
    final.append("## 阶段二（全量版）")
    final.append("")
    final.append("\n".join(L))
    (PROJ / "chain_v1" / "outputs" / "pilot_v1_report.md").write_text(
        "\n".join(final) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
