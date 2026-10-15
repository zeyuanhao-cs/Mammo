#!/usr/bin/env python3
"""eval 公共工具: 读取概率 jsonl + 指标函数 (纯 Python, 无第三方依赖).

兼容两种 probs_norm 结构:
  - VLM 打分文件: 扁平键 "findings.mass": {"0": p0, "1": p1}
  - CNN 输出文件: 嵌套   "findings": {"mass": {"1": p1}}
"""
import json
import math
import random
from collections import defaultdict

FINDINGS = ["mass", "suspicious_calcification", "asymmetry", "architectural_distortion"]
DENSITY_CLASSES = ["A", "B", "C", "D"]
DENSITY_MERGED_CLASSES = ["AB", "C", "D"]
BIRADS_CLASSES = [1, 2, 3, 4, 5]


# ----------------------------- loading -----------------------------

def study_id_of(rec_id: str) -> str:
    return rec_id[:-2] if rec_id.endswith(("_L", "_R")) else rec_id


def _finding_prob(probs_norm: dict, name: str) -> float:
    """从 probs_norm 里取 P(finding=1), 兼容 VLM 扁平 / CNN 嵌套两种结构."""
    # CNN 嵌套: probs_norm["findings"]["mass"]["1"]
    if "findings" in probs_norm and isinstance(probs_norm["findings"], dict):
        v = probs_norm["findings"].get(name)
        if v is not None:
            if "1" in v:
                return float(v["1"])
            if 1 in v:
                return float(v[1])
    # VLM 扁平: probs_norm["findings.mass"]
    k = "findings.%s" % name
    if k in probs_norm:
        v = probs_norm[k]
        if "1" in v:
            return float(v["1"])
        if 1 in v:
            return float(v[1])
    raise KeyError("findings probability not found: %s" % name)


def load_probs_jsonl(path: str) -> list:
    """读取概率 jsonl, 按 id 去重检查, 返回统一结构的 record 列表."""
    recs, seen = [], set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rid = d["id"]
            if rid in seen:
                raise ValueError("duplicate id in %s: %s" % (path, rid))
            seen.add(rid)
            gt = d["ground_truth"]
            greedy = d.get("greedy_prediction") or {}
            pn = d["probs_norm"]
            recs.append({
                "id": rid,
                "study": study_id_of(rid),
                "gt_birads": int(gt["breast_birads"]),
                "gt_density": gt["breast_density"],
                "gt_findings": {k: int(gt["findings"][k]) for k in FINDINGS},
                "p_density": {c: float(pn["breast_density"][c]) for c in DENSITY_CLASSES},
                "p_birads": {str(i): float(pn["breast_birads"][str(i)]) for i in BIRADS_CLASSES},
                "p_findings": {k: _finding_prob(pn, k) for k in FINDINGS},
                "greedy_birads": int(greedy["breast_birads"]) if "breast_birads" in greedy else None,
                "greedy_density": greedy.get("breast_density"),
                "greedy_findings": {k: int(greedy["findings"][k]) for k in FINDINGS} if "findings" in greedy else {},
            })
    return recs


# ----------------------------- basic metrics -----------------------------

def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def binary_metrics(y, yhat):
    """y/yhat: 0/1 列表 -> dict(recall, specificity, precision, f1, acc, tp, fp, fn, tn)."""
    tp = sum(1 for a, b in zip(y, yhat) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(y, yhat) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y, yhat) if a == 1 and b == 0)
    tn = sum(1 for a, b in zip(y, yhat) if a == 0 and b == 0)
    p, r, f = prf(tp, fp, fn)
    n = len(y)
    return {
        "recall": r, "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "precision": p, "f1": f, "acc": (tp + tn) / n if n else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": n, "n_pos": tp + fn,
    }


def confusion(y, yhat, classes):
    """返回 {true_cls: {pred_cls: count}}."""
    cm = {t: {p: 0 for p in classes} for t in classes}
    for a, b in zip(y, yhat):
        cm[a][b] += 1
    return cm


def multiclass_metrics(y, yhat, classes):
    """acc / bACC / macro-F1 / 每类 recall."""
    n = len(y)
    per = {}
    for c in classes:
        tp = sum(1 for a, b in zip(y, yhat) if a == c and b == c)
        fp = sum(1 for a, b in zip(y, yhat) if a != c and b == c)
        fn = sum(1 for a, b in zip(y, yhat) if a == c and b != c)
        p, r, f = prf(tp, fp, fn)
        per[c] = {"precision": p, "recall": r, "f1": f, "support": tp + fn}
    acc = sum(1 for a, b in zip(y, yhat) if a == b) / n if n else 0.0
    recalls = [per[c]["recall"] for c in classes if per[c]["support"] > 0]
    bacc = sum(recalls) / len(recalls) if recalls else 0.0
    f1s = [per[c]["f1"] for c in classes if per[c]["support"] > 0]
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    return {"acc": acc, "bacc": bacc, "macro_f1": macro_f1, "per_class": per, "n": n}


def quadratic_weighted_kappa(y, yhat, classes):
    """二次加权 kappa, classes 须按序排列 (ordinal)."""
    idx = {c: i for i, c in enumerate(classes)}
    k = len(classes)
    cm = [[0] * k for _ in range(k)]
    for a, b in zip(y, yhat):
        cm[idx[a]][idx[b]] += 1
    n = sum(sum(r) for r in cm)
    if n == 0:
        return 0.0
    w = [[(i - j) ** 2 / (k - 1) ** 2 for j in range(k)] for i in range(k)]
    row = [sum(cm[i]) for i in range(k)]
    col = [sum(cm[i][j] for i in range(k)) for j in range(k)]
    e = [[row[i] * col[j] / n for j in range(k)] for i in range(k)]
    num = sum(w[i][j] * cm[i][j] for i in range(k) for j in range(k))
    den = sum(w[i][j] * e[i][j] for i in range(k) for j in range(k))
    return 1.0 - num / den if den else 1.0


# ----------------------------- ranking metrics -----------------------------

def auc(y, score):
    """Mann-Whitney AUC; 无正例/无负例返回 None."""
    pos = [s for a, s in zip(y, score) if a == 1]
    neg = [s for a, s in zip(y, score) if a == 0]
    if not pos or not neg:
        return None
    # rank 法 (含并列)
    vals = sorted(pos + neg)
    # 计算每个值的平均秩
    rank = {}
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[j + 1] == vals[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            rank[vals[t]] = avg
        i = j + 1
    rsum = sum(rank[s] for s in pos)
    n1, n0 = len(pos), len(neg)
    return (rsum - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def pr_auc(y, score):
    """Average Precision; 无正例返回 None."""
    pos_n = sum(y)
    if pos_n == 0 or pos_n == len(y):
        return None
    pairs = sorted(zip(score, y), key=lambda t: -t[0])
    tp = 0
    ap = 0.0
    for i, (s, lab) in enumerate(pairs):
        if lab == 1:
            tp += 1
            ap += tp / (i + 1)
    return ap / pos_n


def ece(y, p, n_bins=15):
    """Expected Calibration Error (等宽分箱)."""
    bins = [[] for _ in range(n_bins)]
    for lab, prob in zip(y, p):
        b = min(int(prob * n_bins), n_bins - 1)
        bins[b].append((lab, prob))
    total = len(y)
    if total == 0:
        return 0.0
    e = 0.0
    for b in bins:
        if not b:
            continue
        m = len(b)
        avg_p = sum(x[1] for x in b) / m
        avg_y = sum(x[0] for x in b) / m
        e += m / total * abs(avg_p - avg_y)
    return e


def ovr_macro_auc(y, probs_list, classes):
    """OvR macro AUC: 每类 one-vs-rest AUC, 只算 y 中实际出现的类 (无正/负例的类跳过).
    probs_list: 每样本 {class: prob}, key 可为 int 或 str (birads 兼容)."""
    aucs = []
    for c in classes:
        yc = [1 if a == c else 0 for a in y]
        if sum(yc) == 0 or sum(yc) == len(yc):
            continue
        s = [p[c if c in p else str(c)] for p in probs_list]
        a = auc(yc, s)
        if a is not None:
            aucs.append(a)
    return sum(aucs) / len(aucs) if aucs else None


def top_label_ece(y, probs_list, classes, n_bins=15):
    """top-label ECE: 预测类 (argmax) 的置信度 vs 预测是否正确, 等宽分箱."""
    idx = {c: i for i, c in enumerate(classes)}
    k = len(classes)
    correct, conf = [], []
    for a, p in zip(y, probs_list):
        vals = [p[c if c in p else str(c)] for c in classes]
        mi = max(range(k), key=lambda i: vals[i])
        conf.append(vals[mi])
        correct.append(1 if idx[a] == mi else 0)
    return ece(correct, conf, n_bins)


# ----------------------------- bootstrap -----------------------------

def bootstrap_ci(records, metric_fn, n_boot=1000, seed=42):
    """按 study 分组的 bootstrap 95% CI. records: load_probs_jsonl 输出; metric_fn(list)->float."""
    groups = defaultdict(list)
    for r in records:
        groups[r["study"]].append(r)
    studies = sorted(groups.keys())
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        sample = []
        for _ in range(len(studies)):
            sample.extend(groups[studies[rng.randrange(len(studies))]])
        v = metric_fn(sample)
        if v is not None:
            stats.append(v)
    if not stats:
        return None, None
    stats.sort()
    lo = stats[int(0.025 * len(stats))]
    hi = stats[min(int(0.975 * len(stats)), len(stats) - 1)]
    return lo, hi


def fmt_ci(v, ci):
    if v is None:
        return "  n/a"
    if ci is None or ci[0] is None:
        return "%.4f" % v
    return "%.4f [%.4f, %.4f]" % (v, ci[0], ci[1])
