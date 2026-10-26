#!/usr/bin/env python3
"""指标级 paired bootstrap（study 级重采样）。

指标: BI-RADS macroAUC / ≥3 AUC / macroF1 / spec90 灵敏度 / spec95 灵敏度 /
      密度 macroAUC / 征象 meanAUC。
配对: C1−B0, C2−B0, C1−C2, C1−Aligner。默认 1000 次，seed 42。
"""
import numpy as np
from scipy.stats import rankdata

FIND = ["mass", "suspicious_calcification", "asymmetry", "architectural_distortion"]
METRICS = ["birads_macro_auc", "ge3_auc", "macro_f1", "bacc", "acc", "recall_c3",
           "sens_spec90", "sens_spec95", "density_auc", "findings_mean_auc",
           "findings_macro_f1"]
N_BOOT, SEED = 1000, 42


def _auc(y, s):
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    pos = y == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    r = rankdata(s)                       # 平均秩处理并列
    return float((r[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _macro_auc(y, prob):
    aucs = [_auc((y == c).astype(int), prob[:, c]) for c in range(prob.shape[1])]
    aucs = [a for a in aucs if a is not None]
    return float(np.mean(aucs)) if aucs else None


def compute_metrics(pack):
    """pack: dict of arrays: y(0-4), bi(n,5), pf(n,4), dens(n,4)|None。"""
    y, bi = pack["y"], pack["bi"]
    out = {}
    out["birads_macro_auc"] = _macro_auc(y, bi)
    ge3 = (y >= 2).astype(int)
    score = 1.0 - bi[:, 0] - bi[:, 1]
    out["ge3_auc"] = _auc(ge3, score)

    pred = bi.argmax(axis=1)
    prec, rec = {}, {}
    for c in range(5):
        tp = int(((y == c) & (pred == c)).sum())
        prec[c] = tp / max(1, int((pred == c).sum()))
        rec[c] = tp / max(1, int((y == c).sum()))
    f1 = [2 * prec[c] * rec[c] / max(1e-9, prec[c] + rec[c]) for c in range(5)]
    out["macro_f1"] = float(np.mean(f1))
    out["bacc"] = float(np.mean([rec[c] for c in range(5)]))
    out["acc"] = float(np.mean(pred == y))
    out["recall_c3"] = float(rec[2])

    for spec, key in ((0.90, "sens_spec90"), (0.95, "sens_spec95")):
        neg = np.sort(score[ge3 == 0])
        if len(neg) and ge3.sum():
            k = max(0, min(len(neg) - 1, int(np.ceil(spec * len(neg))) - 1))
            thr = neg[k]
            out[key] = float(np.mean(score[ge3 == 1] > thr))
        else:
            out[key] = None

    out["density_auc"] = _macro_auc(pack["dens_y"], pack["dens"]) if pack.get("dens") is not None else None
    fa = [_auc(pack["pf_y"][:, j], pack["pf"][:, j]) for j in range(4)]
    fa = [a for a in fa if a is not None]
    out["findings_mean_auc"] = float(np.mean(fa)) if fa else None

    # 征象 macro F1（阈值 0.5）
    f1s = []
    for j in range(4):
        yj = pack["pf_y"][:, j]
        pj = (pack["pf"][:, j] >= 0.5).astype(float)
        tp = float(((yj == 1) & (pj == 1)).sum())
        n_pred, n_true = float(pj.sum()), float(yj.sum())
        prec = tp / n_pred if n_pred else 0.0
        rec = tp / n_true if n_true else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    out["findings_macro_f1"] = float(np.mean(f1s))
    return out


def _pack(rows_by_id, ids):
    """rows: eval_soft 同构 dict；返回按 ids 排列的 numpy 包。"""
    rs = [rows_by_id[i] for i in ids]
    y = np.array([r["gt_birads"] - 1 for r in rs])
    bi = np.array([[r["p_birads"][str(c + 1)] for c in range(5)] for r in rs], dtype=float)
    pf = np.array([[r["p_findings"][k] for k in FIND] for r in rs], dtype=float)
    pfy = np.array([[r["gt_findings"][k] for k in FIND] for r in rs], dtype=float)
    pack = {"y": y, "bi": bi, "pf": pf, "pf_y": pfy, "dens": None, "dens_y": None}
    if all(r.get("probs_norm", {}).get("breast_density") for r in rs):
        cls = ["A", "B", "C", "D"]
        keep = [j for j, r in enumerate(rs) if r.get("gt_density") in cls]
        if keep:
            pack["dens"] = np.array([[r["probs_norm"]["breast_density"].get(c, 0.0) for c in cls]
                                     for r in rs], dtype=float)
            pack["dens_y"] = np.array([cls.index(r["gt_density"]) for r in rs])
    return pack


def bootstrap_pair(rows_a, rows_b, n_boot=N_BOOT, seed=SEED):
    """rows_*: id -> eval_soft 同构 dict。返回 {metric: (diff, lo, hi)}，diff = B − A。"""
    ids = sorted(set(rows_a) & set(rows_b))
    studies = sorted({i.rsplit("_", 1)[0] for i in ids})
    sidx = {s: [j for j, i in enumerate(ids) if i.rsplit("_", 1)[0] == s] for s in studies}
    pa, pb = _pack(rows_a, ids), _pack(rows_b, ids)

    def sub(pack, idx):
        q = {k: (v[idx] if isinstance(v, np.ndarray) else v) for k, v in pack.items()}
        return q

    point_a, point_b = compute_metrics(pa), compute_metrics(pb)
    rng = np.random.default_rng(seed)
    reps = {m: [] for m in METRICS}
    for _ in range(n_boot):
        pick = rng.integers(0, len(studies), len(studies))
        idx = np.concatenate([sidx[studies[g]] for g in pick])
        ma, mb = compute_metrics(sub(pa, idx)), compute_metrics(sub(pb, idx))
        for m in METRICS:
            if ma.get(m) is None or mb.get(m) is None:
                continue
            reps[m].append(mb[m] - ma[m])

    out = {}
    for m in METRICS:
        d = point_b[m] - point_a[m] if (point_a.get(m) is not None and point_b.get(m) is not None) else None
        v = sorted(reps[m])
        if d is None or not v:
            out[m] = (None, None, None)
        else:
            lo = v[max(0, int(0.025 * len(v)))]
            hi = v[min(len(v) - 1, int(0.975 * len(v)))]
            out[m] = (float(d), float(lo), float(hi))
    out["_n_studies"] = len(studies)
    out["_n_ids"] = len(ids)
    out["_point_a"] = {m: point_a.get(m) for m in METRICS}
    out["_point_b"] = {m: point_b.get(m) for m in METRICS}
    return out
