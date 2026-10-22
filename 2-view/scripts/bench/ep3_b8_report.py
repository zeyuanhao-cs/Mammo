#!/usr/bin/env python3
"""汇总 batch=8 流水线：1-epoch / e1 / e2 / e3 的指标、配对 bootstrap 显著性、
计数表与训练 loss 曲线，输出 markdown 报告。"""
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

THIS = Path(__file__).resolve().parent
PROJ = THIS.parent.parent                     # 2-view
sys.path.insert(0, str(PROJ / "scripts" / "eval"))
import common as C  # noqa: E402

OUT_MD = PROJ / "outputs" / "eval" / "ep3_b8_report.md"
OUT_PNG = PROJ / "outputs" / "eval" / "ep3_b8_loss.png"

RUNS = [
    ("1-epoch (b8)", "metrics_probs_os_sub_vt_test_b8.json", "os_sub_vt_test_b8_with_probs.jsonl"),
    ("e1 (b8)", "metrics_probs_vt-ep3-e1_test_b8.json", "vt-ep3-e1_test_b8_with_probs.jsonl"),
    ("e2 (b8)", "metrics_probs_vt-ep3-e2_test_b8.json", "vt-ep3-e2_test_b8_with_probs.jsonl"),
    ("e3 (b8)", "metrics_probs_vt-ep3-e3_test_b8.json", "vt-ep3-e3_test_b8_with_probs.jsonl"),
]
BASE = "1-epoch (b8)"
N_BOOT, SEED = 1000, 42
EP3_STATE = Path("/home/z/zhao/Mammography/model/Qwen3.5-4B-lora-2view-os-sub-vt-ep3/trainer_state.json")
EPOCH_STEPS = [439, 878, 1317]


def g(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def findings_bacc(m):
    vals = []
    for f in C.FINDINGS:
        a = g(m, "findings", f, "at_0.5") or {}
        tp, fp, fn, tn = a.get("tp", 0), a.get("fp", 0), a.get("fn", 0), a.get("tn", 0)
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        vals.append((rec + spec) / 2)
    return sum(vals) / len(vals) if vals else None


def load_metrics(name):
    return json.load(open(PROJ / "outputs" / "eval" / name, encoding="utf-8"))


def load_recs(name):
    return C.load_probs_jsonl(str(PROJ / "outputs" / "scoring" / name))


# ---------------- AUC 定义（与 summarize_probs.py 一致） ----------------
def auc_density(rs):
    return C.ovr_macro_auc([r["gt_density"] for r in rs], [r["p_density"] for r in rs], C.DENSITY_CLASSES)


def auc_birads(rs):
    return C.ovr_macro_auc([r["gt_birads"] for r in rs],
                           [{i: r["p_birads"][str(i)] for i in C.BIRADS_CLASSES} for r in rs],
                           C.BIRADS_CLASSES)


def auc_findings(rs):
    vals = [C.auc([r["gt_findings"][k] for r in rs], [r["p_findings"][k] for r in rs]) for k in C.FINDINGS]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def paired_bootstrap(recs_a, recs_b, n_boot=N_BOOT, seed=SEED):
    """study 级配对 bootstrap：同一次重采样同时算两个模型的 AUC，取差值。"""
    A = {r["id"]: r for r in recs_a}
    B = {r["id"]: r for r in recs_b}
    ids = [i for i in A if i in B]
    groups = defaultdict(list)
    for i in ids:
        groups[A[i]["study"]].append(i)
    studies = sorted(groups)

    field_fns = [("density", auc_density), ("birads", auc_birads), ("findings", auc_findings)]
    stats = {k: [] for k, _ in field_fns}
    rng = random.Random(seed)
    for _ in range(n_boot):
        sample_ids = []
        for _ in range(len(studies)):
            sample_ids.extend(groups[studies[rng.randrange(len(studies))]])
        ra = [A[i] for i in sample_ids]
        rb = [B[i] for i in sample_ids]
        for k, fn in field_fns:
            va, vb = fn(ra), fn(rb)
            if va is not None and vb is not None:
                stats[k].append(vb - va)
    out = {}
    for k, s in stats.items():
        if not s:
            out[k] = (None, None)
            continue
        s.sort()
        lo = s[int(0.025 * len(s))]
        hi = s[min(int(0.975 * len(s)), len(s) - 1)]
        out[k] = (lo, hi)
    return out


def count_table(recs):
    d = {}
    d["dens_A_pred"] = sum(1 for r in recs if r["greedy_density"] == "A")
    d["dens_A_correct"] = sum(1 for r in recs if r["greedy_density"] == "A" and r["gt_density"] == "A")
    d["lesion_gt"] = sum(1 for r in recs if any(r["gt_findings"][k] for k in C.FINDINGS))
    d["lesion_detected"] = sum(1 for r in recs
                               if any(r["gt_findings"][k] for k in C.FINDINGS)
                               and any(r["greedy_findings"].get(k, 0) == 1 for k in C.FINDINGS))
    d["findings"] = {}
    for k in C.FINDINGS:
        pred = sum(1 for r in recs if r["greedy_findings"].get(k, 0) == 1)
        correct = sum(1 for r in recs if r["greedy_findings"].get(k, 0) == 1 and r["gt_findings"][k] == 1)
        d["findings"][k] = (pred, correct)
    d["birads_pred"] = Counter(r["greedy_birads"] for r in recs)
    return d


def loss_curve():
    st = json.load(open(EP3_STATE, encoding="utf-8"))
    hist = [h for h in st["log_history"] if "loss" in h]
    steps = [h["step"] for h in hist]
    losses = [h["loss"] for h in hist]
    marks = []
    for ep, sp in enumerate(EPOCH_STEPS, 1):
        cur = [(s, l) for s, l in zip(steps, losses) if s <= sp]
        marks.append((ep, sp, cur[-1][1] if cur else None))

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=140)
    ax.plot(steps, losses, lw=1.0, color="#2b6cb0", label="train loss")
    for ep, sp, lv in marks:
        ax.axvline(sp, ls="--", lw=0.9, color="#c53030", alpha=0.7)
        ax.annotate(f"ep{ep}  step {sp}\nloss {lv:.4f}", xy=(sp, lv), xytext=(6, 18),
                    textcoords="offset points", fontsize=7.5, color="#c53030")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("vt-ep3 training loss (3 epochs, 439 steps/epoch)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG)
    plt.close(fig)
    return marks


def fmt(v, nd=4):
    return "—" if v is None else f"{v:.{nd}f}"


def main():
    metrics, recs = {}, {}
    for name, mfile, pfile in RUNS:
        metrics[name] = load_metrics(mfile)
        recs[name] = load_recs(pfile)

    names = [n for n, _, _ in RUNS]

    # ---------- 1. 主表 ----------
    lines = ["# vt-ep3 batch=8 评测报告", ""]
    lines.append("流水线: `infer_2view_batched.py` (batch=8, 生成时直接取字段概率) + `summarize_probs.py`; "
                 "测试集 `data/splits/test_local.json` (2000 条)。")
    lines.append("")
    lines.append("## 1. 主表")
    lines.append("")
    head = ("| run | Density AUC | Density macro-F1 | Density bACC | BI-RADS AUC | BI-RADS macro-F1 | "
            "BI-RADS bACC | Findings AUROC | Findings macro-F1 | Findings bACC | any-finding AUC |")
    sep = "|---" * 11 + "|"
    lines += [head, sep]
    for n in names:
        m = metrics[n]
        lines.append(
            f"| {n} | {fmt(g(m, 'density_4class', 'ovr_macro_auc'))} | {fmt(g(m, 'density_4class', 'macro_f1'))} | "
            f"{fmt(g(m, 'density_4class', 'bacc'))} | {fmt(g(m, 'birads_ovr_macro_auc'))} | "
            f"{fmt(g(m, 'birads_5class_probs_argmax', 'macro_f1'))} | "
            f"{fmt(g(m, 'birads_5class_probs_argmax', 'bacc'))} | {fmt(g(m, 'findings_macro_auroc'))} | "
            f"{fmt(g(m, 'findings_macro_f1_at_0.5'))} | {fmt(findings_bacc(m))} | "
            f"{fmt(g(m, 'any_finding_at_0.5', 'auroc'))} |")
    lines.append("")
    lines.append("> AUC = OvR macro AUC（density 4 类 / BI-RADS 5 类）、findings 为 4 个二分类 AUROC 的平均；"
                 "macro-F1 / bACC 基于 probs argmax（findings 用 0.5 阈值，bACC 为 4 个 finding 的 "
                 "(recall+specificity)/2 平均）。")
    lines.append("")

    # ---------- 2. 显著性 ----------
    lines.append("## 2. 显著性：与 1-epoch (b8) 的配对 bootstrap 差值")
    lines.append("")
    lines.append(f"study 级配对 bootstrap，{N_BOOT} 次，seed={SEED}；差值为「该 run − 1-epoch」，单位 AUC；"
                 "CI 不跨 0 标 `*`。")
    lines.append("")
    lines.append("| run | Δ Density AUC [95% CI] | Δ BI-RADS AUC [95% CI] | Δ Findings AUROC [95% CI] |")
    lines.append("|---|---|---|---|")
    sig_count = {n: 0 for n in names}
    deltas = {}
    for n in names[1:]:
        ci = paired_bootstrap(recs[BASE], recs[n])
        cells = []
        d_run = {}
        for k, m in (("density", auc_density), ("birads", auc_birads), ("findings", auc_findings)):
            point = m(recs[n]) - m(recs[BASE])
            lo, hi = ci[k]
            d_run[k] = (point, lo, hi)
            star = " *" if (lo is not None and (lo > 0 or hi < 0)) else ""
            if star:
                sig_count[n] += 1
            cells.append(f"{point:+.4f} [{fmt(lo)}, {fmt(hi)}]{star}")
        deltas[n] = d_run
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    lines.append("")

    # ---------- 3. 计数表 ----------
    counts = {n: count_table(recs[n]) for n in names}
    lines.append("## 3. 计数表（贪心输出的条数）")
    lines.append("")
    lines.append("| 计数项 | " + " | ".join(names) + " |")
    lines.append("|---" * (len(names) + 1) + "|")
    rows = [
        ("密度 A 类：预测数", lambda c: c["dens_A_pred"]),
        ("密度 A 类：预测且真实为 A", lambda c: c["dens_A_correct"]),
        ("有病灶乳房（GT 阳性）总数", lambda c: c["lesion_gt"]),
        ("有病灶乳房：检出（预测任一 finding 阳性）", lambda c: c["lesion_detected"]),
    ]
    for k, lab in [("mass", "mass"), ("suspicious_calcification", "calcification"),
                   ("asymmetry", "asymmetry"), ("architectural_distortion", "arch.distortion")]:
        rows.append((f"finding {lab}：预测阳性数", lambda c, k=k: c["findings"][k][0]))
        rows.append((f"finding {lab}：预测阳性且真实阳性", lambda c, k=k: c["findings"][k][1]))
    for cls in C.BIRADS_CLASSES:
        rows.append((f"BI-RADS {cls}：预测数", lambda c, cls=cls: c["birads_pred"].get(cls, 0)))
    for lab, fn in rows:
        lines.append(f"| {lab} | " + " | ".join(str(fn(counts[n])) for n in names) + " |")
    lines.append("")

    # ---------- 4. loss 曲线 ----------
    lines.append("## 4. 训练 loss 曲线")
    lines.append("")
    try:
        marks = loss_curve()
        lines.append(f"![loss]({OUT_PNG.name})")
        lines.append("")
        lines.append("| epoch | step | loss |")
        lines.append("|---|---|---|")
        for ep, sp, lv in marks:
            lines.append(f"| {ep} | {sp} | {fmt(lv)} |")
    except Exception as e:  # noqa: BLE001
        lines.append(f"（loss 曲线生成失败: {e!r}）")
    lines.append("")

    # ---------- 5. 结论 ----------
    auc_keys = {"density": "Density AUC", "birads": "BI-RADS AUC", "findings": "Findings AUROC"}
    mean_auc = {}
    for n in names:
        vals = [auc_density(recs[n]), auc_birads(recs[n]), auc_findings(recs[n])]
        mean_auc[n] = sum(vals) / len(vals)
    best = max(names, key=lambda n: mean_auc[n])

    f1b = {}
    for n in names:
        m = metrics[n]
        f1b[n] = (g(m, "density_4class", "macro_f1"), g(m, "birads_5class_probs_argmax", "macro_f1"),
                  findings_bacc(m))

    concl = []
    tot_sig = sum(sig_count.values())
    tot_cmp = 3 * (len(names) - 1)
    concl.append(f"e1/e2/e3 相对 1-epoch 的 {tot_cmp} 个 AUC 差值中只有 {tot_sig} 个的 95% CI 不跨 0"
                 f"（{'、'.join(f'{n} {sig_count[n]} 个' for n in names[1:])}），其余差异不显著。")
    best_delta = deltas.get(best)
    if best_delta:
        concl.append(f"按三个字段 AUC 的平均值排序，最好的是 {best}（均值 {mean_auc[best]:.4f}），"
                     f"相对 1-epoch 的差值为 Density {best_delta['density'][0]:+.4f}、"
                     f"BI-RADS {best_delta['birads'][0]:+.4f}、Findings {best_delta['findings'][0]:+.4f}。")
    else:
        concl.append(f"按三个字段 AUC 的平均值排序，最好的是 {best}（均值 {mean_auc[best]:.4f}），即 1-epoch 本身。")
    # F1/bACC 差异 < 2pp 视为无差异
    no_diff = []
    for n in names[1:]:
        cond = all(
            b is not None and a is not None and abs(b - a) < 0.02
            for a, b in zip(f1b[BASE], f1b[n])
        )
        if cond:
            no_diff.append(n)
    concl.append(("macro-F1 / bACC 上 " + "、".join(no_diff) + " 与 1-epoch 的差值全部 < 2 个百分点，视为无差异。")
                 if no_diff else "macro-F1 / bACC 上 e1/e2/e3 至少一项相对 1-epoch 的差值 ≥ 2 个百分点。")
    mono = all(deltas[n]["findings"][0] < 0.005 for n in names[1:])
    concl.append(("继续训练（e1→e3）没有带来 AUC 的实质提升（各 epoch 的 findings AUC 差值均 < 0.005，"
                  "且显著项不足以支撑换模型），" if mono else "继续训练只在部分字段上带来 AUC 变化，")
                 + f"建议以 {best} 作为该 backbone 的最终选择。")

    lines.append("## 5. 结论")
    lines.append("")
    for c in concl[:4]:
        lines.append(f"- {c}")
    lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[out] {OUT_MD}")
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
