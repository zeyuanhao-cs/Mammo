#!/usr/bin/env python3
"""阶段一分析: 软一致性 / 概率扫描 / 软换硬 -> quick_report.md + quick_metrics.json + GATE 判定.

输入 (由 run_pilot.sh 按顺序产出的 jsonl):
  - outputs/pilot/start_hard200.jsonl  起点(1ep) hard 子集, eval_soft（b8 全量只用来挑子集）
  - outputs/pilot/qc1_soft.jsonl       QC1 soft, 软一致性批次 (start 挑的 ≤200)
  - outputs/pilot/qc1_swap.jsonl       QC1 swap_hard, 同一批次
  - outputs/pilot/qc1_scan.jsonl       QC1 scan, 100 (GT>=3 / GT<=2 各 50)
  - outputs/pilot/start_scan.jsonl     起点 scan 对照 (linear 注入)

判定口径 (写进报告):
  flat   = 曲线 max-min < 0.05
  阶跃   = 单点差分最大跳变 > 0.6 且发生在相邻两 p 之间
  STOP   = QC1 扫描曲线 flat 或阶跃, 或 QC1 曲线与起点曲线逐点平均差 < 0.03
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

CV1 = Path(__file__).resolve().parent.parent
P = CV1 / "outputs" / "pilot"


def load(fn):
    f = P / fn
    if not f.exists():
        return []
    return [_norm_row(json.loads(l)) for l in open(f, encoding="utf-8")]


def _norm_row(r):
    """b8 管线（infer_2view_batched）记录缺 p_birads_ge3/max_sign_p，按 eval_soft 同口径补算。"""
    if "p_birads_ge3" not in r or "max_sign_p" not in r:
        pn = r.get("probs_norm") or {}
        bi = pn.get("breast_birads") or {}
        if bi:
            r["p_birads_ge3"] = 1.0 - float(bi.get("1", 0.0)) - float(bi.get("2", 0.0))
        cand = [float((pn.get(f"findings.{k}") or {}).get("1", 0.0))
                for k in ("mass", "suspicious_calcification", "asymmetry", "architectural_distortion")]
        r["max_sign_p"] = max(cand) if cand else None
    return r


def med_ge3(rows):
    v = [r["p_birads_ge3"] for r in rows if r.get("p_birads_ge3") is not None]
    return float(np.median(v)) if v else float("nan")


def spearman(rows, x_key="max_sign_p", y_key="p_birads_ge3"):
    xs = [r[x_key] for r in rows if r.get(x_key) is not None and r.get(y_key) is not None]
    ys = [r[y_key] for r in rows if r.get(x_key) is not None and r.get(y_key) is not None]
    if len(xs) < 3:
        return float("nan")
    return float(spearmanr(xs, ys).statistic)


def curve(scan_rows, ge3):
    by_p = {}
    for r in scan_rows:
        if r["gt_ge3"] != ge3:
            continue
        by_p.setdefault(r["scan_p"], []).append(r["p_birads_ge3"])
    ps = sorted(by_p)
    return ps, [float(np.mean(by_p[p])) for p in ps]


def shape_of(ps, ys):
    if not ys:
        return "none", 0.0, 0.0
    rng = max(ys) - min(ys)
    jumps = [abs(b - a) for a, b in zip(ys, ys[1:])]
    max_jump = max(jumps) if jumps else 0.0
    if rng < 0.05:
        return "flat", rng, max_jump
    if max_jump > 0.6 and max_jump > 0.7 * rng:
        return "step", rng, max_jump
    return "smooth", rng, max_jump


def main():
    # 起点 hard 用 eval_soft 子集输出（与 QC1 同管线同口径）；b8 全量只用来挑子集
    start = load("start_hard200.jsonl")
    qc1_soft = load("qc1_soft.jsonl")
    qc1_swap = load("qc1_swap.jsonl")
    qc1_scan = load("qc1_scan.jsonl")
    start_scan = load("start_scan.jsonl")
    assert start, "缺 start_hard200.jsonl"
    assert qc1_soft and qc1_swap and qc1_scan, "缺 QC1 评测输出"

    soft_ids = [r["id"] for r in qc1_soft]
    smap = {r["id"]: r for r in start}
    start_on_soft = [smap[i] for i in soft_ids if i in smap]

    # ---- 软一致性 ----
    med_start = med_ge3(start_on_soft)
    med_qc1 = med_ge3(qc1_soft)
    sp_start = spearman(start_on_soft)
    sp_qc1 = spearman(qc1_soft)

    # ---- 扫描曲线（起点 5/5 已延后到阶段二末；缺失/不完整时只判 QC1）----
    ps_q, yq_pos = curve(qc1_scan, True)
    _, yq_neg = curve(qc1_scan, False)
    shape_q, rng_q, jump_q = shape_of(ps_q, yq_pos)
    have_start_scan = len(start_scan) >= 1100
    if have_start_scan:
        ps_s, ys_pos = curve(start_scan, True)
        _, ys_neg = curve(start_scan, False)
        shape_s, rng_s, jump_s = shape_of(ps_s, ys_pos)
        mean_diff = float(np.mean(np.abs(np.array(yq_pos) - np.array(ys_pos)))) \
            if len(yq_pos) == len(ys_pos) else float("nan")
    else:
        ps_s, ys_pos, ys_neg = ps_q, [float("nan")] * len(ps_q), [float("nan")] * len(ps_q)
        shape_s, rng_s, jump_s = "deferred", float("nan"), float("nan")
        mean_diff = float("nan")

    # ---- 软换硬 ----
    med_swap = med_ge3(qc1_swap)
    swap_delta = med_qc1 - med_swap

    m = {
        "soft_consistency": {
            "n": len(soft_ids),
            "median_p_ge3_start_hard": med_start, "median_p_ge3_qc1_soft": med_qc1,
            "lift": med_qc1 - med_start,
            "spearman_start": sp_start, "spearman_qc1": sp_qc1,
        },
        "scan": {
            "qc1_shape": shape_q, "qc1_range": rng_q, "qc1_max_jump": jump_q,
            "start_shape": shape_s, "start_range": rng_s, "start_max_jump": jump_s,
            "mean_abs_diff_vs_start": mean_diff,
            "qc1_curve_gt3": dict(zip(map(str, ps_q), yq_pos)),
            "qc1_curve_le2": dict(zip(map(str, ps_q), yq_neg)),
            "start_curve_gt3": dict(zip(map(str, ps_s), ys_pos)) if have_start_scan else {},
            "start_curve_le2": dict(zip(map(str, ps_s), ys_neg)) if have_start_scan else {},
        },
        "swap_hard": {"median_p_ge3_soft": med_qc1, "median_p_ge3_swap": med_swap,
                      "delta": swap_delta},
    }
    (P / "quick_metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=1),
                                          encoding="utf-8")

    # ---- png ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
        for ax, ys_q, ys_s, t in ((axes[0], yq_pos, ys_pos, "GT BI-RADS>=3"),
                                  (axes[1], yq_neg, ys_neg, "GT BI-RADS<=2")):
            ax.plot(ps_q, ys_q, "o-", label=f"QC1 (logit_mlp) [{shape_q}]")
            ax.plot(ps_s, ys_s, "s--", label=f"start (linear) [{shape_s}]")
            ax.set_xlabel("forced P(mass finding = 1)")
            ax.set_title(t)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        axes[0].set_ylabel("mean P(BI-RADS>=3)")
        fig.suptitle("阶段一概率扫描: mass 证据 p -> P(>=3)")
        fig.tight_layout()
        fig.savefig(P / "quick_scan.png", dpi=140)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 画图失败: {e!r}")

    # ---- GATE（新判据，三条同时满足才 PASS；NaN/异常一律 STOP）----
    stop_reasons = []
    scan_diff = (yq_pos[-1] - yq_pos[0]) if len(yq_pos) >= 2 else float("nan")
    checks = [scan_diff, sp_qc1, sp_start, med_qc1, med_start]
    if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in checks):
        stop_reasons.append("指标出现 NaN（扫描/软一致性数据不全）")
    else:
        if not scan_diff > 0.1:
            stop_reasons.append(f"QC1 扫描 p=0→1 的 P(≥3) 均值差 = {scan_diff:+.4f}（需 >0.1）")
        if not (sp_qc1 - sp_start >= 0.1):
            stop_reasons.append(f"Spearman 提升 = {sp_qc1 - sp_start:+.3f}（需 ≥0.1；"
                                f"起点 {sp_start:.3f} → QC1 {sp_qc1:.3f}）")
        if not (med_qc1 > med_start):
            stop_reasons.append(f"P(≥3) 中位未高于起点（{med_start:.4f} → {med_qc1:.4f}）")
    gate = "STOP" if stop_reasons else "PASS"

    lift = med_qc1 - med_start
    L = ["# 阶段一快报（机制验证）", ""]
    L.append(f"**GATE: {gate}**" + (f"（{'；'.join(stop_reasons)}）" if stop_reasons else "——进入阶段二"))
    L.append("")
    L.append("## 开头三问")
    L.append("")
    L.append(f"1. **概率扫描曲线**：起点模型 = "
             + ("deferred（起点扫描已挪到阶段二末，届时补图）" if not have_start_scan
                else f"{shape_s}（max−min={rng_s:.3f}，最大单跳={jump_s:.3f}）")
             + f"；QC1 = **{shape_q}**（max−min={rng_q:.3f}，最大单跳={jump_q:.3f}）；"
             f"QC1 曲线 p=0→1 的 P(≥3) 均值差 = {scan_diff:+.4f}。曲线图见 `quick_scan.png`。")
    L.append(f"2. **软一致性**（n={len(soft_ids)}，起点挑的 pred≤2 且 max_sign>0.1）："
             f"P(≥3) 中位数 起点 hard = {med_start:.4f} → QC1 soft = **{med_qc1:.4f}**"
             f"（{'+' if lift >= 0 else ''}{lift:.4f}）；"
             f"Spearman(P(≥3), max_sign) 起点 = {sp_start:.3f}，QC1 = {sp_qc1:.3f}。")
    L.append(f"3. **软换硬**：QC1 同批乳房 soft P(≥3) 中位 = {med_qc1:.4f} → 换成 argmax token 嵌入 = "
             f"{med_swap:.4f}（Δ = {swap_delta:+.4f}，"
             f"{'变化大' if abs(swap_delta) > 0.1 else '变化不大'}）。")
    L.append("")
    L.append("## 数值明细")
    L.append("")
    L.append("| 项 | 起点(1ep, hard) | QC1(50步, soft) |")
    L.append("|---|---|---|")
    L.append(f"| P(≥3) 中位数 | {med_start:.4f} | {med_qc1:.4f} |")
    L.append(f"| P(≥3) 与 max_sign 的 Spearman | {sp_start:.3f} | {sp_qc1:.3f} |")
    if have_start_scan:
        L.append(f"| 扫描曲线形状 (GT≥3) | {shape_s} (range {rng_s:.3f}) | {shape_q} (range {rng_q:.3f}) |")
        L.append(f"| 扫描 P(=0)→P(=1) 的 P(≥3) 均值 (GT≥3) | {ys_pos[0]:.4f}→{ys_pos[-1]:.4f} | "
                 f"{yq_pos[0]:.4f}→{yq_pos[-1]:.4f} |")
        L.append(f"| 扫描 P(=0)→P(=1) 的 P(≥3) 均值 (GT≤2) | {ys_neg[0]:.4f}→{ys_neg[-1]:.4f} | "
                 f"{yq_neg[0]:.4f}→{yq_neg[-1]:.4f} |")
    else:
        L.append(f"| 扫描曲线形状 (GT≥3) | deferred | {shape_q} (range {rng_q:.3f}) |")
        L.append(f"| 扫描 P(=0)→P(=1) 的 P(≥3) 均值 (GT≥3) | deferred | "
                 f"{yq_pos[0]:.4f}→{yq_pos[-1]:.4f}（Δ={scan_diff:+.4f}） |")
        L.append(f"| 扫描 P(=0)→P(=1) 的 P(≥3) 均值 (GT≤2) | deferred | "
                 f"{yq_neg[0]:.4f}→{yq_neg[-1]:.4f} |")
    L.append(f"| 软换硬 P(≥3) 中位数 | — | {med_swap:.4f} |")
    L.append("")
    L.append("## 口径")
    L.append("")
    L.append("- 起点模型 = `Qwen3.5-4B-lora-2view-os-sub-vt` 1 epoch（checkpoint-439）；"
             "QC1 = 该 adapter + 50 步软证据训练（logit_mlp, small, alpha=1, lam=1, rho_max=0.5, 无 warmup）。")
    L.append("- 软一致性批次：起点 hard 预测 ≤2 且 max 征象概率 >0.1 的前 200 个验证集乳房。")
    L.append("- 扫描：GT≥3 / GT≤2 各 50 个乳房，强制 mass 证据 p=0,0.1,…,1（其余征象按模型自身 p 注入）；"
             "起点模型无 mixer，用 linear 注入 (1−p)E0+pE1 做对照。")
    L.append("- GATE 判据（三条同时满足才 PASS，NaN/报错一律 STOP）："
             "① QC1 扫描 p=0→1 的 P(≥3) 均值差 >0.1；② QC1 的 Spearman 比起点高 ≥0.1；"
             "③ QC1 的 P(≥3) 中位数高于起点。")
    L.append("- 起点扫描（5/5）已挪到阶段二末，用冻结代码从断点续跑；最终报告补起点 vs QC1 对比图。")
    (P / "quick_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\nGATE={gate}", flush=True)
    sys.exit(0 if gate == "PASS" else 3)


if __name__ == "__main__":
    main()
