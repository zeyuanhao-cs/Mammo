#!/usr/bin/env python3
"""第 6 步 (CPU): 统计 + report.md.

要点
- Δ 基准 = **本次重跑**的 occlusion_probs/orig.jsonl (Δ = orig − 遮挡后, 正 = 下降).
- 两套分类并排:
    old = 既有 2000 条测试概率给出的 detected / missed_sign / missed_nosign (第 4 步)
    new = 本次 orig 结果按**完全相同的规则**重新分类
- 所有表 (Δ / Wilcoxon / BI-RADS 迁移 / 病灶 token 数) 都按两套分类各算一遍.
- 附加: missed_sign 内部按预测 BI-RADS = 2 / = 1 拆分.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

BASE = Path("/nesi/project/massey04900/Mammography")
OCC = BASE / "occlusion"
PROBS_DIR = OCC / "occlusion_probs"
EXISTING = Path("/home/zhao/Mammography/Mammo/2-view/outputs/scoring/os_sub_test_with_probs.jsonl")
GROUPS = ("detected", "missed_sign", "missed_nosign")
CONDS = ("orig", "lesion", "ctrl1", "ctrl2")
FINDINGS = ["mass", "suspicious_calcification", "asymmetry", "architectural_distortion"]
TOKENS_PER_IMAGE = 756      # 672x1152 / (16*16) / 4
SIGN_THRESH = 0.1
METRIC_TITLES = {"max_sign_p": "max 征象 P(=1)", "gt_sign_p": "GT 类别征象 P(=1)",
                 "p_birads_ge3": "P(BI-RADS≥3)", "p_birads_ge2": "P(BI-RADS≥2)"}


# ---------------------------------------------------------------- 小工具
def max_view_tokens(entry):
    """单视角口径: 该视角最大病灶框(不外扩)占 PNG 面积比例 x 756, 取 CC/MLO 较大值."""
    best = 0.0
    for v in entry["views"].values():
        for b in v["lesion_boxes"]:
            best = max(best, (b[2] - b[0]) * (b[3] - b[1]) / float(912 * 1520))
    return best * TOKENS_PER_IMAGE


def lesion_type(entry):
    cls = set(entry.get("gt_classes") or [])
    if "mass" in cls:
        return "mass"
    if "suspicious_calcification" in cls:
        return "calcification"
    return "other"


def quartiles(values):
    if not values:
        return {"n": 0, "median": None, "p25": None, "p75": None, "min": None, "max": None}
    a = np.asarray(values, dtype=float)
    return {"n": int(a.size),
            "median": round(float(np.median(a)), 2),
            "p25": round(float(np.percentile(a, 25)), 2),
            "p75": round(float(np.percentile(a, 75)), 2),
            "min": round(float(a.min()), 2),
            "max": round(float(a.max()), 2)}


def load_cond(name):
    p = PROBS_DIR / f"{name}.jsonl"
    out = {}
    if not p.exists():
        return out
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        out[r["id"]] = r
    return out


def med(a):
    a = [v for v in a if v is not None]
    return float(np.median(a)) if a else float("nan")


def fmt(v, nd=4):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{nd}f}"


def fp(v):
    """p 值格式: 过小用 <1e-4."""
    if v is None:
        return "—"
    if v < 1e-4:
        return "<1e-4"
    return f"{v:.4f}"


def wtest(x, y):
    d = [a - b for a, b in zip(x, y) if a is not None and b is not None]
    if len(d) < 8 or all(v == 0 for v in d):
        return None
    try:
        return float(wilcoxon(d, alternative="two-sided").pvalue)
    except Exception:
        return None


def deltas(recs, ids, key):
    o = recs["orig"]
    return {(c, i): (None if o[i][key] is None or recs[c][i][key] is None
                     else o[i][key] - recs[c][i][key])
            for c in ("lesion", "ctrl1", "ctrl2") for i in ids}


# ---------------------------------------------------------------- 分类
def classify_from(probs):
    """probs: id -> {"pred": int, "max_sign": float}; 与第 4 步完全相同的规则."""
    groups = {g: [] for g in GROUPS}
    for bid, r in probs.items():
        if r["pred"] >= 3:
            g = "detected"
        elif r["pred"] == 2 or r["max_sign"] > SIGN_THRESH:
            g = "missed_sign"
        else:
            g = "missed_nosign"
        groups[g].append(bid)
    for g in GROUPS:
        groups[g].sort()
    return groups


def probs_from_orig(recs):
    return {bid: {"pred": r["pred_birads"], "max_sign": r["max_sign_p"]}
            for bid, r in recs["orig"].items()}


def probs_from_existing(ids):
    out = {}
    for line in open(EXISTING, encoding="utf-8"):
        r = json.loads(line)
        if r["id"] not in ids:
            continue
        pn = r["probs_norm"]
        bi = {k: float(pn["breast_birads"][k]) for k in ("1", "2", "3", "4", "5")}
        out[r["id"]] = {"pred": int(max(bi.items(), key=lambda kv: kv[1])[0]),
                        "max_sign": max(float(pn[f"findings.{k}"]["1"]) for k in FINDINGS)}
    return out


# ---------------------------------------------------------------- 单套分类的所有统计
def group_stats(recs, groups, man, single_view, pred_of):
    per_group = {}
    for g in GROUPS:
        ids = [i for i in groups[g] if i in recs["lesion"]]
        if not ids:
            per_group[g] = {"n": 0, "metrics": {}, "transitions": {}, "single_view_subset": []}
            continue
        rows = {}
        for key in ("max_sign_p", "gt_sign_p", "p_birads_ge3", "p_birads_ge2"):
            d = deltas(recs, ids, key)
            les = [d[("lesion", i)] for i in ids]
            c1 = [d[("ctrl1", i)] for i in ids]
            c2 = [d[("ctrl2", i)] for i in ids]
            cm = [np.nanmean([a, b]) if (a is not None and b is not None) else None
                  for a, b in zip(c1, c2)]
            ok = [(a, b) for a, b in zip(les, cm) if a is not None and b is not None]
            rows[key] = {
                "n": len(ids),
                "median_delta_lesion": med(les),
                "median_delta_ctrl1": med(c1),
                "median_delta_ctrl2": med(c2),
                "median_delta_ctrl_mean": med(cm),
                "p_wilcoxon_lesion_vs_ctrl": wtest([a for a, _ in ok], [b for _, b in ok]),
                "frac_lesion_gt_ctrl": (float(np.mean([a > b for a, b in ok])) if ok else None),
                "median_orig": med([recs["orig"][i][key] for i in ids]),
                "median_lesion": med([recs["lesion"][i][key] for i in ids]),
            }
        p_o = {i: recs["orig"][i]["pred_birads"] for i in ids}
        p_l = {i: recs["lesion"][i]["pred_birads"] for i in ids}
        p_c1 = {i: recs["ctrl1"][i]["pred_birads"] for i in ids}
        p_c2 = {i: recs["ctrl2"][i]["pred_birads"] for i in ids}
        trans = {
            "lesion_2to1": sum(1 for i in ids if p_o[i] == 2 and p_l[i] == 1),
            "lesion_ge3_to_le2": sum(1 for i in ids if p_o[i] >= 3 and p_l[i] <= 2),
            "ctrl1_2to1": sum(1 for i in ids if p_o[i] == 2 and p_c1[i] == 1),
            "ctrl1_ge3_to_le2": sum(1 for i in ids if p_o[i] >= 3 and p_c1[i] <= 2),
            "ctrl2_2to1": sum(1 for i in ids if p_o[i] == 2 and p_c2[i] == 1),
            "ctrl2_ge3_to_le2": sum(1 for i in ids if p_o[i] >= 3 and p_c2[i] <= 2),
        }
        sv_rows = []
        for i in [x for x in ids if x in single_view]:
            sv_rows.append({
                "id": i,
                "views_with_box": ("CC" if len(man["breasts"][i]["views"].get("CC", {}).get("lesion_boxes", []))
                                   else "MLO"),
                "delta_lesion": recs["orig"][i]["max_sign_p"] - recs["lesion"][i]["max_sign_p"],
                "delta_ctrl_mean": recs["orig"][i]["max_sign_p"]
                                   - float(np.mean([recs["ctrl1"][i]["max_sign_p"],
                                                    recs["ctrl2"][i]["max_sign_p"]])),
                "pred_orig": p_o[i], "pred_lesion": p_l[i],
            })
        per_group[g] = {"n": len(ids), "metrics": rows, "transitions": trans,
                        "single_view_subset": sv_rows, "ids": ids}
    return per_group


def missed_split(recs, ids, pred_of):
    out = {}
    for tag, pv in (("pred=2", 2), ("pred=1", 1)):
        sub = [i for i in ids if pred_of.get(i) == pv]
        if not sub:
            out[tag] = {"n": 0}
            continue
        d = deltas(recs, sub, "max_sign_p")
        les = [d[("lesion", i)] for i in sub]
        cm = [np.nanmean([d[("ctrl1", i)], d[("ctrl2", i)]]) for i in sub]
        ok = [(a, b) for a, b in zip(les, cm) if a is not None and b is not None]
        d3 = deltas(recs, sub, "p_birads_ge3")
        les3 = [d3[("lesion", i)] for i in sub]
        cm3 = [np.nanmean([d3[("ctrl1", i)], d3[("ctrl2", i)]]) for i in sub]
        out[tag] = {
            "n": len(sub),
            "median_delta_lesion_max_sign": med(les),
            "median_delta_ctrl_max_sign": med(cm),
            "p_wilcoxon_max_sign": wtest([a for a, _ in ok], [b for _, b in ok]),
            "frac_lesion_gt_ctrl": (float(np.mean([a > b for a, b in ok])) if ok else None),
            "median_delta_lesion_p_ge3": med(les3),
            "median_delta_ctrl_p_ge3": med(cm3),
            "p_wilcoxon_p_ge3": wtest(les3, cm3),
            "median_orig_max_sign": med([recs["orig"][i]["max_sign_p"] for i in sub]),
            "ids": sub,
        }
    return out


# ---------------------------------------------------------------- 主流程
def main():
    man = json.load(open(OCC / "occlusion_manifest.json", encoding="utf-8"))
    s4 = json.load(open(OCC / "reports" / "step4.json", encoding="utf-8"))
    recs = {c: load_cond(c) for c in CONDS}
    if not recs["lesion"]:
        raise SystemExit("occlusion_probs/lesion.jsonl 不存在, 先跑第 5 步")
    ids_all = sorted(recs["orig"])
    single_view = [bid for bid in ids_all
                   if (len(man["breasts"][bid]["views"].get("CC", {}).get("lesion_boxes", [])) == 0)
                   != (len(man["breasts"][bid]["views"].get("MLO", {}).get("lesion_boxes", [])) == 0)]

    # ---------- 一致性
    cons_path = OCC / "reports" / "step5_consistency.json"
    cs = json.load(open(cons_path, encoding="utf-8")) if cons_path.exists() else {
        "n_compared": 0, "pred_birads_agreement": None, "pred_birads_disagreement": None,
        "max_sign_p_max_abs_diff": None, "max_sign_p_median_abs_diff": None,
        "p_birads_ge3_max_abs_diff": None, "p_birads_ge3_median_abs_diff": None,
        "consistent": None}

    # ---------- 两套分类
    prob_old = probs_from_existing(set(ids_all))
    prob_new = probs_from_orig(recs)
    groups_old = {g: [i for i in s4["groups"][g] if i in recs["lesion"]] for g in GROUPS}
    groups_new = classify_from(prob_new)

    cross = {g: {h: 0 for h in GROUPS} for g in GROUPS}
    switched = []
    gof = {i: g for g in GROUPS for i in groups_old[g]}
    gnf = {i: g for g in GROUPS for i in groups_new[g]}
    for i in ids_all:
        cross[gof[i]][gnf[i]] += 1
        if gof[i] != gnf[i]:
            switched.append({"id": i, "old": gof[i], "new": gnf[i],
                             "pred_old": prob_old[i]["pred"], "pred_new": prob_new[i]["pred"],
                             "max_sign_old": round(prob_old[i]["max_sign"], 4),
                             "max_sign_new": round(prob_new[i]["max_sign"], 4)})

    stats_old = group_stats(recs, groups_old, man, single_view, {i: v["pred"] for i, v in prob_old.items()})
    stats_new = group_stats(recs, groups_new, man, single_view, {i: v["pred"] for i, v in prob_new.items()})
    pred_new = {i: v["pred"] for i, v in prob_new.items()}
    pred_old = {i: v["pred"] for i, v in prob_old.items()}
    ms_old = missed_split(recs, groups_old["missed_sign"], pred_old)
    ms_new = missed_split(recs, groups_new["missed_sign"], pred_new)

    # ---------- 几何 / token
    geom = {}
    for bid, e in man["breasts"].items():
        mx, tot = 0.0, 0.0
        for v in e["views"].values():
            for b in v["lesion_boxes"]:
                a = (b[2] - b[0]) * (b[3] - b[1]) / float(912 * 1520)
                mx = max(mx, a)
                tot += a
        geom[bid] = {"max_box_area_ratio": mx, "sum_box_area_ratio": tot, "gt_birads": e["gt_birads"]}
    mx_ratio = np.array([v["max_box_area_ratio"] for v in geom.values()])
    geom_stats = {
        "n_breasts": len(geom),
        "max_box_area_ratio": {
            "median": float(np.median(mx_ratio)),
            "p90": float(np.percentile(mx_ratio, 90)),
            "max": float(mx_ratio.max()),
            "n_gt_5pct": int((mx_ratio > 0.05).sum()),
            "n_gt_10pct": int((mx_ratio > 0.10).sum()),
            "n_gt_20pct": int((mx_ratio > 0.20).sum()),
            "n_gt_40pct": int((mx_ratio > 0.40).sum()),
        },
        "top5": sorted(((round(v["max_box_area_ratio"], 4), b) for b, v in geom.items()), reverse=True)[:5],
        "by_gt_birads": {str(k): {
            "n": int(sum(1 for v in geom.values() if v["gt_birads"] == k)),
            "median_max_box_area_ratio": float(np.median(
                [v["max_box_area_ratio"] for v in geom.values() if v["gt_birads"] == k]))
        } for k in (3, 4, 5)},
    }
    tokens = {bid: max_view_tokens(e) for bid, e in man["breasts"].items()}
    ltype = {bid: lesion_type(e) for bid, e in man["breasts"].items()}
    tok = {}
    for tag, gr in (("old", groups_old), ("new", groups_new)):
        tok[tag] = {
            "by_group": {g: quartiles([tokens[b] for b in gr[g]]) for g in GROUPS},
            "by_type": {t: quartiles([tokens[b] for b in tokens if ltype[b] == t])
                        for t in ("mass", "calcification", "other")},
        }

    # ---------- 方法学闸门 (两套分类各看 detected)
    gate = {}
    for tag, st in (("old", stats_old), ("new", stats_new)):
        m = st["detected"]["metrics"]["max_sign_p"]
        m3 = st["detected"]["metrics"]["p_birads_ge3"]
        gate[tag] = {"n": m["n"],
                     "median_delta_lesion_max_sign_p": m["median_delta_lesion"],
                     "median_delta_ctrl_mean_max_sign_p": m["median_delta_ctrl_mean"],
                     "median_delta_lesion_p_birads_ge3": m3["median_delta_lesion"],
                     "pass": bool(m["median_delta_lesion"] > m["median_delta_ctrl_mean"]
                                  and m["median_delta_lesion"] >= 0.05)}

    out = {
        "n_cases": {c: len(v) for c, v in recs.items()},
        "n_orig": len(ids_all),
        "delta_baseline": "本次重跑的 occlusion_probs/orig.jsonl",
        "consistency_vs_existing_2000": cs,
        "classification": {
            "rule": "pred>=3 -> detected; pred<=2 且 (pred==2 或 max_sign_p>0.1) -> missed_sign; 否则 missed_nosign",
            "counts_old": {g: len(groups_old[g]) for g in GROUPS},
            "counts_new": {g: len(groups_new[g]) for g in GROUPS},
            "cross_tab_old_rows": cross,
            "n_switched": len(switched),
            "switched": switched,
        },
        "geometry_png_coords": geom_stats,
        "geometry_per_breast": geom,
        "lesion_tokens_png_view": {"tokens_per_image": TOKENS_PER_IMAGE, **tok},
        "method_gate": gate,
        "groups_old": stats_old,
        "groups_new": stats_new,
        "missed_sign_split": {"old": ms_old, "new": ms_new},
        "n_single_view_breasts": len(single_view),
        "single_view_breasts": single_view,
        "group_ids": {"old": groups_old, "new": groups_new},
    }
    (OCC / "reports" / "step6.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                                encoding="utf-8")
    write_report(out, man, s4)

    cl = out["classification"]
    print("[ok] report.md + reports/step6.json")
    print(f"     分类 旧 {cl['counts_old']}  新 {cl['counts_new']}  换组 {cl['n_switched']}")
    for tag in ("old", "new"):
        g = gate[tag]
        print(f"     闸门[{tag}] detected n={g['n']} Δles={g['median_delta_lesion_max_sign_p']:.3f} "
              f"Δctrl={g['median_delta_ctrl_mean_max_sign_p']:.3f} -> {'PASS' if g['pass'] else 'FAIL'}")
    for tag, st in (("old", stats_old), ("new", stats_new)):
        for g in GROUPS:
            m = st[g]["metrics"].get("max_sign_p")
            if not m:
                continue
            print(f"     [{tag}] {g:14s} n={m['n']:3d} Δles={m['median_delta_lesion']:.3f} "
                  f"Δctrl={m['median_delta_ctrl_mean']:.3f} p={fp(m['p_wilcoxon_lesion_vs_ctrl'])} "
                  f"ratio={fmt(m['frac_lesion_gt_ctrl'], 2)}")
    for tag in ("old", "new"):
        for k, v in ms_new.items() if tag == "new" else ms_old.items():
            if v.get("n"):
                print(f"     missed_sign[{tag}] {k} n={v['n']} Δles={v['median_delta_lesion_max_sign']:.3f} "
                      f"Δctrl={v['median_delta_ctrl_max_sign']:.3f} p={fp(v['p_wilcoxon_max_sign'])}")


# ---------------------------------------------------------------- 报告
def write_report(out, man, s4):
    L = []
    add = L.append
    cs = out["consistency_vs_existing_2000"]
    geo = out["geometry_png_coords"]
    cl = out["classification"]
    so, sn = out["groups_old"], out["groups_new"]
    go, gn = out["method_gate"]["old"], out["method_gate"]["new"]

    add("# 遮挡实验报告（预实验）")
    add("")
    if cs.get("consistent") is False:
        add("> **⚠️ 推理流程与原结果不完全一致**：本次 orig 重跑与既有 2000 条测试集概率对比，"
            f"预测 BI-RADS 一致率 {cs['pred_birads_agreement']:.4f}"
            f"（不一致 {cs['pred_birads_disagreement']:.2%}，阈值 2%），"
            f"max 征象概率中位绝对差 {cs['max_sign_p_median_abs_diff']:.4f}，"
            f"P(BI-RADS≥3) 中位绝对差 {cs['p_birads_ge3_median_abs_diff']:.5f}（阈值 0.01）。"
            "因此**报告中同时给出两套分类**：`old`（既有 2000 条结果）与 `new`（本次 orig 结果）。"
            "所有 Δ 都用本次 orig 做基准，四个遮挡条件在同一次推理内完成，Δ 的比较不受该不一致影响。")
        add("")
    add("- 模型: `Qwen3.5-4B` + `Qwen3.5-4B-lora-2view-os-sub`（视觉塔冻结，LoRA r=8 仅作用于 LLM）")
    add("- 输入: 两视角 CC + MLO，PNG 原始 912×1520；模型输入 smart_resize 到 **672×1152**"
        "（774144 px，`image_max_pixels=786432`；patch 16 / merge 2 → **756 视觉 token/图，两视角 1512**），贪心解码，batch=8")
    add("- 样本: 测试集 GT BI-RADS≥3 的 192 个乳房 × 4 种情况（orig / lesion / ctrl1 / ctrl2）")
    add("- Δ 定义: **Δ = 本次 orig − 遮挡后**（正 = 概率下降）；主指标 `max_sign_p` = 4 个征象 P(=1) 的最大值")
    add("- 统计: 配对 Wilcoxon 符号秩检验（双侧，配对单位 = 乳房），对照取 ctrl1/ctrl2 的 Δ 平均")
    add("- 两套分类规则完全一致：`pred≥3 → detected`；`pred≤2 且（pred=2 或 max_sign_p>0.1）→ missed_sign`；其余 `missed_nosign`")
    add("")

    add("### 坐标口径（重要）")
    add("")
    add("- `vindr_detection_v1_folds.csv` 的 `xmin/ymin/xmax/ymax` 是 DICOM 原始坐标，**不能**按宽高比换算到 PNG"
        "（本仓库的 PNG 是裁剪后再缩放的，非等比且带偏移）。")
    add("- `resized_*` **就是现用 912×1520 PNG 的像素坐标**（第 1 步用叠加图人工核对：resized 框压在病灶上，"
        "按原图比例换算的框整体偏离）。本实验所有矩形都在该坐标系下，只做越界裁剪。")
    add("")

    # ---------------- 0
    add("## 0. 方法学检查")
    add("")
    add("### 0.1 推理管线一致性（本次 orig vs 既有 2000 条）")
    add("")
    add("| 项 | 值 |")
    add("|---|---|")
    add(f"| 对比样本数 | {cs['n_compared']} |")
    add(f"| 贪心 BI-RADS 预测一致率 | {fmt(cs['pred_birads_agreement'])} |")
    add(f"| 不一致比例（阈值 2%） | {fmt(cs['pred_birads_disagreement'])} |")
    add(f"| max 征象 P(=1) 最大/中位绝对差（阈值 0.01） | {cs['max_sign_p_max_abs_diff']:.2e} / "
        f"{cs['max_sign_p_median_abs_diff']:.2e} |")
    add(f"| P(BI-RADS≥3) 最大/中位绝对差 | {cs['p_birads_ge3_max_abs_diff']:.2e} / "
        f"{cs['p_birads_ge3_median_abs_diff']:.2e} |")
    if cs.get("consistent") is True:
        verdict = "一致"
    elif cs.get("consistent") is False:
        verdict = "不一致"
    else:
        verdict = "未判定"
    add(f"| 判定 | **{verdict}** |")
    dg = cs.get("diagnosis")
    if dg:
        add(f"| 不一致样本的 BI-RADS margin 中位 | {fmt(dg['median_margin_disagree'], 3)}"
            f"（全样本 {fmt(dg['median_margin_all'], 3)}） |")
        add(f"| \\|Δ max 征象 P(=1)\\| > 0.2 的样本数 | {dg['n_abs_dmaxsign_gt_0_2']} |")
        add(f"| \\|Δ(P4+P5)\\| > 0.1 的样本数（max） | {dg['n_abs_diff_p45_gt_0_1']}"
            f"（{fmt(dg['max_abs_diff_p_birads_45'], 3)}） |")
    add("")
    if dg and dg.get("disagreements"):
        add("| 不一致样本 | 旧预测 | 新预测 | margin(旧/新) | \\|Δmax_sign\\| | Δ(P4+P5) |")
        add("|---|---|---|---|---|---|")
        for d in dg["disagreements"]:
            add(f"| {d['id'][:12]}… | {d['old_pred']} | {d['new_pred']} | "
                f"{d['margin_old']:.3f} / {d['margin_new']:.3f} | "
                f"{d['abs_diff_max_sign_p']:.3f} | {d['diff_p_birads_45']:+.3f} |")
        add("")

    add("### 0.2 病灶大小（PNG 坐标下的面积占比）")
    add("")
    add("| 项 | 值 |")
    add("|---|---|")
    add(f"| 每乳房最大病灶框面积 / 整图 | 中位 {geo['max_box_area_ratio']['median']*100:.2f}%，"
        f"p90 {geo['max_box_area_ratio']['p90']*100:.2f}%，max {geo['max_box_area_ratio']['max']*100:.2f}% |")
    add(f"| 占比 > 5% / 10% / 20% / 40% 的乳房数 | {geo['max_box_area_ratio']['n_gt_5pct']} / "
        f"{geo['max_box_area_ratio']['n_gt_10pct']} / {geo['max_box_area_ratio']['n_gt_20pct']} / "
        f"{geo['max_box_area_ratio']['n_gt_40pct']} |")
    add("| 按 GT BI-RADS（n / 中位占比） | " + "；".join(
        f"{k}: {v['n']} / {v['median_max_box_area_ratio']*100:.2f}%"
        for k, v in geo["by_gt_birads"].items()) + " |")
    add(f"| 最大的 5 个 | " + "；".join(f"{b[-6:] if len(b) > 6 else b} {r*100:.1f}%"
                                     for r, b in geo["top5"]) + " |")
    add("")
    add("- 遮挡矩形 = 病灶框每边外扩 20%，面积约为原框的 **1.96 倍**。")

    add("")
    add("### 0.2b 病灶 token 数（单视角口径）")
    add("")
    add(f"- 口径：`该视角最大病灶框（不外扩）面积 / (912×1520) × {out['lesion_tokens_png_view']['tokens_per_image']}`，"
        "每乳房取 CC/MLO 较大值；756 = 672×1152 的视觉 token 数/图（774144/256/4）。")
    add("- 类型口径：含 mass → mass；否则含 suspicious_calcification → calcification；否则 other。")
    add("")
    add("| 分组（old） | n | 中位 | p25 | p75 | 分组（new） | n | 中位 | p25 | p75 |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    for g in GROUPS:
        a, b = out["lesion_tokens_png_view"]["old"]["by_group"][g], \
               out["lesion_tokens_png_view"]["new"]["by_group"][g]
        add(f"| {g} | {a['n']} | {fmt(a['median'],1)} | {fmt(a['p25'],1)} | {fmt(a['p75'],1)} "
            f"| {g} | {b['n']} | {fmt(b['median'],1)} | {fmt(b['p25'],1)} | {fmt(b['p75'],1)} |")
    add("")
    add("| 病灶类型 | n | 中位 | p25 | p75 | min | max |")
    add("|---|---|---|---|---|---|---|")
    for t in ("mass", "calcification", "other"):
        s = out["lesion_tokens_png_view"]["new"]["by_type"][t]
        add(f"| {t} | {s['n']} | {fmt(s['median'],1)} | {fmt(s['p25'],1)} | {fmt(s['p75'],1)} "
            f"| {fmt(s['min'],1)} | {fmt(s['max'],1)} |")
    add("")
    add("| 类型 × 分组（new） | n | 中位 | p25 | p75 |")
    add("|---|---|---|---|---|")
    tokpb = {b: (tokens_of(man, b), ltype_of(man, b)) for b in man["breasts"]}
    for t in ("mass", "calcification", "other"):
        for g in GROUPS:
            vals = [tokpb[b][0] for b in out["group_ids"]["new"][g] if tokpb[b][1] == t]
            if not vals:
                continue
            q = quartiles(vals)
            add(f"| {t} / {g} | {q['n']} | {fmt(q['median'],1)} | {fmt(q['p25'],1)} | {fmt(q['p75'],1)} |")
    add("")

    add("### 0.3 detected 组是否能被遮挡打下去")
    add("")
    add("| 分类 | n | 中位 Δ(lesion) max_sign | 中位 Δ(ctrl) max_sign | 中位 Δ(lesion) P(≥3) | 判定 |")
    add("|---|---|---|---|---|---|")
    for tag, g in (("old", go), ("new", gn)):
        add(f"| {tag} | {g['n']} | {g['median_delta_lesion_max_sign_p']:.4f} | "
            f"{g['median_delta_ctrl_mean_max_sign_p']:.4f} | "
            f"{g['median_delta_lesion_p_birads_ge3']:.4f} | **{'通过' if g['pass'] else '未通过'}** |")
    add("")
    if not (go["pass"] and gn["pass"]):
        add("> **警告：detected 组遮病灶后概率没有明显下降（或不如对照），遮挡方法/坐标映射可能有问题，"
            "其他组不下结论。**")
        add("")

    # ---------------- 1 分类
    add("## 1. 分类：新旧对比")
    add("")
    add("| 组 | old（既有 2000 条） | new（本次 orig） | 差 |")
    add("|---|---|---|---|")
    for g in GROUPS:
        add(f"| {g} | {cl['counts_old'][g]} | {cl['counts_new'][g]} | "
            f"{cl['counts_new'][g] - cl['counts_old'][g]:+d} |")
    add(f"| 合计 | {sum(cl['counts_old'].values())} | {sum(cl['counts_new'].values())} | — |")
    add("")
    add(f"**换组乳房数：{cl['n_switched']} / 192**")
    add("")
    add("| old ↓ / new → | detected | missed_sign | missed_nosign |")
    add("|---|---|---|---|")
    for g in GROUPS:
        row = cl["cross_tab_old_rows"][g]
        add(f"| {g} | " + " | ".join(str(row[h]) for h in GROUPS) + " |")
    add("")
    if cl["switched"]:
        add("| 换组样本 | old | new | pred(旧→新) | max_sign(旧→新) |")
        add("|---|---|---|---|---|")
        for x in cl["switched"]:
            add(f"| {x['id'][:12]}… | {x['old']} | {x['new']} | "
                f"{x['pred_old']}→{x['pred_new']} | {x['max_sign_old']:.3f}→{x['max_sign_new']:.3f} |")
        add("")

    # ---------------- 2 Δ 表
    add("## 2. Δ 表（新旧两套分类并排）")
    add("")
    for key, title in METRIC_TITLES.items():
        add(f"### 2.{list(METRIC_TITLES).index(key)+1} {title}")
        add("")
        add("| 组 | old n | old Δlesion | old Δctrl | old p | old 比例 | "
            "new n | new Δlesion | new Δctrl | new p | new 比例 |")
        add("|---|---|---|---|---|---|---|---|---|---|---|")
        for g in GROUPS:
            mo = so[g]["metrics"].get(key)
            mn = sn[g]["metrics"].get(key)
            if not mo or not mn:
                continue
            add(f"| {g} | {mo['n']} | {fmt(mo['median_delta_lesion'])} | "
                f"{fmt(mo['median_delta_ctrl_mean'])} | {fp(mo['p_wilcoxon_lesion_vs_ctrl'])} | "
                f"{fmt(mo['frac_lesion_gt_ctrl'],2)} | {mn['n']} | {fmt(mn['median_delta_lesion'])} | "
                f"{fmt(mn['median_delta_ctrl_mean'])} | {fp(mn['p_wilcoxon_lesion_vs_ctrl'])} | "
                f"{fmt(mn['frac_lesion_gt_ctrl'],2)} |")
        add("")
        add("（Δlesion / Δctrl 为组内中位数；比例 = 「病灶下降 > 对照下降」的乳房占比；"
            "ctrl 取 ctrl1/ctrl2 的 Δ 平均）")
        add("")

    # ---------------- 3 迁移
    add("## 3. 遮病灶后 BI-RADS 迁移（新旧并排）")
    add("")
    add("| 分类 | 组 | n | 2→1 (lesion) | ≥3→≤2 (lesion) | 2→1 (ctrl1) | ≥3→≤2 (ctrl1) | 2→1 (ctrl2) | ≥3→≤2 (ctrl2) |")
    add("|---|---|---|---|---|---|---|---|---|")
    for tag, st in (("old", so), ("new", sn)):
        for g in GROUPS:
            t = st[g]["transitions"]
            if not t:
                continue
            add(f"| {tag} | {g} | {st[g]['n']} | {t['lesion_2to1']} | {t['lesion_ge3_to_le2']} | "
                f"{t['ctrl1_2to1']} | {t['ctrl1_ge3_to_le2']} | {t['ctrl2_2to1']} | {t['ctrl2_ge3_to_le2']} |")
    add("")

    # ---------------- 4 missed_sign 拆分
    add("## 4. missed_sign 内部：按预测 BI-RADS = 2 / = 1 拆分")
    add("")
    add("| 分类 | 子组 | n | 原图 max_sign | Δlesion | Δctrl | Wilcoxon p | 比例 |")
    add("|---|---|---|---|---|---|---|---|")
    for tag, ms in (("old", out["missed_sign_split"]["old"]), ("new", out["missed_sign_split"]["new"])):
        for k in ("pred=2", "pred=1"):
            v = ms[k]
            if not v.get("n"):
                add(f"| {tag} | {k} | 0 | — | — | — | — | — |")
                continue
            add(f"| {tag} | {k} | {v['n']} | {fmt(v['median_orig_max_sign'])} | "
                f"{fmt(v['median_delta_lesion_max_sign'])} | {fmt(v['median_delta_ctrl_max_sign'])} | "
                f"{fp(v['p_wilcoxon_max_sign'])} | {fmt(v['frac_lesion_gt_ctrl'],2)} |")
    add("")
    add("| 分类 | 子组 | Δlesion P(≥3) | Δctrl P(≥3) | Wilcoxon p |")
    add("|---|---|---|---|---|")
    for tag, ms in (("old", out["missed_sign_split"]["old"]), ("new", out["missed_sign_split"]["new"])):
        for k in ("pred=2", "pred=1"):
            v = ms[k]
            if not v.get("n"):
                continue
            add(f"| {tag} | {k} | {fmt(v['median_delta_lesion_p_ge3'])} | "
                f"{fmt(v['median_delta_ctrl_p_ge3'])} | {fp(v['p_wilcoxon_p_ge3'])} |")
    add("")

    # ---------------- 5 单视角
    add(f"## 5. 只有一个视角有病灶框的乳房（{out['n_single_view_breasts']} 个，单独列出）")
    add("")
    add("Δ 与分类无关（Δ 由本次 orig 与遮挡结果直接算出）；旧组/新组并列。")
    add("")
    add("| id | 有框视角 | Δlesion | Δctrl均 | pred 原→遮病灶后 | old 组 | new 组 |")
    add("|---|---|---|---|---|---|---|")
    oldmap = {x["id"]: x for g in GROUPS for x in so[g]["single_view_subset"]}
    newmap = {x["id"]: x for g in GROUPS for x in sn[g]["single_view_subset"]}
    for bid in sorted(oldmap):
        x = newmap.get(bid, oldmap[bid])
        og = next((g for g in GROUPS if bid in out["group_ids"]["old"][g]), "—")
        ng = next((g for g in GROUPS if bid in out["group_ids"]["new"][g]), "—")
        add(f"| {bid} | {x['views_with_box']} | {fmt(x['delta_lesion'])} | {fmt(x['delta_ctrl_mean'])} | "
            f"{x['pred_orig']}→{x['pred_lesion']} | {og} | {ng} |")
    add("")

    # ---------------- 6 口径
    add("## 6. 结论（可直接引用）")
    add("")
    m_no = so["detected"]["metrics"]["max_sign_p"]
    m_nn = sn["detected"]["metrics"]["max_sign_p"]
    ms_o = so["missed_sign"]["metrics"]["max_sign_p"]
    ms_n = sn["missed_sign"]["metrics"]["max_sign_p"]
    mn_o = so["missed_nosign"]["metrics"]["max_sign_p"]
    mn_n = sn["missed_nosign"]["metrics"]["max_sign_p"]
    g3_o = so["detected"]["metrics"]["p_birads_ge3"]
    t_det_n = sn["detected"]["transitions"]
    t_det_o = so["detected"]["transitions"]
    t_ms_n = sn["missed_sign"]["transitions"]
    ratio_ms = (ms_n["median_delta_lesion"] / m_nn["median_delta_lesion"]
                if m_nn["median_delta_lesion"] else float("nan"))
    add(f"1. **遮挡手段有效**：detected 组 `max_sign_p` 中位 Δ = {m_nn['median_delta_lesion']:.3f}"
        f"（对照 {m_nn['median_delta_ctrl_mean']:.3f}），`P(BI-RADS≥3)` 中位 Δ = "
        f"{sn['detected']['metrics']['p_birads_ge3']['median_delta_lesion']:.3f}；"
        f"{t_det_n['lesion_ge3_to_le2']}/{sn['detected']['n']} 从 ≥3 跌到 ≤2（对照 {t_det_n['ctrl1_ge3_to_le2']} 与 "
        f"{t_det_n['ctrl2_ge3_to_le2']}）。")
    add(f"2. **missed_sign 的「怀疑」确有病灶来源，但强度低一个量级**：中位 Δ = "
        f"{ms_n['median_delta_lesion']:.3f}（对照 {ms_n['median_delta_ctrl_mean']:.3f}，"
        f"Wilcoxon p {fp(ms_n['p_wilcoxon_lesion_vs_ctrl'])}，"
        f"{ms_n['frac_lesion_gt_ctrl']*100:.0f}% 的乳房病灶下降大于对照），"
        f"但只有 detected 组的 {ratio_ms:.2f}×；遮病灶后 {t_ms_n['lesion_2to1']}/{sn['missed_sign']['n']} "
        f"从 BI-RADS 2 退到 1（对照 {t_ms_n['ctrl1_2to1']} 与 {t_ms_n['ctrl2_2to1']}）。")
    add(f"3. **missed_nosign 基本不来自病灶**：中位 Δ = {mn_n['median_delta_lesion']:.4f}"
        f"（对照 {mn_n['median_delta_ctrl_mean']:.4f}），虽 p = {fp(mn_n['p_wilcoxon_lesion_vs_ctrl'])} "
        "但效应量实质为零；该类乳房的输出对病灶遮挡不敏感。")
    add(f"4. **新旧分类不改变结论**：{cl['n_switched']}/192 换组（detected {cl['counts_old']['detected']}→"
        f"{cl['counts_new']['detected']}，missed_sign {cl['counts_old']['missed_sign']}→"
        f"{cl['counts_new']['missed_sign']}，missed_nosign {cl['counts_old']['missed_nosign']}→"
        f"{cl['counts_new']['missed_nosign']}），三组 Δ 中位数几乎不变"
        f"（{m_no['median_delta_lesion']:.3f}→{m_nn['median_delta_lesion']:.3f}、"
        f"{ms_o['median_delta_lesion']:.3f}→{ms_n['median_delta_lesion']:.3f}、"
        f"{mn_o['median_delta_lesion']:.4f}→{mn_n['median_delta_lesion']:.4f}）。")
    sp2 = out["missed_sign_split"]["new"]["pred=2"]
    sp1 = out["missed_sign_split"]["new"]["pred=1"]
    if sp2.get("n") and sp1.get("n"):
        add(f"5. **missed_sign 内部**：预测=2 子组 Δ = {sp2['median_delta_lesion_max_sign']:.3f}"
            f"（对照 {sp2['median_delta_ctrl_max_sign']:.3f}，p {fp(sp2['p_wilcoxon_max_sign'])}，n={sp2['n']}）"
            f"高于预测=1 子组 {sp1['median_delta_lesion_max_sign']:.3f}"
            f"（对照 {sp1['median_delta_ctrl_max_sign']:.3f}，p {fp(sp1['p_wilcoxon_max_sign'])}，n={sp1['n']}）；"
            "两者都显著，说明「被压到 1」的那部分也仍有可测的病灶依赖。")
    add(f"6. **只有单视角有框的 {out['n_single_view_breasts']} 个乳房**已单独列出（表 5）："
        "这些样本另一视角无遮挡，Δ 被稀释，解释时需排除。")
    add("")
    add("## 7. 口径与已知限制")
    add("")
    add("- 遮挡: lesion 框每边外扩 20%（裁到边界内）；对照与病灶框同尺寸、随机放在 Otsu 乳腺区内"
        "（≥80% 面积在乳腺内、不与任何外扩病灶框重叠），每个病灶框每个对照条件各 1 个；"
        "填充 = 框外 16px 环中位灰度；seed 42。")
    add("- 只有一个视角有框的乳房，另一视角没有可遮的病灶（也不产生对照），该视角输入保持不变，"
        "因此这类样本的 Δ 天生被稀释，单独列表。")
    add("- 两套分类只是同一批 192 个乳房在 old / new 概率下的分组，Δ、Wilcoxon、迁移、token 数的样本集合随之不同；"
        "对照均为配对（同一乳房同一视角）。")
    add("- 病灶框极大的样本（外扩后占整图 >20%）遮挡本身近似抹掉大片组织，Δ 的解释力有限。")
    add("")
    (OCC / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def tokens_of(man, bid):
    return max_view_tokens(man["breasts"][bid])


def ltype_of(man, bid):
    return lesion_type(man["breasts"][bid])


if __name__ == "__main__":
    main()
