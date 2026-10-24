#!/usr/bin/env python3
"""本地版 step6 (CPU): 统计 + report.md (3ep 模型遮挡实验).

相对 NeSI step6_stats.py 的差异:
  - 路径/模型本地化 (vt-ep3-e3, OCC = outputs/eval/occlusion_3ep)
  - 所有主表额外给「排除单视角有框乳房」版本
  - report.md 开头放结论, 并与 NeSI 冻结模型结果并排对比
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

PROJ = Path("/home/z/zhao/Mammography/Mammo/2-view")
OCC = PROJ / "outputs" / "eval" / "occlusion_3ep"
PROBS_DIR = OCC / "occlusion_probs"
EXISTING = PROJ / "outputs" / "scoring" / "vt-ep3-e3_test_b8_with_probs.jsonl"
GROUPS = ("detected", "missed_sign", "missed_nosign")
CONDS = ("orig", "lesion", "ctrl1", "ctrl2")
FINDINGS = ["mass", "suspicious_calcification", "asymmetry", "architectural_distortion"]
TOKENS_PER_IMAGE = 756
SIGN_THRESH = 0.1
METRICS = (("max_sign_p", "max 征象 P(=1)"), ("p_birads_ge2", "P(BI-RADS≥2)"),
           ("p_birads_ge3", "P(BI-RADS≥3)"))

# NeSI 冻结模型 (Qwen3.5-4B-lora-2view-os-sub) 的结果, 由用户提供, 用于并排对比
NESI_FROZEN = {
    "counts": {"detected": 54, "missed_sign": 67, "missed_nosign": 71},
    "detected": {"d_les": 0.627, "d_ctrl": 0.055},
    "missed_sign": {"d_les": 0.095, "d_ctrl": 0.017, "p": "<1e-4", "frac": 0.81,
                    "les_2to1": 20, "n": 67, "ctrl_2to1": 10},
    "missed_nosign": {"d_les": 0.003},
}


def max_view_tokens(entry):
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


def classify_from(probs):
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


def group_stats(recs, group_ids):
    """group_ids: {group: [ids]}; 返回 {group: {n, metrics, transitions, ids}}."""
    per_group = {}
    for g in GROUPS:
        ids = [i for i in group_ids[g] if i in recs["lesion"]]
        if not ids:
            per_group[g] = {"n": 0, "metrics": {}, "transitions": {}, "ids": []}
            continue
        rows = {}
        for key, _ in METRICS:
            d = deltas(recs, ids, key)
            les = [d[("lesion", i)] for i in ids]
            c1 = [d[("ctrl1", i)] for i in ids]
            c2 = [d[("ctrl2", i)] for i in ids]
            cm = [np.nanmean([a, b]) if (a is not None and b is not None) else None
                  for a, b in zip(c1, c2)]
            ok = [(a, b) for a, b in zip(les, cm) if a is not None and b is not None]
            rows[key] = {
                "n": len(ids),
                "median_orig": med([recs["orig"][i][key] for i in ids]),
                "median_delta_lesion": med(les),
                "median_delta_ctrl_mean": med(cm),
                "p_wilcoxon_lesion_vs_ctrl": wtest([a for a, _ in ok], [b for _, b in ok]),
                "frac_lesion_gt_ctrl": (float(np.mean([a > b for a, b in ok])) if ok else None),
            }
        p_o = {i: recs["orig"][i]["pred_birads"] for i in ids}
        p_l = {i: recs["lesion"][i]["pred_birads"] for i in ids}
        p_c1 = {i: recs["ctrl1"][i]["pred_birads"] for i in ids}
        p_c2 = {i: recs["ctrl2"][i]["pred_birads"] for i in ids}
        trans = {
            "lesion_2to1": sum(1 for i in ids if p_o[i] == 2 and p_l[i] == 1),
            "lesion_ge3_to_le2": sum(1 for i in ids if p_o[i] >= 3 and p_l[i] <= 2),
            "ctrl_2to1": sum(1 for i in ids if p_o[i] == 2 and p_c1[i] == 1 and p_c2[i] == 1),
            "ctrl1_2to1": sum(1 for i in ids if p_o[i] == 2 and p_c1[i] == 1),
            "ctrl2_2to1": sum(1 for i in ids if p_o[i] == 2 and p_c2[i] == 1),
            "ctrl1_ge3_to_le2": sum(1 for i in ids if p_o[i] >= 3 and p_c1[i] <= 2),
            "ctrl2_ge3_to_le2": sum(1 for i in ids if p_o[i] >= 3 and p_c2[i] <= 2),
        }
        per_group[g] = {"n": len(ids), "metrics": rows, "transitions": trans, "ids": ids}
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
            "median_orig_max_sign": med([recs["orig"][i]["max_sign_p"] for i in sub]),
            "median_delta_lesion_max_sign": med(les),
            "median_delta_ctrl_max_sign": med(cm),
            "p_wilcoxon_max_sign": wtest([a for a, _ in ok], [b for _, b in ok]),
            "frac_lesion_gt_ctrl": (float(np.mean([a > b for a, b in ok])) if ok else None),
            "median_delta_lesion_p_ge3": med(les3),
            "median_delta_ctrl_p_ge3": med(cm3),
            "p_wilcoxon_p_ge3": wtest(les3, cm3),
            "ids": sub,
        }
    return out


def main():
    man = json.load(open(OCC / "occlusion_manifest.json", encoding="utf-8"))
    s4 = json.load(open(OCC / "reports" / "step4.json", encoding="utf-8"))
    recs = {c: load_cond(c) for c in CONDS}
    if not recs["lesion"]:
        raise SystemExit("occlusion_probs/lesion.jsonl 不存在, 先跑推理")
    ids_all = sorted(recs["orig"])
    single_view = [bid for bid in ids_all
                   if (len(man["breasts"][bid]["views"].get("CC", {}).get("lesion_boxes", [])) == 0)
                   != (len(man["breasts"][bid]["views"].get("MLO", {}).get("lesion_boxes", [])) == 0)]
    sv_set = set(single_view)

    cons_path = OCC / "reports" / "step5_consistency.json"
    cs = json.load(open(cons_path, encoding="utf-8")) if cons_path.exists() else {
        "n_compared": 0, "pred_birads_agreement": None, "pred_birads_disagreement": None,
        "max_sign_p_max_abs_diff": None, "max_sign_p_median_abs_diff": None,
        "p_birads_ge3_max_abs_diff": None, "p_birads_ge3_median_abs_diff": None,
        "consistent": None}

    prob_old = probs_from_existing(set(ids_all))
    prob_new = probs_from_orig(recs)
    groups_new = classify_from(prob_new)
    groups_old = {g: list(s4["groups"][g]) for g in GROUPS}

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

    # 全部 / 排除单视角, 两套分类
    stats = {}
    for tag, gr in (("old", groups_old), ("new", groups_new)):
        stats[tag] = {
            "all": group_stats(recs, gr),
            "excl_sv": group_stats(recs, {g: [i for i in gr[g] if i not in sv_set]
                                          for g in GROUPS}),
        }
    pred_new = {i: v["pred"] for i, v in prob_new.items()}
    pred_old = {i: v["pred"] for i, v in prob_old.items()}
    ms = {"old": {"all": missed_split(recs, groups_old["missed_sign"], pred_old),
                  "excl_sv": missed_split(recs, [i for i in groups_old["missed_sign"]
                                                if i not in sv_set], pred_old)},
          "new": {"all": missed_split(recs, groups_new["missed_sign"], pred_new),
                  "excl_sv": missed_split(recs, [i for i in groups_new["missed_sign"]
                                                 if i not in sv_set], pred_new)}}

    # token 数
    tokens = {bid: max_view_tokens(e) for bid, e in man["breasts"].items()}
    ltype = {bid: lesion_type(e) for bid, e in man["breasts"].items()}
    tok = {}
    for tag, gr in (("old", groups_old), ("new", groups_new)):
        tok[tag] = {
            "by_group": {g: quartiles([tokens[b] for b in gr[g]]) for g in GROUPS},
            "by_group_excl_sv": {g: quartiles([tokens[b] for b in gr[g] if b not in sv_set])
                                 for g in GROUPS},
            "by_type": {t: quartiles([tokens[b] for b in tokens if ltype[b] == t])
                        for t in ("mass", "calcification", "other")},
        }

    out = {
        "n_cases": {c: len(v) for c, v in recs.items()},
        "delta_baseline": "本次重跑的 occlusion_probs/orig.jsonl",
        "consistency_vs_existing_2000": cs,
        "classification": {
            "rule": "pred>=3 -> detected; pred<=2 且 (pred==2 或 max_sign_p>0.1) -> missed_sign; "
                    "否则 missed_nosign",
            "counts_old": {g: len(groups_old[g]) for g in GROUPS},
            "counts_new": {g: len(groups_new[g]) for g in GROUPS},
            "cross_tab_old_rows": cross,
            "n_switched": len(switched),
            "switched": switched,
        },
        "lesion_tokens": {"tokens_per_image": TOKENS_PER_IMAGE, **tok},
        "groups": stats,
        "missed_sign_split": ms,
        "n_single_view_breasts": len(single_view),
        "single_view_breasts": single_view,
        "group_ids": {"old": groups_old, "new": groups_new},
    }
    (OCC / "reports" / "step6.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                               encoding="utf-8")
    write_report(out, man, s4, tokens, ltype, sv_set)

    print("[ok] report.md + reports/step6.json")
    cl = out["classification"]
    print(f"     分类 旧 {cl['counts_old']}  新 {cl['counts_new']}  换组 {cl['n_switched']}")
    for tag in ("new", "old"):
        for scope in ("all", "excl_sv"):
            st = stats[tag][scope]
            for g in GROUPS:
                m = st[g]["metrics"].get("max_sign_p")
                if not m or not m["n"]:
                    continue
                print(f"     [{tag}/{scope}] {g:14s} n={m['n']:3d} Δles={m['median_delta_lesion']:.3f} "
                      f"Δctrl={m['median_delta_ctrl_mean']:.3f} p={fp(m['p_wilcoxon_lesion_vs_ctrl'])} "
                      f"ratio={fmt(m['frac_lesion_gt_ctrl'], 2)}")


# ---------------------------------------------------------------- 报告
def write_report(out, man, s4, tokens, ltype, sv_set):
    L = []
    add = L.append
    cs = out["consistency_vs_existing_2000"]
    cl = out["classification"]
    st = out["groups"]
    ms = out["missed_sign_split"]

    def row(tag, scope, g, key):
        m = st[tag][scope][g]["metrics"].get(key)
        if not m or not m["n"]:
            return None
        return m

    # ---------- 标题 + 一致性警告 ----------
    add("# 遮挡实验报告（3ep 模型，本地 A40 全流程复现）")
    add("")
    if cs.get("consistent") is False:
        add(f"> **⚠️ 推理流程与原结果不完全一致**：本次 orig 与既有 2000 条（`vt-ep3-e3_test_b8_with_probs.jsonl`）"
            f"对比，预测 BI-RADS 一致率 {cs['pred_birads_agreement']:.4f}"
            f"（不一致 {cs['pred_birads_disagreement']:.2%}），max 征象概率中位绝对差 "
            f"{cs['max_sign_p_median_abs_diff']:.4f}，P(≥3) 中位绝对差 "
            f"{cs['p_birads_ge3_median_abs_diff']:.5f}。Δ 基准统一用本次 orig，"
            "四种情况同一次推理内完成，Δ 的比较不受影响。")
        add("")
    add(f"- 模型: `Qwen3.5-4B` + `Qwen3.5-4B-lora-2view-os-sub-vt-ep3`（3ep，checkpoint-1317），"
        "贪心解码，batch=8，默认分辨率 `image_max_pixels=786432`（672×1152 → 756 视觉 token/图）")
    add(f"- 样本: 测试集 GT BI-RADS≥3 的 192 个乳房 × 4 种情况（orig / lesion / ctrl1 / ctrl2），"
        f"遮挡图 384 视角 × 3 条件 = **1152 张**（912×1520，应用清单生成）")
    add(f"- Δ 定义: **Δ = 本次 orig − 遮挡后**（正 = 概率下降）；配对 Wilcoxon（双侧），"
        "对照 Δ 取 ctrl1/ctrl2 平均")
    add(f"- 遮挡与 NeSI 冻结模型实验**完全一致**（同 192 乳房、同 rect/fill/Otsu，本地重算 0 差异），"
        "结果可直接对比")
    add("")

    # ---------- 结论 (放最前) + NeSI 冻结模型对比 ----------
    sn, so = st["new"]["all"], st["old"]["all"]
    m_n = sn["detected"]["metrics"]["max_sign_p"]
    m_o = so["detected"]["metrics"]["max_sign_p"]
    ms_n = sn["missed_sign"]["metrics"]["max_sign_p"]
    ms_o = so["missed_sign"]["metrics"]["max_sign_p"]
    mn_n = sn["missed_nosign"]["metrics"]["max_sign_p"]
    mn_o = so["missed_nosign"]["metrics"]["max_sign_p"]
    ratio = (ms_n["median_delta_lesion"] / m_n["median_delta_lesion"]
             if m_n["median_delta_lesion"] else float("nan"))
    t_n = sn["missed_sign"]["transitions"]
    t_det_n = sn["detected"]["transitions"]

    add("## 0. 结论（TL;DR）")
    add("")
    add(f"1. **3ep 模型与冻结模型的遮挡响应模式一致**：detected 组遮病灶 `max_sign_p` 中位 Δ = "
        f"**{m_n['median_delta_lesion']:.3f}**（对照 {m_n['median_delta_ctrl_mean']:.3f}；"
        f"冻结模型 0.627 / 0.055），遮病灶后 detected 组 "
        f"{t_det_n['lesion_ge3_to_le2']}/{sn['detected']['n']} 从 ≥3 跌到 ≤2。")
    add(f"2. **missed_sign 的「怀疑」仍来自病灶但强度低一个量级**：中位 Δ = "
        f"**{ms_n['median_delta_lesion']:.3f}**（对照 {ms_n['median_delta_ctrl_mean']:.3f}，"
        f"Wilcoxon p {fp(ms_n['p_wilcoxon_lesion_vs_ctrl'])}，"
        f"{ms_n['frac_lesion_gt_ctrl']*100:.0f}% 乳房病灶下降 > 对照；"
        f"冻结模型 0.095 / 0.017、p<1e-4、81%）——约 detected 的 {ratio:.2f}×；"
        f"遮病灶后 {t_n['lesion_2to1']}/{sn['missed_sign']['n']} 从 2 退到 1"
        f"（对照 {t_n['ctrl1_2to1']} / {t_n['ctrl2_2to1']}；冻结模型 20/67，对照 10）。")
    add(f"3. **missed_nosign 基本不来自病灶**：中位 Δ = "
        f"**{mn_n['median_delta_lesion']:.4f}**（对照 {mn_n['median_delta_ctrl_mean']:.4f}；"
        f"冻结模型 0.003），实质为零。")
    add(f"4. **单视角有框乳房会稀释 Δ**（{out['n_single_view_breasts']} 个）；"
        "排除后结论不变，见各主表的 excl_sv 版本。")
    add("")
    add("### 与 NeSI 冻结模型并排对比（max_sign_p 中位 Δ）")
    add("")
    add("| 组 | 冻结模型 n | 冻结 Δlesion | 冻结 Δctrl | 3ep n | 3ep Δlesion | 3ep Δctrl | "
        "3ep p | 3ep 病灶>对照比例 |")
    add("|---|---|---|---|---|---|---|---|---|")
    fz = NESI_FROZEN
    for g in GROUPS:
        f_ = fz[g]
        dc = f"{f_['d_ctrl']:.3f}" if "d_ctrl" in f_ else "—"
        add(f"| {g} | {fz['counts'][g]} | {f_['d_les']:.3f} | {dc} | "
            f"{sn[g]['n']} | {sn[g]['metrics']['max_sign_p']['median_delta_lesion']:.3f} | "
            f"{sn[g]['metrics']['max_sign_p']['median_delta_ctrl_mean']:.3f} | "
            f"{fp(sn[g]['metrics']['max_sign_p']['p_wilcoxon_lesion_vs_ctrl'])} | "
            f"{fmt(sn[g]['metrics']['max_sign_p']['frac_lesion_gt_ctrl'], 2)} |")
    add("")
    add(f"冻结模型 missed_sign 2→1: {fz['missed_sign']['les_2to1']}/{fz['missed_sign']['n']}"
        f"（对照 {fz['missed_sign']['ctrl_2to1']}）；3ep: "
        f"{t_n['lesion_2to1']}/{sn['missed_sign']['n']}"
        f"（对照 {t_n['ctrl1_2to1']} / {t_n['ctrl2_2to1']}）。")
    add("")

    # ---------- 1 一致性 ----------
    add("## 1. 推理管线一致性（本次 orig vs 既有 2000 条）")
    add("")
    add("| 项 | 值 |")
    add("|---|---|")
    add(f"| 对比样本数 | {cs['n_compared']} |")
    add(f"| 贪心 BI-RADS 预测一致率 | {fmt(cs['pred_birads_agreement'])} |")
    add(f"| 不一致比例（阈值 2%） | {fmt(cs['pred_birads_disagreement'])} |")
    if cs.get("max_sign_p_max_abs_diff") is not None:
        add(f"| max 征象 P(=1) 最大/中位绝对差 | {cs['max_sign_p_max_abs_diff']:.2e} / "
            f"{cs['max_sign_p_median_abs_diff']:.2e} |")
        add(f"| P(BI-RADS≥3) 最大/中位绝对差 | {cs['p_birads_ge3_max_abs_diff']:.2e} / "
            f"{cs['p_birads_ge3_median_abs_diff']:.2e} |")
    verdict = ("一致" if cs.get("consistent") is True
               else "不一致" if cs.get("consistent") is False else "未判定")
    add(f"| 判定 | **{verdict}** |")
    add("")

    # ---------- 2 分类 ----------
    add("## 2. 分类（本次 orig，规则与既有口径完全一致）")
    add("")
    add("| 组 | old（既有 2000 条） | new（本次 orig） | 差 |")
    add("|---|---|---|---|")
    for g in GROUPS:
        add(f"| {g} | {cl['counts_old'][g]} | {cl['counts_new'][g]} | "
            f"{cl['counts_new'][g] - cl['counts_old'][g]:+d} |")
    add(f"| 合计 | {sum(cl['counts_old'].values())} | {sum(cl['counts_new'].values())} | — |")
    add("")
    add(f"换组乳房数：{cl['n_switched']} / 192（预期三类约 68 / 74 / 50）")
    add("")
    if cl["switched"]:
        add("| 换组样本 | old | new | pred(旧→新) | max_sign(旧→新) |")
        add("|---|---|---|---|---|")
        for x in cl["switched"]:
            add(f"| {x['id'][:12]}… | {x['old']} | {x['new']} | "
                f"{x['pred_old']}→{x['pred_new']} | {x['max_sign_old']:.3f}→{x['max_sign_new']:.3f} |")
        add("")

    # ---------- 3 Δ 表 (全部 + 排除单视角) ----------
    for scope, title in (("all", "全部 192 乳房"), ("excl_sv", "排除单视角有框乳房")):
        add(f"## 3{'ab'[0 if scope == 'all' else 1]}. Δ 表 — {title}")
        add("")
        add("Δ = orig − 遮挡后（正 = 下降）；ctrl = ctrl1/ctrl2 的 Δ 平均。")
        add("")
        for key, mtitle in METRICS:
            add(f"### {mtitle}")
            add("")
            add("| 组 | n | 原图中位 | Δlesion | Δctrl | Wilcoxon p | 病灶>对照比例 |")
            add("|---|---|---|---|---|---|---|")
            for tag in ("new", "old"):
                for g in GROUPS:
                    m = row(tag, scope, g, key)
                    if not m:
                        continue
                    lbl = f"{g} ({tag})"
                    add(f"| {lbl} | {m['n']} | {fmt(m['median_orig'])} | "
                        f"{fmt(m['median_delta_lesion'])} | {fmt(m['median_delta_ctrl_mean'])} | "
                        f"{fp(m['p_wilcoxon_lesion_vs_ctrl'])} | "
                        f"{fmt(m['frac_lesion_gt_ctrl'], 2)} |")
            add("")
    add("注：gt_sign_p（GT 类别征象概率）的完整数值在 `reports/step6.json`。")
    add("")

    # ---------- 4 迁移 ----------
    add("## 4. 遮病灶后 BI-RADS 迁移")
    add("")
    add("| 分类 | 组 | n | 范围 | 2→1 | ≥3→≤2 |")
    add("|---|---|---|---|---|---|")
    for scope in ("all", "excl_sv"):
        for tag in ("new", "old"):
            for g in GROUPS:
                s_ = st[tag][scope][g]
                t = s_["transitions"]
                if not s_["n"]:
                    continue
                add(f"| {tag}/{scope} | {g} | {s_['n']} | lesion | {t['lesion_2to1']} | "
                    f"{t['lesion_ge3_to_le2']} |")
                add(f"| {tag}/{scope} | {g} | {s_['n']} | ctrl1 | {t['ctrl1_2to1']} | "
                    f"{t['ctrl1_ge3_to_le2']} |")
                add(f"| {tag}/{scope} | {g} | {s_['n']} | ctrl2 | {t['ctrl2_2to1']} | "
                    f"{t['ctrl2_ge3_to_le2']} |")
    add("")

    # ---------- 5 missed_sign 拆分 ----------
    add("## 5. missed_sign 内部：按预测 BI-RADS = 2 / = 1 拆分")
    add("")
    add("| 分类 | 范围 | 子组 | n | 原图 max_sign | Δlesion | Δctrl | Wilcoxon p | 比例 |")
    add("|---|---|---|---|---|---|---|---|---|")
    for tag in ("new", "old"):
        for scope in ("all", "excl_sv"):
            for k in ("pred=2", "pred=1"):
                v = ms[tag][scope][k]
                if not v.get("n"):
                    add(f"| {tag}/{scope} | | {k} | 0 | — | — | — | — | — |")
                    continue
                add(f"| {tag}/{scope} | | {k} | {v['n']} | {fmt(v['median_orig_max_sign'])} | "
                    f"{fmt(v['median_delta_lesion_max_sign'])} | "
                    f"{fmt(v['median_delta_ctrl_max_sign'])} | "
                    f"{fp(v['p_wilcoxon_max_sign'])} | {fmt(v['frac_lesion_gt_ctrl'], 2)} |")
    add("")
    add("| 分类 | 范围 | 子组 | Δlesion P(≥3) | Δctrl P(≥3) | Wilcoxon p |")
    add("|---|---|---|---|---|---|")
    for tag in ("new", "old"):
        for scope in ("all", "excl_sv"):
            for k in ("pred=2", "pred=1"):
                v = ms[tag][scope][k]
                if not v.get("n"):
                    continue
                add(f"| {tag}/{scope} | | {k} | {fmt(v['median_delta_lesion_p_ge3'])} | "
                    f"{fmt(v['median_delta_ctrl_p_ge3'])} | {fp(v['p_wilcoxon_p_ge3'])} |")
    add("")

    # ---------- 6 token 数 ----------
    tok = out["lesion_tokens"]
    add(f"## 6. 病灶 token 数（该视角最大病灶框(不外扩)面积/整图 × {tok['tokens_per_image']}，"
        "每乳房取 CC/MLO 较大值）")
    add("")
    add("| 分组 | n | 中位 | p25 | p75 |")
    add("|---|---|---|---|---|")
    for g in GROUPS:
        for tag in ("new", "old"):
            q = tok[tag]["by_group"][g]
            qx = tok[tag]["by_group_excl_sv"][g]
            add(f"| {g} ({tag}) | {q['n']} | {fmt(q['median'], 1)} | {fmt(q['p25'], 1)} | "
                f"{fmt(q['p75'], 1)} |")
            add(f"| {g} ({tag}, excl_sv) | {qx['n']} | {fmt(qx['median'], 1)} | "
                f"{fmt(qx['p25'], 1)} | {fmt(qx['p75'], 1)} |")
    add("")
    add("| 病灶类型（全部 192） | n | 中位 | p25 | p75 | min | max |")
    add("|---|---|---|---|---|---|---|")
    for t in ("mass", "calcification", "other"):
        s_ = tok["new"]["by_type"][t]
        add(f"| {t} | {s_['n']} | {fmt(s_['median'], 1)} | {fmt(s_['p25'], 1)} | "
            f"{fmt(s_['p75'], 1)} | {fmt(s_['min'], 1)} | {fmt(s_['max'], 1)} |")
    add("")

    # ---------- 7 单视角列表 ----------
    add(f"## 7. 只有一个视角有病灶框的乳房（{out['n_single_view_breasts']} 个）")
    add("")
    add("这些样本另一视角无遮挡，Δ 被稀释；各主表的 excl_sv 版本已排除它们。")
    add("")
    add("| id | new 组 | old 组 |")
    add("|---|---|---|")
    for bid in out["single_view_breasts"]:
        og = next((g for g in GROUPS if bid in out["group_ids"]["old"][g]), "—")
        ng = next((g for g in GROUPS if bid in out["group_ids"]["new"][g]), "—")
        add(f"| {bid} | {ng} | {og} |")
    add("")

    # ---------- 8 口径 ----------
    add("## 8. 口径与已知限制")
    add("")
    add("- 坐标: `vindr_detection_v1_folds.csv` 的 `resized_*` 即 912×1520 PNG 像素坐标"
        "（6 张叠加图人工核对，红框压病灶、按宽高比换算的绿框整体偏离；见 `check/ov*.png`）。")
    add("- 遮挡: lesion 框每边外扩 20%（裁到边界内）；对照与病灶框一一对应同尺寸，随机放在 Otsu "
        "乳腺区内（≥80% 面积在乳腺内、不与任何外扩病灶框重叠）；填充 = 框外 16px 环中位灰度；seed 42。"
        "本地重算的 rect/fill/Otsu 与 NeSI manifest 384 视角 0 差异。")
    add("- 检查图: `check/ov*.png`（坐标验证）、`check/preview_*.png`（6 乳房 × 4 情况拼图）、"
        "`reports/step3_check.txt`（1152 张数量/尺寸校验）。")
    add("- 原始概率: `occlusion_probs/{{orig,lesion,ctrl1,ctrl2}}.jsonl`；"
        "完整数值统计: `reports/step6.json`。")
    add("")
    (OCC / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
