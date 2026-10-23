#!/usr/bin/env python3
"""patch-zoom 测试集评测.

输入: 三个尺度 infer_patch.py 输出的 probs jsonl (同一 test patch 集合)
输出: json 结果 + markdown 片段 (所有数字 6 位小数)

指标:
  - 4 个 findings 的 AUC (全 test patch: 阳性块 + 2x 阴性块);
  - 0.5 阈值下的 recall / precision;
  - recall 再按病灶面积三分位切点 (train 阳性块算出, 存在 stats.json) 分三档;
  - 1x vs 0.74x, 2x vs 1x 的 AUC 差值: 按乳房重采样配对 bootstrap (1000 次, seed 42).
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

CLASSES = ["mass", "suspicious_calcification", "asymmetry", "architectural_distortion"]
PATCH_DIR = Path("/nesi/project/massey04900/Mammography/patch_zoom")


def load(path):
    recs = [json.loads(l) for l in open(path, encoding="utf-8")]
    seen = set()
    for r in recs:
        assert r["id"] not in seen, f"duplicate id {r['id']}"
        seen.add(r["id"])
    return recs


def auc_score(y, s):
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    npos, nneg = y.sum(), (1 - y).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ss = s[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        avg = (i + j + 2) / 2.0  # 1-based 平均秩
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def build_table(recs, meta):
    """id -> (labels, p, breast, area_ratio); 概率缺失的记录计入 skipped."""
    out, skipped = {}, []
    for r in recs:
        m = meta.get(r["id"])
        pn = r.get("probs_norm")
        if m is None or not isinstance(pn, dict) or any(pn.get(k) is None for k in CLASSES):
            skipped.append(r["id"])
            continue
        p = {}
        for k in CLASSES:
            v = pn[k]
            p[k] = float(v["1"]) / (float(v["0"]) + float(v["1"])) if "0" in v else float(v["1"])
        out[r["id"]] = {
            "labels": m["labels"], "p": p, "breast": m["breast"],
            "area": m.get("area_ratio"), "kind": m["kind"],
        }
    return out, skipped


def summarize(tab, ids, q1, q2, tag):
    y = {k: [tab[i]["labels"][k] for i in ids] for k in CLASSES}
    res = {"tag": tag, "n_patches": len(ids),
           "n_pos_per_class": {k: int(sum(y[k])) for k in CLASSES}}
    per = {}
    for k in CLASSES:
        s = [tab[i]["p"][k] for i in ids]
        auc = auc_score(y[k], s)
        pred = [1 if v >= 0.5 else 0 for v in s]
        tp = sum(1 for a, b in zip(y[k], pred) if a == 1 and b == 1)
        fp = sum(1 for a, b in zip(y[k], pred) if a == 0 and b == 1)
        fn = sum(1 for a, b in zip(y[k], pred) if a == 1 and b == 0)
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        # 面积三档 recall (只在阳性块里分档)
        bin_vals = []
        for lo, hi in ((0.0, q1), (q1, q2), (q2, float("inf"))):
            sel = [i for i in ids if tab[i]["kind"] == "pos"
                   and tab[i]["area"] is not None and lo < tab[i]["area"] <= hi]
            n_pos = len(sel)
            if n_pos:
                hit = sum(1 for i in sel if tab[i]["p"][k] >= 0.5)
                bin_vals.append({"n": n_pos, "recall": hit / n_pos})
            else:
                bin_vals.append({"n": 0, "recall": float("nan")})
        per[k] = {"auc": auc, "recall@0.5": rec, "precision@0.5": prec,
                  "tp": tp, "fp": fp, "fn": fn, "recall_by_area": bin_vals}
    res["per_class"] = per
    return res


def f6(x):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.6f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", nargs="+", required=True,
                    help="x074/x1/x2 的 probs jsonl, 按顺序")
    ap.add_argument("--scales", nargs="+", required=True)
    ap.add_argument("--meta", default=str(PATCH_DIR / "meta_test.jsonl"))
    ap.add_argument("--stats", default=str(PATCH_DIR / "stats.json"))
    ap.add_argument("--out", required=True, help="输出 json")
    ap.add_argument("--md", default=None, help="输出 markdown 片段")
    ap.add_argument("--report", default=None, help="输出完整报告 markdown")
    ap.add_argument("--smoke", default=None, help="run_all.py 产出的 smoke_metrics.json")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    meta = {json.loads(l)["id"]: json.loads(l) for l in open(args.meta, encoding="utf-8")}
    stats = json.load(open(args.stats, encoding="utf-8"))
    q1 = stats["area_tertiles"]["q1"]
    q2 = stats["area_tertiles"]["q2"]

    tabs, common, skipped_all = {}, None, {}
    for scale, path in zip(args.scales, args.probs):
        recs = load(path)
        tab, skipped = build_table(recs, meta)
        if skipped:
            print(f"[warn] {scale}: {len(skipped)} 条概率缺失, 已跳过 (e.g. {skipped[:3]})")
        skipped_all[scale] = len(skipped)
        tabs[scale] = tab
        ks = set(tab)
        common = ks if common is None else (common & ks)
    ids = sorted(common)
    print(f"[data] common patches = {len(ids)} / {len(meta)}")

    results = {"q1": q1, "q2": q2, "n_common": len(ids),
               "n_skipped": skipped_all, "scales": {}}
    for scale in args.scales:
        results["scales"][scale] = summarize(tabs[scale], ids, q1, q2, scale)

    results["bootstrap"] = {}
    pairs = [("x1", "x074"), ("x2", "x1")]
    for ka, kb in pairs:
        if ka in tabs and kb in tabs:
            for cls in CLASSES:
                key = f"{ka}_minus_{kb}:{cls}"
                results["bootstrap"][key] = paired_bootstrap_auc_same(
                    tabs[ka], tabs[kb], ids, cls, args.n_boot, args.seed)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("[out]", args.out)

    if args.md:
        write_md(results, args.scales, args.md)
        print("[md]", args.md)
    if args.report:
        write_report(results, args.scales, args.report, stats,
                     json.load(open(args.smoke, encoding="utf-8")) if args.smoke else None)
        print("[report]", args.report)
    return results


def paired_bootstrap_auc_same(tab_a, tab_b, ids, cls, n_boot, seed):
    """AUC(a) - AUC(b), 同一批 id, 按乳房重采样."""
    breast2ids = defaultdict(list)
    for i in ids:
        breast2ids[tab_a[i]["breast"]].append(i)
    breasts = sorted(breast2ids)
    y = {i: tab_a[i]["labels"][cls] for i in ids}
    sa = {i: tab_a[i]["p"][cls] for i in ids}
    sb = {i: tab_b[i]["p"][cls] for i in ids}
    full_a = auc_score([y[i] for i in ids], [sa[i] for i in ids])
    full_b = auc_score([y[i] for i in ids], [sb[i] for i in ids])
    rng = random.Random(seed)
    deltas = []
    for _ in range(n_boot):
        samp = []
        for _ in breasts:
            samp.extend(breast2ids[breasts[rng.randrange(len(breasts))]])
        yy = [y[i] for i in samp]
        if sum(yy) == 0 or sum(yy) == len(yy):
            continue
        d = auc_score(yy, [sa[i] for i in samp]) - auc_score(yy, [sb[i] for i in samp])
        if not np.isnan(d):
            deltas.append(d)
    deltas = np.asarray(deltas)
    return {"auc_a": float(full_a), "auc_b": float(full_b),
            "delta": float(full_a - full_b), "n_boot_valid": int(len(deltas)),
            "ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))]
            if len(deltas) else [float("nan"), float("nan")]}


def write_md(res, scales, path):
    Path(path).write_text("\n".join(md_lines(res, scales)), encoding="utf-8")


def md_lines(res, scales):
    C = CLASSES
    L = ["## 测试集评测（224×224 局部块，正:负 = 1:2）", ""]
    L.append(f"面积三分位切点（train 阳性块）：q1={f6(res['q1'])}，q2={f6(res['q2'])}；"
             f"三档 = [0,q1] / (q1,q2] / (q2,1]。")
    L.append(f"三组共用同一批 test patch：{res['n_common']} 块"
             f"（概率缺失被跳过：{res.get('n_skipped', {})}）。")
    L.append("")
    L.append("### 4 类 AUC（全 test patch）")
    L.append("")
    L.append("| scale | " + " | ".join(C) + " |")
    L.append("|---" * (len(C) + 1) + "|")
    for s in scales:
        per = res["scales"][s]["per_class"]
        L.append(f"| {s} | " + " | ".join(f6(per[k]["auc"]) for k in C) + " |")
    L.append("")
    L.append("### 0.5 阈值 recall / precision")
    L.append("")
    L.append("| scale | class | n_pos | recall | precision |")
    L.append("|---|---|---|---|---|")
    for s in scales:
        per = res["scales"][s]["per_class"]
        for k in C:
            L.append(f"| {s} | {k} | {res['scales'][s]['n_pos_per_class'][k]} | "
                     f"{f6(per[k]['recall@0.5'])} | {f6(per[k]['precision@0.5'])} |")
    L.append("")
    L.append("### recall 按病灶面积三档")
    L.append("")
    L.append("| scale | class | 小(n) | 小 recall | 中(n) | 中 recall | 大(n) | 大 recall |")
    L.append("|---|---|---|---|---|---|---|---|")
    for s in scales:
        per = res["scales"][s]["per_class"]
        for k in C:
            b = per[k]["recall_by_area"]
            L.append(f"| {s} | {k} | " + " | ".join(
                f"{x['n']} | {f6(x['recall'])}" for x in b) + " |")
    L.append("")
    L.append("### AUC 差值配对 bootstrap（按乳房重采样，1000 次，seed 42）")
    L.append("")
    L.append("| 比较 | class | AUC(A) | AUC(B) | Δ(A-B) | 95% CI |")
    L.append("|---|---|---|---|---|---|")
    for key, v in res["bootstrap"].items():
        pair, cls = key.split(":")
        L.append(f"| {pair} | {cls} | {f6(v['auc_a'])} | {f6(v['auc_b'])} | "
                 f"{f6(v['delta'])} | [{f6(v['ci95'][0])}, {f6(v['ci95'][1])}] |")
    return L


def write_report(res, scales, path, stats, smoke):
    """完整报告: 数据 / 训练设置 / 冒烟实测 / 结果 / 说明."""
    C = CLASSES
    L = ["# 局部放大（patch-zoom）能否让小病灶被认出来", "",
         "诊断实验：用 Kaggle VinDr 的 912×1520 原图裁 224×224 局部块，分别以 "
         "0.74×（160×160）、1×（224×224）、2×（448×448）三种输入分辨率训练 "
         "Qwen3.5-4B LoRA（解冻视觉塔，2 epoch），比较 4 个 findings 的判别能力。", ""]

    tr, te = stats["train"], stats["test"]
    L += ["## 1. 数据", "",
          "| 集合 | 阳性块 | 阴性块 | 阳性乳房 | 阴性乳房 | 框>224 被排除 |",
          "|---|---|---|---|---|---|",
          f"| train | {tr['n_pos_kept']} | {tr['n_neg_kept']} | {tr['n_breasts_pos']} | "
          f"{tr['n_breasts_neg']} | {tr['excluded_big']} |",
          f"| test | {te['n_pos_kept']} | {te['n_neg_kept']} | {te['n_breasts_pos']} | "
          f"{te['n_breasts_neg']} | {te['excluded_big']} |", ""]
    L += ["每类阳性块（multi-hot）：", "",
          "| 集合 | " + " | ".join(C) + " |", "|---" * (len(C) + 1) + "|"]
    for name, d in (("train", tr), ("test", te)):
        L.append(f"| {name} | " + " | ".join(str(d["pos_by_class"].get(k, 0)) for k in C) + " |")
    at = stats["area_tertiles"]
    L += ["", f"面积三分位切点（train 阳性块）：q1={f6(at['q1'])}，q2={f6(at['q2'])}；"
              f"test 三档计数 {at['test_counts']}（小/中/大）。",
          f"train/test 乳房交集 {stats['breast_overlap_train_test']}（应为 0）。", ""]

    L += ["## 2. 训练设置", "",
          "- Qwen3.5-4B + LoRA（r=8, alpha=16, dropout=0.05, target=all），"
          "`freeze_vision_tower: false`（视觉塔纳入 LoRA），bf16，cosine + 5% warmup，lr 5e-5。",
          "- per_device_batch=2，grad_accum=8（等效 batch 16），2 epoch（1995 块 → 250 步），"
          "`save_steps=50`。",
          "- 三个尺度共用同一批 224×224 块，仅用 `image_min_pixels=image_max_pixels` 强制输入像素预算："
          "x074=25600（160×160，25 视觉 token）、x1=50176（224×224，49 token）、"
          "x2=200704（448×448，196 token）。",
          "- 输出只含 4 个 findings 的 0/1；概率用 b8 贪心生成的逐步 logits 在字段取值 token 处取 softmax。", ""]

    if smoke:
        L += ["## 3. 冒烟实测（每尺度 20 步训练 + 16 块推理）", "",
              "| scale | 每步耗时(s) | 峰值显存(MiB) | 20 步墙钟(s) | 推理 16 块(s) | 模型加载(s) |",
              "|---|---|---|---|---|---|"]
        for s in scales:
            m = smoke.get(s)
            if not m or not m.get("ok"):
                L.append(f"| {s} | 失败 | - | - | - | - |")
                continue
            t, i = m["train"], m["infer"]
            sp = f"{t['s_per_step']:.3f}" if t.get("s_per_step") else "n/a"
            L.append(f"| {s} | {sp} | {t.get('peak_mem_mib')} | {t['wall_s']:.0f} | "
                     f"{i['wall_s']:.0f} | {i['load_s']:.1f} |")
        L += ["", "训练在 genoa 分区单张 H100 上运行；20 步墙钟含预处理与模型加载。", ""]

    L += ["## 4. 结果", ""]
    L += md_lines(res, scales)
    L += ["", "## 5. 说明", "",
          "- 阴性块为阳性 2 倍，其中一半来自 BI-RADS 1 正常乳房的乳腺区域（224×224 均值 > 80），"
          "一半来自有病灶乳房中与所有病灶框都不重叠的区域。",
          "- 训练/测试按 study 划分，同一乳房不跨集合；bootstrap 按乳房重采样。",
          "- 部分类别的 test 阳性块数量很少（见上表 n_pos），其 AUC/召回区间会较宽，"
          "`nan` 表示该类别在重采样子集中无正/负样本。",
          "- 所有数字保留 6 位小数。", ""]
    Path(path).write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
