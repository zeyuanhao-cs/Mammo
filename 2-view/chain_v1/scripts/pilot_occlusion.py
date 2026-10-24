#!/usr/bin/env python3
"""阶段二遮挡机制分析: B0(hard) / C1(soft) 在 occl_{cond}.jsonl 上的 Δ 与联动比.

三类按 B0 在原图(orig)上的预测划分; 每类报告:
  原图 P(>=2)/P(>=3)/max_sign 中位数; 遮病灶 Δ、遮对照 Δ(两对照平均);
  联动比 = ΔP(>=3) / Δmax_sign。
输出: chain_v1/outputs/pilot/occlusion_summary.json (+ md 片段 pilot_occlusion.md)
"""
import json
from pathlib import Path

import numpy as np

PROJ = Path("/home/z/zhao/Mammography/Mammo/2-view")
P = PROJ / "chain_v1" / "outputs" / "pilot"


def load(fn):
    f = P / fn
    if not f.exists():
        return {}
    return {json.loads(l)["id"]: json.loads(l) for l in open(f, encoding="utf-8")}


def med(v):
    v = [x for x in v if x is not None]
    return float(np.median(v)) if v else float("nan")


def main():
    out = {}
    for model in ("b0", "c1"):
        orig = load(f"occl_{model}_orig.jsonl")
        if not orig:
            raise SystemExit(f"缺 occl_{model}_orig.jsonl")
        lesion = load(f"occl_{model}_lesion.jsonl")
        c1 = load(f"occl_{model}_ctrl1.jsonl")
        c2 = load(f"occl_{model}_ctrl2.jsonl")

        # 三类按 B0 原图预测划分（两组同一批乳房）
        b0_orig = load("occl_b0_orig.jsonl")
        cats = {"detected": [], "missed_sign": [], "missed_nosign": []}
        for rid, r in b0_orig.items():
            if r["pred_birads"] >= 3:
                cats["detected"].append(rid)
            elif r["pred_birads"] == 2 or r["max_sign_p"] > 0.1:
                cats["missed_sign"].append(rid)
            else:
                cats["missed_nosign"].append(rid)

        def delta(rows_o, rows_l, key):
            return [rows_o[i][key] - rows_l[i][key]
                    for i in rows_o if i in rows_l and rows_o[i][key] is not None
                    and rows_l[i][key] is not None]

        mmodel = {}
        for g, ids in cats.items():
            ids = [i for i in ids if i in lesion and i in c1 and i in c2]
            entry = {"n": len(ids)}
            for key in ("p_birads_ge2", "p_birads_ge3", "max_sign_p"):
                d_les = delta({i: orig[i] for i in ids}, {i: lesion[i] for i in ids}, key)
                d_c = [(orig[i][key] - np.mean([c1[i][key], c2[i][key]]))
                       for i in ids if c1[i][key] is not None and c2[i][key] is not None]
                entry[key] = {
                    "median_orig": med([orig[i][key] for i in ids]),
                    "median_delta_lesion": med(d_les),
                    "median_delta_ctrl": med(d_c),
                }
            dl3 = entry["p_birads_ge3"]["median_delta_lesion"]
            dls = entry["max_sign_p"]["median_delta_lesion"]
            entry["linkage_ratio"] = (dl3 / dls) if dls and not np.isnan(dls) and dls > 1e-6 else None
            mmodel[g] = entry
        out[model] = {"categories_by_b0_orig": {g: len(v) for g, v in cats.items()},
                      "groups": mmodel}

    (P / "occlusion_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                              encoding="utf-8")

    L = ["## 遮挡机制分析（B0 hard / C1 soft，遮挡图与 occlusion_3ep 完全一致）", "",
         "| 模型 | 组 | n | 原图 P(≥3) | Δ病灶 P(≥3) | Δ对照 P(≥3) | 原图 max_sign | "
         "Δ病灶 max_sign | Δ对照 max_sign | 联动比 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for model in ("b0", "c1"):
        for g in ("detected", "missed_sign", "missed_nosign"):
            e = out[model]["groups"][g]
            f4 = lambda v: "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"
            L.append(f"| {model} | {g} | {e['n']} | {f4(e['p_birads_ge3']['median_orig'])} | "
                     f"{f4(e['p_birads_ge3']['median_delta_lesion'])} | "
                     f"{f4(e['p_birads_ge3']['median_delta_ctrl'])} | "
                     f"{f4(e['max_sign_p']['median_orig'])} | "
                     f"{f4(e['max_sign_p']['median_delta_lesion'])} | "
                     f"{f4(e['max_sign_p']['median_delta_ctrl'])} | "
                     f"{f4(e['linkage_ratio'])} |")
    L += ["", "联动比 = ΔP(≥3) / Δmax_sign（遮病灶），衡量征象概率下降传导到 BI-RADS 的比例。"
              "P(≥2) 的数值在 `occlusion_summary.json`。"]
    (P / "pilot_occlusion.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
