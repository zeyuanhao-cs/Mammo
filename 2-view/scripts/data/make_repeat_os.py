#!/usr/bin/env python3
"""
Repeat-based oversampling: 基于 breast_birads + positive findings 的重复采样。

规则:
  - r_c = max(1, sqrt(0.1 / f_c)), f_c 基于原始 train_raw.json 统计
  - r_i = min(4, max(r_birads, r_positive_findings...))  # 多标签取 max, 不相乘, cap=4
  - 小数 repeat factor 用 stochastic rounding; seed=42; 最后 shuffle
"""
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

SEED = 42
CAP = 4
TARGET_FREQ = 0.1

SRC = Path("/nesi/project/massey04900/Mammography/train_raw.json")
DST = Path("/nesi/project/massey04900/Mammography/train_repeat_os_birads_findings_cap4.json")
FINDINGS = ["mass", "suspicious_calcification", "asymmetry", "architectural_distortion"]


def main():
    data = json.loads(SRC.read_text(encoding="utf-8"))
    n = len(data)

    # ---- f_c: 基于 train_raw.json 的原始统计 ----
    cnt_birads = Counter(s["output"]["breast_birads"] for s in data)
    cnt_pos = Counter()
    for s in data:
        for k in FINDINGS:
            if s["output"]["findings"][k] == 1:
                cnt_pos[k] += 1

    # ---- 每类 repeat factor (cap 前) ----
    r_birads = {c: max(1.0, math.sqrt(TARGET_FREQ / (c_ / n))) for c, c_ in cnt_birads.items()}
    r_find = {k: max(1.0, math.sqrt(TARGET_FREQ / (c_ / n))) for k, c_ in cnt_pos.items()}

    print("=== class repeat factors (before cap) ===")
    for c in sorted(r_birads):
        print(f"  BI-RADS {c}: f={cnt_birads[c]/n:.4f}  r={r_birads[c]:.4f}")
    for k in FINDINGS:
        print(f"  {k}: f={cnt_pos[k]/n:.4f}  r={r_find[k]:.4f}")

    rng = random.Random(SEED)

    def stochastic_round(r: float) -> int:
        fl = math.floor(r)
        frac = r - fl
        return fl + (1 if rng.random() < frac else 0)

    # ---- 逐样本重复 ----
    out = []
    rep_sum_birads = defaultdict(lambda: [0, 0])  # birads -> [sum_rep, n_samples]
    rep_sum_find = defaultdict(lambda: [0, 0])    # finding -> [sum_rep, n_pos_samples]
    for s in data:
        o = s["output"]
        r = r_birads[o["breast_birads"]]
        for k in FINDINGS:
            if o["findings"][k] == 1:
                r = max(r, r_find[k])
        r = min(float(CAP), r)
        rep = stochastic_round(r)
        rep_sum_birads[o["breast_birads"]][0] += rep
        rep_sum_birads[o["breast_birads"]][1] += 1
        for k in FINDINGS:
            if o["findings"][k] == 1:
                rep_sum_find[k][0] += rep
                rep_sum_find[k][1] += 1
        out.extend([s] * rep)

    rng.shuffle(out)
    DST.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- 报告 ----
    m = len(out)
    print(f"\n=== result ===")
    print(f"N_original = {n}, N_new = {m}  ({m/n:.2f}x)")

    new_birads = Counter(s["output"]["breast_birads"] for s in out)
    print("\nBI-RADS:")
    for c in sorted(new_birads):
        srep, nsamp = rep_sum_birads[c]
        print(f"  {c}: {cnt_birads[c]} -> {new_birads[c]}  ({new_birads[c]/cnt_birads[c]:.2f}x, avg_rep={srep/nsamp:.2f})")

    new_density = Counter(s["output"]["breast_density"] for s in out)
    old_density = Counter(s["output"]["breast_density"] for s in data)
    print("\nbreast_density (未参与 repeat factor 计算, 仅随源样本放大):")
    for c in sorted(old_density):
        print(f"  {c}: {old_density[c]} -> {new_density.get(c, 0)}  ({new_density.get(c,0)/old_density[c]:.2f}x)")

    print("\nfindings 阳性数:")
    for k in FINDINGS:
        new_pos = sum(1 for s in out if s["output"]["findings"][k] == 1)
        srep, nsamp = rep_sum_find[k]
        print(f"  {k}: {cnt_pos[k]} -> {new_pos}  ({new_pos/cnt_pos[k]:.2f}x, avg_rep={srep/nsamp:.2f})")

    def all_zero(s):
        return all(s["output"]["findings"][k] == 0 for k in FINDINGS)

    old_z = sum(1 for s in data if all_zero(s))
    new_z = sum(1 for s in out if all_zero(s))
    print(f"\nall-zero findings: {old_z} -> {new_z}  ({new_z/old_z:.2f}x)")

    print(f"\nOutput: {DST}")


if __name__ == "__main__":
    main()
