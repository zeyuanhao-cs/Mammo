#!/usr/bin/env python3
"""
第 2 步: 在剩余 train (train_sub_raw.json, 6399 条) 上重新生成 Repeat-OS 训练集。

规则与原 make_repeat_os.py 完全一致:
  - r_c = max(1, sqrt(0.1 / f_c)), f_c 在剩余 train 上重新统计
  - r_i = min(4, max(r_birads, r_positive_findings...))  # 多标签取 max, 不相乘, cap=4
  - stochastic rounding; seed=42; 最后 shuffle
  - density 不参与 repeat factor
  - val 样本绝不进入训练集 (输入即 train_sub_raw.json, 并额外校验与 val_raw.json 无交集)

输出 (2-view/data/):
  - train_sub_repeat_os.json        # 原始格式 OS 训练集
  - train_llamafactory_os_sub.json  # alpaca 格式 (LLaMA-Factory 训练用)
  - dataset_info.json               # 追加注册 vindr_2view_train_os_sub
"""

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

SEED = 42
CAP = 4
TARGET_FREQ = 0.1

THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR.parent / "data"
RAW_DIR = DATA_DIR / "raw"
LF_DIR = DATA_DIR / "llamafactory"
SRC = RAW_DIR / "train_sub_raw.json"
VAL_FILE = RAW_DIR / "val_raw.json"
DST_RAW = RAW_DIR / "train_sub_repeat_os.json"
DST_ALPACA = LF_DIR / "train_llamafactory_os_sub.json"
INFO_FILE = DATA_DIR / "dataset_info.json"
PROMPT_FILE = Path("/home/zhao/Mammography/prompt.txt")

FINDINGS = ["mass", "suspicious_calcification", "asymmetry", "architectural_distortion"]


def load_instruction() -> str:
    text = PROMPT_FILE.read_text(encoding="utf-8").strip()
    n = text.count("<image>")
    if n == 1:
        return text.replace("<image>", "<image>\n<image>", 1)
    assert n == 2, f"prompt 需要 1 或 2 个 <image> 占位符, 实际 {n}"
    return text


def make_output(gt: dict) -> str:
    """训练目标: 与 prompt 要求的 JSON 结构一致 (键序一致)."""
    obj = {
        "breast_density": gt["breast_density"],
        "findings": {k: int(gt["findings"][k]) for k in FINDINGS},
        "breast_birads": int(gt["breast_birads"]),
    }
    return json.dumps(obj, ensure_ascii=False)


def main():
    data = json.loads(SRC.read_text(encoding="utf-8"))
    n = len(data)

    # ---- val 泄漏校验 ----
    val_ids = {s["id"] for s in json.loads(VAL_FILE.read_text(encoding="utf-8"))}
    assert not ({s["id"] for s in data} & val_ids), "train_sub 与 val 存在交集!"

    # ---- f_c: 基于剩余 train 重新统计 ----
    cnt_birads = Counter(s["output"]["breast_birads"] for s in data)
    cnt_pos = Counter()
    for s in data:
        for k in FINDINGS:
            if s["output"]["findings"][k] == 1:
                cnt_pos[k] += 1

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
    rep_sum_birads = defaultdict(lambda: [0, 0])
    rep_sum_find = defaultdict(lambda: [0, 0])
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

    # ---- val 泄漏二次校验 (重复后) ----
    assert not any(s["id"] in val_ids for s in out), "OS 输出中混入 val 样本!"

    DST_RAW.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- alpaca 格式 + 注册 ----
    instruction = load_instruction()
    alpaca = [
        {
            "instruction": instruction,
            "input": "",
            "output": make_output(s["output"]),
            "images": list(s["images"]),
        }
        for s in out
    ]
    DST_ALPACA.write_text(json.dumps(alpaca, ensure_ascii=False), encoding="utf-8")

    info = json.loads(INFO_FILE.read_text(encoding="utf-8"))
    info["vindr_2view_train_os_sub"] = {
        "file_name": DST_ALPACA.name,
        "formatting": "alpaca",
        "columns": {"prompt": "instruction", "query": "input", "response": "output", "images": "images"},
    }
    INFO_FILE.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

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
    print("\nbreast_density (未参与 repeat factor, 仅随源样本放大):")
    for c in sorted(old_density):
        print(f"  {c}: {old_density[c]} -> {new_density.get(c, 0)}  ({new_density.get(c,0)/old_density[c]:.2f}x)")

    print("\nfindings 阳性数:")
    for k in FINDINGS:
        new_pos = sum(1 for s in out if s["output"]["findings"][k] == 1)
        srep, nsamp = rep_sum_find[k]
        print(f"  {k}: {cnt_pos[k]} -> {new_pos}  ({new_pos/cnt_pos[k]:.2f}x, avg_rep={srep/nsamp:.2f})")

    old_z = sum(1 for s in data if all(s["output"]["findings"][k] == 0 for k in FINDINGS))
    new_z = sum(1 for s in out if all(s["output"]["findings"][k] == 0 for k in FINDINGS))
    print(f"\nall-zero findings: {old_z} -> {new_z}  ({new_z/old_z:.2f}x, 占比 {old_z/n:.1%} -> {new_z/m:.1%})")

    print(f"\nOutput: {DST_RAW}\n        {DST_ALPACA}\n        dataset_info.json += vindr_2view_train_os_sub")


if __name__ == "__main__":
    main()
