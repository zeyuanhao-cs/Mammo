#!/usr/bin/env python3
"""Repeat-OS（含 density）：在 train_sub_raw_local.json 上做 repeat-based oversampling。

与 make_repeat_os_sub.py 的唯一区别：**density 也参与 repeat factor**。
  - r_c   = max(1, sqrt(TARGET_FREQ / f_c))，f_c 在剩余 train 上统计
  - BI-RADS 用单标签 r_c；4 个 finding 各用自己阳性样本的 r_c；
    density 用 A/B/C/D 的 r_c
  - 每条样本 r_i = min(CAP, max(r_birads, 各阳性 finding 的 r, r_density))
    （多标签取 max，不相乘）
  - stochastic rounding; seed=42; 最后 shuffle
  - val 样本绝不进入训练集（输入即 train_sub_raw_local.json，并校验与 val_raw_local.json 无交集）

注意：repeat factor 的 rng 调用与旧脚本一样是"每条样本抽一次"，因此开 density 后
随机序列位置不变，差异只来自 r 的取值变化。

输出:
  - 2-view/data/splits/train_sub_repeat_os_density_local.json   # 原始格式
  - 2-view/data/llamafactory/train_llamafactory_os_sub_density_local.json  # alpaca
  - 2-view/data/dataset_info.json += vindr_2view_train_os_sub_density_local

用法:
  python scripts/data/make_repeat_os_sub_density.py
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
PROJ_DIR = THIS_DIR.parent.parent                    # 2-view
DATA_DIR = PROJ_DIR / "data"
RAW_DIR = DATA_DIR / "splits"
LF_DIR = DATA_DIR / "llamafactory"

SRC = RAW_DIR / "train_sub_raw_local.json"           # 6,399 条
VAL_FILE = RAW_DIR / "val_raw_local.json"
OLD = RAW_DIR / "train_sub_repeat_os_local.json"     # 旧 OS（不含 density），仅用于对比报告
DST_RAW = RAW_DIR / "train_sub_repeat_os_density_local.json"
DST_ALPACA = LF_DIR / "train_llamafactory_os_sub_density_local.json"
INFO_FILE = DATA_DIR / "dataset_info.json"
PROMPT_FILE = PROJ_DIR / "dataset_build" / "prompt.txt"
DATASET_NAME = "vindr_2view_train_os_sub_density_local"

FINDINGS = ["mass", "suspicious_calcification", "asymmetry", "architectural_distortion"]
DENSITY = ["A", "B", "C", "D"]
BIRADS = [1, 2, 3, 4, 5]


def load_instruction() -> str:
    text = PROMPT_FILE.read_text(encoding="utf-8").strip()
    n = text.count("<image>")
    if n == 1:
        return text.replace("<image>", "<image>\n<image>", 1)
    assert n == 2, f"prompt 需要 1 或 2 个 <image> 占位符, 实际 {n}"
    return text


def make_output(gt: dict) -> str:
    """训练目标: 与 prompt 要求的 JSON 结构一致 (键序: density -> findings -> birads)."""
    obj = {
        "breast_density": gt["breast_density"],
        "findings": {k: int(gt["findings"][k]) for k in FINDINGS},
        "breast_birads": int(gt["breast_birads"]),
    }
    return json.dumps(obj, ensure_ascii=False)


def dist(data):
    """返回 (n, density/birADS/pos/all-zero 计数)."""
    n = len(data)
    d = {
        "density": Counter(s["output"]["breast_density"] for s in data),
        "birads": Counter(int(s["output"]["breast_birads"]) for s in data),
        "pos": {k: sum(1 for s in data if int(s["output"]["findings"][k]) == 1) for k in FINDINGS},
        "zero": sum(1 for s in data if all(int(s["output"]["findings"][k]) == 0 for k in FINDINGS)),
    }
    return n, d


def main():
    data = json.loads(SRC.read_text(encoding="utf-8"))
    n = len(data)

    val_ids = {s["id"] for s in json.loads(VAL_FILE.read_text(encoding="utf-8"))}
    assert not ({s["id"] for s in data} & val_ids), "train_sub 与 val 存在交集!"

    # ---- f_c ----
    cnt_birads = Counter(int(s["output"]["breast_birads"]) for s in data)
    cnt_density = Counter(s["output"]["breast_density"] for s in data)
    cnt_pos = Counter()
    for s in data:
        for k in FINDINGS:
            if int(s["output"]["findings"][k]) == 1:
                cnt_pos[k] += 1

    r_birads = {c: max(1.0, math.sqrt(TARGET_FREQ / (c_ / n))) for c, c_ in cnt_birads.items()}
    r_density = {c: max(1.0, math.sqrt(TARGET_FREQ / (c_ / n))) for c, c_ in cnt_density.items()}
    r_find = {k: max(1.0, math.sqrt(TARGET_FREQ / (c_ / n))) for k, c_ in cnt_pos.items()}

    print("=== class repeat factors (before cap) ===")
    for c in sorted(cnt_birads):
        print(f"  BI-RADS {c}: f={cnt_birads[c]/n:.4f}  r={r_birads[c]:.4f}")
    for k in FINDINGS:
        print(f"  {k}: f={cnt_pos[k]/n:.4f}  r={r_find[k]:.4f}")
    print("  --- density (本次新增参与) ---")
    for c in DENSITY:
        if c in cnt_density:
            print(f"  density {c}: f={cnt_density[c]/n:.4f}  r={r_density[c]:.4f}")

    rng = random.Random(SEED)

    def stochastic_round(r: float) -> int:
        fl = math.floor(r)
        frac = r - fl
        return fl + (1 if rng.random() < frac else 0)

    out = []
    rep_by_density = defaultdict(list)
    for s in data:
        o = s["output"]
        r = max(r_birads[int(o["breast_birads"])], r_density[o["breast_density"]])
        for k in FINDINGS:
            if int(o["findings"][k]) == 1:
                r = max(r, r_find[k])
        r = min(float(CAP), r)
        rep = stochastic_round(r)
        rep_by_density[o["breast_density"]].append(rep)
        out.extend([s] * rep)

    rng.shuffle(out)

    assert not any(s["id"] in val_ids for s in out), "OS 输出中混入 val 样本!"

    DST_RAW.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

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
    info[DATASET_NAME] = {
        "file_name": f"llamafactory/{DST_ALPACA.name}",
        "formatting": "alpaca",
        "columns": {"prompt": "instruction", "query": "input", "response": "output", "images": "images"},
    }
    INFO_FILE.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------------- 对比报告 ----------------
    n_new, new = dist(out)
    n_old, old = dist(json.loads(OLD.read_text(encoding="utf-8"))) if OLD.exists() else (0, None)
    n_raw, raw = dist(data)

    L = []
    L.append("## 训练集对比 (source / 旧 OS / 新 OS+density)")
    L.append("")
    L.append(f"- 源数据: `{SRC.name}` = {n_raw} 条")
    L.append(f"- 旧: `{OLD.name}` = {n_old} 条 ({n_old / n_raw:.2f}x)")
    L.append(f"- 新: `{DST_RAW.name}` = {n_new} 条 ({n_new / n_raw:.2f}x)")
    L.append("")
    L.append("| 类别 | 源 count | 源 % | 旧 OS count | 旧 OS % | 新 OS count | 新 OS % |")
    L.append("|---|---|---|---|---|---|---|")

    def row(label, cv, ov, nv):
        f = lambda c: "-" if c is None else f"{c}"
        p = lambda c: "-" if c is None else f"{c / n_raw * 100:.1f}%"
        po = "-" if ov is None else f"{ov / n_old * 100:.1f}%" if n_old else "-"
        pn = f"{nv / n_new * 100:.1f}%"
        L.append(f"| {label} | {cv} | {p(cv)} | {f(ov)} | {po} | {nv} | {pn} |")

    for c in DENSITY:
        row(f"density {c}", raw["density"].get(c, 0),
            old["density"].get(c, 0) if old else None, new["density"].get(c, 0))
    for c in BIRADS:
        row(f"BI-RADS {c}", raw["birads"].get(c, 0),
            old["birads"].get(c, 0) if old else None, new["birads"].get(c, 0))
    for k in FINDINGS:
        row(f"{k}=1", raw["pos"][k], old["pos"][k] if old else None, new["pos"][k])
    row("findings 全 0", raw["zero"], old["zero"] if old else None, new["zero"])

    L.append("")
    L.append("density 的平均重复次数 (新):")
    L.append("")
    L.append("| density | 源条数 | 平均 rep | 新条数 | 放大倍数 |")
    L.append("|---|---|---|---|---|")
    for c in DENSITY:
        reps = rep_by_density.get(c, [])
        if not reps:
            L.append(f"| {c} | 0 | - | 0 | - |")
            continue
        L.append(f"| {c} | {len(reps)} | {sum(reps)/len(reps):.3f} | "
                 f"{new['density'].get(c, 0)} | {new['density'].get(c, 0)/len(reps):.2f}x |")

    md = "\n".join(L)
    (RAW_DIR / "repeat_os_density_report.md").write_text(md + "\n", encoding="utf-8")
    print()
    print(md)
    print(f"\nOutput: {DST_RAW}\n        {DST_ALPACA}\n"
          f"        dataset_info.json += {DATASET_NAME}")


if __name__ == "__main__":
    main()
