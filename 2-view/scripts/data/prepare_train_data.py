#!/usr/bin/env python3
"""
把 2-view train_raw.json 全量转成 LLaMA-Factory alpaca 训练格式 (不做任何采样):
  7999 条全部保留, 阳性:阴性 ≈ 1:9.4, 反映真实筛查分布。

输出 (2-view/):
  - train_llamafactory_full.json     # alpaca 格式, 含 images 列表 (绝对路径)
  - dataset_info.json                # LLaMA-Factory 数据集注册
"""

import json
from collections import Counter
from pathlib import Path

TRAIN_FILE = Path("/nesi/project/massey04900/Mammography/train_raw.json")
PROMPT_FILE = Path("/home/zhao/Mammography/prompt.txt")
THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR.parent.parent / "data"
OUT_FILE = DATA_DIR / "llamafactory" / "train_llamafactory_full.json"
INFO_FILE = DATA_DIR / "dataset_info.json"

FINDINGS = ["mass", "suspicious_calcification", "asymmetry", "architectural_distortion"]


def load_instruction() -> str:
    text = PROMPT_FILE.read_text(encoding="utf-8").strip()
    n = text.count("<image>")
    if n == 1:
        return text.replace("<image>", "<image>\n<image>", 1)
    assert n == 2, f"prompt 需要 1 或 2 个 <image> 占位符, 实际 {n}"
    return text


def make_output(gt: dict) -> str:
    """训练目标: 与 prompt v2 要求的 JSON 结构完全一致 (键序一致)."""
    obj = {
        "breast_density": gt["breast_density"],
        "findings": {k: int(gt["findings"][k]) for k in FINDINGS},
        "breast_birads": int(gt["breast_birads"]),
    }
    return json.dumps(obj, ensure_ascii=False)


def main():
    instruction = load_instruction()

    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)
    print(f"raw train samples: {len(raw)}")

    data = []
    for s in raw:
        data.append(
            {
                "instruction": instruction,
                "input": "",
                "output": make_output(s["output"]),
                "images": list(s["images"]),
            }
        )

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"saved {len(data)} samples -> {OUT_FILE}")

    info = {
        "vindr_2view_train": {
            "file_name": OUT_FILE.name,
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "images": "images",
            },
        }
    }
    with open(INFO_FILE, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"saved dataset_info -> {INFO_FILE}")

    # 统计保存后的分布
    birads = Counter(json.loads(d["output"])["breast_birads"] for d in data)
    dens = Counter(json.loads(d["output"])["breast_density"] for d in data)
    find_pos = {k: sum(1 for d in data if json.loads(d["output"])["findings"][k] == 1) for k in FINDINGS}
    print("BI-RADS:", dict(sorted(birads.items())))
    print("density:", dict(sorted(dens.items())))
    print("finding positives:", find_pos)


if __name__ == "__main__":
    main()
