#!/usr/bin/env python3
"""
VinDr-Mammo 推理数据构建器 (参数化, 覆盖式)

用法:
    python3 vindr_infer/build_mllm.py --prompt direct
    python3 vindr_infer/build_mllm.py --prompt icl
    python3 vindr_infer/build_mllm.py --prompt cot

输入:
    <DATA_ROOT>/vlm_dataset_labels_only/test.json  实验数据集 (4000 条, GT 标签)
    <DATA_ROOT>/vlm_dataset/test.json              按索引取 image_path (顺序与 labels_only 一致)
    vindr_infer/prompts/<name>.yaml                prompt 配置 (direct/icl/cot)

输出 (覆盖式):
    vindr_infer/mllm_infer.json        LLaMA-Factory mllm(sharegpt) 格式, 每次覆盖

设计:
    - 实验数据集 = vlm_dataset_labels_only (breast_birads + breast_density + findings)
    - image_path 从完整版 vlm_dataset/test.json 按索引取 (两者同一生成逻辑, 顺序一致)
    - 每次推理前跑此脚本注入 prompt, 输出固定文件 (覆盖)
    - ICL 示例图像路径 + GT 内联在本文件 (ICL_EXAMPLES 常量)
    - dataset_info.json 注册名 vindr_infer, file_name 用绝对路径, 不污染 data/
"""
import argparse
import json
import os
from pathlib import Path

import yaml

# 路径
DATA_ROOT = Path("/Users/haozeyuan/Desktop/phd/vlm/9_9/vindr-mammo")  # 图像根
THIS_DIR = Path(__file__).resolve().parent                           # vindr_infer/
# 实验数据集 = labels_only (GT 标签); image_path 从完整版按索引取 (顺序一致)
GT_DATA = DATA_ROOT / "vlm_dataset_labels_only" / "test.json"
IMG_DATA = DATA_ROOT / "vlm_dataset" / "test.json"
PROMPTS_DIR = THIS_DIR / "prompts"
OUT_MLLM = THIS_DIR / "mllm_infer.json"

# ICL few-shot 示例 (内联, 来自 train 集, 1 正常 + 1 Mass, 已验证图像存在)
ICL_EXAMPLES = [
    {
        # 示例1: 正常病例 (No Finding)
        "image_path": "images_png/b8d273e8601f348d3664778dae0e7e0b/d8125545210c08e1b1793a5af6458ee2.png",
        "ground_truth": {
            "breast_birads": 2,
            "breast_density": "C",
            "findings": [
                {"finding_category": "No Finding", "finding_birads": None, "bbox": None}
            ],
        },
    },
    {
        # 示例2: 含明确 Mass 病灶
        "image_path": "images_png/89524e5f372d9aff8ed43b4ef29c1435/5a94dd668eaa9865b907450c37db6ecc.png",
        "ground_truth": {
            "breast_birads": 3,
            "breast_density": "C",
            "findings": [
                {"finding_category": "Mass", "finding_birads": 3,
                 "bbox": [0.1533, 0.5741, 0.3126, 0.6601]}
            ],
        },
    },
]


def to_abs(rel_or_abs):
    """相对路径转绝对 (基于 DATA_ROOT); 绝对路径原样返回."""
    if os.path.isabs(rel_or_abs):
        return rel_or_abs
    return str(DATA_ROOT / rel_or_abs)


def load_prompt(name):
    with open(PROMPTS_DIR / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def build_user_content(cfg, target_img_abs):
    """组装 (content_text, images_list). <image> 数必须 == images 数."""
    prompt = cfg["prompt"]
    mode = cfg.get("mode", "direct")

    if mode == "icl":
        ex1, ex2 = ICL_EXAMPLES[0], ICL_EXAMPLES[1]
        ex1_img = to_abs(ex1["image_path"])
        ex2_img = to_abs(ex2["image_path"])
        ex1_json = json.dumps(ex1["ground_truth"], ensure_ascii=False, indent=2)
        ex2_json = json.dumps(ex2["ground_truth"], ensure_ascii=False, indent=2)
        text = (prompt
                .replace("[Image 1]", "<image>").replace("[JSON 1]", ex1_json)
                .replace("[Image 2]", "<image>").replace("[JSON 2]", ex2_json)
                .replace("[Image]", "<image>"))
        images = [ex1_img, ex2_img, target_img_abs]
    else:
        # direct / cot: prompt 已内置 <image> 占位符
        text = prompt
        images = [target_img_abs]

    n = text.count("<image>")
    assert n == len(images), f"[{mode}] <image> 占位符数({n}) != images 数({len(images)})"
    return text, images


def main():
    parser = argparse.ArgumentParser(description="构建 VinDr 推理 mllm 数据 (覆盖式)")
    parser.add_argument("--prompt", required=True, choices=["direct", "icl", "cot"],
                        help="prompt 策略: direct / icl / cot")
    args = parser.parse_args()

    # 实验数据集 = labels_only (GT); image_path 从完整版按索引取
    gt_data = json.load(open(GT_DATA))
    img_data = json.load(open(IMG_DATA))
    assert len(gt_data) == len(img_data), \
        f"长度不一致: labels_only={len(gt_data)} vs 完整版={len(img_data)}"
    print(f"实验数据: {len(gt_data)} 条 <- {GT_DATA.name} (labels_only)")
    print(f"  image_path 从 {IMG_DATA.name} 按索引取 (顺序一致)")

    cfg = load_prompt(args.prompt)

    records = []
    for gt, img in zip(gt_data, img_data):
        img_abs = to_abs(img["image_path"])
        assert os.path.exists(img_abs), f"图像不存在: {img_abs}"
        content, images = build_user_content(cfg, img_abs)
        records.append({
            "messages": [{"role": "user", "content": content}],
            "images": images,
        })

    # 覆盖式写入
    with open(OUT_MLLM, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    sample = records[0]
    print(f"[{args.prompt}] {len(records)} 条 -> {OUT_MLLM} (覆盖)")
    print(f"  每条: <image>={sample['messages'][0]['content'].count('<image>')}, "
          f"images={len(sample['images'])}")
    print(f"  -> 下一步: python vindr_infer/run_{args.prompt}.py")


if __name__ == "__main__":
    main()
