#!/usr/bin/env python3
"""
VinDr-Mammo 推理数据构造 (单文件)

职责: prompts (direct/icl/cot) + ICL 示例 + labels_only 与 vlm_dataset 按索引对齐
      -> 生成 infer_data.<prompt>.<split>.json

说明:
  - 实验数据集 vlm_dataset_labels_only 无 image_id, 按索引从完整版 vlm_dataset 取
    image_path (两数据集同一生成逻辑, 顺序已验证 100% 一致)
  - GT 与 ICL 示例均做类别归一化: 原始 10 类 -> 4 主类 + Associated Feature,
    No Finding 跳过 (findings 为空数组)
  - 输出为 LLaMA-Factory ChatModel.chat 兼容格式: messages (纯文本含 <image> 占位符)
    + images (绝对路径列表); ground_truth 一并存入便于对比

用法:
    python3 build_data.py --prompt direct --limit 10
    python3 build_data.py --prompt icl    --limit 0   # 0 = 全量
"""
import argparse
import json
from pathlib import Path

# ============================== CONFIG ==============================
DATA_ROOT = Path("/hy-tmp/9_9/vindr-mammo")
THIS_DIR = Path(__file__).resolve().parent
SPLIT = "test"
# ====================================================================

# ------------------------------ Prompts ------------------------------
# direct: 直接预测; <image> 为图像占位符
PROMPT_DIRECT = """You are an expert mammography assistant.

<image>

Analyze the provided mammography image and predict:

1. Breast BI-RADS:
Choose one of [1, 2, 3, 4, 5].

2. Breast density:
Choose one of ["A", "B", "C", "D"].

3. All visible findings.
For each finding, predict:
- finding_category: one of
  ["Mass", "Calcification", "Asymmetry", "Architectural Distortion", "Associated Feature"]
- finding_birads: one of [3, 4, 5, null]
- bbox: [x_min, y_min, x_max, y_max]

Use "Associated Feature" for non-primary mammographic signs.
Use null for finding_birads only when the category is "Associated Feature".

Bounding-box coordinates must be normalized to [0, 1], with (0, 0) at the top-left and (1, 1) at the bottom-right.
Use the tightest possible bounding box around the visible finding itself.
Do not include surrounding normal breast tissue or the broader dense fibroglandular region.

If no abnormal finding is present, return an empty "findings" array.
Return ONLY valid JSON with exactly these fields:
"breast_birads", "breast_density", "findings".

Each item in "findings" must contain exactly:
"finding_category", "finding_birads", "bbox".

Do not output explanations or additional text."""

# icl: 2 few-shot 示例 (1 正常 + 1 Mass), 任务指令与 direct 完全一致
# 占位符: [Image 1]/[Image 2] 示例图, [JSON 1]/[JSON 2] 示例 GT, [Image] 目标图
PROMPT_ICL = """You are an expert mammography assistant.

I will first show you two annotated examples, then ask you to annotate a new image.

Example 1:
[Image 1]
Annotation:
[JSON 1]

Example 2:
[Image 2]
Annotation:
[JSON 2]

Now analyze the provided mammography image and predict:

1. Breast BI-RADS:
Choose one of [1, 2, 3, 4, 5].

2. Breast density:
Choose one of ["A", "B", "C", "D"].

3. All visible findings.
For each finding, predict:
- finding_category: one of
  ["Mass", "Calcification", "Asymmetry", "Architectural Distortion", "Associated Feature"]
- finding_birads: one of [3, 4, 5, null]
- bbox: [x_min, y_min, x_max, y_max]

Use "Associated Feature" for non-primary mammographic signs.
Use null for finding_birads only when the category is "Associated Feature".

Bounding-box coordinates must be normalized to [0, 1], with (0, 0) at the top-left and (1, 1) at the bottom-right.
Use the tightest possible bounding box around the visible finding itself.
Do not include surrounding normal breast tissue or the broader dense fibroglandular region.

If no abnormal finding is present, return an empty "findings" array.
Return ONLY valid JSON with exactly these fields:
"breast_birads", "breast_density", "findings".

Each item in "findings" must contain exactly:
"finding_category", "finding_birads", "bbox".

Do not output explanations or additional text.

The image to annotate:
[Image]"""

# cot: 分 5 步推理后输出 JSON
PROMPT_COT = """<image>You are an expert mammography assistant.
Analyze the mammography image step by step:
1. Assess breast BI-RADS and breast density.
2. Identify all visible findings.
3. Classify each finding.
4. Assign a BI-RADS category to each finding.
5. Localize each finding with a bounding box.

Bounding box:
[x_min, y_min, x_max, y_max]
Coordinates are normalized to [0, 1], with (0, 0) at the top-left.

Finally, return ONLY valid JSON:
{
  "breast_birads": <integer>,
  "breast_density": "<A/B/C/D>",
  "findings": [
    {
      "finding_category": "<category>",
      "finding_birads": <integer>,
      "bbox": [<x_min>, <y_min>, <x_max>, <y_max>]
    }
  ]
}"""

PROMPTS = {"direct": PROMPT_DIRECT, "icl": PROMPT_ICL, "cot": PROMPT_COT}

# --------------------------- ICL few-shot 示例 ---------------------------
ICL_EXAMPLES = [
    {   # 示例1: 正常病例 (No Finding)
        "image_path": "images_png/b8d273e8601f348d3664778dae0e7e0b/d8125545210c08e1b1793a5af6458ee2.png",
        "ground_truth": {
            "breast_birads": 2,
            "breast_density": "C",
            "findings": [],
        },
    },
    {   # 示例2: 含明确 Mass 病灶
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


# --------------------------- 类别归一化 ---------------------------
# 原始 10 类 -> 4 主类 + Associated Feature; No Finding 不作为 finding (跳过)
CATEGORY_MAP = {
    "Mass": "Mass",
    "Suspicious Calcification": "Calcification",

    "Asymmetry": "Asymmetry",
    "Focal Asymmetry": "Asymmetry",
    "Global Asymmetry": "Asymmetry",

    "Architectural Distortion": "Architectural Distortion",

    "Skin Thickening": "Associated Feature",
    "Suspicious Lymph Node": "Associated Feature",
    "Nipple Retraction": "Associated Feature",
    "Skin Retraction": "Associated Feature",
}

PRIMARY_CATEGORIES = {
    "Mass",
    "Calcification",
    "Asymmetry",
    "Architectural Distortion",
}


def convert_findings(raw_findings):
    new_findings = []

    for f in raw_findings:
        cat = f.get("finding_category")

        # No Finding 不作为 finding，直接跳过
        if cat == "No Finding":
            continue

        if cat not in CATEGORY_MAP:
            raise ValueError(f"Unknown finding_category: {cat}")

        new_cat = CATEGORY_MAP[cat]

        new_f = {
            "finding_category": new_cat,
            "finding_birads": f.get("finding_birads"),
            "bbox": f.get("bbox"),
        }

        # Associated Feature 不要求 finding_birads
        if new_cat == "Associated Feature":
            new_f["finding_birads"] = None

        new_findings.append(new_f)

    return new_findings


def convert_gt(gt):
    """对整条 GT 应用类别归一化 (返回新 dict)."""
    g = dict(gt)
    g["findings"] = convert_findings(g["findings"])
    return g


def build_user_content(mode, target_img_abs):
    """组装 (user_text, images): 文本含 <image> 占位符, images 为绝对路径列表.
    (LLaMA-Factory ChatModel.chat 的输入格式)"""
    if mode == "icl":
        ex1, ex2 = ICL_EXAMPLES
        text = (PROMPTS["icl"]
                .replace("[JSON 1]", json.dumps(convert_gt(ex1["ground_truth"]), ensure_ascii=False, indent=2))
                .replace("[JSON 2]", json.dumps(convert_gt(ex2["ground_truth"]), ensure_ascii=False, indent=2))
                .replace("[Image 1]", "<image>").replace("[Image 2]", "<image>")
                .replace("[Image]", "<image>"))
        images = [str(DATA_ROOT / ex1["image_path"]),
                  str(DATA_ROOT / ex2["image_path"]),
                  target_img_abs]
    else:
        text = PROMPTS[mode]
        images = [target_img_abs]

    assert text.count("<image>") == len(images), \
        f"[{mode}] <image> 占位符数({text.count('<image>')}) != images 数({len(images)})"
    return text, images


def main():
    ap = argparse.ArgumentParser(description="构造 VinDr 推理数据")
    ap.add_argument("--prompt", required=True, choices=["direct", "icl", "cot"])
    ap.add_argument("--limit", type=int, default=0, help="只取前 N 条 (0 = 全量)")
    ap.add_argument("--split", default=SPLIT)
    args = ap.parse_args()

    gt_data = json.load(open(DATA_ROOT / "vlm_dataset_labels_only" / f"{args.split}.json"))
    img_data = json.load(open(DATA_ROOT / "vlm_dataset" / f"{args.split}.json"))
    assert len(gt_data) == len(img_data), "labels_only 与 vlm_dataset 长度不一致"
    n = len(gt_data) if args.limit <= 0 else min(args.limit, len(gt_data))

    records = []
    for idx in range(n):
        gt, img = gt_data[idx], img_data[idx]
        img_abs = str(DATA_ROOT / img["image_path"])
        assert Path(img_abs).exists(), f"图像不存在: {img_abs}"
        text, images = build_user_content(args.prompt, img_abs)
        records.append({
            "index": idx,
            "image_id": img["image_id"],
            "image_path": img["image_path"],
            "prompt": args.prompt,
            "messages": [{"role": "user", "content": text}],
            "images": images,
            "ground_truth": convert_gt(gt),
        })

    out = THIS_DIR / f"infer_data.{args.prompt}.5cls.{args.split}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"[{args.prompt}] {n} 条 -> {out}")
    print(f"  每条 images={len(records[0]['images'])}, 含 ground_truth")
    print(f"  -> 下一步: python3 infer.py --prompt {args.prompt} --split {args.split}")


if __name__ == "__main__":
    main()
