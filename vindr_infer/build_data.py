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

Analyze the mammography image and return ONLY valid JSON.

Predict:
- breast_birads: one of [1, 2, 3, 4, 5]
- breast_density: one of ["A", "B", "C", "D"]
- findings: all visible abnormal findings.

Density guide:
A = almost entirely fatty.
B = scattered fibroglandular density.
C = heterogeneously dense.
D = extremely dense.

Each finding must contain:
- finding_category: one of ["Mass", "Calcification", "Asymmetry", "Architectural Distortion", "Associated Feature"]
- finding_birads: one of [3, 4, 5, null]
- bbox: [x_min, y_min, x_max, y_max], normalized to [0, 1]

Rules:
- Only output findings=[] when no abnormal finding is visible; otherwise include all visible abnormalities.
- Choose the most appropriate category; do not default every abnormality to Mass.
- Use Calcification for suspicious bright spots/clusters, Asymmetry for asymmetric density, Architectural Distortion for tissue distortion/spiculation, and Associated Feature for secondary signs such as skin/nipple retraction, skin thickening, or suspicious lymph node.
- Use finding_birads=null only for Associated Feature; otherwise use one of [3, 4, 5].
- If findings is empty, breast_birads should be 1 or 2; if findings is not empty, breast_birads should usually match the highest finding_birads.
- Use a tight normalized bbox around the abnormal finding itself.

Return ONLY a JSON object with exactly these keys:
"breast_birads", "breast_density", "findings".

Each finding object must have exactly these keys:
"finding_category", "finding_birads", "bbox".

Do not output explanations, markdown, code fences, or extra text."""

# 占位符: [Image 1]/[Image 2] 示例图, [JSON 1]/[JSON 2] 示例 GT, [Image] 目标图
PROMPT_ICL = """You are an expert mammography assistant.

I will first show you four annotated examples, then ask you to annotate a new image.

Example 1:
[Image 1]
Annotation:
[JSON 1]

Example 2:
[Image 2]
Annotation:
[JSON 2]

Example 3:
[Image 3]
Annotation:
[JSON 3]

Example 4:
[Image 4]
Annotation:
[JSON 4]

Now analyze the new mammography image.

[Image]

Return ONLY valid JSON.

Predict:
- breast_birads: one of [1, 2, 3, 4, 5]
- breast_density: one of ["A", "B", "C", "D"]
- findings: all visible abnormal findings.

Density guide:
A = almost entirely fatty.
B = scattered fibroglandular density.
C = heterogeneously dense.
D = extremely dense.

Each finding must contain:
- finding_category: one of ["Mass", "Calcification", "Asymmetry", "Architectural Distortion", "Associated Feature"]
- finding_birads: one of [3, 4, 5, null]
- bbox: [x_min, y_min, x_max, y_max], normalized to [0, 1]

Rules:
- Only output findings=[] when no abnormal finding is visible; otherwise include all visible abnormalities.
- Choose Mass for a localized opacity or mass-like lesion; use Asymmetry when there is no clear mass boundary.
- Use Calcification for suspicious bright spots/clusters, Architectural Distortion for tissue distortion/spiculation, and Associated Feature for secondary signs such as skin/nipple retraction, skin thickening, or suspicious lymph node.
- Use finding_birads=null only for Associated Feature; otherwise use one of [3, 4, 5].
- If findings is empty, breast_birads should be 1 or 2.
- If findings is not empty, breast_birads should usually match the highest finding_birads.
- Use a tight normalized bbox around the abnormal finding itself.
- Do not copy example coordinates; estimate bbox from the new image.

Return ONLY a JSON object with exactly these keys:
"breast_birads", "breast_density", "findings".

Each finding object must have exactly these keys:
"finding_category", "finding_birads", "bbox".

Do not output explanations, markdown, code fences, or extra text."""

# cot: 分 5 步推理后输出 JSON
PROMPT_COT = """You are an expert mammography assistant.

<image>

Analyze the mammography image using the following diagnostic reasoning process:

1. Observation:
Assess breast density and scan the entire breast for suspicious regions.

2. Assessment:
For each suspicious region, evaluate its visual appearance and determine whether it is a Mass, Calcification, Asymmetry, Architectural Distortion, or Associated Feature. Estimate a tight bounding box around the abnormality.

3. Diagnostic synthesis:
Use the identified findings to determine finding-level BI-RADS and the overall breast BI-RADS.

Perform these steps carefully, then return ONLY valid JSON.

Predict:
- breast_birads: one of [1, 2, 3, 4, 5]
- breast_density: one of ["A", "B", "C", "D"]
- findings: all visible abnormal findings.

Density guide:
A = almost entirely fatty.
B = scattered fibroglandular density.
C = heterogeneously dense.
D = extremely dense.

Each finding must contain:
- finding_category: one of ["Mass", "Calcification", "Asymmetry", "Architectural Distortion", "Associated Feature"]
- finding_birads: one of [3, 4, 5, null]
- bbox: [x_min, y_min, x_max, y_max], normalized to [0, 1]

Rules:
- Use findings=[] only when no abnormal finding is visible.
- Choose Mass for a localized opacity or mass-like lesion; use Asymmetry when there is no clear mass boundary.
- Use Calcification for suspicious bright spots or clusters.
- Use Architectural Distortion for tissue distortion or disrupted architecture.
- Use finding_birads=null only for Associated Feature; otherwise use one of [3, 4, 5].
- If findings is empty, breast_birads should be 1 or 2.
- If findings is not empty, breast_birads should usually match the highest finding_birads.
- Use a tight normalized bbox around the abnormal finding itself.

Return ONLY a JSON object with exactly these keys:
"breast_birads", "breast_density", "findings".

Do not output reasoning, explanations, markdown, code fences, or extra text."""

PROMPTS = {"direct": PROMPT_DIRECT, "icl": PROMPT_ICL, "cot": PROMPT_COT}

# --------------------------- ICL few-shot 示例 ---------------------------
ICL_EXAMPLES = [
        { 
        "image_path": "images_png/dd75639d26941066bcee87059be269fb/df9bc222001bfa55330b55d68804b3b6.png",
        "json": {
            "breast_birads": 1,
            "breast_density": "C",
            "findings": []
            }
        },
        {
        "image_path": "images_png/d71aeaa60e6a14df6b9adccd012cbb90/841802bee1fc1a6b4bde97a754713da8.png",
        "json": {
            "breast_birads": 3,
            "breast_density": "B",
            "findings": [
                {
                    "finding_category": "Mass",
                    "finding_birads": 3,
                    "bbox": [0.2708, 0.6129, 0.4836, 0.6812]
                }
            ]
        }
    },
    {
        "image_path": "images_png/3087f2712d14867142d383ebac1e15b4/5c0272f3dc9c06f9edb8b8a749a6cf8b.png",
        "json": {
            "breast_birads": 4,
            "breast_density": "C",
            "findings": [
                {
                    "finding_category": "Calcification",
                    "finding_birads": 4,
                    "bbox": [0.367, 0.368, 0.5959, 0.4152]
                }
            ]
        }
    },
    {
        "image_path": "images_png/b5a6b304d769cdd45a7ac4253f653e5e/228bbafbc7cd167675e4d534785a56ef.png",
        "json": {
            "breast_birads": 3,
            "breast_density": "C",
            "findings": [
                {
                    "finding_category": "Asymmetry",
                    "finding_birads": 3,
                    "bbox": [0.1409, 0.5981, 0.2655, 0.6456]
                }
            ]
        }
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
        ex1, ex2, ex3, ex4 = ICL_EXAMPLES
        text = (PROMPTS["icl"]
                .replace("[JSON 1]", json.dumps(ex1["json"], ensure_ascii=False, indent=2))
                .replace("[JSON 2]", json.dumps(ex2["json"], ensure_ascii=False, indent=2))
                .replace("[JSON 3]", json.dumps(ex3["json"], ensure_ascii=False, indent=2))
                .replace("[JSON 4]", json.dumps(ex4["json"], ensure_ascii=False, indent=2))

                .replace("[Image 1]", "<image>")
                .replace("[Image 2]", "<image>")
                .replace("[Image 3]", "<image>")
                .replace("[Image 4]", "<image>")

                .replace("[Image]", "<image>"))
        images = [
            str(DATA_ROOT / ex1["image_path"]),
            str(DATA_ROOT / ex2["image_path"]),
            str(DATA_ROOT / ex3["image_path"]),
            str(DATA_ROOT / ex4["image_path"]),

            target_img_abs,
        ]
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

    out = THIS_DIR / f"infer_data.{args.prompt}.{args.split}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"[{args.prompt}] {n} 条 -> {out}")
    print(f"  每条 images={len(records[0]['images'])}, 含 ground_truth")
    print(f"  -> 下一步: python3 infer.py --prompt {args.prompt} --split {args.split}")


if __name__ == "__main__":
    main()
