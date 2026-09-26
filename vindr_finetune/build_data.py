#!/usr/bin/env python3
"""
VinDr-Mammo 训练数据构造 (复用 vindr_infer/build_data.py 的 prompt 与类别归一化)

职责: vlm_dataset/{split}.json -> LLaMA-Factory alpaca (instruction/output/images) 训练格式
      - instruction: user prompt (含 <image> 占位符)
      - output:      归一化后的 GT JSON (10 类 -> 4 主类 + Associated Feature,
                     No Finding -> findings=[])
      - images:      相对路径列表 (相对于 THIS_DIR=vindr_finetune/, 即 "images_png/study/img.png")
                     LLaMA-Factory 解析时相对于 dataset_dir, 配 dataset_dir: vindr_finetune 即可

说明:
- 不再设 input 字段 (LLaMA-Factory alpaca 格式 input 是 optional, 删了更简洁;
  训练时 prompt = instruction, 等价于原来 instruction + "\\n" + "")
- 划分: train + test 两档 (无 val, 与 Mammo-CLIP 官方一致)
- images 默认用相对路径 (相对于 THIS_DIR), 加 --absolute 可退回绝对路径

输出: vindr_finetune/data/<prompt>_<split>.json

用法:
    # 本地生成 (默认相对路径, 图片已在 vindr_finetune/images_png/ 下)
    python3 build_data.py --prompt direct

    # 远程生成 (同样相对路径, 只要布局一致即可跨机器复用)
    python3 build_data.py --prompt direct

    # 退回绝对路径 (需指定 --data-root, 远程路径校验会缺图属正常)
    python3 build_data.py --prompt direct --absolute \\
        --data-root /hy-tmp/9_9/vindr-mammo

    # 限量
    python3 build_data.py --prompt direct --limit 50
    python3 build_data.py --prompt icl --splits train

配套 dataset_info.json (放在 vindr_finetune/dataset_info.json):
    {
      "vindr_direct_train": {
        "file_name": "data/direct_train.json",
        "columns": {"prompt": "instruction", "response": "output", "images": "images"}
      },
      "vindr_direct_test": {
        "file_name": "data/direct_test.json",
        "columns": {"prompt": "instruction", "response": "output", "images": "images"}
      }
    }

配套 trial.yaml 加一行:
    dataset_dir: vindr_finetune
"""
import argparse
import json
from pathlib import Path

# ============================== CONFIG ==============================
# 图片目录 = THIS_DIR/images_png (用户已把 images_png 拉到 vindr_finetune/ 下)
# vlm_dataset JSON 仍从外部 data_root 读 (默认本地 vindr-mammo, 可用 --data-root 覆盖)
THIS_DIR = Path(__file__).resolve().parent          # = Mammo/vindr_finetune/
IMG_DIR = THIS_DIR / "images_png"                    # 图片所在 (相对路径基准)
OUT_DIR = THIS_DIR / "data"
DEFAULT_DATA_ROOT = Path("/Users/haozeyuan/Desktop/phd/vlm/9_9/vindr-mammo")  # vlm_dataset JSON 所在
DEFAULT_SPLITS = ["train", "test"]  # 已切到 train+test 无 val (与 Mammo-CLIP 一致)
# ====================================================================

# ------------------------------ Prompts ------------------------------
# 与 vindr_infer/build_data.py 保持一致
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
- Use Calcification for suspicious bright spots/clusters, Asymmetry for asymmetric density, Architectural Distortion for tissue distortion/spiculation, and Associated Feature for secondary signs such as skin/nipple retraction, skin thickening, or suspicious lymph node.
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
# 与 vindr_infer/build_data.py 保持一致
ICL_EXAMPLES = [
    {
        "image_path": "images_png/dd75639d26941066bcee87059be269fb/df9bc222001bfa55330b55d68804b3b6.png",
        "json": {
            "breast_birads": 1,
            "breast_density": "C",
            "findings": []
        },
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
        },
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
        },
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
    g = {
        "breast_birads": gt.get("breast_birads"),
        "breast_density": gt.get("breast_density"),
        "findings": convert_findings(gt.get("findings", [])),
    }
    return g


def build_user_content(mode, target_img_path, absolute=False):
    """组装 (user_text, images):
    - target_img_path: 目标图的相对路径 (相对于 THIS_DIR, 如 "images_png/study/img.png")
    - absolute=False: images 返回相对路径 (相对于 THIS_DIR), LLaMA-Factory 配 dataset_dir: vindr_finetune
    - absolute=True:  images 返回绝对路径 (向后兼容, 远程训练用)
    与 vindr_infer/build_data.py 逻辑一致"""
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
        if absolute:
            images = [
                str(THIS_DIR / ex1["image_path"]),
                str(THIS_DIR / ex2["image_path"]),
                str(THIS_DIR / ex3["image_path"]),
                str(THIS_DIR / ex4["image_path"]),
                str(THIS_DIR / target_img_path),
            ]
        else:
            images = [
                ex1["image_path"],
                ex2["image_path"],
                ex3["image_path"],
                ex4["image_path"],
                target_img_path,
            ]
    else:
        text = PROMPTS[mode]
        images = [str(THIS_DIR / target_img_path) if absolute else target_img_path]

    assert text.count("<image>") == len(images), \
        f"[{mode}] <image> 占位符数({text.count('<image>')}) != images 数({len(images)})"
    return text, images


def main():
    ap = argparse.ArgumentParser(description="构造 VinDr LLaMA-Factory SFT 训练数据")
    ap.add_argument("--prompt", required=True, choices=["direct", "icl", "cot"])
    ap.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS,
                    choices=["train", "test"])
    ap.add_argument("--limit", type=int, default=0, help="每个 split 只取前 N 条 (0 = 全量)")
    ap.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT),
                    help=f"vlm_dataset JSON 所在目录 (默认 {DEFAULT_DATA_ROOT})")
    ap.add_argument("--absolute", action="store_true",
                    help="images 用绝对路径 (默认相对路径, 相对于 THIS_DIR)")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 图片校验基准目录: 相对模式校验 THIS_DIR/images_png, 绝对模式校验 data_root/images_png
    img_check_root = (THIS_DIR if not args.absolute else data_root)
    print(f"[模式] images={'绝对路径' if args.absolute else '相对路径(相对于 vindr_finetune/)'}")
    print(f"[校验] 图片目录: {img_check_root / 'images_png'}")

    for split in args.splits:
        data = json.load(open(data_root / "vlm_dataset" / f"{split}.json"))
        n = len(data) if args.limit <= 0 else min(args.limit, len(data))

        records = []
        n_img_missing = 0
        for idx in range(n):
            rec = data[idx]
            # rec["image_path"] 形如 "images_png/study/img.png"
            # 相对模式: 直接写入; 绝对模式: 拼 data_root 前缀
            target_img_path = rec["image_path"] if not args.absolute else str(data_root / rec["image_path"])

            # 校验图片存在 (仅本地校验, 缺失只警告不中断)
            check_path = img_check_root / rec["image_path"]
            if not check_path.exists():
                n_img_missing += 1
                if n_img_missing <= 3:
                    print(f"  [warn] 图像不存在: {check_path}")

            text, images = build_user_content(args.prompt, rec["image_path"], absolute=args.absolute)
            gt = convert_gt(rec)

            # LLaMA-Factory alpaca 多模态 SFT 格式 (无 input 字段, prompt 全在 instruction)
            records.append({
                "instruction": text,
                "output": json.dumps(gt, ensure_ascii=False),
                "images": images,
            })

        out = OUT_DIR / f"{args.prompt}_{split}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        n_empty = sum(1 for r in records
                      if json.loads(r["output"])["findings"] == [])
        print(f"[{args.prompt}/{split}] {n} 条 (阴性 findings=[] 共 {n_empty}, 图缺失 {n_img_missing}) -> {out}")

    print("\n下一步: 确认 vindr_finetune/dataset_info.json 已注册, trial.yaml 含 'dataset_dir: vindr_finetune'")


if __name__ == "__main__":
    main()
