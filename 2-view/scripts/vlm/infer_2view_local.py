#!/usr/bin/env python3
"""
2-view (CC + MLO) 推理：Qwen3.5-4B base 模型零样本推理 VinDr-Mammo 乳房级测试集。

适配要点（相对于 vindr_finetune/infer_full_test.py）：
  - 模型路径: /home/z/zhao/Mammography/model/Qwen3.5-4B (NeSI)
  - 数据: /nesi/project/massey04900/Mammography/test.json (2-view 乳房级样本, 2000 条)
  - instruction 来自 /home/zhao/Mammography/prompt.txt，
    单个 <image> 占位符自动扩展为两个 (LLaMA-Factory 要求占位符数与图片数一致)
  - 默认推理前 500 条, 支持 --limit 调整; 按 id 断点续推
  - 不加载 LoRA (基座零样本); 如有 adapter 可用 --adapter 指定
  - flash_attn 使用 sdpa (本环境无 flash-attn wheel)
  - 视觉分辨率与训练 trial.yaml 对齐: image_max_pixels=786432, min=512*512

输出:
  2-view/outputs/predictions/predictions_{model}{suffix}_{limit}.jsonl
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm
from llamafactory.chat import ChatModel

# ============================== CONFIG ==============================
MODELS = {
    "qwen3.5-4b": {
        "path": "/home/z/zhao/Mammography/model/Qwen3.5-4B",
        "template": "qwen3_5",
        "enable_thinking": False,
    },
    "medgemma-4b-it": {
        "path": "/nesi/project/massey04900/Mammography/model/medgemma-4b-it",
        "template": "gemma3",
        "enable_thinking": False,
    },
}
DEFAULT_MODEL = "qwen3.5-4b"
DATA_FILE = Path("/home/z/zhao/Mammography/Mammo/2-view/data/splits/test_local.json")
PROMPT_FILE = Path("/home/z/zhao/Mammography/Mammo/2-view/dataset_build/prompt.txt")
INFER_DTYPE = "bfloat16"

# 与训练 (trial.yaml) 对齐的视觉分辨率
IMAGE_MAX_PIXELS = 786432
IMAGE_MIN_PIXELS = 512 * 512

MAX_NEW_TOKENS = 256
DEFAULT_LIMIT = 500

THIS_DIR = Path(__file__).resolve().parent          # 2-view/scripts/vlm
PROJ_DIR = THIS_DIR.parent.parent                    # 2-view
OUT_DIR = PROJ_DIR / "outputs" / "predictions"
RUNCFG_DIR = PROJ_DIR / "outputs" / "run_configs"
# ====================================================================

THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", flags=re.DOTALL)


def parse_prediction(raw: str):
    """Extract a JSON object from model output. Return None on failure."""
    if not raw:
        return None

    text = THINK_RE.sub("", raw).strip()

    m = FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def load_instruction(prompt_file: Path = None) -> str:
    """读 prompt 文件 (默认 prompt.txt); 确保恰好 2 个 <image> 占位符 (CC + MLO).

    - 若只有 1 个占位符: 自动扩展为 2 个
    - 若已有 2 个 (如新 prompt 中 "CC view: <image>\\nMLO view: <image>"): 原样保留
    - 其他数量: 报错
    """
    prompt_file = Path(prompt_file) if prompt_file is not None else PROMPT_FILE
    text = prompt_file.read_text(encoding="utf-8").strip()
    n = text.count("<image>")
    if n == 1:
        return text.replace("<image>", "<image>\n<image>", 1)
    if n == 2:
        return text
    raise AssertionError(f"prompt 需要 1 或 2 个 <image> 占位符, 实际 {n} 个: {PROMPT_FILE}")


def load_records(path: Path, instruction: str) -> list:
    """把 2-view test.json 转成推理用 records."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    assert isinstance(raw, list) and raw, f"empty/invalid data: {path}"

    records = []
    for i, r in enumerate(raw):
        assert "id" in r and "images" in r and "output" in r, f"missing keys at {i}"
        assert len(r["images"]) == 2, f"expected 2 images at {i}, got {len(r['images'])}"
        for p in r["images"]:
            assert os.path.isfile(p), f"image not found: {p}"
        records.append(
            {
                "index": i,
                "id": r["id"],
                "messages": [{"role": "user", "content": instruction}],
                "images": list(r["images"]),
                "ground_truth": r["output"],
            }
        )
    return records


def load_done_ids(out_file: Path):
    done = set()
    if not out_file.exists():
        return done

    with open(out_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                done.add(item["id"])
            except Exception:
                pass
    return done


def main():
    parser = argparse.ArgumentParser(description="2-view 推理 on VinDr test.json")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, choices=list(MODELS.keys()),
                        help=f"模型标签, 默认 {DEFAULT_MODEL}; 可选 {list(MODELS.keys())}")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help=f"推理条数, 默认 {DEFAULT_LIMIT}")
    parser.add_argument("--adapter", type=str, default=None, help="可选 LoRA adapter 路径")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--template", type=str, default=None,
                        help="覆盖模型默认模板 (如 LoRA 推理时用 qwen3_5_nothink 与训练一致)")
    parser.add_argument("--out-suffix", type=str, default="",
                        help="输出文件名后缀, 如 _v2prompt, 用于区分不同 prompt 版本")
    parser.add_argument("--data", type=str, default=str(DATA_FILE), help="数据文件路径")
    parser.add_argument("--prompt-file", type=str, default=str(PROMPT_FILE),
                        help="prompt 文件路径 (默认 dataset_build/prompt.txt)")
    args = parser.parse_args()

    model_path = MODELS[args.model]["path"]
    template = MODELS[args.model]["template"]
    enable_thinking = MODELS[args.model]["enable_thinking"]

    assert torch.cuda.is_available(), "CUDA unavailable"
    torch.cuda.set_device(0)
    print(f"[cuda] device: {torch.cuda.get_device_name(0)}")
    print(f"[torch] version: {torch.__version__}, cuda: {torch.version.cuda}")

    # ---------------------- dataset ----------------------
    instruction = load_instruction(Path(args.prompt_file))
    records = load_records(Path(args.data), instruction)
    if args.limit > 0:
        records = records[: args.limit]
    out_file = OUT_DIR / f"predictions_{args.model}{args.out_suffix}_{len(records)}.jsonl"
    print(f"[data] {args.data} -> 前 {len(records)} 条")
    print(f"[data] instruction ({len(instruction)} chars), 2 个 <image> 占位符")

    done = load_done_ids(out_file)
    pending = [rec for rec in records if rec["id"] not in done]
    print(f"[data] done={len(done)} pending={len(pending)}")

    # ---------------------- model ----------------------
    model_args = dict(
        model_name_or_path=model_path,
        template=template,
        trust_remote_code=True,
        infer_backend="huggingface",
        infer_dtype=INFER_DTYPE,
        image_max_pixels=IMAGE_MAX_PIXELS,
        image_min_pixels=IMAGE_MIN_PIXELS,
        enable_thinking=enable_thinking,
        flash_attn="sdpa",
    )
    if args.adapter:
        model_args["adapter_name_or_path"] = args.adapter
        model_args["finetuning_type"] = "lora"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUNCFG_DIR.mkdir(parents=True, exist_ok=True)
    with open(RUNCFG_DIR / f"run_config_{args.model}{args.out_suffix}.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": model_path,
                "model_tag": args.model,
                "adapter": args.adapter,
                "template": template,
                "samples": len(records),
                "max_new_tokens": args.max_new_tokens,
                "image_max_pixels": IMAGE_MAX_PIXELS,
                "image_min_pixels": IMAGE_MIN_PIXELS,
                "enable_thinking": False,
                "do_sample": False,
                "infer_dtype": INFER_DTYPE,
                "flash_attn": "sdpa",
                "data": str(args.data),
                "prompt_file": str(args.prompt_file),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(f"[model] loading: {model_path}" + (f" + LoRA {args.adapter}" if args.adapter else " (base, zero-shot)"))
    print(f"[model] template={template} dtype={INFER_DTYPE} flash_attn=sdpa thinking={enable_thinking}")

    t_model = time.perf_counter()
    chat_model = ChatModel(args=model_args)
    torch.cuda.synchronize()
    print(f"[model] loaded in {time.perf_counter() - t_model:.1f}s")

    if not pending:
        print("[infer] nothing to do")
        return

    # ---------------------- inference ----------------------
    n_run = 0
    n_parse_fail = 0
    elapsed_sum = 0.0
    t_all = time.perf_counter()

    with open(out_file, "a", encoding="utf-8") as fout:
        pbar = tqdm(pending, total=len(pending), desc="2view_infer", unit="case", dynamic_ncols=True)
        for rec in pbar:
            t0 = time.perf_counter()

            try:
                responses = chat_model.chat(
                    messages=rec["messages"],
                    images=rec["images"],
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                )
                pred_raw = responses[0].response_text.strip()
                pred_json = parse_prediction(pred_raw)
                error = None
            except Exception as e:
                pred_raw = ""
                pred_json = None
                error = repr(e)

            sec = time.perf_counter() - t0
            elapsed_sum += sec
            n_run += 1

            if pred_json is None:
                n_parse_fail += 1

            record = {
                "id": rec["id"],
                "index": rec["index"],
                "prediction_raw": pred_raw,
                "prediction_json": pred_json,
                "ground_truth": rec["ground_truth"],
                "seconds": round(sec, 2),
            }
            if error is not None:
                record["error"] = error

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            if n_run % 10 == 0:
                os.fsync(fout.fileno())

            avg_sec = elapsed_sum / n_run
            pbar.set_postfix(sec=f"{sec:.1f}", avg=f"{avg_sec:.1f}", fail=n_parse_fail)

        fout.flush()
        os.fsync(fout.fileno())

    total_sec = time.perf_counter() - t_all
    print()
    print("==================== SUMMARY ====================")
    print(f"new={n_run} parse_fail={n_parse_fail} avg={elapsed_sum / max(n_run, 1):.2f}s/case")
    print(f"TOTAL inference time: {total_sec / 60:.1f} min ({total_sec / 3600:.2f} h)")
    print(f"Output: {out_file}")


if __name__ == "__main__":
    main()
