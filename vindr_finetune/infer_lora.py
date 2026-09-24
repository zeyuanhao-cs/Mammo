#!/usr/bin/env python3
"""
VinDr-Mammo VLM 推理 - LoRA 微调后模型版本 (LLaMA-Factory, GPU/BF16)

与 vindr_infer/infer_3prompts_500.py 对齐:
  - 默认推理每个 prompt 的前 500 条 (与基线 500 条实验相同的数据子集)
  - tqdm 进度条 + postfix (耗时/BI-RADS/findings/失败数)
  - 加载基座模型 + LoRA adapter (finetuning_type=lora), 只加载一次
  - 测试数据复用 vindr_infer/build_data.py 生成的 infer_data.<prompt>.<split>.json
    (与基线推理完全相同的输入, 保证公平对比)
  - flash_attn 默认 sdpa (本环境未安装 flash-attn)

用法:
    python3 infer_lora.py --prompt direct --split test            # 前 500 条
    python3 infer_lora.py --prompt direct --split test --limit 0  # 全量
    python3 infer_lora.py --prompt icl --split test --limit 50
    python3 infer_lora.py --prompt direct --split test --adapter /path/to/other_lora
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
BASE_MODEL_PATH = "/hy-tmp/9_9/Qwen/Qwen3.5-4B"
DEFAULT_ADAPTER = "/hy-tmp/9_9/Qwen/Qwen_lora_1000"
DEFAULT_LIMIT = 500  # 与 infer_3prompts_500.py 相同: 每个取前 500 条

# 与基线 infer.py 保持一致: qwen3_5 模板 + enable_thinking=False
TEMPLATE = "qwen3_5"

THIS_DIR = Path(__file__).resolve().parent

# 复用 vindr_infer 的推理数据 (与基线相同的输入)
INFER_DIR = THIS_DIR.parent / "vindr_infer"
DATA_FILE = INFER_DIR / "infer_data.{prompt}.{split}.json"

OUTPUT_ROOT = THIS_DIR / "outputs"

MAX_NEW_TOKENS = 256

# 与基线 infer.py 相同的图像设置 (912x1520 不会被 1536*1536 上限缩放, 实际全分辨率)
IMAGE_MAX_PIXELS = 1536 * 1536
IMAGE_MIN_PIXELS = 512 * 512
IMAGE_TAG = "1536"

INFER_DTYPE = "bfloat16"
# ====================================================================


THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", flags=re.DOTALL)


def gpu_mem(prefix: str):
    """打印 CUDA 显存状态。"""
    if not torch.cuda.is_available():
        print(f"[gpu] {prefix}: CUDA not available", flush=True)
        return

    free, total = torch.cuda.mem_get_info()
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()

    print(
        f"[gpu] {prefix}: "
        f"allocated={allocated / 1024**3:.2f}GB, "
        f"reserved={reserved / 1024**3:.2f}GB, "
        f"free={free / 1024**3:.2f}GB, "
        f"total={total / 1024**3:.2f}GB",
        flush=True,
    )


def parse_prediction(raw: str):
    """从模型原始输出中提取 JSON。解析失败返回 None。"""
    if raw is None:
        return None

    text = THINK_RE.sub("", raw).strip()

    # 提取 ```json ... ``` 中的内容
    m = FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()

    # 直接尝试解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 截取第一个 { 到最后一个 }
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return None

    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def get_eval_json(pred_json):
    """
    兼容两种输出：
    1. direct / icl / hidden-cot:
       {"breast_birads": ..., "breast_density": ..., "findings": [...]}

    2. visible-cot:
       {"reasoning": ..., "final_json": {"breast_birads": ..., ...}}
    """
    if isinstance(pred_json, dict) and isinstance(pred_json.get("final_json"), dict):
        return pred_json["final_json"]
    return pred_json


def validate_record(rec: dict):
    """检查数据是否符合 ChatModel 输入。"""
    assert "messages" in rec and isinstance(rec["messages"], list), "record 缺少 messages"
    assert "images" in rec and isinstance(rec["images"], list), "record 缺少 images"
    assert "ground_truth" in rec, "record 缺少 ground_truth"

    content = rec["messages"][0]["content"]
    n_placeholders = content.count("<image>")
    n_images = len(rec["images"])

    assert n_placeholders == n_images, (
        f"index={rec.get('index')} <image> 数量({n_placeholders}) != images 数量({n_images})"
    )


def main():
    parser = argparse.ArgumentParser(description="VinDr-Mammo LoRA 微调模型推理")

    parser.add_argument("--prompt", required=True, choices=["direct", "icl", "cot"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--model", default=BASE_MODEL_PATH, help="基座模型路径")
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER, help="LoRA adapter 路径")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="只推理前 N 条；0 表示全量")

    # 本环境未安装 flash-attn, 默认 sdpa; 装好后可加 --flash-attn fa2 提速
    parser.add_argument("--flash-attn", default="sdpa", choices=["sdpa", "fa2", "disabled", "auto"])

    parser.add_argument("--debug-gpu", action="store_true", help="逐条打印 GPU 显存状态")
    parser.add_argument("--fsync-every", type=int, default=10, help="每 N 条 fsync 一次；0 表示只在最后 fsync")

    args = parser.parse_args()

    # ---------------------- CUDA 检查 ----------------------
    assert torch.cuda.is_available(), "CUDA 不可用：当前环境没有检测到 GPU"

    torch.cuda.set_device(0)

    print("[cuda] available:", torch.cuda.is_available())
    print("[cuda] device_name:", torch.cuda.get_device_name(0))
    print("[torch] version:", torch.__version__)

    gpu_mem("before data load")

    # ---------------------- 读取数据 ----------------------
    data_file = Path(str(DATA_FILE).format(prompt=args.prompt, split=args.split))
    assert data_file.exists(), f"数据文件不存在: {data_file}，请先在 vindr_infer 下运行 build_data.py"

    records = json.load(open(data_file, "r", encoding="utf-8"))

    if args.limit > 0:
        records = records[:args.limit]

    assert len(records) > 0, "records 为空"

    for rec in records[:5]:
        validate_record(rec)

    print(f"[data] file={data_file}")
    print(f"[data] records={len(records)}")
    print(f"[data] first_index={records[0]['index']} last_index={records[-1]['index']}")

    # ---------------------- 输出目录 ----------------------
    adapter_tag = Path(args.adapter).name  # e.g. Qwen_lora_1000
    attn_tag = args.flash_attn
    run_name = f"{args.prompt}_{adapter_tag}_{IMAGE_TAG}_{attn_tag}"

    out_dir = OUTPUT_ROOT / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / "predictions.jsonl"
    config_file = out_dir / "run_config.json"

    # ---------------------- 断点续推 ----------------------
    done = set()

    if out_file.exists():
        with open(out_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    done.add(item["index"])
                except Exception:
                    pass

    if done:
        print(f"[resume] 已完成 {len(done)} 条，将跳过这些 index")

    # ---------------------- LLaMA-Factory 模型参数 ----------------------
    model_args = dict(
        model_name_or_path=args.model,
        adapter_name_or_path=args.adapter,
        template=TEMPLATE,
        trust_remote_code=True,
        infer_backend="huggingface",
        finetuning_type="lora",

        infer_dtype=INFER_DTYPE,

        # 图像设置 (与基线 infer.py 一致)
        image_max_pixels=IMAGE_MAX_PIXELS,
        image_min_pixels=IMAGE_MIN_PIXELS,

        # 关闭 thinking
        enable_thinking=False,

        flash_attn=args.flash_attn,
    )

    # ---------------------- 保存配置快照 ----------------------
    run_config = {
        "task_version": "5cls_v1_lora",
        "base_model": args.model,
        "adapter": args.adapter,
        "template": TEMPLATE,
        "prompt": args.prompt,
        "split": args.split,
        "data_file": str(data_file),
        "output_file": str(out_file),
        "n_records_this_run": len(records),
        "max_new_tokens": args.max_new_tokens,
        "image_max_pixels": IMAGE_MAX_PIXELS,
        "image_min_pixels": IMAGE_MIN_PIXELS,
        "enable_thinking": False,
        "do_sample": False,
        "infer_backend": "huggingface",
        "finetuning_type": "lora",
        "infer_dtype": INFER_DTYPE,
        "flash_attn": args.flash_attn,
    }

    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    print(f"[config] saved to {config_file}")

    # ---------------------- 加载模型 ----------------------
    print(f"[model] base={args.model}")
    print(f"[model] adapter={args.adapter}")
    print(f"[model] template={TEMPLATE}, enable_thinking=False")
    print(f"[model] infer_dtype={INFER_DTYPE}, flash_attn={args.flash_attn}")

    gpu_mem("before model load")
    t0 = time.time()

    chat_model = ChatModel(args=model_args)

    torch.cuda.synchronize()
    print(f"[model] loaded in {time.time() - t0:.1f}s")
    gpu_mem("after model load")

    # ---------------------- 推理 ----------------------
    pending = [rec for rec in records if rec.get("index") not in done]
    print(f"[infer] selected={len(records)} done={len(done)} pending={len(pending)}")

    fout = open(out_file, "a", encoding="utf-8")

    n_run = 0
    n_parse_fail = 0
    elapsed_sum = 0.0
    t_start = time.time()

    try:
        pbar = tqdm(
            pending,
            total=len(pending),
            desc=f"{args.prompt:>6}",
            unit="img",
            dynamic_ncols=True,
            leave=True,
        )

        for rec in pbar:
            idx = rec["index"]

            validate_record(rec)

            t1 = time.perf_counter()

            try:
                if args.debug_gpu:
                    torch.cuda.synchronize()
                    gpu_mem(f"before index={idx}")

                responses = chat_model.chat(
                    messages=rec["messages"],
                    images=rec["images"],
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                )

                if args.debug_gpu:
                    torch.cuda.synchronize()
                    gpu_mem(f"after index={idx}")

                pred_raw = responses[0].response_text.strip()
                pred_json = parse_prediction(pred_raw)
                eval_json = get_eval_json(pred_json)
                error = None

            except Exception as e:
                pred_raw = ""
                pred_json = None
                eval_json = None
                error = repr(e)

            sec = time.perf_counter() - t1
            elapsed_sum += sec
            n_run += 1

            if eval_json is None:
                n_parse_fail += 1

            record = {
                "index": idx,
                "image_id": rec.get("image_id"),
                "image_path": rec.get("image_path"),
                "prompt": args.prompt,
                "prediction_raw": pred_raw,
                "prediction_json": pred_json,
                "eval_json": eval_json,
                "ground_truth": rec["ground_truth"],
                "seconds": round(sec, 2),
            }
            if error is not None:
                record["error"] = error

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

            if args.fsync_every > 0 and n_run % args.fsync_every == 0:
                os.fsync(fout.fileno())

            # 进度条 postfix: 单条耗时/平均耗时/pred_birads/findings 数/解析失败数
            avg_sec = elapsed_sum / n_run
            pred_birads = eval_json.get("breast_birads") if isinstance(eval_json, dict) else "?"
            n_findings = (
                len(eval_json.get("findings", []))
                if isinstance(eval_json, dict) and isinstance(eval_json.get("findings"), list)
                else "?"
            )

            pbar.set_postfix(
                sec=f"{sec:.1f}",
                avg=f"{avg_sec:.1f}",
                birads=pred_birads,
                findings=n_findings,
                fail=n_parse_fail,
            )

    finally:
        fout.flush()
        os.fsync(fout.fileno())
        fout.close()

    avg_sec = elapsed_sum / max(n_run, 1)
    total_sec = time.time() - t_start

    print()
    print(f"[done] 新推理 {n_run} 条 (selected={len(records)}, 已跳过 {len(done)})")
    print(f"[done] parse_fail {n_parse_fail} 条")
    print(f"[done] avg {avg_sec:.2f}s/img, total {total_sec / 60:.1f} min")
    print(f"[done] results: {out_file}")


if __name__ == "__main__":
    main()
