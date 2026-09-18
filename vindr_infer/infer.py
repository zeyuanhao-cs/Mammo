#!/usr/bin/env python3
"""
VinDr-Mammo VLM 推理 (5cls version, LLaMA-Factory, GPU/BF16)

职责:
  - 读取 build_data.py 生成的 infer_data.<prompt>.5cls.<split>.json
  - 通过 LLaMA-Factory ChatModel 逐条推理
  - 使用 LLaMA-Factory 参数 infer_dtype=bfloat16 / flash_attn=fa2
  - 逐条保存 JSONL，支持中断续跑

用法:
    python infer.py --prompt direct --split test --limit 5 --max-new-tokens 256
    python infer.py --prompt icl    --split test --limit 5 --max-new-tokens 256

如果 flash_attn 报错:
    python infer.py --prompt direct --split test --limit 5 --max-new-tokens 256 --no-flash-attn
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import torch
from llamafactory.chat import ChatModel


# ============================== CONFIG ==============================
MODEL_PATH = "/hy-tmp/9_9/Qwen/Qwen3.5-4B"

# Qwen3.5 原生模板，thinking 用 enable_thinking=False 关闭
TEMPLATE = "qwen3_5"

THIS_DIR = Path(__file__).resolve().parent

# 读取 5cls 新文件
DATA_FILE = THIS_DIR / "infer_data.{prompt}.5cls.{split}.json"

OUTPUT_ROOT = THIS_DIR / "outputs"

# JSON 输出很短，256 通常够用；不够再改回 512
MAX_NEW_TOKENS = 256

# 先保持 1536，确认 GPU 跑通后再决定是否降到 1024
IMAGE_MAX_PIXELS = 1024 * 1024
IMAGE_MIN_PIXELS = 512 * 512

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


def validate_record(rec: dict):
    """检查 build_data 生成的数据是否符合 ChatModel 输入。"""
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
    parser = argparse.ArgumentParser(description="VinDr-Mammo VLM 推理 5cls GPU/BF16 版本")

    # 暂时只跑 direct / icl，cot prompt 还没同步 5cls
    parser.add_argument("--prompt", required=True, choices=["direct", "icl"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--limit", type=int, default=0, help="只推理前 N 条；0 表示全量")

    # 默认使用 LLaMA-Factory 的 flash_attn=fa2；报错时加 --no-flash-attn
    parser.add_argument("--no-flash-attn", action="store_true", help="禁用 flash_attn")

    args = parser.parse_args()

    # ---------------------- CUDA 检查 ----------------------
    assert torch.cuda.is_available(), "CUDA 不可用：当前环境没有检测到 GPU"

    torch.cuda.set_device(0)

    print("[cuda] available:", torch.cuda.is_available())
    print("[cuda] device_count:", torch.cuda.device_count())
    print("[cuda] current_device:", torch.cuda.current_device())
    print("[cuda] device_name:", torch.cuda.get_device_name(0))
    print("[torch] version:", torch.__version__)
    print("[torch] cuda version:", torch.version.cuda)

    gpu_mem("before data load")

    # ---------------------- 读取 5cls 数据 ----------------------
    data_file = Path(str(DATA_FILE).format(prompt=args.prompt, split=args.split))
    assert data_file.exists(), f"数据文件不存在: {data_file}，请先运行 build_data.py"

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
    attn_tag = "noflash" if args.no_flash_attn else "fa2"
    run_name = f"{args.prompt}_5cls_qwen35_4b_1536_gpu_{attn_tag}_v2"

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
        template=TEMPLATE,
        trust_remote_code=True,
        infer_backend="huggingface",
        finetuning_type="full",

        # LLaMA-Factory 可识别参数
        infer_dtype=INFER_DTYPE,

        # 图像设置
        image_max_pixels=IMAGE_MAX_PIXELS,
        image_min_pixels=IMAGE_MIN_PIXELS,

        # 关闭 thinking
        enable_thinking=False,
    )

    if args.no_flash_attn:
        model_args["flash_attn"] = "disabled"
    else:
        model_args["flash_attn"] = "fa2"

    # ---------------------- 保存配置快照 ----------------------
    run_config = {
        "task_version": "5cls_v1",
        "model": args.model,
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
        "finetuning_type": "full",
        "infer_dtype": INFER_DTYPE,
        "flash_attn": model_args.get("flash_attn"),
    }

    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    print(f"[config] saved to {config_file}")

    # ---------------------- 加载模型 ----------------------
    print(f"[model] loading {args.model}")
    print(f"[model] template={TEMPLATE}, enable_thinking=False")
    print(f"[model] infer_dtype={INFER_DTYPE}")
    print(f"[model] flash_attn={model_args.get('flash_attn')}")
    print(f"[image] max_pixels={IMAGE_MAX_PIXELS}, min_pixels={IMAGE_MIN_PIXELS}")

    gpu_mem("before model load")
    t0 = time.time()

    chat_model = ChatModel(args=model_args)

    torch.cuda.synchronize()
    print(f"[model] loaded in {time.time() - t0:.1f}s")
    gpu_mem("after model load")

    # ---------------------- 推理 ----------------------
    fout = open(out_file, "a", encoding="utf-8")

    n_run = 0
    n_parse_fail = 0
    t_start = time.time()

    for rec in records:
        idx = rec["index"]

        if idx in done:
            continue

        validate_record(rec)

        t1 = time.time()

        try:
            torch.cuda.synchronize()
            gpu_mem(f"before index={idx}")

            responses = chat_model.chat(
                messages=rec["messages"],
                images=rec["images"],
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
            )

            torch.cuda.synchronize()
            gpu_mem(f"after index={idx}")

            pred_raw = responses[0].response_text.strip()

        except Exception as e:
            pred_raw = ""
            pred_json = None
            sec = time.time() - t1

            record = {
                "index": idx,
                "image_id": rec.get("image_id"),
                "image_path": rec.get("image_path"),
                "prompt": args.prompt,
                "prediction_raw": pred_raw,
                "prediction_json": pred_json,
                "ground_truth": rec["ground_truth"],
                "seconds": round(sec, 1),
                "error": repr(e),
            }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            os.fsync(fout.fileno())

            n_run += 1
            n_parse_fail += 1

            print(
                f"[{idx + 1}/{len(records)}] ERROR {sec:.1f}s {repr(e)}",
                flush=True,
            )
            continue

        sec = time.time() - t1
        pred_json = parse_prediction(pred_raw)

        if pred_json is None:
            n_parse_fail += 1

        record = {
            "index": idx,
            "image_id": rec.get("image_id"),
            "image_path": rec.get("image_path"),
            "prompt": args.prompt,
            "prediction_raw": pred_raw,
            "prediction_json": pred_json,
            "ground_truth": rec["ground_truth"],
            "seconds": round(sec, 1),
        }

        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        fout.flush()
        os.fsync(fout.fileno())

        n_run += 1

        avg = (time.time() - t_start) / max(n_run, 1)
        status = "OK" if pred_json is not None else "PARSE_FAIL"

        pred_birads = pred_json.get("breast_birads") if isinstance(pred_json, dict) else "?"
        gt_birads = rec["ground_truth"].get("breast_birads")

        n_findings = None
        if isinstance(pred_json, dict) and isinstance(pred_json.get("findings"), list):
            n_findings = len(pred_json["findings"])

        print(
            f"[{idx + 1}/{len(records)}] "
            f"{sec:.1f}s avg={avg:.1f}s json={status} "
            f"pred_birads={pred_birads} gt_birads={gt_birads} "
            f"pred_findings={n_findings}",
            flush=True,
        )

    fout.close()

    print()
    print(f"[done] 新推理 {n_run} 条")
    print(f"[done] parse_fail {n_parse_fail} 条")
    print(f"[done] results: {out_file}")


if __name__ == "__main__":
    main()