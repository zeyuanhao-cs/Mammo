#!/usr/bin/env python3
"""
等待 LoRA 训练完成后，自动对 direct_test.json 全量推理。

流程:
  1. 轮询等待训练结束 (llamafactory-cli 进程退出 + 最终 adapter 保存)
  2. 加载基座模型 + LoRA adapter
  3. 全量推理 vindr_finetune/data/direct_test.json (4000 条)

参数与 vindr_infer/infer_3prompts_500.py 保持一致:
  - template: qwen3_5, thinking 关闭
  - bfloat16
  - image_max_pixels = 1536*1536, image_min_pixels = 512*512
  - max_new_tokens = 256, do_sample = False
  - 输出解析/断点续推逻辑一致

差异:
  - flash_attn 使用 fa2 (已安装 flash-attn 2.7.4.post1 预编译 wheel)
  - 加载 LoRA: adapter_name_or_path + finetuning_type=lora

用法:
  python3 infer_full_test.py            # 等训练完成后自动推理
  python3 infer_full_test.py --no-wait  # 跳过等待直接推理 (训练已结束时)

输出:
  vindr_finetune/outputs/direct_full/predictions.jsonl
"""

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm
from llamafactory.chat import ChatModel

# ============================== CONFIG ==============================
MODEL_PATH = "/hy-tmp/9_9/Qwen/Qwen3.5-4B"
ADAPTER_PATH = "/hy-tmp/9_9/Qwen/qwen-4b/lora"
TEMPLATE = "qwen3_5"
INFER_DTYPE = "bfloat16"

# 与训练 (trial.yaml) 对齐的视觉分辨率, 消除训练/推理分辨率偏移
IMAGE_MAX_PIXELS = 786432
IMAGE_MIN_PIXELS = 512 * 512

MAX_NEW_TOKENS = 256

THIS_DIR = Path(__file__).resolve().parent
DATA_FILE = THIS_DIR / "data" / "direct_test.json"
OUT_DIR = THIS_DIR / "outputs" / "direct_full"
OUT_FILE = OUT_DIR / "predictions.jsonl"

TRAIN_LOG = THIS_DIR / "train.log"
POLL_INTERVAL = 60  # 秒
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


def training_process_alive() -> bool:
    try:
        out = subprocess.run(
            ["pgrep", "-f", "llamafactory-cli train"],
            capture_output=True,
            text=True,
        )
        return out.returncode == 0
    except Exception:
        return False


def wait_for_training():
    adapter = Path(ADAPTER_PATH) / "adapter_model.safetensors"
    print(f"[wait] 训练输出目录: {ADAPTER_PATH}")
    print(f"[wait] 轮询间隔: {POLL_INTERVAL}s, Ctrl+C 可中断")

    n_polls = 0
    while True:
        alive = training_process_alive()
        done = adapter.is_file()

        if n_polls % 5 == 0:
            print(
                f"[wait] {time.strftime('%H:%M:%S')} "
                f"训练进程运行中={alive}, 最终adapter已保存={done}"
            )

        if not alive:
            if done:
                print("[wait] 训练进程已退出且最终 adapter 已保存, 开始推理")
                return
            else:
                print(
                    "[wait] 警告: 训练进程已退出但未找到最终 adapter "
                    f"({adapter})。可能训练失败, 30s 后重新检查..."
                )
                time.sleep(30)
                if not training_process_alive() and not adapter.is_file():
                    raise RuntimeError(
                        "训练已结束但未保存最终 adapter, 中止推理。"
                        f"请检查训练日志: {TRAIN_LOG}"
                    )
                continue

        time.sleep(POLL_INTERVAL)
        n_polls += 1


def load_records(path: Path) -> list:
    """把 LLaMA-Factory 格式的 direct_test.json 转成推理用 records."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    assert isinstance(raw, list) and raw, f"empty/invalid data: {path}"

    records = []
    for i, r in enumerate(raw):
        assert "instruction" in r and "output" in r and "images" in r, f"missing keys at {i}"
        images = [os.path.join(str(THIS_DIR), p) for p in r["images"]]
        for p in images:
            assert os.path.isfile(p), f"image not found: {p}"
        records.append(
            {
                "index": i,
                "messages": [{"role": "user", "content": r["instruction"]}],
                "images": images,
                "ground_truth": r["output"],
            }
        )
    return records


def load_done_indices(out_file: Path):
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
                done.add(item["index"])
            except Exception:
                pass
    return done


def main():
    parser = argparse.ArgumentParser(description="等待训练完成后全量推理 direct_test.json")
    parser.add_argument("--no-wait", action="store_true", help="跳过等待训练")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--limit", type=int, default=0, help="0 = 全量")
    args = parser.parse_args()

    if not args.no_wait:
        wait_for_training()

    assert torch.cuda.is_available(), "CUDA unavailable"
    torch.cuda.set_device(0)
    print(f"[cuda] device: {torch.cuda.get_device_name(0)}")
    print(f"[torch] version: {torch.__version__}, cuda: {torch.version.cuda}")

    # ---------------------- dataset ----------------------
    records = load_records(DATA_FILE)
    if args.limit > 0:
        records = records[: args.limit]
    print(f"[data] {DATA_FILE} -> {len(records)} samples (全量)")

    done = load_done_indices(OUT_FILE)
    pending = [rec for rec in records if rec["index"] not in done]
    print(f"[data] done={len(done)} pending={len(pending)}")

    # ---------------------- model ----------------------
    model_args = dict(
        model_name_or_path=MODEL_PATH,
        adapter_name_or_path=ADAPTER_PATH,
        finetuning_type="lora",
        template=TEMPLATE,
        trust_remote_code=True,
        infer_backend="huggingface",
        infer_dtype=INFER_DTYPE,
        image_max_pixels=IMAGE_MAX_PIXELS,
        image_min_pixels=IMAGE_MIN_PIXELS,
        enable_thinking=False,
        flash_attn="fa2",
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": MODEL_PATH,
                "adapter": ADAPTER_PATH,
                "template": TEMPLATE,
                "samples": len(records),
                "max_new_tokens": args.max_new_tokens,
                "image_max_pixels": IMAGE_MAX_PIXELS,
                "image_min_pixels": IMAGE_MIN_PIXELS,
                "enable_thinking": False,
                "do_sample": False,
                "infer_dtype": INFER_DTYPE,
                "flash_attn": "fa2",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(f"[model] loading: {MODEL_PATH} + LoRA {ADAPTER_PATH}")
    print(f"[model] template={TEMPLATE} dtype={INFER_DTYPE} flash_attn=fa2 thinking=False")

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

    with open(OUT_FILE, "a", encoding="utf-8") as fout:
        pbar = tqdm(pending, total=len(pending), desc="direct_full", unit="img", dynamic_ncols=True)
        for rec in pbar:
            idx = rec["index"]
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
                "index": idx,
                "prompt": "direct",
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
    print(f"new={n_run} parse_fail={n_parse_fail} avg={elapsed_sum / max(n_run, 1):.2f}s/img")
    print(f"TOTAL inference time: {total_sec / 60:.1f} min ({total_sec / 3600:.2f} h)")
    print(f"Output: {OUT_FILE}")


if __name__ == "__main__":
    main()
