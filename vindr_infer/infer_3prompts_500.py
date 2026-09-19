#!/usr/bin/env python3
"""
VinDr-Mammo: sequential inference for Direct / ICL / CoT with one model load.

Expected files in the same directory:
  infer_data.direct.test.json
  infer_data.icl.test.json
  infer_data.cot.test.json

Default: run first 500 samples from each file (1500 total).
Outputs:
  outputs_500/direct/predictions.jsonl
  outputs_500/icl/predictions.jsonl
  outputs_500/cot/predictions.jsonl

Features:
  - Load model only once
  - tqdm progress bars (per prompt + global)
  - Resume from existing JSONL outputs
  - JSON extraction / parse fallback
  - BF16 + FlashAttention2 by default
  - Qwen3.5 template with thinking disabled

Examples:
  python3 infer_3prompts_500.py
  python3 infer_3prompts_500.py --limit 500
  python3 infer_3prompts_500.py --no-flash-attn
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
MODEL_PATH = "/hy-tmp/9_9/Qwen/Qwen3.5-4B"
TEMPLATE = "qwen3_5"
INFER_DTYPE = "bfloat16"

# Keep the same visual resolution across Direct / ICL / CoT.
IMAGE_MAX_PIXELS = 1536 * 1536
IMAGE_MIN_PIXELS = 512 * 512

MAX_NEW_TOKENS = 256
DEFAULT_LIMIT = 500

THIS_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = THIS_DIR / "outputs_500"

DATA_FILES = {
    "direct": THIS_DIR / "infer_data.direct.test.json",
    "icl": THIS_DIR / "infer_data.icl.test.json",
    "cot": THIS_DIR / "infer_data.cot.test.json",
}

# Run simpler / faster prompts first; ICL last is also fine, but this order
# makes it easy to compare Direct -> ICL -> CoT in one pass.
# PROMPT_ORDER = ["direct", "icl", "cot"]
PROMPT_ORDER = ["icl"]

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


def get_eval_json(pred_json):
    """Support both normal schema and optional {reasoning, final_json} CoT."""
    if isinstance(pred_json, dict) and isinstance(pred_json.get("final_json"), dict):
        return pred_json["final_json"]
    return pred_json


def validate_record(rec: dict):
    assert "messages" in rec and isinstance(rec["messages"], list), "missing messages"
    assert "images" in rec and isinstance(rec["images"], list), "missing images"
    assert "ground_truth" in rec, "missing ground_truth"
    assert len(rec["messages"]) > 0, "messages is empty"

    content = rec["messages"][0].get("content", "")
    n_placeholders = content.count("<image>")
    n_images = len(rec["images"])

    assert n_placeholders == n_images, (
        f"index={rec.get('index')} <image> count={n_placeholders} "
        f"but images count={n_images}"
    )


def load_records(path: Path, limit: int):
    assert path.exists(), f"data file not found: {path}"

    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)

    assert isinstance(records, list) and records, f"empty/invalid data: {path}"

    if limit > 0:
        records = records[:limit]

    for rec in records[: min(10, len(records))]:
        validate_record(rec)

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
                # Ignore broken/truncated lines instead of killing resume.
                pass
    return done


def gpu_info():
    print(f"[cuda] device: {torch.cuda.get_device_name(0)}")
    print(f"[torch] version: {torch.__version__}")
    print(f"[torch] cuda: {torch.version.cuda}")


def infer_one_prompt(
    chat_model,
    prompt_name: str,
    records: list,
    output_root: Path,
    max_new_tokens: int,
    fsync_every: int,
    global_bar,
):
    # out_dir = output_root / prompt_name
    out_dir = output_root / ("new_icl" if prompt_name == "icl" else prompt_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / "predictions.jsonl"
    done = load_done_indices(out_file)

    pending = [rec for rec in records if rec.get("index") not in done]

    print()
    print(f"[{prompt_name}] data={DATA_FILES[prompt_name]}")
    print(f"[{prompt_name}] selected={len(records)} done={len(done)} pending={len(pending)}")
    print(f"[{prompt_name}] output={out_file}")

    if not pending:
        print(f"[{prompt_name}] nothing to do")
        global_bar.update(len(records))
        return {
            "prompt": prompt_name,
            "selected": len(records),
            "new": 0,
            "parse_fail": 0,
            "avg_sec": 0.0,
            "out_file": str(out_file),
        }

    n_run = 0
    n_parse_fail = 0
    elapsed_sum = 0.0

    # If resuming, count already-completed records toward the global bar.
    global_bar.update(len(records) - len(pending))

    with open(out_file, "a", encoding="utf-8") as fout:
        pbar = tqdm(
            pending,
            total=len(pending),
            desc=f"{prompt_name:>6}",
            unit="img",
            dynamic_ncols=True,
            leave=True,
        )

        for rec in pbar:
            idx = rec["index"]
            validate_record(rec)
            t0 = time.perf_counter()

            try:
                responses = chat_model.chat(
                    messages=rec["messages"],
                    images=rec["images"],
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                )
                pred_raw = responses[0].response_text.strip()
                pred_json = parse_prediction(pred_raw)
                eval_json = get_eval_json(pred_json)
                error = None
            except Exception as e:
                pred_raw = ""
                pred_json = None
                eval_json = None
                error = repr(e)

            sec = time.perf_counter() - t0
            elapsed_sum += sec
            n_run += 1

            if eval_json is None:
                n_parse_fail += 1

            record = {
                "index": idx,
                "image_id": rec.get("image_id"),
                "image_path": rec.get("image_path"),
                "prompt": prompt_name,
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

            if fsync_every > 0 and n_run % fsync_every == 0:
                os.fsync(fout.fileno())

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
            global_bar.update(1)

        fout.flush()
        os.fsync(fout.fileno())

    avg_sec = elapsed_sum / max(n_run, 1)
    print(
        f"[{prompt_name}] finished: new={n_run}, parse_fail={n_parse_fail}, "
        f"avg={avg_sec:.2f}s/img"
    )

    return {
        "prompt": prompt_name,
        "selected": len(records),
        "new": n_run,
        "parse_fail": n_parse_fail,
        "avg_sec": avg_sec,
        "out_file": str(out_file),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run Direct + ICL + CoT inference sequentially with one model load"
    )
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="samples per prompt")
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--no-flash-attn", action="store_true")
    parser.add_argument(
        "--fsync-every",
        type=int,
        default=10,
        help="fsync every N newly written rows; 0 = only at prompt end",
    )
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA unavailable"
    torch.cuda.set_device(0)
    gpu_info()

    # ---------------------- Load all three datasets ----------------------
    datasets = {}
    for prompt_name in PROMPT_ORDER:
        datasets[prompt_name] = load_records(DATA_FILES[prompt_name], args.limit)
        print(
            f"[data] {prompt_name}: {DATA_FILES[prompt_name].name} "
            f"-> {len(datasets[prompt_name])} samples"
        )

    total_selected = sum(len(v) for v in datasets.values())
    print(f"[data] total selected={total_selected}")

    # ---------------------- Model config ----------------------
    model_args = dict(
        model_name_or_path=args.model,
        template=TEMPLATE,
        trust_remote_code=True,
        infer_backend="huggingface",
        finetuning_type="full",
        infer_dtype=INFER_DTYPE,
        image_max_pixels=IMAGE_MAX_PIXELS,
        image_min_pixels=IMAGE_MIN_PIXELS,
        enable_thinking=False,
        flash_attn="disabled" if args.no_flash_attn else "fa2",
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    run_config = {
        "model": args.model,
        "template": TEMPLATE,
        "prompts": PROMPT_ORDER,
        "limit_per_prompt": args.limit,
        "max_new_tokens": args.max_new_tokens,
        "image_max_pixels": IMAGE_MAX_PIXELS,
        "image_min_pixels": IMAGE_MIN_PIXELS,
        "enable_thinking": False,
        "do_sample": False,
        "infer_dtype": INFER_DTYPE,
        "flash_attn": model_args["flash_attn"],
        "files": {k: str(v) for k, v in DATA_FILES.items()},
    }
    with open(OUTPUT_ROOT / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    # ---------------------- Load model once ----------------------
    print()
    print(f"[model] loading: {args.model}")
    print(
        f"[model] template={TEMPLATE} dtype={INFER_DTYPE} "
        f"flash_attn={model_args['flash_attn']} thinking=False"
    )
    print(
        f"[image] min_pixels={IMAGE_MIN_PIXELS} max_pixels={IMAGE_MAX_PIXELS}"
    )

    t_model = time.perf_counter()
    chat_model = ChatModel(args=model_args)
    torch.cuda.synchronize()
    print(f"[model] loaded in {time.perf_counter() - t_model:.1f}s")

    # ---------------------- Inference ----------------------
    summaries = []
    t_all = time.perf_counter()

    with tqdm(
        total=total_selected,
        desc=" TOTAL",
        unit="img",
        dynamic_ncols=True,
        position=0,
    ) as global_bar:
        for prompt_name in PROMPT_ORDER:
            summary = infer_one_prompt(
                chat_model=chat_model,
                prompt_name=prompt_name,
                records=datasets[prompt_name],
                output_root=OUTPUT_ROOT,
                max_new_tokens=args.max_new_tokens,
                fsync_every=args.fsync_every,
                global_bar=global_bar,
            )
            summaries.append(summary)

    total_sec = time.perf_counter() - t_all

    print("\n==================== SUMMARY ====================")
    for s in summaries:
        print(
            f"{s['prompt']:>6}: selected={s['selected']:4d} "
            f"new={s['new']:4d} parse_fail={s['parse_fail']:3d} "
            f"avg={s['avg_sec']:.2f}s/img"
        )
        print(f"       -> {s['out_file']}")

    print(f"TOTAL inference time: {total_sec / 60:.1f} min ({total_sec / 3600:.2f} h)")
    print(f"Outputs: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
