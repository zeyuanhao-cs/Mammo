#!/usr/bin/env python3
"""批量推理 + 打分合一（infer_2view_local.py + score_2view.py 的合并版）。

动机:
  - infer_2view_local.py 是 batch=1，每样本 ~2.9 s；
  - score_2view.py 为了取字段概率，会把「prompt + 模型自己生成的 completion」再前向一次，
    图片重新过一遍视觉编码器（每样本额外 ~0.8 s）。
  - 贪心生成时每步的 logits 与 teacher-forcing 在相同上下文下的 logits 是同一个东西，
    因此可以直接从 generate(output_scores=True) 里读字段概率，省掉第二次前向。

实现要点:
  - prompts 左 padding，batch_size 默认 8（tokenizer.padding_side 已由 engine 设为 left）；
  - 生成参数与原来完全一致: do_sample=False, num_beams=1, repetition_penalty=1.0,
    max_new_tokens 默认 256, 分辨率沿用 IMAGE_MAX/MIN_PIXELS；
  - 字段概率: 在生成的 token 序列里定位该字段取值的 token，取该步 softmax，
    候选取值 token 的解析方式与 score_2view.score_one 一致（context + 候选值 重新 tokenize，
    与原 prefix 求公共前缀）。

输出:
  - --out-pred:  与 infer_2view_local.py 同 schema 的预测 jsonl
  - --out-probs: 与 score_2view.py 同 schema 的概率 jsonl（可直接喂 summarize_probs.py）
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import GenerationConfig

import sys

THIS_DIR = Path(__file__).resolve().parent
PROJ_DIR = THIS_DIR.parent.parent
sys.path.insert(0, str(THIS_DIR))

from infer_2view_local import (  # noqa: E402
    IMAGE_MAX_PIXELS, IMAGE_MIN_PIXELS, MAX_NEW_TOKENS, MODELS,
    load_instruction, load_records, parse_prediction,
)
from score_2view import FIELD_SPECS, common_prefix_len, find_token_index  # noqa: E402
from llamafactory.chat import ChatModel  # noqa: E402

OUT_PRED_DIR = PROJ_DIR / "outputs" / "predictions"
OUT_PROB_DIR = PROJ_DIR / "outputs" / "scoring"


def build_prompt(template, tokenizer, processor, instruction, images):
    """与 engine._process_args / score_one 完全一致的 prompt 构建."""
    messages = [{"role": "user", "content": instruction}]
    messages = template.mm_plugin.process_messages(messages, images, [], [], processor)
    paired = messages + [{"role": "assistant", "content": ""}]
    prompt_ids, _ = template.encode_oneturn(tokenizer, paired, None, None)
    prompt_ids, _ = template.mm_plugin.process_token_ids(
        prompt_ids, None, images, [], [], tokenizer, processor
    )
    return prompt_ids, images


def to_tensors(mm_inputs, model):
    """engine._process_args 里的 mm_inputs -> tensor 转换."""
    mm_kwargs = {}
    for key, value in mm_inputs.items():
        if isinstance(value, list) and isinstance(value[0], torch.Tensor):
            value = torch.stack(value)
        elif isinstance(value, list) and isinstance(value[0], list) and isinstance(value[0][0], torch.Tensor):
            value = torch.stack([torch.stack(v) for v in value])
        elif not isinstance(value, torch.Tensor):
            value = torch.tensor(value)

        if torch.is_floating_point(value):
            value = value.to(model.dtype)

        if key == "second_per_grid_ts":
            mm_kwargs[key] = value.tolist()
        else:
            mm_kwargs[key] = value.to(model.device)
    return mm_kwargs


def span_of_token(pieces):
    spans, pos = [], 0
    for p in pieces:
        spans.append((pos, pos + len(p)))
        pos += len(p)
    return spans


def field_probs_from_scores(tokenizer, gen_ids, pieces, score_rows):
    """用生成时的逐步 logits 复刻 score_one 的字段概率."""
    text = "".join(pieces)
    spans = span_of_token(pieces)
    probs, probs_norm, argmax_match = {}, {}, {}

    def tok_index_of(char_pos):
        for i, (s, e) in enumerate(spans):
            if s <= char_pos < e:
                return i
        return None

    for field, (regex, cands) in FIELD_SPECS.items():
        m = regex.search(text)
        tok_idx = tok_index_of(m.start(1)) if m else None
        if tok_idx is None or tok_idx >= len(score_rows):
            probs[field] = None
            probs_norm[field] = None
            argmax_match[field] = None
            continue

        row = score_rows[tok_idx].float()
        sm = torch.softmax(row, dim=-1)
        argmax_match[field] = int(row.argmax()) == int(gen_ids[tok_idx])

        context_str = text[: spans[tok_idx][0]]
        prefix_ids = gen_ids[:tok_idx]
        p = {}
        for c in cands:
            full = tokenizer(context_str + c, add_special_tokens=False)["input_ids"]
            common = common_prefix_len(full, prefix_ids)
            cand_toks = full[common:]
            if not cand_toks:
                p[c] = 0.0
                continue
            p[c] = float(sm[cand_toks[0]])

        probs[field] = p
        z = sum(p.values())
        probs_norm[field] = {c: (v / z if z > 0 else 0.0) for c, v in p.items()} if z > 0 else None

    return probs, probs_norm, argmax_match


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-4b", choices=list(MODELS.keys()))
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--prompt-file", default=str(PROJ_DIR / "dataset_build" / "prompt.txt"))
    ap.add_argument("--out-suffix", required=True)
    ap.add_argument("--out-pred", default=None)
    ap.add_argument("--out-probs", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--tag", default="")
    ap.add_argument("--image-max-pixels", type=int, default=IMAGE_MAX_PIXELS,
                    help=f"视觉最长边像素预算, 默认 {IMAGE_MAX_PIXELS}; 1386240=912x1520 原生")
    ap.add_argument("--image-min-pixels", type=int, default=IMAGE_MIN_PIXELS,
                    help=f"视觉最短边像素预算, 默认 {IMAGE_MIN_PIXELS}")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA unavailable"
    torch.cuda.set_device(0)

    instruction = load_instruction(Path(args.prompt_file))
    records = load_records(Path(args.data), instruction)
    if args.limit > 0:
        records = records[: args.limit]

    out_pred = Path(args.out_pred) if args.out_pred else (
        OUT_PRED_DIR / f"predictions_{args.model}{args.out_suffix}_{len(records)}.jsonl"
    )
    out_probs = Path(args.out_probs) if args.out_probs else (
        OUT_PROB_DIR / f"{args.out_suffix.lstrip('_')}_with_probs.jsonl"
    )
    out_pred.parent.mkdir(parents=True, exist_ok=True)
    out_probs.parent.mkdir(parents=True, exist_ok=True)

    done_pred = set()
    if out_pred.exists():
        for line in open(out_pred, encoding="utf-8"):
            try:
                done_pred.add(json.loads(line)["id"])
            except Exception:
                pass
    done_probs = set()
    if out_probs.exists():
        for line in open(out_probs, encoding="utf-8"):
            try:
                done_probs.add(json.loads(line)["id"])
            except Exception:
                pass

    pending = [r for r in records if r["id"] not in done_pred or r["id"] not in done_probs]
    print(f"[data] total={len(records)} pending={len(pending)} batch={args.batch_size}")

    model_args = dict(
        model_name_or_path=MODELS[args.model]["path"],
        template=MODELS[args.model]["template"],
        trust_remote_code=True,
        infer_backend="huggingface",
        infer_dtype="bfloat16",
        image_max_pixels=args.image_max_pixels,
        image_min_pixels=args.image_min_pixels,
        enable_thinking=False,
        flash_attn="sdpa",
        adapter_name_or_path=args.adapter,
        finetuning_type="lora",
    )
    t0 = time.perf_counter()
    cm = ChatModel(args=model_args)
    engine = cm.engine
    model, tokenizer, processor, template = engine.model, engine.tokenizer, engine.processor, engine.template
    torch.cuda.synchronize()
    print(f"[model] loaded in {time.perf_counter() - t0:.1f}s (adapter={args.adapter})")

    eos_ids = template.get_stop_token_ids(tokenizer)
    gen_cfg = GenerationConfig(
        do_sample=False,
        num_beams=1,
        repetition_penalty=1.0,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=eos_ids,
        pad_token_id=tokenizer.pad_token_id,
    )

    pad_id = tokenizer.pad_token_id
    t_all = time.perf_counter()
    n_cases = 0
    n_batch = 0
    t_gen = 0.0

    fpred = open(out_pred, "a", encoding="utf-8")
    fprobs = open(out_probs, "a", encoding="utf-8")

    pbar = tqdm(total=len(pending), desc=f"batch_infer[{args.tag}]", unit="case", dynamic_ncols=True)
    for start in range(0, len(pending), args.batch_size):
        chunk = pending[start : start + args.batch_size]
        ids_list, imgs_list = [], []
        for rec in chunk:
            ids, imgs = build_prompt(template, tokenizer, processor, instruction, list(rec["images"]))
            ids_list.append(ids)
            imgs_list.append(imgs)

        lens = [len(x) for x in ids_list]
        maxlen = max(lens)
        input_ids = torch.full((len(chunk), maxlen), pad_id, dtype=torch.long)
        attn = torch.zeros((len(chunk), maxlen), dtype=torch.long)
        for i, ids in enumerate(ids_list):
            input_ids[i, maxlen - len(ids) :] = torch.tensor(ids, dtype=torch.long)
            attn[i, maxlen - len(ids) :] = 1
        input_ids = input_ids.to(model.device)
        attn = attn.to(model.device)

        flat_images = [im for imgs in imgs_list for im in imgs]
        mm_inputs = template.mm_plugin.get_mm_inputs(
            images=flat_images, videos=[], audios=[],
            imglens=[len(x) for x in imgs_list], vidlens=[0] * len(chunk), audlens=[0] * len(chunk),
            batch_ids=ids_list, processor=processor,
        )
        mm_kwargs = to_tensors(mm_inputs, model)

        t1 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                generation_config=gen_cfg,
                return_dict_in_generate=True,
                output_scores=True,
                **mm_kwargs,
            )
        torch.cuda.synchronize()
        t_gen += time.perf_counter() - t1

        seqs = out.sequences[:, maxlen:]
        scores = out.scores  # tuple(len=steps) of [B, V]

        for i, rec in enumerate(chunk):
            gids = seqs[i].tolist()
            if eos_ids:
                for j, t in enumerate(gids):
                    if t in eos_ids:
                        gids = gids[:j]
                        break
            pieces = [tokenizer.decode([t], skip_special_tokens=True) for t in gids]
            raw = "".join(pieces).strip()
            pred_json = parse_prediction(raw)

            rows = [s[i] for s in scores[: len(gids)]] if scores else []
            probs, probs_norm, argmax_match = field_probs_from_scores(
                tokenizer, gids, pieces, rows
            )

            fpred.write(json.dumps({
                "id": rec["id"],
                "index": rec["index"],
                "prediction_raw": raw,
                "prediction_json": pred_json,
                "ground_truth": rec["ground_truth"],
                "seconds": None,
                "batch_size": len(chunk),
            }, ensure_ascii=False) + "\n")
            fprobs.write(json.dumps({
                "id": rec["id"],
                "ground_truth": rec["ground_truth"],
                "greedy_prediction": pred_json,
                "argmax_match": argmax_match,
                "probs": probs,
                "probs_norm": probs_norm,
            }, ensure_ascii=False) + "\n")

        fpred.flush()
        fprobs.flush()
        n_batch += 1
        n_cases += len(chunk)
        pbar.update(len(chunk))
        pbar.set_postfix(gen=f"{t_gen / n_cases:.2f}s/case")

    pbar.close()
    fpred.close()
    fprobs.close()

    total = time.perf_counter() - t_all
    print()
    print(f"[done] cases={n_cases} batches={n_batch} total={total / 60:.1f} min "
          f"({total / 3600:.2f} h), gen={t_gen:.1f}s -> {t_gen / max(n_cases, 1):.2f}s/case "
          f"(batch={args.batch_size})")
    print(f"[out] pred={out_pred}")
    print(f"[out] probs={out_probs}")


if __name__ == "__main__":
    main()
