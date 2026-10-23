#!/usr/bin/env python3
"""patch-zoom 推理: 单张 224 局部块 -> 4 个 findings 的 0/1 概率.

沿用 b8 流水线 (infer_2view_batched.py) 的做法:
  - 贪心生成 + output_scores=True, 直接从每步 logits 取字段取值 token 的 softmax,
    不重复前向图像;
  - 图像像素预算由 --image-max-pixels / --image-min-pixels 固定为训练同款
    (0.74x=25600, 1x=50176, 2x=200704), 保证推理与训练分辨率一致.

输入: patch_zoom/meta_test.jsonl (id, labels, ...)
输出: {"id", "ground_truth", "greedy_prediction", "argmax_match", "probs", "probs_norm"}
      probs/probs_norm 为扁平键: mass / suspicious_calcification /
      asymmetry / architectural_distortion -> {"0": p0, "1": p1}
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import GenerationConfig

from llamafactory.chat import ChatModel

THIS_DIR = Path(__file__).resolve().parent
PROJ_DIR = THIS_DIR.parent.parent
MODEL_PATH = "/nesi/project/massey04900/Mammography/model/Qwen3.5-4B"
TEMPLATE = "qwen3_5"
PROMPT_FILE = PROJ_DIR / "prompts" / "patch_zoom_prompt.txt"
PATCH_DIR = Path("/nesi/project/massey04900/Mammography/patch_zoom")
FINDINGS = ["mass", "suspicious_calcification", "asymmetry", "architectural_distortion"]
FIELD_SPECS = {k: (re.compile(rf'"{k}"\s*:\s*([01])'), ["0", "1"]) for k in FINDINGS}
THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", flags=re.DOTALL)


def load_instruction(prompt_file):
    text = Path(prompt_file).read_text(encoding="utf-8").strip()
    assert text.count("<image>") == 1, f"prompt 需要恰好 1 个 <image>, 实际 {text.count('<image>')}"
    return text


def build_prompt(template, tokenizer, processor, instruction, images):
    messages = [{"role": "user", "content": instruction}]
    messages = template.mm_plugin.process_messages(messages, images, [], [], processor)
    paired = messages + [{"role": "assistant", "content": ""}]
    prompt_ids, _ = template.encode_oneturn(tokenizer, paired, None, None)
    prompt_ids, _ = template.mm_plugin.process_token_ids(
        prompt_ids, None, images, [], [], tokenizer, processor
    )
    return prompt_ids, images


def to_tensors(mm_inputs, model):
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


def parse_json_text(raw):
    text = THINK_RE.sub("", raw or "").strip()
    m = FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e == -1 or e <= s:
        return None
    try:
        return json.loads(text[s : e + 1])
    except Exception:
        return None


def common_prefix_len(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def field_probs_from_scores(tokenizer, gen_ids, pieces, score_rows):
    """与 score_2view / infer_2view_batched 同一算法: 在字段取值 token 位置取 softmax."""
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
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--scale", required=True, choices=["x074", "x1", "x2"])
    ap.add_argument("--data", default=str(PATCH_DIR / "meta_test.jsonl"))
    ap.add_argument("--prompt-file", default=str(PROMPT_FILE))
    ap.add_argument("--out-probs", required=True)
    ap.add_argument("--image-max-pixels", type=int, default=0)
    ap.add_argument("--image-min-pixels", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    BUDGET = {"x074": 25600, "x1": 50176, "x2": 200704}
    budget = BUDGET[args.scale]
    maxp = args.image_max_pixels or budget
    minp = args.image_min_pixels or budget

    assert torch.cuda.is_available(), "CUDA unavailable"
    torch.cuda.set_device(0)
    print(f"[cuda] {torch.cuda.get_device_name(0)}")

    instruction = load_instruction(args.prompt_file)
    recs = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    if args.limit > 0:
        recs = recs[: args.limit]
    for r in recs:
        r["images"] = [str(PATCH_DIR / "images" / "test" / f"{r['id']}.png")]
        r["ground_truth"] = r["labels"]

    out_probs = Path(args.out_probs)
    out_probs.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_probs.exists():
        for line in open(out_probs, encoding="utf-8"):
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    pending = [r for r in recs if r["id"] not in done]
    print(f"[data] total={len(recs)} done={len(done)} pending={len(pending)} "
          f"budget={minp}-{maxp} batch={args.batch_size}")

    model_args = dict(
        model_name_or_path=MODEL_PATH,
        template=TEMPLATE,
        trust_remote_code=True,
        infer_backend="huggingface",
        infer_dtype="bfloat16",
        image_max_pixels=maxp,
        image_min_pixels=minp,
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
    print(f"[model] loaded in {time.perf_counter() - t0:.1f}s")

    eos_ids = template.get_stop_token_ids(tokenizer)
    gen_cfg = GenerationConfig(
        do_sample=False, num_beams=1, repetition_penalty=1.0,
        max_new_tokens=96, eos_token_id=eos_ids, pad_token_id=tokenizer.pad_token_id,
    )
    pad_id = tokenizer.pad_token_id
    t_all = time.perf_counter()

    fout = open(out_probs, "a", encoding="utf-8")
    pbar = tqdm(total=len(pending), desc=f"patch_infer[{args.scale}]", unit="patch")
    for start in range(0, len(pending), args.batch_size):
        chunk = pending[start : start + args.batch_size]
        ids_list, imgs_list = [], []
        for rec in chunk:
            ids, imgs = build_prompt(template, tokenizer, processor, instruction, list(rec["images"]))
            ids_list.append(ids)
            imgs_list.append(imgs)

        maxlen = max(len(x) for x in ids_list)
        input_ids = torch.full((len(chunk), maxlen), pad_id, dtype=torch.long)
        attn = torch.zeros((len(chunk), maxlen), dtype=torch.long)
        for i, ids in enumerate(ids_list):
            input_ids[i, maxlen - len(ids) :] = torch.tensor(ids, dtype=torch.long)
            attn[i, maxlen - len(ids) :] = 1
        input_ids, attn = input_ids.to(model.device), attn.to(model.device)

        flat_images = [im for imgs in imgs_list for im in imgs]
        mm_inputs = template.mm_plugin.get_mm_inputs(
            images=flat_images, videos=[], audios=[],
            imglens=[len(x) for x in imgs_list], vidlens=[0] * len(chunk),
            audlens=[0] * len(chunk), batch_ids=ids_list, processor=processor,
        )
        mm_kwargs = to_tensors(mm_inputs, model)

        with torch.inference_mode():
            out = model.generate(
                input_ids=input_ids, attention_mask=attn, generation_config=gen_cfg,
                return_dict_in_generate=True, output_scores=True, **mm_kwargs,
            )

        seqs = out.sequences[:, maxlen:]
        scores = out.scores
        for i, rec in enumerate(chunk):
            gids = seqs[i].tolist()
            if eos_ids:
                for j, t in enumerate(gids):
                    if t in eos_ids:
                        gids = gids[:j]
                        break
            pieces = [tokenizer.decode([t], skip_special_tokens=True) for t in gids]
            raw = "".join(pieces).strip()
            rows = [s[i] for s in scores[: len(gids)]] if scores else []
            probs, probs_norm, argmax_match = field_probs_from_scores(tokenizer, gids, pieces, rows)
            fout.write(json.dumps({
                "id": rec["id"],
                "ground_truth": rec["ground_truth"],
                "greedy_prediction": parse_json_text(raw),
                "prediction_raw": raw,
                "argmax_match": argmax_match,
                "probs": probs,
                "probs_norm": probs_norm,
            }, ensure_ascii=False) + "\n")
        fout.flush()
        pbar.update(len(chunk))
        pbar.set_postfix(avg=f"{(time.perf_counter() - t_all) / max(pbar.n, 1):.2f}s")

    pbar.close()
    fout.close()
    total = time.perf_counter() - t_all
    print(f"[done] {len(pending)} patches in {total / 60:.1f} min -> {out_probs}")


if __name__ == "__main__":
    main()
