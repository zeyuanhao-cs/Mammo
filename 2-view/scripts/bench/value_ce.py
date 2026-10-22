#!/usr/bin/env python3
"""取值 token 的 teacher-forcing 平均 CE loss。
GT completion 用训练时的输出格式 (density, findings, breast_birads)。
只统计 6 个字段取值 token 的 CE (每个字段取其值首个 token), 输出每样本每字段 CE 与均值。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

PROJ = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJ / "scripts" / "vlm"))
from score_2view import FIELD_SPECS, find_token_index, load_instruction  # noqa: E402
from llamafactory.chat import ChatModel  # noqa: E402

FINDINGS = ["mass", "suspicious_calcification", "asymmetry", "architectural_distortion"]
IMAGE_MAX_PIXELS = 786432
IMAGE_MIN_PIXELS = 512 * 512
TEMPLATE = "qwen3_5"


def gt_text(out: dict) -> str:
    return json.dumps({
        "breast_density": out["breast_density"],
        "findings": {k: int(out["findings"][k]) for k in FINDINGS},
        "breast_birads": int(out["breast_birads"]),
    }, ensure_ascii=False)


def build_prompt(template, tokenizer, processor, instruction, images):
    messages = [{"role": "user", "content": instruction}]
    messages = template.mm_plugin.process_messages(messages, images, [], [], processor)
    paired = messages + [{"role": "assistant", "content": ""}]
    prompt_ids, _ = template.encode_oneturn(tokenizer, paired, None, None)
    prompt_ids, _ = template.mm_plugin.process_token_ids(prompt_ids, None, images, [], [], tokenizer, processor)
    return prompt_ids


def to_mm_kwargs(template, processor, model, images, prompt_ids, input_ids):
    mm = template.mm_plugin.get_mm_inputs(
        images=images, videos=[], audios=[], imglens=[len(images)], vidlens=[0], audlens=[0],
        batch_ids=[prompt_ids], processor=processor,
    )
    kw = {}
    for key, value in mm.items():
        if isinstance(value, list) and isinstance(value[0], torch.Tensor):
            value = torch.stack(value)
        elif isinstance(value, list) and isinstance(value[0], list) and isinstance(value[0][0], torch.Tensor):
            value = torch.stack([torch.stack(v) for v in value])
        elif not isinstance(value, torch.Tensor):
            value = torch.tensor(value)
        if torch.is_floating_point(value):
            value = value.to(model.dtype)
        kw[key] = value.tolist() if key == "second_per_grid_ts" else value.to(model.device)
    if hasattr(processor, "create_mm_token_type_ids"):
        mtt = processor.create_mm_token_type_ids([input_ids])
        if not isinstance(mtt, torch.Tensor):
            mtt = torch.tensor(mtt)
        kw["mm_token_type_ids"] = mtt.to(model.device)
    return kw


def value_ce(model, tokenizer, processor, template, instruction, rec):
    images = list(rec["images"])
    prompt_ids = build_prompt(template, tokenizer, processor, instruction, images)
    prompt_len = len(prompt_ids)

    completion = gt_text(rec["output"])
    enc = tokenizer(completion, add_special_tokens=False, return_offsets_mapping=True)
    comp_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]

    input_ids = prompt_ids + comp_ids
    ids_t = torch.tensor([input_ids], device=model.device)
    attn = torch.ones_like(ids_t)
    mm_kwargs = to_mm_kwargs(template, processor, model, images, prompt_ids, input_ids)

    K = len(comp_ids) + 1
    with torch.inference_mode():
        try:
            logits = model(input_ids=ids_t, attention_mask=attn, use_cache=False,
                           logits_to_keep=K, **mm_kwargs).logits[0]
        except TypeError:
            logits = model(input_ids=ids_t, attention_mask=attn, use_cache=False,
                           num_logits_to_keep=K, **mm_kwargs).logits[0]
    logits = logits.float()
    assert logits.shape[0] == K, (logits.shape, K)

    logp = torch.log_softmax(logits, dim=-1)
    per_field = {}
    for field, (regex, _cands) in FIELD_SPECS.items():
        m = regex.search(completion)
        if not m:
            per_field[field] = None
            continue
        tok = find_token_index(offsets, m.start(1))
        if tok is None:
            per_field[field] = None
            continue
        # 行 r 对应 seq 位置 prompt_len-1+r, 预测 completion token r
        val = -float(logp[tok, comp_ids[tok]])
        per_field[field] = val
    vals = [v for v in per_field.values() if v is not None]
    return per_field, (sum(vals) / len(vals) if vals else None), prompt_len, len(comp_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model-path", default="/home/z/zhao/Mammography/model/Qwen3.5-4B")
    ap.add_argument("--prompt-file", default=str(PROJ / "dataset_build" / "prompt.txt"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    instruction = load_instruction(Path(args.prompt_file))
    raw = json.load(open(args.data, encoding="utf-8"))
    if args.limit > 0:
        raw = raw[: args.limit]

    out_path = Path(args.out)
    done = set()
    if out_path.exists():
        for line in open(out_path, encoding="utf-8"):
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    pending = [r for r in raw if r["id"] not in done]
    print(f"[data] n={len(raw)} done={len(done)} pending={len(pending)}", flush=True)

    model_args = dict(
        model_name_or_path=args.model_path, template=TEMPLATE, trust_remote_code=True,
        infer_backend="huggingface", infer_dtype="bfloat16",
        image_max_pixels=IMAGE_MAX_PIXELS, image_min_pixels=IMAGE_MIN_PIXELS,
        enable_thinking=False, flash_attn="sdpa",
        adapter_name_or_path=args.adapter, finetuning_type="lora",
    )
    t0 = time.perf_counter()
    cm = ChatModel(args=model_args)
    eng = cm.engine
    model, tokenizer, processor, template = eng.model, eng.tokenizer, eng.processor, eng.template
    torch.cuda.synchronize()
    print(f"[model] loaded {time.perf_counter() - t0:.1f}s", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()
    with open(out_path, "a", encoding="utf-8") as f:
        for i, rec in enumerate(pending):
            per_field, mean_ce, plen, clen = value_ce(
                model, tokenizer, processor, template, instruction, rec)
            f.write(json.dumps({"id": rec["id"], "ce": per_field, "mean_ce": mean_ce,
                                "prompt_len": plen, "comp_len": clen}, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 50 == 0:
                print(f"[{args.tag}] {i+1}/{len(pending)} "
                      f"{(time.perf_counter() - t_all) / (i + 1):.2f}s/case", flush=True)
    print(f"[done] {args.tag} n={len(pending)} {time.perf_counter() - t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
