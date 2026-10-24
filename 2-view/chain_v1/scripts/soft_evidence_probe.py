#!/usr/bin/env python3
"""软证据探针（chain_v1）。

每样本三次前向（全部在本脚本内完成）：
  hard   : 模型自生成 JSON，不替换
  soft   : 同一份 JSON，把 4 个 finding 取值 token **本身所在位置**的输入嵌入换成 (1-p)·E[0]+p·E[1]
  teacher: GT 征象构成的前缀
soft vs hard 的所有比较只用本脚本的 hard / soft；b8 记录的概率只用于提供 p，
以及做一个自检（hard 与 b8 的 BI-RADS argmax 一致率、平均绝对差）。

位置约定：
  - 替换位置 = 序列中该取值 token 的**自身位置**（seq = len(prompt) + comp_idx），
    即被后续 token 看到的那个 hidden state（decoder 第 0 层入口）。
  - 读取 BI-RADS 分布的行 = 序列中 BI-RADS 取值 token 的自身位置（该行预测该 token）。
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch

PROJ = Path(__file__).resolve().parent.parent.parent  # chain_v1/scripts -> 2-view
sys.path.insert(0, str(PROJ / "scripts" / "vlm"))
from score_2view import FIELD_SPECS, find_token_index, load_instruction  # noqa: E402
from llamafactory.chat import ChatModel  # noqa: E402

FIND = ["mass", "suspicious_calcification", "asymmetry", "architectural_distortion"]
TEMPLATE = "qwen3_5"
IMAGE_MAX_PIXELS = 786432
IMAGE_MIN_PIXELS = 512 * 512
BIRADS_IDS = [16, 17, 18, 19, 20]   # '1'..'5'
ID0, ID1 = 15, 16                   # '0' / '1'


def build_prompt(template, tokenizer, processor, instruction, images):
    messages = [{"role": "user", "content": instruction}]
    messages = template.mm_plugin.process_messages(messages, images, [], [], processor)
    paired = messages + [{"role": "assistant", "content": ""}]
    pids, _ = template.encode_oneturn(tokenizer, paired, None, None)
    pids, _ = template.mm_plugin.process_token_ids(pids, None, images, [], [], tokenizer, processor)
    return pids


def to_mm_kwargs(template, processor, model, images, prompt_ids):
    mm = template.mm_plugin.get_mm_inputs(
        images=images, videos=[], audios=[], imglens=[len(images)], vidlens=[0], audlens=[0],
        batch_ids=[prompt_ids], processor=processor)
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
    return kw


def gt_text(gt):
    return json.dumps({
        "breast_density": gt["breast_density"],
        "findings": {k: int(gt["findings"][k]) for k in FIND},
        "breast_birads": int(gt["breast_birads"]),
    }, ensure_ascii=False)


def field_token_pos(text, offsets):
    out = {}
    for field, (pat, _c) in FIELD_SPECS.items():
        m = re.search(pat, text)
        out[field] = find_token_index(offsets, m.start(1)) if m else None
    return out


def forward_birads(model, tokenizer, processor, layer0, mm_kwargs, prompt_ids, text, soft=None):
    """soft: {field: p}，替换该字段取值 token 自身位置；返回 (5 维分布, 位置 dict, comp_ids, offsets)"""
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    comp_ids, offs = enc["input_ids"], enc["offset_mapping"]
    pos = field_token_pos(text, offs)
    if pos.get("breast_birads") is None:
        return None, pos, comp_ids, offs

    input_ids = prompt_ids + comp_ids
    L = len(input_ids)
    K = len(comp_ids) + 1
    ids_t = torch.tensor([input_ids], device=model.device)
    attn = torch.ones_like(ids_t)
    kw = dict(mm_kwargs)
    if hasattr(processor, "create_mm_token_type_ids"):
        mtt = processor.create_mm_token_type_ids([input_ids])
        if not isinstance(mtt, torch.Tensor):
            mtt = torch.tensor(mtt)
        kw["mm_token_type_ids"] = mtt.to(model.device)

    handle = None
    if soft:
        emb_w = model.get_input_embeddings().weight
        e0, e1 = emb_w[ID0].detach(), emb_w[ID1].detach()
        plan = [(len(prompt_ids) + pos[f], float(p)) for f, p in soft.items()
                if pos.get(f) is not None and pos[f] < len(comp_ids)]

        def hook(module, args):
            hs = args[0].clone()
            for seq_pos, p in plan:
                hs[0, seq_pos] = ((1.0 - p) * e0 + p * e1).to(hs.dtype)
            return (hs,) + tuple(args[1:])

        handle = layer0.register_forward_pre_hook(hook)

    # position_ids 修复（同 soft_decode.rope_prefill）：多图 prefill 必须显式传
    # text arange + get_rope_index 视觉位置，否则与 b8(generate) 管线 rope 不一致
    # （实测概率差可达 0.148）。
    B, L = ids_t.shape
    text_pos = torch.arange(L, device=ids_t.device).view(1, -1).expand(B, -1)
    grid, mtti = kw.get("image_grid_thw"), kw.get("mm_token_type_ids")
    if mtti is not None and grid is not None:
        vis_pos, _ = model.model.get_rope_index(ids_t, mtti, grid, None, attention_mask=attn)
    else:
        vis_pos = text_pos.unsqueeze(0).expand(3, -1, -1)
    pos4 = torch.cat([text_pos.unsqueeze(0), vis_pos], dim=0)

    try:
        with torch.inference_mode():
            try:
                out = model(input_ids=ids_t, attention_mask=attn, position_ids=pos4,
                            use_cache=False, logits_to_keep=K, **kw)
            except TypeError:
                out = model(input_ids=ids_t, attention_mask=attn, position_ids=pos4,
                            use_cache=False, num_logits_to_keep=K, **kw)
    finally:
        if handle is not None:
            handle.remove()

    row = out.logits[0, pos["breast_birads"]].float()
    five = torch.softmax(row[BIRADS_IDS], dim=-1)
    return [round(float(x), 8) for x in five], pos, comp_ids, offs


def print_context(text, comp_ids, offs, pos, prompt_len, tag, fields):
    """打印每个被替换位置前后各 3 个 token 的解码文本。"""
    print(f"  [{tag}] prompt_len={prompt_len}, comp_len={len(comp_ids)}")
    for f in fields:
        i = pos.get(f)
        if i is None:
            print(f"    {f}: 位置未找到")
            continue
        lo, hi = max(0, i - 3), min(len(comp_ids), i + 4)
        ctx = []
        for j in range(lo, hi):
            mark = ">>" if j == i else "  "
            s, e = offs[j]
            ctx.append(f"{mark}[{j}]{text[s:e]!r}")
        print(f"    {f}: comp_idx={i} seq_pos={prompt_len + i} 值token={text[offs[i][0]:offs[i][1]]!r} "
              f"tok_id={comp_ids[i]}")
        print(f"        上下文: " + " ".join(ctx))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--probs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--print-context", type=int, default=0, help="前 N 条打印替换位置上下文")
    ap.add_argument("--sens-out", default="")
    ap.add_argument("--sens-n", type=int, default=20)
    args = ap.parse_args()

    instruction = load_instruction(Path(PROJ / "dataset_build" / "prompt.txt"))
    data = {r["id"]: r for r in json.load(open(args.data, encoding="utf-8"))}
    b8 = {}
    for line in open(args.probs, encoding="utf-8"):
        line = line.strip()
        if line:
            d = json.loads(line)
            b8[d["id"]] = d
    ids = [i for i in data if i in b8]
    if args.limit:
        ids = ids[: args.limit]

    out_path = Path(args.out)
    done = set()
    if out_path.exists():
        for line in open(out_path, encoding="utf-8"):
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    pending = [i for i in ids if i not in done]
    print(f"[{args.split}] total={len(ids)} done={len(done)} pending={len(pending)}", flush=True)

    model_args = dict(
        model_name_or_path="/home/z/zhao/Mammography/model/Qwen3.5-4B", template=TEMPLATE,
        trust_remote_code=True, infer_backend="huggingface", infer_dtype="bfloat16",
        image_max_pixels=IMAGE_MAX_PIXELS, image_min_pixels=IMAGE_MIN_PIXELS,
        enable_thinking=False, flash_attn="sdpa",
        adapter_name_or_path=args.adapter, finetuning_type="lora")
    cm = ChatModel(args=model_args)
    eng = cm.engine
    model, tokenizer, processor, template = eng.model, eng.tokenizer, eng.processor, eng.template
    layer0 = model.model.language_model.layers[0]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    agree = 0
    mad_sum = 0.0
    mad_max = 0.0
    n_chk = 0

    with open(out_path, "a", encoding="utf-8") as f:
        for n, sid in enumerate(pending):
            rec = data[sid]
            b = b8[sid]
            pn = b["probs_norm"]
            p = {f"findings.{k}": float(pn[f"findings.{k}"]["1"]) for k in FIND}
            greedy = b.get("greedy_prediction") or {}
            soft_text = json.dumps(greedy, ensure_ascii=False)
            teacher_text = gt_text(b["ground_truth"])

            images = list(rec["images"])
            pids = build_prompt(template, tokenizer, processor, instruction, images)
            mmk = to_mm_kwargs(template, processor, model, images, pids)

            soft5, pos, comp_ids, offs = forward_birads(
                model, tokenizer, processor, layer0, mmk, pids, soft_text, soft=p)
            hard5, _, _, _ = forward_birads(
                model, tokenizer, processor, layer0, mmk, pids, soft_text, soft=None)
            teach5, _, _, _ = forward_birads(
                model, tokenizer, processor, layer0, mmk, pids, teacher_text, soft=None)

            if n < args.print_context:
                print(f"\n[sample {n}] {sid}  greedy={soft_text}")
                print_context(soft_text, comp_ids, offs, pos, len(pids), "soft", list(p.keys()))

            b8_birads = [float(pn["breast_birads"][str(i)]) for i in range(1, 6)]
            if hard5 and b8_birads:
                n_chk += 1
                agree += int(max(range(5), key=lambda i: hard5[i]) == max(range(5), key=lambda i: b8_birads[i]))
                d = sum(abs(x - y) for x, y in zip(hard5, b8_birads)) / 5
                mad_sum += d
                mad_max = max(mad_max, d)

            rec_out = {
                "id": sid,
                "p_findings": p,
                "hard_birads": hard5,
                "soft_birads": soft5,
                "teacher_birads": teach5,
                "b8_birads": b8_birads,
                "gt_birads": int(b["ground_truth"]["breast_birads"]),
                "gt_findings": {k: int(b["ground_truth"]["findings"][k]) for k in FIND},
                "greedy_findings": {k: int(greedy.get("findings", {}).get(k, 0)) for k in FIND},
                "greedy_density": greedy.get("breast_density"),
                "gt_density": b["ground_truth"]["breast_density"],
            }
            f.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
            f.flush()

            if (n + 1) % 100 == 0:
                el = time.perf_counter() - t0
                print(f"[{args.split}] {n+1}/{len(pending)} {el/(n+1):.2f}s/case "
                      f"hard-vs-b8 argmax 一致 {agree}/{n_chk}", flush=True)

    if n_chk:
        print(f"[自检 {args.split}] hard vs b8: argmax 一致率 {agree}/{n_chk} = {agree/n_chk:.4f}, "
              f"平均绝对差 {mad_sum/n_chk:.6f} (单例最大 {mad_max:.6f})", flush=True)
        Path(str(out_path).replace(".jsonl", "_selfcheck.json")).write_text(json.dumps({
            "split": args.split, "n": n_chk,
            "argmax_agree": agree / n_chk,
            "mean_abs_diff": mad_sum / n_chk,
            "max_sample_abs_diff": mad_max,
        }, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- 敏感性：mass 的 p 从 0 到 1 ----
    if args.sens_out:
        import random
        rng = random.Random(42)
        pool = sorted(data.keys() & b8.keys())
        sens_ids = rng.sample(pool, min(args.sens_n, len(pool)))
        grid = [i / 10 for i in range(11)]
        rows = []
        for sid in sens_ids:
            rec, b = data[sid], b8[sid]
            pn = b["probs_norm"]
            p = {f"findings.{k}": float(pn[f"findings.{k}"]["1"]) for k in FIND}
            text = json.dumps(b.get("greedy_prediction") or {}, ensure_ascii=False)
            images = list(rec["images"])
            pids = build_prompt(template, tokenizer, processor, instruction, images)
            mmk = to_mm_kwargs(template, processor, model, images, pids)
            curve = []
            for g in grid:
                p2 = dict(p)
                p2["findings.mass"] = g
                five, _, _, _ = forward_birads(model, tokenizer, processor, layer0, mmk, pids, text, soft=p2)
                curve.append(sum(five[2:]) if five else None)
            rows.append({"id": sid, "p_ge3": curve, "gt_birads": int(b["ground_truth"]["breast_birads"]),
                         "p_mass": p["findings.mass"], "gt_mass": int(b["ground_truth"]["findings"]["mass"])})
            print(f"[sens] {sid} done", flush=True)
        Path(args.sens_out).write_text(json.dumps({"grid": grid, "rows": rows}, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    print("[done]", args.split, flush=True)


if __name__ == "__main__":
    main()
