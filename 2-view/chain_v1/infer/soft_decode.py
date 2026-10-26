#!/usr/bin/env python3
"""soft-decode：b8 贪心解码，但 finding 取值 token 用软嵌入作为下一步输入（输出文本仍是 argmax）。

batch=1。流程：
  prefill(prompt + 图像) → past/logits
  循环：tok = argmax(logits)；若 tok 正好补全某个 finding 的取值（文本以 `"mass": 0` 结尾），
        就用本步 logits 算 p = sigmoid(l1 - l0)，下一步输入嵌入 = (1-p)E["0"] + pE["1"]，
        否则正常把 tok 作为下一步输入。
p=0/1 时混合退化为 E["0"]/E["1"]，与普通贪心逐 token 等价（见 tests/test_soft_decode.py）。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent.parent.parent))
from chain_v1.train.soft_mixer import SoftMode, soft_vector  # noqa: E402

from typing import Dict, List, Optional, Sequence

import torch

PROJ = Path(__file__).resolve().parent.parent.parent      # chain_v1/infer -> 2-view
sys.path.insert(0, str(PROJ / "scripts" / "vlm"))

FIND = ["mass", "suspicious_calcification", "asymmetry", "architectural_distortion"]
TYPE_IDX = {k: i for i, k in enumerate(FIND)}
ID0, ID1 = 15, 16
_VALUE_RE = {k: re.compile(rf'"{k}"\s*:\s*([01])') for k in FIND}


def mix_embedding(emb_weight: torch.Tensor, p: float, id0: int, id1: int, dtype) -> torch.Tensor:
    """(1-p)·E[id0] + p·E[id1]。"""
    e0, e1 = emb_weight[id0].detach(), emb_weight[id1].detach()
    return ((1.0 - p) * e0 + p * e1).to(dtype)


def completed_finding_field(text: str, names: Sequence[str] = FIND) -> Optional[str]:
    """若文本刚刚以某个 finding 的取值结尾，返回该 finding 名。"""
    for k in names:
        m = _VALUE_RE[k].search(text)
        if m and m.end() == len(text):
            return k
    return None


def rope_prefill(model, prompt_ids, kw):
    """prefill 并构造与 transformers generate 完全一致的 rope 位置:
    - prefill: [4,B,L] = text arange + get_rope_index 的 3 通道视觉位置（视觉 token 紧凑）
    - 返回 (out, rope_deltas)，decode 步 position_ids = past_kv_len + delta
      （修 bug：原先 prefill 不传 position_ids、decode 用 past_len——视觉 token 的 rope
      位置与 b8 管线(generate)不同，实测概率差达 0.148，是管线间数值不一致的主因。）
    """
    device = prompt_ids.device
    B, L = prompt_ids.shape[0], prompt_ids.shape[-1]
    mm_tti = kw.get("mm_token_type_ids")
    grid = kw.get("image_grid_thw")
    text_pos = torch.arange(L, device=device).view(1, -1).expand(B, -1)
    if mm_tti is not None and grid is not None:
        vis_pos, deltas = model.model.get_rope_index(
            prompt_ids, mm_tti, grid, None, attention_mask=torch.ones_like(prompt_ids))
    else:
        vis_pos = text_pos.unsqueeze(0).expand(3, -1, -1)
        deltas = torch.zeros(B, 1, dtype=torch.long, device=device)
    pos4 = torch.cat([text_pos.unsqueeze(0), vis_pos], dim=0)
    out = model(input_ids=prompt_ids,
                attention_mask=torch.ones_like(prompt_ids),
                position_ids=pos4, use_cache=True, **kw)
    return out, deltas


@torch.inference_mode()
def soft_decode(
    model,
    tokenizer,
    text_of=None,
    prompt_ids: Optional[torch.Tensor] = None,
    mm_kwargs: Optional[Dict] = None,
    max_new_tokens: int = 256,
    stop_ids: Optional[Sequence[int]] = None,
    soft: bool = True,
    soft_mode: str = SoftMode.LOGIT_MLP,
    mixer=None,
    id0: int = ID0,
    id1: int = ID1,
    names: Sequence[str] = FIND,
    force_p: Optional[Dict[str, float]] = None,
    force_hard: bool = False,
    force_base_token: bool = False,   # 强制 p 时 base token 也跟随 p（扫描路径用）
    prefill: Optional[Dict] = None,
    return_rows: bool = False,
):
    """返回 {"tokens", "text", "soft_events"}；return_rows=True 时另含 "rows"（每步 logits, cpu float32）。

    force_p    : {field: p} 强制该字段的软注入概率（老师模式=全部 GT；扫描=只覆盖一个字段），
                 其余字段仍按模型自身 p 注入。文本照常按 argmax 生成。
    force_hard : True 时注入向量 = E[argmax token]（delta=0），即"软换硬"对照；
                 p 仍按软管线记录。
    prefill    : {"past","logits","seq_len","text_len"} 复用已算好的 prefill（同一 prompt 多次
                 生成时省掉视觉塔与 prefill），生成部分各分支独立。
    """
    text_of = text_of or (lambda ids: tokenizer.decode(ids, skip_special_tokens=True))
    force_p = dict(force_p or {})
    dt = next(model.parameters()).dtype
    if prefill is not None:
        past = prefill["past"]
        logits = prefill["logits"]
        seq_len = prefill["seq_len"]
        deltas = prefill.get("rope_deltas")
        if deltas is None:
            deltas = torch.zeros(1, dtype=torch.long, device=logits.device)
    else:
        device = prompt_ids.device
        kw = dict(mm_kwargs or {})
        for k, v in list(kw.items()):
            if torch.is_tensor(v) and v.device != device:
                kw[k] = v.to(device)
        out, deltas = rope_prefill(model, prompt_ids, kw)
        past = out.past_key_values
        logits = out.logits[0, -1].float()
        seq_len = int(prompt_ids.shape[1])

    gen: List[int] = []
    events = []
    stop = set(stop_ids or [])
    rows: List[torch.Tensor] = []

    def next_pos():
        """decode 位置 = past_kv 长度 + rope_delta（与 transformers generate 的
        prepare_inputs_for_generation 完全一致；生成的都是 text token，单通道广播到 3 通道）。"""
        pl = past.get_seq_length()
        return (torch.tensor([[[pl + int(deltas.reshape(-1)[0])]]],
                             dtype=torch.long, device=logits.device))

    for step in range(max_new_tokens):
        rows.append(logits.detach().to("cpu", torch.float32))
        tok = int(torch.argmax(logits))
        if tok in stop:
            break
        gen.append(tok)
        text = text_of(gen)

        field = completed_finding_field(text, names) if soft else None
        if field is not None:
            p = float(torch.sigmoid(logits[id1] - logits[id0]))
            if field in force_p:
                p = float(force_p[field])
            events.append((step, field, p))
            if force_hard:
                # 硬 token 注入：若该字段给了 force_p，则 token 跟随强制概率
                # （p>0.5 → id1"1"，否则 id0"0"）；未给 force_p 时沿用模型自身 argmax
                # （与既有 swap_hard 行为一致）。
                tid = (id1 if p > 0.5 else id0) if field in force_p else tok
                vec = model.get_input_embeddings().weight[tid].to(dt)
            else:
                # 强制 p 的路径（扫描/teacher）：base token 也跟随强制 p
                # （p>0.5 → "1"），避免"强制 p=1 但仍以模型自己判的 E["0"] 作基座"的错配
                base_tok = (id1 if p > 0.5 else id0) if (force_base_token and field in force_p) else tok
                vec = soft_vector(mixer, model.get_input_embeddings().weight, p,
                                  TYPE_IDX.get(field, 0), base_tok, id0, id1, dt, soft_mode,
                                  gate=getattr(mixer, "gate", "none"))
            emb = vec.to(dt).view(1, 1, -1)  # 修 bug: logit_mlp 路径 soft_vector 返回 fp32(mixer 权重)，须转模型 dtype
            seq_len += 1
            out = model(inputs_embeds=emb, position_ids=next_pos(),
                        attention_mask=torch.ones((1, seq_len), device=emb.device),
                        past_key_values=past, use_cache=True)
        else:
            nxt = torch.tensor([[tok]], device=logits.device)
            seq_len += 1
            out = model(input_ids=nxt, position_ids=next_pos(),
                        attention_mask=torch.ones((1, seq_len), device=nxt.device),
                        past_key_values=past, use_cache=True)

        past = out.past_key_values
        logits = out.logits[0, -1].float()

    res = {"tokens": gen, "text": text_of(gen), "soft_events": events}
    if return_rows:
        res["rows"] = rows
    return res


def _merge_mm(mm_list):
    """把每条 [1, ...] 的 mm_kwargs 合并成 [B, ...]（pixel_values/image_grid_thw/
    mm_token_type_ids/second_per_grid_ts 沿 batch/图像维拼接）。"""
    out = {}
    keys = set()
    for mm in mm_list:
        keys.update(mm)
    for k in keys:
        vals = [mm[k] for mm in mm_list if k in mm]
        if k in ("pixel_values", "image_grid_thw", "second_per_grid_ts"):
            out[k] = torch.cat(vals, dim=0)
        elif k == "mm_token_type_ids":
            out[k] = torch.cat(vals, dim=0)
        else:
            try:
                out[k] = torch.cat(vals, dim=0)
            except RuntimeError:
                out[k] = vals[0]
    return out


@torch.inference_mode()
def decode_batch(model, tokenizer, cases, names: Sequence[str] = FIND,
                 stop_ids=None, max_new_tokens: int = 256, soft: bool = False,
                 force_hard: bool = False, force_base_token: bool = False, mixer=None,
                 soft_mode: str = SoftMode.LOGIT_MLP, text_of=None,
                 return_rows: bool = False):
    """批内并行软解码（等长 prompt；b8 管线两视角图分辨率固定 → 天然等长）。

    cases: list of dict:
      prompt_ids : List[int] / Tensor [L]
      mm_kwargs  : 单条 mm inputs（[1, ...] 张量，eval_soft 用 get_mm_inputs 构造）
      force_p    : {field: p}（可选；老师/扫描模式）
    其余语义与单条 soft_decode 一致；注入按条独立，已完成条用 pad 前进、输出丢弃。
    返回 list of {"tokens","text","soft_events"[,"rows"]}。
    """
    B = len(cases)
    assert B >= 1
    text_of = text_of or (lambda ids: tokenizer.decode(ids, skip_special_tokens=True))
    dt = next(model.parameters()).dtype
    emb_w = model.get_input_embeddings().weight
    device = model.device
    def _ids_list(c):
        t = c["prompt_ids"]
        if torch.is_tensor(t):
            t = t.reshape(-1).tolist()
        return [int(x) for x in t]

    lens = {len(_ids_list(c)) for c in cases}
    assert len(lens) == 1, f"批内 prompt 长度不一致: {lens}（请回退逐条 decode）"
    L = lens.pop()
    pad_id = (next(iter(stop_ids)) if stop_ids else _ids_list(cases[0])[0])
    id0, id1 = ID0, ID1
    stop = set(stop_ids or [])
    fps = [dict(c.get("force_p") or {}) for c in cases]
    soft_flags = [soft and not force_hard for _ in range(B)]

    # ---- 批 prefill ----
    ids = torch.tensor([_ids_list(c) for c in cases], device=device)
    mm_list = []
    for c in cases:
        kw = dict(c.get("mm_kwargs") or {})
        clean = {}
        for k, v in kw.items():
            if torch.is_tensor(v):
                clean[k] = v.to(device) if v.is_floating_point() else v.to(device)
            else:
                clean[k] = v
        mm_list.append(clean)
    merged = _merge_mm(mm_list)
    out, deltas = rope_prefill(model, ids, merged)
    past = out.past_key_values
    logits = out.logits[:, -1, :].float()
    seq_len = L
    delta0 = int(deltas.reshape(-1)[0])   # 批内等长且同图配置 → delta 相同

    gens: List[List[int]] = [[] for _ in range(B)]
    texts: List[str] = [""] * B
    events: List[list] = [[] for _ in range(B)]
    rows: List[List[torch.Tensor]] = [[] for _ in range(B)]
    done = [False] * B

    def next_pos_b():
        pl = past.get_seq_length()
        # 与 generate 的 decode 一致: text 位置 + rope_delta（单通道广播到 3 通道）
        return torch.full((1, B, 1), pl + delta0, dtype=torch.long, device=device)

    for step in range(max_new_tokens):
        if return_rows:
            for j in range(B):
                if not done[j]:
                    rows[j].append(logits[j].detach().to("cpu", torch.float32))
        toks = logits.argmax(dim=-1)
        embs = [None] * B
        for j in range(B):
            if done[j]:
                continue
            t = int(toks[j])
            if t in stop:
                done[j] = True
                continue
            gens[j].append(t)
            texts[j] = text_of(gens[j])
            field = completed_finding_field(texts[j], names) if soft_flags[j] else None
            if field is not None:
                p = float(torch.sigmoid(logits[j][id1] - logits[j][id0]))
                if field in fps[j]:
                    p = float(fps[j][field])
                events[j].append((step, field, p))
                if force_hard:
                    tid = (id1 if p > 0.5 else id0) if field in fps[j] else t
                    embs[j] = emb_w[tid].to(dt).view(-1)
                else:
                    base_t = (id1 if p > 0.5 else id0) if (force_base_token and field in fps[j]) else t
                    embs[j] = soft_vector(mixer, emb_w, p, TYPE_IDX.get(field, 0),
                                          base_t, id0, id1, dt, soft_mode,
                                          gate=getattr(mixer, "gate", "none")).to(dt).view(-1)
            else:
                embs[j] = emb_w[t].to(dt).view(-1)
        if all(done):
            break
        stack = torch.stack([
            (embs[j] if embs[j] is not None else emb_w[pad_id].to(dt).view(-1))
            for j in range(B)])
        seq_len += 1
        out = model(inputs_embeds=stack.unsqueeze(1), position_ids=next_pos_b(),
                    attention_mask=torch.ones((B, seq_len), dtype=torch.long,
                                              device=device),
                    past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :].float()

    res_list = []
    for j in range(B):
        r = {"tokens": gens[j], "text": texts[j], "soft_events": events[j]}
        if return_rows:
            r["rows"] = rows[j]
        res_list.append(r)
    return res_list


def main():  # pragma: no cover - 需要 GPU，留给人工验证
    import argparse
    sys.path.insert(0, str(PROJ / "scripts" / "vlm"))
    from score_2view import load_instruction
    from llamafactory.chat import ChatModel

    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--data", default=str(PROJ / "data/splits/test_local.json"))
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--no-soft", action="store_true")
    ap.add_argument("--soft-mode", choices=["linear", "logit_mlp"], default="logit_mlp")
    ap.add_argument("--soft-mixer", default="", help="训练产出的 soft_mixer.pt 路径（logit_mlp 必需）")
    args = ap.parse_args()

    cm = ChatModel(args=dict(
        model_name_or_path="/home/z/zhao/Mammography/model/Qwen3.5-4B", template="qwen3_5",
        trust_remote_code=True, infer_backend="huggingface", infer_dtype="bfloat16",
        image_max_pixels=786432, image_min_pixels=512 * 512, enable_thinking=False,
        flash_attn="sdpa", adapter_name_or_path=args.adapter, finetuning_type="lora"))
    eng = cm.engine
    tokenizer, processor, template, model = eng.tokenizer, eng.processor, eng.template, eng.model
    instr = load_instruction(PROJ / "dataset_build" / "prompt.txt")
    data = json.load(open(args.data, encoding="utf-8"))[: args.limit]

    for rec in data:
        images = list(rec["images"])
        messages = template.mm_plugin.process_messages(
            [{"role": "user", "content": instr}], images, [], [], processor)
        pids, _ = template.encode_oneturn(tokenizer, messages + [{"role": "assistant", "content": ""}], None, None)
        pids, _ = template.mm_plugin.process_token_ids(pids, None, images, [], [], tokenizer, processor)
        mm = template.mm_plugin.get_mm_inputs(images=images, videos=[], audios=[], imglens=[len(images)],
                                             vidlens=[0], audlens=[0], batch_ids=[pids], processor=processor)
        kw = {}
        for k, v in mm.items():
            if isinstance(v, list) and isinstance(v[0], torch.Tensor):
                v = torch.stack(v)
            elif isinstance(v, list) and isinstance(v[0], list) and isinstance(v[0][0], torch.Tensor):
                v = torch.stack([torch.stack(x) for x in v])
            elif not isinstance(v, torch.Tensor):
                v = torch.tensor(v)
            if torch.is_floating_point(v):
                v = v.to(model.dtype)
            kw[k] = v.tolist() if k == "second_per_grid_ts" else v.to(model.device)
        if hasattr(processor, "create_mm_token_type_ids"):
            mtt = processor.create_mm_token_type_ids([pids])
            kw["mm_token_type_ids"] = (mtt if isinstance(mtt, torch.Tensor) else torch.tensor(mtt)).to(model.device)

        ids = torch.tensor([pids], device=model.device)
        mixer = None
        if args.soft_mode == "logit_mlp" and not args.no_soft:
            from chain_v1.train.soft_mixer import load_soft_mixer
            assert args.soft_mixer, "logit_mlp 需要 --soft-mixer"
            mixer = load_soft_mixer(args.soft_mixer, device=model.device, dtype=torch.float32)
        r = soft_decode(model, tokenizer, prompt_ids=ids, mm_kwargs=kw,
                        max_new_tokens=args.max_new_tokens, soft=not args.no_soft,
                        soft_mode=args.soft_mode, mixer=mixer)
        print(f"[{rec.get('id')}] soft={not args.no_soft} events={r['soft_events'][:4]} "
              f"text={r['text'][:120]}")


if __name__ == "__main__":  # pragma: no cover
    main()
