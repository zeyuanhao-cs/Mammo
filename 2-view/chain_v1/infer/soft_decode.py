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
) -> Dict:
    """返回 {"tokens": [...], "text": str, "soft_events": [(step, field, p)]}"""
    text_of = text_of or (lambda ids: tokenizer.decode(ids, skip_special_tokens=True))
    device = prompt_ids.device
    kw = dict(mm_kwargs or {})

    out = model(input_ids=prompt_ids, attention_mask=torch.ones_like(prompt_ids),
                use_cache=True, **kw)
    past = out.past_key_values
    logits = out.logits[0, -1].float()
    seq_len = int(prompt_ids.shape[1])

    gen: List[int] = []
    events = []
    stop = set(stop_ids or [])

    for step in range(max_new_tokens):
        tok = int(torch.argmax(logits))
        if tok in stop:
            break
        gen.append(tok)
        text = text_of(gen)

        field = completed_finding_field(text, names) if soft else None
        if field is not None:
            p = float(torch.sigmoid(logits[id1] - logits[id0]))
            events.append((step, field, p))
            dt = next(model.parameters()).dtype
            vec = soft_vector(mixer, model.get_input_embeddings().weight, p,
                              TYPE_IDX.get(field, 0), tok, id0, id1, dt, soft_mode)
            emb = vec.view(1, 1, -1)
            seq_len += 1
            out = model(inputs_embeds=emb, attention_mask=torch.ones((1, seq_len), device=device),
                        past_key_values=past, use_cache=True)
        else:
            nxt = torch.tensor([[tok]], device=device)
            seq_len += 1
            out = model(input_ids=nxt, attention_mask=torch.ones((1, seq_len), device=device),
                        past_key_values=past, use_cache=True)

        past = out.past_key_values
        logits = out.logits[0, -1].float()

    return {"tokens": gen, "text": text_of(gen), "soft_events": events}


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
