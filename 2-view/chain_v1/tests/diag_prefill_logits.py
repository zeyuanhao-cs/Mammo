#!/usr/bin/env python3
"""定位 soft_decode(新 rope) vs b8 generate 的首 token logits 差异来源."""
import json
import sys
from pathlib import Path

import torch

PROJ = Path("/home/z/zhao/Mammography/Mammo/2-view")
sys.path.insert(0, str(PROJ / "scripts" / "vlm"))
sys.path.insert(0, str(PROJ))
from chain_v1.infer.soft_decode import rope_prefill  # noqa: E402
from chain_v1.infer.eval_soft import build_prompt_one, stop_ids_of  # noqa: E402
from score_2view import load_instruction  # noqa: E402
from llamafactory.chat import ChatModel  # noqa: E402
from transformers import GenerationConfig  # noqa: E402

cm = ChatModel(args=dict(
    model_name_or_path="/home/z/zhao/Mammography/model/Qwen3.5-4B", template="qwen3_5",
    trust_remote_code=True, infer_backend="huggingface", infer_dtype="bfloat16",
    image_max_pixels=786432, image_min_pixels=512 * 512, enable_thinking=False,
    flash_attn="sdpa",
    adapter_name_or_path="/home/z/zhao/Mammography/model/Qwen3.5-4B-lora-2view-os-sub-vt/checkpoint-439",
    finetuning_type="lora"))
eng = cm.engine
tokenizer, processor, template, model = eng.tokenizer, eng.processor, eng.template, eng.model
model.eval()
instr = load_instruction(PROJ / "dataset_build" / "prompt.txt")
stops = stop_ids_of(template, tokenizer)
data = json.load(open(PROJ / "data/splits/val_raw_local.json", encoding="utf-8"))[:1]
rec = data[0]
prompt_ids, kw = build_prompt_one(template, tokenizer, processor, instr, list(rec["images"]),
                                  dtype=model.dtype, device=next(model.parameters()).device)
print(f"prompt_ids shape={tuple(prompt_ids.shape)} kw={sorted(kw)}", flush=True)

gen_cfg = GenerationConfig(do_sample=False, num_beams=1, repetition_penalty=1.0,
                           max_new_tokens=1, eos_token_id=sorted(stops)[0],
                           pad_token_id=tokenizer.pad_token_id)

with torch.inference_mode():
    # A) 我们的 rope_prefill
    out, deltas = rope_prefill(model, prompt_ids, kw)
    la = out.logits[0, -1].float()
    ta = la.topk(5)
    print(f"[A rope_prefill] delta={deltas.tolist()} top5={ta.indices.tolist()} "
          f"vals={[round(v,4) for v in ta.values.tolist()]}", flush=True)

    # B) generate 1 步
    g = model.generate(input_ids=prompt_ids, attention_mask=torch.ones_like(prompt_ids),
                       generation_config=gen_cfg, return_dict_in_generate=True,
                       output_scores=True, **kw)
    lb = g.scores[0][0].float()
    tb = lb.topk(5)
    print(f"[B generate   ] top5={tb.indices.tolist()} "
          f"vals={[round(v,4) for v in tb.values.tolist()]}", flush=True)

    d = (la - lb).abs()
    print(f"[diff] max|Δlogit|={d.max().item():.4f}  mean={d.mean().item():.5f}  "
          f"argmax_same={int(la.argmax()) == int(lb.argmax())}", flush=True)
