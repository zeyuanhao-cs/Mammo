#!/usr/bin/env python3
"""诊断 soft_decode 的 rope/position_ids 报错: 分步测 prefill 与 decode."""
import json
import sys
from pathlib import Path

import torch

CV1 = Path(__file__).resolve().parent.parent
PROJ = CV1.parent
sys.path.insert(0, str(PROJ / "scripts" / "vlm"))
sys.path.insert(0, str(PROJ))

from llamafactory.chat import ChatModel  # noqa: E402
from score_2view import load_instruction  # noqa: E402

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
data = json.load(open(PROJ / "data/splits/val_raw_local.json", encoding="utf-8"))[:1]
rec = data[0]
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
    kw["mm_token_type_ids"] = (mtt if torch.is_tensor(mtt) else torch.tensor(mtt)).to(model.device)

ids = torch.tensor([pids], device=model.device)
print(f"prompt_len={len(pids)} kw keys={sorted(kw)}", flush=True)

with torch.inference_mode():
    # 1) prefill
    try:
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True, **kw)
        print(f"[1] prefill OK, logits {tuple(out.logits.shape)}", flush=True)
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"[1] prefill FAIL: {e!r}", flush=True)
        raise SystemExit(1)

    # 2) decode 1 步: 不带 position_ids
    tok = int(out.logits[0, -1].argmax())
    try:
        out2 = model(input_ids=torch.tensor([[tok]], device=model.device),
                     attention_mask=torch.ones((1, len(pids) + 1), device=model.device),
                     past_key_values=out.past_key_values, use_cache=True)
        print(f"[2] decode w/o position_ids OK", flush=True)
    except Exception as e:
        print(f"[2] decode w/o position_ids FAIL: {type(e).__name__}: {e}", flush=True)

    # 3) decode 1 步: 带 4D position_ids（text 通道 + 3 个 mrope 通道全 +1）
    try:
        past_len = out.past_key_values.get_seq_length()
        pos = torch.full((4, 1, 1), past_len, dtype=torch.long, device=model.device)
        out3 = model(input_ids=torch.tensor([[tok]], device=model.device),
                     attention_mask=torch.ones((1, len(pids) + 1), device=model.device),
                     position_ids=pos,
                     past_key_values=out.past_key_values, use_cache=True)
        print(f"[3] decode with 4D position_ids OK, past={past_len}", flush=True)
    except Exception as e:
        print(f"[3] FAIL: {type(e).__name__}: {e}", flush=True)

    # 4) 10 步循环带 position_ids
    try:
        past = out.past_key_values
        logits = out.logits[0, -1].float()
        seq = len(pids)
        for step in range(10):
            t = int(logits.argmax())
            pl = past.get_seq_length()
            pos = torch.full((4, 1, 1), pl, dtype=torch.long, device=model.device)
            o = model(input_ids=torch.tensor([[t]], device=model.device),
                      attention_mask=torch.ones((1, seq + 1), device=model.device),
                      position_ids=pos, past_key_values=past, use_cache=True)
            past = o.past_key_values
            logits = o.logits[0, -1].float()
            seq += 1
        print("[4] 10-step loop OK", flush=True)
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"[4] FAIL: {type(e).__name__}: {e}", flush=True)
