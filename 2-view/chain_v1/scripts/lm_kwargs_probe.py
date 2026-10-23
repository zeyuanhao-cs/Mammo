#!/usr/bin/env python3
"""打印第 1 次前向时 language_model 实际收到的 args/kwargs 键名与形状（真实模型）。"""
import sys
from pathlib import Path

import torch

CV1 = Path(__file__).resolve().parent.parent
PROJ = CV1.parent
sys.path.insert(0, str(PROJ / "scripts" / "vlm"))
sys.path.insert(0, str(CV1.parent))

from llamafactory.chat import ChatModel  # noqa: E402
from chain_v1.train.core import find_text_model  # noqa: E402
from chain_v1.train.data import SoftKDDataset, build_inputs, gt_text  # noqa: E402
from score_2view import load_instruction  # noqa: E402

DATA = PROJ / "data" / "llamafactory" / "train_llamafactory_os_sub_local.json"


def shape(x):
    if torch.is_tensor(x):
        return f"Tensor{tuple(x.shape)} {x.dtype}"
    if isinstance(x, (list, tuple)):
        return f"{type(x).__name__}[{len(x)}]"
    return repr(x)


def main():
    cm = ChatModel(args=dict(
        model_name_or_path="/home/z/zhao/Mammography/model/Qwen3.5-4B", template="qwen3_5_nothink",
        trust_remote_code=True, infer_backend="huggingface", infer_dtype="bfloat16",
        image_max_pixels=786432, image_min_pixels=512 * 512, enable_thinking=False,
        flash_attn="sdpa",
        adapter_name_or_path="/home/z/zhao/Mammography/model/Qwen3.5-4B-lora-2view-os-sub-vt",
        finetuning_type="lora"))
    eng = cm.engine
    model, tok, proc, tpl = eng.model, eng.tokenizer, eng.processor, eng.template
    model.eval()

    ds = SoftKDDataset(str(DATA), tok, tpl, limit=1)
    s = ds[0]
    built = build_inputs(s, tok, tpl, proc)
    images = built["images"]
    mm = tpl.mm_plugin.get_mm_inputs(images=images, videos=[], audios=[], imglens=[len(images)],
                                    vidlens=[0], audlens=[0], batch_ids=[built["input_ids"]], processor=proc)
    kw = {}
    for k, v in mm.items():
        if isinstance(v, list) and isinstance(v[0], torch.Tensor):
            v = torch.stack(v)
        elif not isinstance(v, torch.Tensor):
            v = torch.tensor(v)
        if torch.is_floating_point(v):
            v = v.to(model.dtype)
        kw[k] = v.tolist() if k == "second_per_grid_ts" else v.to(model.device)
    if hasattr(proc, "create_mm_token_type_ids"):
        mtt = proc.create_mm_token_type_ids([built["input_ids"]])
        kw["mm_token_type_ids"] = (mtt if torch.is_tensor(mtt) else torch.tensor(mtt)).to(model.device)

    lm = find_text_model(model)
    print("language_model =", type(lm).__name__)
    box = {}

    def pre(module, args, kwargs):
        box["args"] = args
        box["kwargs"] = dict(kwargs)

    h = lm.register_forward_pre_hook(pre, with_kwargs=True)
    ids = torch.tensor([built["input_ids"]], device=model.device)
    with torch.inference_mode():
        model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
              logits_to_keep=59, **kw)
    h.remove()

    print(f"\nargs ({len(box.get('args', ())) }):")
    for i, a in enumerate(box.get("args", ())):
        print(f"  [{i}] {shape(a)}")
    print(f"\nkwargs ({len(box.get('kwargs', {}))}):")
    for k, v in box.get("kwargs", {}).items():
        print(f"  {k:<24} {shape(v)}")

    if torch.is_tensor(box["kwargs"].get("position_ids")):
        pos = box["kwargs"]["position_ids"]
        print(f"\nposition_ids shape={tuple(pos.shape)} dtype={pos.dtype} "
              f"first={pos[..., :6].tolist()}")
    print("\ninputs_embeds requires_grad:", getattr(box["kwargs"].get("inputs_embeds"), "requires_grad", None))


if __name__ == "__main__":
    main()
