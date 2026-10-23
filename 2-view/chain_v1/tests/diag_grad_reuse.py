#!/usr/bin/env python3
"""诊断：复用 vs 不复用时 mixer 梯度的量级（1 样本，eval，无 dropout）。"""
import sys
from pathlib import Path

import torch

CV1 = Path(__file__).resolve().parent.parent
PROJ = CV1.parent
sys.path.insert(0, str(PROJ / "scripts" / "vlm"))
sys.path.insert(0, str(CV1.parent))

from llamafactory.chat import ChatModel  # noqa: E402
from chain_v1.train.core import (capture_fused_embeds, find_text_model,  # noqa: E402
                                 first_decoder_layer, make_vec_hook)
from chain_v1.train.data import SoftKDDataset, build_inputs  # noqa: E402
from chain_v1.train.soft_mixer import SoftMixer, base_id_from_p, soft_vector  # noqa: E402


def main():
    cm = ChatModel(args=dict(
        model_name_or_path="/home/z/zhao/Mammography/model/Qwen3.5-4B", template="qwen3_5_nothink",
        trust_remote_code=True, infer_backend="huggingface", infer_dtype="bfloat16",
        image_max_pixels=786432, image_min_pixels=512 * 512, enable_thinking=False, flash_attn="sdpa",
        adapter_name_or_path="/home/z/zhao/Mammography/model/Qwen3.5-4B-lora-2view-os-sub-vt",
        finetuning_type="lora"))
    eng = cm.engine
    model, tok, proc, tpl = eng.model, eng.tokenizer, eng.processor, eng.template
    model.eval()
    torch.manual_seed(42)
    mixer = SoftMixer(2560, 64, 4, init="small").to(model.device)
    lm = find_text_model(model)
    layer0 = first_decoder_layer(model)
    lm_head = getattr(model, "lm_head", None) or model.get_output_embeddings()
    ds = SoftKDDataset(str(PROJ / "data/llamafactory/train_llamafactory_os_sub_local.json"), tok, tpl, limit=1)
    s = ds[0]
    built = build_inputs(s, tok, tpl, proc)
    P, L = built["prompt_len"], len(built["labels"])
    K = L - P + 1
    ids = torch.tensor([built["input_ids"]], device=model.device)
    att = torch.ones_like(ids)
    mm = tpl.mm_plugin.get_mm_inputs(images=built["images"], videos=[], audios=[],
                                    imglens=[len(built["images"])], vidlens=[0], audlens=[0],
                                    batch_ids=[built["input_ids"]], processor=proc)
    mkw = {}
    for k, v in mm.items():
        if isinstance(v, list) and isinstance(v[0], torch.Tensor):
            v = torch.stack(v)
        elif not isinstance(v, torch.Tensor):
            v = torch.tensor(v)
        if torch.is_floating_point(v):
            v = v.to(model.dtype)
        mkw[k] = v.tolist() if k == "second_per_grid_ts" else v.to(model.device)
    if hasattr(proc, "create_mm_token_type_ids"):
        mtt = proc.create_mm_token_type_ids([built["input_ids"]])
        mkw["mm_token_type_ids"] = (mtt if torch.is_tensor(mtt) else torch.tensor(mtt)).to(model.device)

    probe = [6, s["birads_pos"]] + list(s["finding_pos"])

    def run(use_reuse):
        model.zero_grad(set_to_none=True)
        mixer.zero_grad(set_to_none=True)
        box, cap = capture_fused_embeds(lm)
        out1 = model(input_ids=ids, attention_mask=att, use_cache=False, logits_to_keep=K, **mkw)
        cap.remove()
        p = torch.sigmoid(out1.logits[0, list(s["finding_pos"]), 16].float()
                          - out1.logits[0, list(s["finding_pos"]), 15].float())
        vecs = [soft_vector(mixer, model.get_input_embeddings().weight, p[j], j,
                            base_id_from_p(p[j]), 15, 16,
                            model.get_input_embeddings().weight.dtype, "logit_mlp") for j in range(4)]
        if use_reuse:
            hs = box["e"].clone()
            for j in range(4):
                hs[0, P + s["finding_pos"][j]] = vecs[j].to(hs.dtype)
            a2 = list(box["args"])
            k2 = dict(box["kwargs"])
            if a2 and torch.is_tensor(a2[0]):
                a2[0] = hs
            else:
                k2["inputs_embeds"] = hs
            o2 = lm(*a2, **k2)
            hidden = getattr(o2, "last_hidden_state", o2[0])
            logits2 = lm_head(hidden[:, -K:])
        else:
            h = make_vec_hook(layer0, [P + s["finding_pos"][j] for j in range(4)], vecs)
            try:
                o2 = model(input_ids=ids, attention_mask=att, use_cache=False, logits_to_keep=K, **mkw)
            finally:
                h.remove()
            logits2 = o2.logits
        (logits2[0, probe].float().mean()).backward()
        return {n: q.grad.detach().float().norm().item() if q.grad is not None else None
                for n, q in mixer.named_parameters()

                }, logits2[0, probe].float().tolist()

    gA, lA = run(True)
    gB, lB = run(False)
    print("logits A:", [round(x, 6) for x in lA])
    print("logits B:", [round(x, 6) for x in lB])
    for k in gA:
        print(f"{k:<16} |gradA|={gA[k]:.4e}  |gradB|={gB[k]:.4e}  rel_diff={abs(gA[k]-gB[k])/max(gA[k],1e-12):.3e}")


if __name__ == "__main__":
    main()
