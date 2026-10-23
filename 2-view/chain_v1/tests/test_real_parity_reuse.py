#!/usr/bin/env python3
"""真实模型上的复用一致性测试（GPU，eval 模式，无 dropout）。

A) 复用：第 2 次前向复用第 1 次传给 language_model 的全部实参，仅替换 inputs_embeds
B) 不复用：整模型再跑一次（hook 注入）
比较：BI-RADS / 4 个 finding / density 位置 logits 最大差、position_ids 是否逐位相同、各参数梯度最大差。
输出 chain_v1/outputs/real_parity_reuse.md
"""
import json
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
from chain_v1.train.data import FIND_FIELDS, SoftKDDataset, build_inputs  # noqa: E402
from chain_v1.train.soft_mixer import (SoftMixer, base_id_from_p,  # noqa: E402
                                      soft_vector)

N = 3
OUT = CV1 / "outputs" / "real_parity_reuse.md"


def mm_kwargs(tpl, proc, model, images, ids):
    mm = tpl.mm_plugin.get_mm_inputs(images=images, videos=[], audios=[], imglens=[len(images)],
                                    vidlens=[0], audlens=[0], batch_ids=[ids], processor=proc)
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
        mtt = proc.create_mm_token_type_ids([ids])
        kw["mm_token_type_ids"] = (mtt if torch.is_tensor(mtt) else torch.tensor(mtt)).to(model.device)
    return kw


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
    ds = SoftKDDataset(str(PROJ / "data/llamafactory/train_llamafactory_os_sub_local.json"),
                       tok, tpl, limit=N)

    rows = []
    for i in range(N):
        s = ds[i]
        built = build_inputs(s, tok, tpl, proc)
        P, L = built["prompt_len"], len(built["labels"])
        K = L - P + 1
        ids = torch.tensor([built["input_ids"]], device=model.device)
        att = torch.ones_like(ids)
        mkw = mm_kwargs(tpl, proc, model, built["images"], built["input_ids"])

        # ---- 第 1 次前向 + 捕获 ----
        box, cap = capture_fused_embeds(lm)
        with torch.no_grad():
            out1 = model(input_ids=ids, attention_mask=att, use_cache=False, logits_to_keep=K, **mkw)
        cap.remove()
        logits1 = out1.logits
        p = torch.sigmoid(logits1[0, [s["finding_pos"][j] for j in range(4)], 16].float()
                          - logits1[0, [s["finding_pos"][j] for j in range(4)], 15].float())
        vecs = [soft_vector(mixer, model.get_input_embeddings().weight, p[j], j,
                            base_id_from_p(p[j]), 15, 16, model.get_input_embeddings().weight.dtype,
                            "logit_mlp") for j in range(4)]
        hs = box["e"].clone()
        for j in range(4):
            hs[0, P + s["finding_pos"][j]] = vecs[j].to(hs.dtype)

        # ---- A：复用 ----
        a2 = list(box["args"])
        k2 = dict(box["kwargs"])
        if a2 and torch.is_tensor(a2[0]):
            a2[0] = hs
        else:
            k2["inputs_embeds"] = hs
        with torch.no_grad():
            outA = lm(*a2, **k2)
        hiddenA = getattr(outA, "last_hidden_state", outA[0])
        logitsA = lm_head(hiddenA[:, -K:])

        # ---- B：不复用（整模型 + hook）----
        seq_pos = [P + s["finding_pos"][j] for j in range(4)]
        h = make_vec_hook(layer0, seq_pos, vecs)
        boxB, capB = capture_fused_embeds(lm)
        try:
            with torch.no_grad():
                outB = model(input_ids=ids, attention_mask=att, use_cache=False,
                             logits_to_keep=K, **mkw)
        finally:
            h.remove()
            capB.remove()
        logitsB = outB.logits

        posA = k2.get("position_ids")
        posB = boxB.get("kwargs", {}).get("position_ids")
        pos_same = (posA is not None and posB is not None and torch.equal(posA, posB))
        probe = {"density": 6, "birads": s["birads_pos"]}
        for j, name in enumerate(FIND_FIELDS):
            probe[name] = s["finding_pos"][j]
        diffs = {k: float((logitsA[0, v].float() - logitsB[0, v].float()).abs().max())
                 for k, v in probe.items()}
        rows.append({"i": i, "pos_same": pos_same, "pos_shape": tuple(posA.shape) if posA is not None else None,
                     "diffs": diffs, "max": max(diffs.values())})
        print(f"[{i}] position_ids identical={pos_same} max_logit_diff={max(diffs.values()):.3e} "
              f"per-field={ {k: round(v, 6) for k, v in diffs.items()} }", flush=True)

    # ---- 梯度（样本 0）----
    s = ds[0]
    built = build_inputs(s, tok, tpl, proc)
    P, L = built["prompt_len"], len(built["labels"])
    K = L - P + 1
    ids = torch.tensor([built["input_ids"]], device=model.device)
    att = torch.ones_like(ids)
    mkw = mm_kwargs(tpl, proc, model, built["images"], built["input_ids"])
    probe_idx = [6, s["birads_pos"]] + list(s["finding_pos"])

    def grads_for(use_reuse):
        model.zero_grad(set_to_none=True)
        mixer.zero_grad(set_to_none=True)
        box, cap = capture_fused_embeds(lm)
        out1 = model(input_ids=ids, attention_mask=att, use_cache=False, logits_to_keep=K, **mkw)
        cap.remove()
        p = torch.sigmoid(out1.logits[0, list(s["finding_pos"]), 16].float()
                          - out1.logits[0, list(s["finding_pos"]), 15].float())
        vecs_ = [soft_vector(mixer, model.get_input_embeddings().weight, p[j], j,
                             base_id_from_p(p[j]), 15, 16,
                             model.get_input_embeddings().weight.dtype, "logit_mlp") for j in range(4)]
        if use_reuse:
            hs = box["e"].clone()
            for j in range(4):
                hs[0, P + s["finding_pos"][j]] = vecs_[j].to(hs.dtype)
            a2 = list(box["args"])
            k2 = dict(box["kwargs"])
            if a2 and torch.is_tensor(a2[0]):
                a2[0] = hs
            else:
                k2["inputs_embeds"] = hs
            out2 = lm(*a2, **k2)
            hidden = getattr(out2, "last_hidden_state", out2[0])
            logits2 = lm_head(hidden[:, -K:])
        else:
            h = make_vec_hook(layer0, [P + s["finding_pos"][j] for j in range(4)], vecs_)
            try:
                out2 = model(input_ids=ids, attention_mask=att, use_cache=False,
                             logits_to_keep=K, **mkw)
            finally:
                h.remove()
            logits2 = out2.logits
        logits2[0, probe_idx].float().sum().backward()
        return {n: (p_.grad.detach().clone() if p_.grad is not None else None)
                for n, p_ in list(model.named_parameters()) + [("mixer." + n, q) for n, q in mixer.named_parameters()]}

    gA = grads_for(True)
    gB = grads_for(False)
    gd = {}
    for k in gB:
        if gA.get(k) is None or gB[k] is None:
            continue
        gd[k] = float((gA[k] - gB[k]).abs().max())
    worst_key = max(gd, key=gd.get) if gd else None

    lines = ["# 真实模型复用一致性（eval，无 dropout）", "",
             f"- 模型：Qwen3.5-4B + `os-sub-vt` adapter，bf16，786432，样本数 {N}",
             "- A = 复用第 1 次语言模型实参（只换 inputs_embeds）；B = 整模型重跑 + hook", "",
             "| # | position_ids 逐位相同 | position_ids shape | density | BI-RADS | mass | calc | asymmetry | arch | 最大差 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        d = r["diffs"]
        lines.append(f"| {r['i']} | {r['pos_same']} | {r['pos_shape']} | {d['density']:.3e} | {d['birads']:.3e} | "
                     f"{d['findings.mass']:.3e} | {d['findings.suspicious_calcification']:.3e} | "
                     f"{d['findings.asymmetry']:.3e} | {d['findings.architectural_distortion']:.3e} | "
                     f"**{r['max']:.3e}** |")
    lines += ["", f"**梯度（样本 0）**：比较 {len(gd)} 个参数张量，最大差 **{gd[worst_key]:.3e}**"
                  f"（{worst_key}）" if gd else "**梯度**：无可用梯度", "",
              "注：训练模式下两次前向共享同一个 dropout 掩码 —— 这是复用与不复用之间**唯一**预期的行为差异；"
              "eval 模式下该差异不存在，因此本表差异应为 0 或 bf16 噪声。", ""]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
