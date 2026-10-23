#!/usr/bin/env python3
"""前向一致性测试（需要 GPU + 1-epoch 适配器，eval 模式）：
   A  : 新代码数据路径（data.build_inputs + to_tensors），forward 不传 position_ids
   B1 : LLaMA-Factory collator（SFTDataCollatorWith4DAttentionMask）产出的 batch，含 position_ids / rope_deltas
   B2 : 同 B1 的 batch，但去掉 position_ids / rope_deltas（用于定位差异来源）

比较：completion 平均 CE（loss）与 BI-RADS / 4 个 finding 取值位置的 logits 最大绝对差。
输出：chain_v1/outputs/forward_parity.md + 控制台摘要。
"""
import json
import sys
from pathlib import Path

import torch
from transformers import Trainer

HERE = Path(__file__).resolve().parent
CV1 = HERE.parent
PROJ = CV1.parent
sys.path.insert(0, str(PROJ / "scripts" / "vlm"))
sys.path.insert(0, str(CV1.parent))

from llamafactory.chat import ChatModel  # noqa: E402
from llamafactory.data.collator import SFTDataCollatorWith4DAttentionMask  # noqa: E402
from chain_v1.train.data import (FIND_FIELDS, SoftKDDataset, build_inputs,  # noqa: E402
                                 to_tensors)

ADAPTER = "/home/z/zhao/Mammography/model/Qwen3.5-4B-lora-2view-os-sub-vt"
DATA = PROJ / "data" / "llamafactory" / "train_llamafactory_os_sub_local.json"
OUT = CV1 / "outputs" / "forward_parity.md"
N = 3


def ce_and_probe(model, logits, labels, prompt_len, positions):
    """completion 平均 CE 与指定位置 logits（position 为 completion 内下标）。"""
    K = logits.shape[1]
    comp = labels[:, prompt_len:prompt_len + K - 1].to(logits.device)
    rows = logits[:, :K - 1].float()
    ce = torch.nn.functional.cross_entropy(rows.reshape(-1, rows.shape[-1]), comp.reshape(-1))
    picks = {name: logits[0, idx].float().clone() for name, idx in positions.items()}
    return float(ce), picks


def forward_once(model, batch):
    kw = {k: (v.to(model.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    kw.pop("labels", None)
    rd = kw.pop("rope_deltas", None)
    inp = kw.pop("input_ids")
    kw.pop("attention_mask", None)
    attn = batch["attention_mask"].to(model.device)
    try:
        out = model(input_ids=inp, attention_mask=attn, use_cache=False,
                    logits_to_keep=batch["_K"], rope_deltas=rd, **kw)
    except TypeError:
        out = model(input_ids=inp, attention_mask=attn, use_cache=False,
                    logits_to_keep=batch["_K"], **kw)
    return out.logits


def main():
    cm = ChatModel(args=dict(
        model_name_or_path="/home/z/zhao/Mammography/model/Qwen3.5-4B", template="qwen3_5_nothink",
        trust_remote_code=True, infer_backend="huggingface", infer_dtype="bfloat16",
        image_max_pixels=786432, image_min_pixels=512 * 512, enable_thinking=False,
        flash_attn="sdpa", adapter_name_or_path=ADAPTER, finetuning_type="lora"))
    eng = cm.engine
    model, tokenizer, processor, template = eng.model, eng.tokenizer, eng.processor, eng.template
    model.eval()

    ds = SoftKDDataset(str(DATA), tokenizer, template, limit=N)
    collator = SFTDataCollatorWith4DAttentionMask(
        tokenizer=tokenizer, template=template, processor=processor, model=model,
        padding=False, pad_to_multiple_of=None, label_pad_token_id=-100,
        return_tensors="pt", compute_dtype=torch.bfloat16)

    rows = []
    for i in range(len(ds)):
        s = ds[i]
        built = build_inputs(s, tokenizer, template, processor)
        P = built["prompt_len"]
        K = len(built["labels"]) - P + 1
        positions = {f: s["finding_pos"][j] for j, f in enumerate(FIND_FIELDS)}
        positions["birads"] = s["birads_pos"]

        # ---- A：新代码路径 ----
        ta = to_tensors(built, processor, template, model)
        A_logits = forward_once(model, {**{k: v for k, v in ta.items() if k in
                                          ("input_ids", "attention_mask", "labels")},
                                       **ta["mm_kwargs"], "_K": K})
        ce_a, pk_a = ce_and_probe(model, A_logits, ta["labels"].cpu(), P, positions)

        # ---- B1 / B2：LLaMA-Factory collator ----
        feats = [{"input_ids": built["input_ids"], "labels": built["labels"],
                  "attention_mask": [1] * len(built["input_ids"]), "images": built["images"]}]
        batch = collator(feats)
        has_pos = "position_ids" in batch
        batch["_K"] = K
        B1_logits = forward_once(model, batch)
        ce_b1, pk_b1 = ce_and_probe(model, B1_logits, batch["labels"].cpu(), P, positions)

        # B2：去掉 LF 预算的 position_ids/rope_deltas，改用 mm_token_type_ids（即 A 的机制）
        b2 = {k: v for k, v in batch.items() if k not in ("position_ids", "rope_deltas")}
        if hasattr(processor, "create_mm_token_type_ids"):
            mtt = processor.create_mm_token_type_ids([built["input_ids"]])
            b2["mm_token_type_ids"] = (mtt if torch.is_tensor(mtt) else torch.tensor(mtt)).to(model.device)
        B2_logits = forward_once(model, b2)
        ce_b2, pk_b2 = ce_and_probe(model, B2_logits, batch["labels"].cpu(), P, positions)

        d1 = max(float((pk_a[k] - pk_b1[k]).abs().max()) for k in positions)
        d2 = max(float((pk_a[k] - pk_b2[k]).abs().max()) for k in positions)
        pos_shape = tuple(batch["position_ids"].shape) if has_pos else None
        rows.append((i, P, K, ce_a, ce_b1, ce_b2, d1, d2, pos_shape))
        print(f"[{i}] P={P} K={K} pos_ids={pos_shape} CE A/B1/B2 = {ce_a:.6f}/{ce_b1:.6f}/{ce_b2:.6f} "
              f"| logits maxdiff A-B1={d1:.6f} A-B2={d2:.6f}", flush=True)

    lines = ["# 前向一致性测试（新代码 vs LLaMA-Factory collator/forward）", "",
             f"- 适配器：`{ADAPTER}`（1-epoch 解冻版），模型 eval 模式，bf16，786432",
             f"- 样本：`{DATA.name}` 前 {N} 条", "",
             "| # | prompt_len | comp_len | CE(A 新代码) | CE(B1 LF+pos_ids) | CE(B2 LF 去 pos_ids) "
             "| logits 最大差 A-B1 | logits 最大差 A-B2 | position_ids shape |",
             "|---|---|---|---|---|---|---|---|---|"]
    for i, P, K, ca, cb1, cb2, d1, d2, ps in rows:
        lines.append(f"| {i} | {P} | {K-1} | {ca:.6f} | {cb1:.6f} | {cb2:.6f} | {d1:.6f} | {d2:.6f} | "
                     f"{ps if ps else '未传'} |")
    worst1 = max(r[6] for r in rows)
    worst2 = max(r[7] for r in rows)
    verdict = ("一致（A == B2 且 B1 差异显著 → position_ids 是唯一来源，需在新代码里补 3D rope position_ids）"
               if worst2 < 0.05 and worst1 > 0.05 else
               "一致（三条路径 logits 差都很小）" if worst1 < 0.05 else
               "不一致，需排查")
    lines += ["", f"**结论**：A-B1 最大差 {worst1:.6f}，A-B2 最大差 {worst2:.6f} → {verdict}", ""]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
