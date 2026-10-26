#!/usr/bin/env python3
"""C3m 训练后诊断（只读）：视角 A（自身征象 + linear 软证据）vs 视角 M（unknown 向量）。

样本：训练集全部 GT>=3 + 随机 300 个 GT<=2（seed 42）。
每例 3 次前向：pass1（teacher forcing，取模型自身 p）→ 视角 A → 视角 M。
输出：outputs/c3m_teacher_diag.md
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJ = Path("/home/z/zhao/Mammography/Mammo/2-view")
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / "scripts" / "vlm"))

from llamafactory.chat import ChatModel  # noqa: E402
from chain_v1.train.core import birads_dist, first_decoder_layer, finding_p  # noqa: E402
from chain_v1.train.data import build_inputs, gt_text, positions_in_text, to_tensors, FIND_FIELDS  # noqa: E402
from chain_v1.train.soft_mixer import SoftMode, soft_vector  # noqa: E402

BIRADS = None  # 从 core 默认取


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="/home/z/zhao/Mammography/model/chain_v1_C3m")
    ap.add_argument("--unknown", default="",
                    help="soft_unknown.pt 路径；空=用初始化 (E0+E1)/2（训练未保存时）")
    ap.add_argument("--data", default=str(PROJ / "data/llamafactory/train_llamafactory_os_sub_local.json"))
    ap.add_argument("--n-neg", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-new-tokens", type=int, default=1)
    ap.add_argument("--out", default=str(PROJ / "chain_v1/outputs/pilot/c3m_teacher_diag.md"))
    a = ap.parse_args()

    torch.cuda.set_device(0)
    cm = ChatModel(args=dict(
        model_name_or_path="/home/z/zhao/Mammography/model/Qwen3.5-4B", template="qwen3_5",
        trust_remote_code=True, infer_backend="huggingface", infer_dtype="bfloat16",
        image_max_pixels=786432, image_min_pixels=512 * 512, enable_thinking=False,
        flash_attn="sdpa", adapter_name_or_path=a.adapter, finetuning_type="lora"))
    eng = cm.engine
    tokenizer, processor, template, model = eng.tokenizer, eng.processor, eng.template, eng.model
    model.eval()
    emb_w = model.get_input_embeddings().weight
    id0, id1 = 15, 16
    ids5 = [int(t) for t in tokenizer("12345", add_special_tokens=False)["input_ids"]]
    if len(ids5) != 5:
        ids5 = [tokenizer.convert_tokens_to_ids(ch) for ch in ["1", "2", "3", "4", "5"]]
    print(f"[birads ids] {ids5}", flush=True)

    unknown = None
    if a.unknown and Path(a.unknown).exists():
        blob = torch.load(a.unknown, map_location="cpu")
        unknown = blob["unknown"].float().to(emb_w.device)
        print(f"[unknown] 已加载 {tuple(unknown.shape)} ← {a.unknown}", flush=True)
    else:
        init = (emb_w[id0].detach().float() + emb_w[id1].detach().float()) / 2.0
        unknown = init.unsqueeze(0).repeat(4, 1)
        print("[unknown] 未找到保存的向量，使用初始化 (E0+E1)/2 —— 视角 M 为『初值版』",
              flush=True)

    # ---- 样本选择 ----
    recs = json.load(open(a.data, encoding="utf-8"))
    pos_g3, neg = [], []
    for r in recs:
        o = r["output"]
        o = json.loads(o) if isinstance(o, str) else o
        (pos_g3 if int(o["breast_birads"]) >= 3 else neg).append(r)
    rng = np.random.default_rng(a.seed)
    neg_pick = [neg[i] for i in rng.choice(len(neg), size=min(a.n_neg, len(neg)), replace=False)]
    pick = pos_g3 + neg_pick
    print(f"[data] GT>=3 {len(pos_g3)} + GT<=2 {len(neg_pick)} = {len(pick)} 例", flush=True)

    instr_default = recs[0]["instruction"]
    layer0 = first_decoder_layer(model)
    model_dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device

    out = []
    for k, rec in enumerate(pick, 1):
        o = rec["output"]
        o = json.loads(o) if isinstance(o, str) else o
        gt_b = int(o["breast_birads"])
        text = gt_text(o)
        comp_ids, pos = positions_in_text(text, tokenizer)
        sample = {
            "instruction": rec["instruction"] or instr_default,
            "images": rec["images"],
            "completion_text": text,
            "comp_ids": comp_ids,
            "finding_pos": [pos[f] for f in FIND_FIELDS],
            "birads_pos": pos["breast_birads"],
            "gt_findings": {kk: int(o["findings"][kk]) for kk in FIND_FIELDS},
            "gt_birads": gt_b,
        }
        built = build_inputs(sample, tokenizer, template, processor)
        t = to_tensors(built, processor, template, model=model)
        input_ids, attn = t["input_ids"], t["attention_mask"]
        fpos, bpos = t["finding_pos"], t["birads_pos"]
        K = int((t["labels"] != -100).sum().item()) + 1
        common = dict(input_ids=input_ids, attention_mask=attn, use_cache=False, **t["mm_kwargs"])

        with torch.no_grad():
            try:
                logits1 = model(**common, logits_to_keep=K).logits
            except TypeError:
                logits1 = model(**common, num_logits_to_keep=K).logits
            p = finding_p(logits1, fpos, id0, id1, False)[0]
            prompt_len = int(input_ids.shape[1]) - K + 1
            seq_pos = [prompt_len + int(x) for x in fpos[0].tolist()]

            def run_inject(vecs):
                def hook(module, args):
                    hs = args[0].clone()
                    for sp, v in zip(seq_pos, vecs):
                        hs[0, sp] = v.to(hs.dtype)
                    return (hs,) + tuple(args[1:])
                h = layer0.register_forward_pre_hook(hook)
                try:
                    try:
                        return model(**common, logits_to_keep=K).logits
                    except TypeError:
                        return model(**common, num_logits_to_keep=K).logits
                finally:
                    h.remove()

            vecA = [soft_vector(None, emb_w, p[j], j, id0, id0, id1, model_dtype, SoftMode.LINEAR)
                    for j in range(len(FIND_FIELDS))]
            vecM = [unknown[j].to(model_dtype) for j in range(len(FIND_FIELDS))]
            logitsA = run_inject(vecA)
            logitsM = run_inject(vecM)

            def dist(logits):
                rows = logits[torch.arange(1), bpos][0]
                z = rows[ids5].float()
                pr = torch.softmax(z, dim=-1)
                return pr.cpu().numpy()

            pa, pm = dist(logitsA), dist(logitsM)

        gi = gt_b - 1
        out.append({"id": rec.get("id", f"idx{k}"), "gt": gt_b,
                    "pa_gt": float(pa[gi]), "pm_gt": float(pm[gi]),
                    "pa": pa.tolist(), "pm": pm.tolist()})
        if k % 50 == 0:
            print(f"  {k}/{len(pick)}", flush=True)

    # ---- 统计 ----
    pri = np.zeros(5)
    for r in recs:
        oo = r["output"]; oo = json.loads(oo) if isinstance(oo, str) else oo
        pri[int(oo["breast_birads"]) - 1] += 1
    pri = pri / pri.sum()

    def kl(p, q):
        p = np.asarray(p, float); q = np.asarray(q, float)
        m = (p > 0) & (q > 0)
        return float(np.sum(p[m] * np.log(p[m] / q[m])))

    L = ["# C3m 训练后视角 A / M 诊断（只读）", "",
         f"- adapter：`{a.adapter}`",
         f"- unknown：{'已加载 ' + a.unknown if a.unknown and Path(a.unknown).exists() else '未保存 → 使用初始化 (E0+E1)/2（初值版）'}",
         f"- 样本：GT≥3 全部 {len(pos_g3)} + 随机 GT≤2 {len(neg_pick)}（seed {a.seed}）= {len(pick)}",
         "- 每例 3 次前向（pass1 取模型自身 p → 视角 A linear 注入 → 视角 M 注入 unknown）", "",
         "| GT | n | margin 中位 | margin≥0.05 比例 | A 更高 | M 更高 | 无分歧(<0.05) |",
         "|---|---|---|---|---|---|---|"]
    for gt in range(1, 6):
        sub = [r for r in out if r["gt"] == gt]
        if not sub:
            L.append(f"| {gt} | 0 | — | — | — | — | — |")
            continue
        d = np.array([r["pa_gt"] - r["pm_gt"] for r in sub])
        marg = np.abs(d)
        L.append(f"| {gt} | {len(sub)} | {np.median(marg):.4f} | {np.mean(marg >= 0.05):.1%} | "
                 f"{np.mean(d >= 0.05):.1%} | {np.mean(d <= -0.05):.1%} | {np.mean(marg < 0.05):.1%} |")

    pa_mean = np.mean([r["pa"] for r in out], axis=0)
    pm_mean = np.mean([r["pm"] for r in out], axis=0)
    L += ["", "## 与训练集 BI-RADS 先验的 KL", "",
          f"- 训练集先验 π = {np.round(pri, 4).tolist()}",
          f"- 视角 A 平均分布 = {np.round(pa_mean, 4).tolist()} → KL(A‖π) = **{kl(pa_mean, pri):.4f}**",
          f"- 视角 M 平均分布 = {np.round(pm_mean, 4).tolist()} → KL(M‖π) = **{kl(pm_mean, pri):.4f}**",
          f"- 逐样本 KL 中位：A {np.median([kl(r['pa'], pri) for r in out]):.4f}，"
          f"M {np.median([kl(r['pm'], pri) for r in out]):.4f}", ""]

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"[done] {a.out}")
    with open("/tmp/c3m_diag_rows.json", "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
