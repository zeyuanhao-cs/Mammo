#!/usr/bin/env python3
"""soft_mixer 单元测试（CPU）：
  A) zero 初始化：第 2 次前向 == "把 finding 取值换成模型自己 argmax token" 的硬前向
  B) GT=1 但 p=0.15 时，注入位置的 base 必须是 E["0"]（不看 GT）
  C) small 初始化（N(0,1e-3)）：报告与硬 token 输出的最大差
  D) p 从 0.05 → 0.2：对 p 的梯度非零
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

CV1 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CV1.parent))

from chain_v1.train import soft_mixer as _sm  # noqa: E402
from chain_v1.train.core import SoftKDConfig, soft_kd_loss  # noqa: E402
from chain_v1.train.soft_mixer import (SoftMixer, base_id_from_p,  # noqa: E402
                                       soft_vector)

ID0, ID1 = 15, 16
BIR = [16, 17, 18, 19, 20]


class TinyLayer(nn.Module):
    def forward(self, hidden_states, **kw):
        return hidden_states


class TinyLM(nn.Module):
    def __init__(self, vocab=32, d=16, n_layers=2):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([TinyLayer() for _ in range(n_layers)])
        self.head = nn.Linear(d, vocab, bias=False)
        self.model = SimpleNamespace(language_model=SimpleNamespace(layers=self.layers))

    def get_input_embeddings(self):
        return self.emb

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None,
                use_cache=False, logits_to_keep=None, num_logits_to_keep=None, **kw):
        h = self.emb(input_ids) if input_ids is not None else inputs_embeds
        for l in self.layers:
            h = l(h)
        logits = self.head(h)
        k = logits_to_keep or num_logits_to_keep
        if k:
            logits = logits[:, -k:]
        return SimpleNamespace(logits=logits)


def build_batch(P=10, C=8, V=32, seed=0, force_gt=None):
    g = torch.Generator().manual_seed(seed)
    comp = torch.randint(3, 12, (1, C), generator=g)
    if force_gt is not None:
        comp[0, 2] = force_gt[0]
        comp[0, 3] = force_gt[1]
        comp[0, 4] = force_gt[2]
        comp[0, 5] = force_gt[3]
    prompt = torch.randint(3, 12, (1, P), generator=g)
    ids = torch.cat([prompt, comp], dim=1)
    labels = torch.cat([torch.full((1, P), -100, dtype=torch.long), comp], dim=1)
    return ids, labels, torch.tensor([[2, 3, 4, 5]]), torch.tensor([6])


def q_of(model, ids, att, K, bpos):
    with torch.inference_mode():
        logits = model(input_ids=ids, attention_mask=att, logits_to_keep=K).logits
    return torch.softmax(logits[0, bpos][BIR].float(), dim=-1)


def test_zero_init_equals_own_argmax_hard():
    torch.manual_seed(0)
    model = TinyLM()
    mixer = SoftMixer(16, 8, 4, init="zero")
    ids, labels, fpos, bpos = build_batch()
    att = torch.ones_like(ids)
    P = 10
    K = labels.shape[1] - P + 1
    cfg = SoftKDConfig(soft_mode="logit_mlp")

    _, hard = soft_kd_loss(model, ids, att, labels, fpos, bpos, cfg=cfg, rho=0.0, use_second=False)
    p = hard["p"].detach()

    # 参考：把 finding 取值 token 换成模型自己的 argmax token，走普通前向
    ids_ref = ids.clone()
    for j in range(fpos.shape[1]):
        ids_ref[0, P + int(fpos[0, j])] = ID1 if float(p[0, j]) > 0.5 else ID0
    q_ref = q_of(model, ids_ref, att, K, int(bpos[0]))

    _, soft = soft_kd_loss(model, ids, att, labels, fpos, bpos, cfg=cfg, rho=1.0,
                           use_second=True, mixer=mixer)
    d = float((soft["qS"].detach() - q_ref).abs().max())
    assert d < 1e-7, f"zero-init 应等于'模型自己硬征象'前向，实际 maxdiff={d}"
    print(f"test_zero_init_equals_own_argmax_hard: OK  qS-q_ref max={d:.2e}")


def test_base_uses_model_not_gt():
    torch.manual_seed(0)
    model = TinyLM()
    mixer = SoftMixer(16, 8, 4, init="zero")
    # 4 个 finding 的 GT 全部写成 "1"（id=16）
    ids, labels, fpos, bpos = build_batch(force_gt=(ID1, ID1, ID1, ID1))
    att = torch.ones_like(ids)
    cfg = SoftKDConfig(soft_mode="logit_mlp")

    seen = []
    orig = _sm.soft_vector

    def spy(*a, **kw):
        # 签名: (mixer, emb_w, p, type_idx, base_id, id0, id1, dtype, mode)
        seen.append((float(a[2]), int(a[4])))
        return orig(*a, **kw)

    _sm.soft_vector = spy
    try:
        soft_kd_loss(model, ids, att, labels, fpos, bpos, cfg=cfg, rho=1.0,
                     use_second=True, mixer=mixer)
    finally:
        _sm.soft_vector = orig

    assert seen, "未捕获到 soft_vector 调用"
    for pv, bid in seen:
        expect = ID1 if pv > 0.5 else ID0
        assert bid == expect, f"p={pv} 时 base 应为 {expect}，实际 {bid}（GT 是 1）"
    low = [pv for pv, _ in seen if pv < 0.5]
    print(f"test_base_uses_model_not_gt: OK  n={len(seen)}, p<0.5 的 {len(low)} 个位置 base 均为 '0'"
          f"（GT 全为 1）")

    # 直接验证 GT=1、p=0.15 的向量 = E["0"]
    p15 = torch.tensor(0.15)
    b = base_id_from_p(p15)
    assert b == ID0
    vec = soft_vector(mixer, model.emb.weight, p15, 0, b, ID0, ID1, torch.float32, "logit_mlp")
    assert torch.equal(vec, model.emb.weight[ID0]), "p=0.15 时注入向量应等于 E['0']"
    print("  p=0.15, GT=1 → base=E['0']：OK")


def test_small_init_gap():
    torch.manual_seed(0)
    model = TinyLM()
    mixer = SoftMixer(16, 8, 4, init="small")
    ids, labels, fpos, bpos = build_batch()
    att = torch.ones_like(ids)
    P = 10
    K = labels.shape[1] - P + 1
    cfg = SoftKDConfig(soft_mode="logit_mlp")

    _, hard = soft_kd_loss(model, ids, att, labels, fpos, bpos, cfg=cfg, rho=0.0, use_second=False)
    p = hard["p"].detach()
    ids_ref = ids.clone()
    for j in range(fpos.shape[1]):
        ids_ref[0, P + int(fpos[0, j])] = ID1 if float(p[0, j]) > 0.5 else ID0
    q_ref = q_of(model, ids_ref, att, K, int(bpos[0]))

    _, soft = soft_kd_loss(model, ids, att, labels, fpos, bpos, cfg=cfg, rho=1.0,
                           use_second=True, mixer=mixer)
    dq = float((soft["qS"].detach() - q_ref).abs().max())
    deltas = [float((soft_vector(mixer, model.emb.weight, p[0, j], j,
                                base_id_from_p(p[0, j]), ID0, ID1, torch.float32, "logit_mlp")
                    - model.emb.weight[base_id_from_p(p[0, j])]).abs().max()) for j in range(4)]
    print(f"test_small_init_gap: OK  与硬 token 前向的 q maxdiff={dq:.3e}；"
          f"|Δemb|max={max(deltas):.3e}（hidden=16, |E|≈{float(model.emb.weight.abs().max()):.3f}）")
    assert dq < 0.05, "small 初始化不应显著改变输出"


def test_p_gradient_nonzero():
    torch.manual_seed(1)
    emb = nn.Embedding(32, 16)
    mixer = SoftMixer(16, 8, 4, init="small")
    grads = {}
    for pv in (0.05, 0.2):
        p = torch.tensor(pv, requires_grad=True)
        vec = soft_vector(mixer, emb.weight, p, 0, ID0, ID0, ID1, torch.float32, "logit_mlp")
        vec.sum().backward()
        grads[pv] = float(p.grad)
        assert abs(grads[pv]) > 0, f"p={pv} 梯度为 0"
    assert grads[0.05] != grads[0.2]
    print(f"test_p_gradient_nonzero: OK  dL/dp(0.05)={grads[0.05]:.6f}, dL/dp(0.20)={grads[0.2]:.6f}")


if __name__ == "__main__":
    test_zero_init_equals_own_argmax_hard()
    test_base_uses_model_not_gt()
    test_small_init_gap()
    test_p_gradient_nonzero()
