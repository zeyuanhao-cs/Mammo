#!/usr/bin/env python3
"""单元测试：第 2 次前向「复用第 1 次融合嵌入」vs「重跑视觉塔」，BI-RADS logits 与各参数梯度是否一致。

CPU + 极小 VL 桩模型（emb → visual 融合 → text → lm_head）。
复用路径：在 text model 入口截获融合后的 inputs_embeds（保留计算图），第 2 次只跑 LM；
梯度仍会经这份嵌入回传到 emb 与 visual。
"""
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

CV1 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CV1.parent))

from chain_v1.train.core import SoftKDConfig, soft_kd_loss  # noqa: E402
from chain_v1.train.soft_mixer import SoftMixer  # noqa: E402

D = 16
V = 32


class TinyText(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(D, D) for _ in range(2)])

    def forward(self, inputs_embeds=None, attention_mask=None, use_cache=False,
                mm_token_type_ids=None, **kw):
        h = inputs_embeds
        for l in self.layers:
            h = torch.tanh(l(h))
        return SimpleNamespace(last_hidden_state=h)


class TinyVisual(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, D)

    def forward(self, pixel_values):
        return self.proj(pixel_values)


class _Wrap(nn.Module):
    def __init__(self, lm):
        super().__init__()
        self.language_model = lm


class TinyVL(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.visual = TinyVisual()
        self.lm_head = nn.Linear(D, V, bias=False)
        self.model = _Wrap(TinyText())          # 唯一引用，避免 named_modules 去重把别名藏掉

    @property
    def text(self):
        return self.model.language_model

    def get_input_embeddings(self):
        return self.emb

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, pixel_values=None,
                logits_to_keep=None, num_logits_to_keep=None, mm_token_type_ids=None,
                use_cache=False, **kw):
        h = self.emb(input_ids) if input_ids is not None else inputs_embeds
        if pixel_values is not None:
            v = self.visual(pixel_values)              # [B, n_img, D]，模拟"融合图像特征"
            h = h.clone()
            h[:, :v.shape[1]] = h[:, :v.shape[1]] + v
        out = self.text(inputs_embeds=h, attention_mask=attention_mask)
        logits = self.lm_head(out.last_hidden_state)
        k = logits_to_keep or num_logits_to_keep
        if k:
            logits = logits[:, -k:]
        return SimpleNamespace(logits=logits)


def build_case(seed=0, P=8, C=6):
    g = torch.Generator().manual_seed(seed)
    comp = torch.randint(3, 12, (1, C), generator=g)
    prompt = torch.randint(3, 12, (1, P), generator=g)
    ids = torch.cat([prompt, comp], dim=1)
    labels = torch.cat([torch.full((1, P), -100, dtype=torch.long), comp], dim=1)
    fpos = torch.tensor([[1, 2, 3, 4]])
    bpos = torch.tensor([C - 1])
    mm = {"pixel_values": torch.randn(1, 4, 3, generator=g)}
    return ids, labels, fpos, bpos, mm


def run(model, mixer, inputs, reuse):
    ids, labels, fpos, bpos, mm = inputs
    loss, parts = soft_kd_loss(model, ids, torch.ones_like(ids), labels, fpos, bpos,
                               mm_kwargs=mm, cfg=SoftKDConfig(soft_mode="logit_mlp"),
                               rho=1.0, use_second=True, mixer=mixer, reuse_embeds=reuse)
    return loss, parts


def grads_of(model, mixer, loss):
    for p in list(model.parameters()) + list(mixer.parameters()):
        p.grad = None
    loss.backward()
    g = {f"model:{n}": p.grad.detach().clone() for n, p in model.named_parameters()}
    g.update({f"mixer:{n}": p.grad.detach().clone() for n, p in mixer.named_parameters()})
    return g


def main():
    torch.manual_seed(0)
    base = TinyVL()
    mixer0 = SoftMixer(D, 8, 4, init="small")
    inputs = build_case()

    m1, mix1 = copy.deepcopy(base), copy.deepcopy(mixer0)
    m2, mix2 = copy.deepcopy(base), copy.deepcopy(mixer0)

    loss1, p1 = run(m1, mix1, inputs, reuse=True)
    loss2, p2 = run(m2, mix2, inputs, reuse=False)
    assert p1["reuse"] is True and p2["reuse"] is False, "复用开关未按预期生效"

    d_logits = float((p1["birads_logits"] - p2["birads_logits"]).abs().max())
    d_loss = abs(float(loss1) - float(loss2))

    g1 = grads_of(m1, mix1, loss1)
    g2 = grads_of(m2, mix2, loss2)
    assert set(g1) == set(g2)
    dg = {k: float((g1[k] - g2[k]).abs().max()) for k in g1}
    worst_key = max(dg, key=dg.get)

    print(f"BI-RADS logits 最大差 : {d_logits:.3e}")
    print(f"loss 差              : {d_loss:.3e}")
    print(f"梯度最大差            : {dg[worst_key]:.3e}  ({worst_key})")
    print("分组梯度最大差：")
    for tag in ("emb", "visual", "text", "lm_head", "mixer"):
        keys = [k for k in dg if tag in k]
        if keys:
            print(f"  {tag:<8} {max(dg[k] for k in keys):.3e}")

    assert d_logits < 1e-6, "复用与不复用的 BI-RADS logits 应一致"
    assert d_loss < 1e-6
    assert max(dg.values()) < 1e-6, "两路径的梯度应一致"
    print("test_embed_reuse: OK（复用融合嵌入不改变前向/梯度）")


if __name__ == "__main__":
    main()
