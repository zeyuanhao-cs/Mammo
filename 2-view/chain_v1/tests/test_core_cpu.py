#!/usr/bin/env python3
"""单元测试 2、3（CPU，极小随机模型）：
  2) --detach-p 开/关时 p 是否带梯度；
  3) rho=0（不做第 2 次前向）时损失与标准 SFT CE 完全一致；
     另外顺带验证 soft 路径（use_second=True）能跑通且 p∈[0,1]。
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

CV1 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CV1.parent))

from chain_v1.train.core import (SoftKDConfig, finding_p, first_decoder_layer,  # noqa: E402
                                rho_at, soft_kd_loss)
from chain_v1.train.soft_mixer import SoftMixer  # noqa: E402


class TinyLayer(nn.Module):
    def forward(self, hidden_states, **kw):
        return hidden_states


class TinyLM(nn.Module):
    """满足 core 的最小接口：get_input_embeddings / forward(logits_to_keep)。"""

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


def build_batch(P=10, C=8, V=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    comp = torch.randint(3, 12, (1, C), generator=g)
    prompt = torch.randint(3, 12, (1, P), generator=g)
    input_ids = torch.cat([prompt, comp], dim=1)
    labels = torch.cat([torch.full((1, P), -100, dtype=torch.long), comp], dim=1)
    finding_pos = torch.tensor([[2, 3, 4, 5]])
    birads_pos = torch.tensor([6])
    return input_ids, labels, finding_pos, birads_pos


def test_detach_p():
    torch.manual_seed(0)
    logits = torch.randn(1, 12, 32, requires_grad=True)
    pos = torch.tensor([[2, 3, 4, 5]])
    p_grad = finding_p(logits, pos, 15, 16, detach=False)
    p_nograd = finding_p(logits, pos, 15, 16, detach=True)
    assert p_grad.requires_grad, "detach_p=False 时 p 应带梯度"
    assert not p_nograd.requires_grad, "detach_p=True 时 p 不应带梯度"
    pd_ = p_grad.detach()
    assert p_grad.shape == (1, 4) and float(pd_.min()) >= 0 and float(pd_.max()) <= 1
    p_grad.sum().backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0, "梯度未回传到 logits"
    print("test_detach_p: OK  (detach=False requires_grad=True / detach=True requires_grad=False)")


def test_rho0_equals_sft():
    torch.manual_seed(0)
    model = TinyLM()
    input_ids, labels, finding_pos, birads_pos = build_batch()
    P = 10
    C = 8

    # 参考：标准 SFT（completion 段平均 CE）
    ref_logits = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids)).logits
    ref = F.cross_entropy(ref_logits[:, P - 1:P + C - 1].reshape(-1, 32),
                          labels[:, P:P + C].reshape(-1))

    cfg = SoftKDConfig()
    loss, parts = soft_kd_loss(model, input_ids, torch.ones_like(input_ids), labels,
                               finding_pos, birads_pos, mm_kwargs={}, cfg=cfg,
                               rho=0.0, use_second=False)
    assert abs(float(loss) - float(ref)) < 1e-6, (float(loss), float(ref))
    assert float(parts["ce2"]) == 0.0 and float(parts["kl"]) == 0.0
    assert rho_at(0, 1000, cfg) == 0.0 and rho_at(int(1000 * cfg.rho_warmup_frac), 1000, cfg) == cfg.rho_max
    print(f"test_rho0_equals_sft: OK  (loss={float(loss):.6f} == SFT CE={float(ref):.6f})")

    # soft 路径冒烟（logit_mlp 需要 mixer；此极小模型无视觉塔，reuse 会回退）
    mixer = SoftMixer(16, 8, 4, init="small")
    loss2, parts2 = soft_kd_loss(model, input_ids, torch.ones_like(input_ids), labels,
                                 finding_pos, birads_pos, mm_kwargs={}, cfg=cfg,
                                 rho=1.0, use_second=True, mixer=mixer)
    p = parts2["p"].detach()
    assert 0.0 <= float(p.min()) and float(p.max()) <= 1.0
    assert torch.isfinite(loss2), "soft 路径 loss 非有限"
    assert abs(float(parts2["qS"].sum()) - 1.0) < 1e-5
    loss2.backward()
    print(f"test_soft_path: OK  (loss={float(loss2):.6f}, ce1={float(parts2['ce1']):.6f}, "
          f"ce2={float(parts2['ce2']):.6f}, kl={float(parts2['kl']):.6f})")


if __name__ == "__main__":
    test_detach_p()
    test_rho0_equals_sft()
