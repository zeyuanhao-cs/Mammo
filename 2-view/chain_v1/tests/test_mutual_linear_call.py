#!/usr/bin/env python3
"""真实调用 soft_kd_loss 的单测：linear 注入 + --mutual（mixer=None，unknown 单独可训练）。

用极小的 stub 模型（emb → 1 层 decoder → head），确保覆盖：
  - soft_kd_loss 的真实签名（含 unknown）与 mutual 分支真的被执行；
  - linear 无 mixer 时 soft_vector 正常；
  - 梯度能同时传到 unknown（视角 M）和 p（经 logits1 的模型参数）；
  - 损失组成 = ce1 + α(ce2+ce_m) + λ·kl_mutual（不含 kl_T）。
"""
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn

sys.path.insert(0, "/home/z/zhao/Mammography/Mammo/2-view")
from chain_v1.train.core import SoftKDConfig, soft_kd_loss  # noqa: E402
from chain_v1.train.soft_mixer import SoftMode  # noqa: E402

torch.manual_seed(0)
V, H, NF = 32, 16, 4
ID0, ID1 = 15, 16
BIRADS = (20, 21, 22, 23, 24)

ok = True


def check(name, cond, detail=""):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
    ok = ok and bool(cond)


class StubInner(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Identity()])


class StubModel(nn.Module):
    """最小可跑结构：emb → layers[0]（hook 注入点）→ head。"""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, H)
        self.language_model = StubInner()
        self.head = nn.Linear(H, V, bias=False)

    def get_input_embeddings(self):
        return self.emb

    def get_output_embeddings(self):
        return self.head

    def forward(self, input_ids=None, attention_mask=None, logits_to_keep=None, **kw):
        h = self.emb(input_ids)
        for layer in self.language_model.layers:
            h = layer(h)
        logits = self.head(h)
        if logits_to_keep:
            logits = logits[:, -int(logits_to_keep):]
        return SimpleNamespace(logits=logits)


model = StubModel()
L, C = 8, 4                     # prompt 4 + completion 4
input_ids = torch.randint(0, V, (1, L))
input_ids[0, -C:] = torch.tensor([ID0, ID1, ID0, BIRADS[2]])   # 末位是 BI-RADS 取值
labels = torch.full((1, L), -100)
labels[0, -C:] = input_ids[0, -C:]
finding_pos = torch.tensor([[0, 1, 2, 3]])      # logits 行号（K-1=4 行内）
birads_pos = torch.tensor([3])
attention_mask = torch.ones_like(input_ids)

cfg = SoftKDConfig(alpha=0.05, lam=0.05, mutual=True, soft_mode=SoftMode.LINEAR,
                   rho_max=1.0, rho_warmup_frac=0.0, id0=ID0, id1=ID1,
                   birads_ids=BIRADS, detach_p=False)
cfg = cfg.resolved()

unknown = nn.Parameter((model.emb.weight[ID0].detach() + model.emb.weight[ID1].detach()) / 2)

print("[1] linear + mutual 真实调用（mixer=None）")
loss, parts = soft_kd_loss(model, input_ids=input_ids, attention_mask=attention_mask,
                           labels=labels, finding_pos=finding_pos, birads_pos=birads_pos,
                           mm_kwargs={}, cfg=cfg, rho=1.0, use_second=True, mixer=None,
                           reuse_embeds=False, unknown=unknown)
check("loss 有限", torch.isfinite(loss), f"(loss={float(loss):.4f})")
check("mutual 分支执行（ce_m>0 且计数和为窗口样本数）",
      float(parts["ce_m"]) > 0 and (parts["n_am"] + parts["n_ma"] + parts["n_none"]) == 1,
      f"ce_m={float(parts['ce_m']):.4f}, n_am/na/none="
      f"{parts['n_am']}/{parts['n_ma']}/{parts['n_none']}")

print("\n[2] 损失组成 = ce1 + α(ce2+ce_m) + λ·kl_mutual（不含 kl_T）")
exp = float(parts["ce1"]) + 0.05 * (float(parts["ce2"]) + float(parts["ce_m"])) \
    + 0.05 * float(parts["kl_mutual"])
check("复算一致（不含 kl_T）", abs(exp - float(loss)) < 1e-6,
      f"(loss={float(loss):.6f}, 复算={exp:.6f}, kl_T={float(parts['kl']):.4f})")

print("\n[3] 梯度：unknown 与 p（模型参数）都能拿到")
model.zero_grad(set_to_none=True)
if unknown.grad is not None:
    unknown.grad = None
loss.backward()
g_unknown = float(unknown.grad.abs().sum()) if unknown.grad is not None else 0.0
g_head = float(model.head.weight.grad.abs().sum())
check("unknown 有梯度", g_unknown > 0, f"|grad|={g_unknown:.3e}")
check("模型参数（p 的来源）有梯度", g_head > 0, f"|grad_head|={g_head:.3e}")
check("qT 仍是 detached（kl_T 不进梯度）", parts["qT"].grad_fn is None)

print("\n[5] extra_metrics（日志量）与 parts 一致，且 loss 公式闭合")
met = {}
loss5, parts5 = soft_kd_loss(model, input_ids=input_ids, attention_mask=attention_mask,
                             labels=labels, finding_pos=finding_pos, birads_pos=birads_pos,
                             mm_kwargs={}, cfg=cfg, rho=1.0, use_second=True, mixer=None,
                             reuse_embeds=False, unknown=unknown, extra_metrics=met)
check("日志里有 ce_m / kl_mutual / kl_T",
      all(k in met for k in ("ce_m", "kl_mutual", "kl_T")), f"keys={sorted(met)}")
check("日志 ce_m 非零且与 parts 一致",
      met["ce_m"] > 0 and abs(met["ce_m"] - float(parts5["ce_m"])) < 1e-6,
      f"(log={met['ce_m']:.6f}, parts={float(parts5['ce_m']):.6f})")
lhs = met["ce1"] + 0.05 * (met["ce2"] + met["ce_m"]) + 0.05 * met["kl_mutual"]
check("loss = ce1+α(ce2+ce_m)+λ·kl_mutual（窗口量闭合，1e-6）",
      abs(lhs - float(loss5)) < 1e-6,
      f"(loss={float(loss5):.6f}, 复算={lhs:.6f}, kl_T={met['kl_T']:.4f} 不参与)")

print("\n[4] 关掉 mutual：同样的调用不触发视角 M")
cfg2 = SoftKDConfig(alpha=0.05, lam=0.05, mutual=False, soft_mode=SoftMode.LINEAR,
                    rho_max=1.0, id0=ID0, id1=ID1, birads_ids=BIRADS).resolved()
loss2, parts2 = soft_kd_loss(model, input_ids=input_ids, attention_mask=attention_mask,
                             labels=labels, finding_pos=finding_pos, birads_pos=birads_pos,
                             mm_kwargs={}, cfg=cfg2, rho=1.0, use_second=True, mixer=None,
                             reuse_embeds=False, unknown=unknown)
exp2 = float(parts2["ce1"]) + 0.05 * float(parts2["ce2"]) + 0.05 * float(parts2["kl"])
check("mutual=False 时 ce_m=0 且不含 mutual 项",
      float(parts2["ce_m"]) == 0.0 and abs(exp2 - float(loss2)) < 1e-6,
      f"(loss={float(loss2):.6f})")

print("\n[6] kl_T 的图：keep_gt_teacher=True 有梯度 / False 无梯度")
cfg_t = SoftKDConfig(alpha=0.05, lam=0.05, mutual=True, keep_gt_teacher=True,
                     soft_mode=SoftMode.LINEAR, rho_max=1.0, id0=ID0, id1=ID1,
                     birads_ids=BIRADS).resolved()
model.zero_grad(set_to_none=True)
loss_t, parts_t = soft_kd_loss(model, input_ids=input_ids, attention_mask=attention_mask,
                               labels=labels, finding_pos=finding_pos, birads_pos=birads_pos,
                               mm_kwargs={}, cfg=cfg_t, rho=1.0, use_second=True, mixer=None,
                               reuse_embeds=False, unknown=unknown)
check("keep=True: kl_T 带图", parts_t["kl_raw"].requires_grad is True,
      f"(requires_grad={parts_t['kl_raw'].requires_grad})")
check("keep=True: teacher 侧 qT 仍 detached", parts_t["qT"].grad_fn is None)
(0.05 * parts_t["kl_raw"]).backward()
g_student = float(model.head.weight.grad.abs().sum()) if model.head.weight.grad is not None else 0.0
check("keep=True: 仅 λ·kl_T 反传也能到学生参数", g_student > 0, f"|grad|={g_student:.3e}")
check("keep=True: loss 含 λ·kl_T",
      abs(float(loss_t) - (float(parts_t["ce1"]) + 0.05 * (float(parts_t["ce2"]) + float(parts_t["ce_m"]))
                           + 0.05 * (float(parts_t["kl"]) + float(parts_t["kl_mutual"])))) < 1e-6,
      f"(loss={float(loss_t):.6f})")

cfg_f = SoftKDConfig(alpha=0.05, lam=0.05, mutual=True, keep_gt_teacher=False,
                     soft_mode=SoftMode.LINEAR, rho_max=1.0, id0=ID0, id1=ID1,
                     birads_ids=BIRADS).resolved()
loss_f, parts_f = soft_kd_loss(model, input_ids=input_ids, attention_mask=attention_mask,
                               labels=labels, finding_pos=finding_pos, birads_pos=birads_pos,
                               mm_kwargs={}, cfg=cfg_f, rho=1.0, use_second=True, mixer=None,
                               reuse_embeds=False, unknown=unknown)
check("keep=False: kl_T 无图（no_grad）", parts_f["kl_raw"].requires_grad is False,
      f"(requires_grad={parts_f['kl_raw'].requires_grad})")
check("keep=False: loss 不含 λ·kl_T",
      abs(float(loss_f) - (float(parts_f["ce1"]) + 0.05 * (float(parts_f["ce2"]) + float(parts_f["ce_m"]))
                           + 0.05 * float(parts_f["kl_mutual"]))) < 1e-6,
      f"(loss={float(loss_f):.6f}, kl_T={float(parts_f['kl']):.4f} 未参与)")

print("\nMUTUAL_LINEAR_CALL_" + ("OK" if ok else "FAILED"))
sys.exit(0 if ok else 1)
