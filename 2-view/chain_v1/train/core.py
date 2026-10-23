#!/usr/bin/env python3
"""chain_v1 训练核心：两次前向的软证据 + KD 损失。

与具体模型/数据无关（只要 forward 支持 `logits_to_keep`），便于用极小模型在 CPU 上做单元测试。

序列约定
--------
input_ids = [prompt | completion]，labels 在 prompt 段为 -100、completion 段为 GT token id（连续后缀）。
`finding_pos` / `birads_pos` 均为**相对 completion 起点**的 token 下标。
由于 `logits_to_keep=K`（K = completion 长度 + 1）时第 r 行对应"位置 L-K+r"，即预测 completion 第 r 个 token，
所以行号 = completion 内下标，可直接索引。

损失
----
    L = CE1 + alpha * CE2(birads) + lam * KL(q_T || q_S)
两次前向共享同一份输入；第 2 次仅在 decoder 第 0 层入口把 4 个 finding 取值位置的 hidden state
换成 (1-p)E["0"] + p E["1"]（等价于替换输入嵌入）。第 2 次是否启用由 rho 控制。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

ID0, ID1 = 15, 16                      # '0' / '1'
BIRADS_IDS = (16, 17, 18, 19, 20)      # '1'..'5'


@dataclass
class SoftKDConfig:
    alpha: float = 1.0        # CE2 权重
    lam: float = 1.0          # KL 权重
    rho_max: float = 0.5      # 第 2 次前向的最大启用概率
    rho_warmup_frac: float = 0.2
    detach_p: bool = False    # True 时 p 不回传梯度
    no_kd: bool = False       # lam = 0
    no_soft: bool = False     # 关闭第 2 次前向（纯 SFT 基线）
    soft_mode: str = "logit_mlp"   # linear | logit_mlp
    id0: int = ID0
    id1: int = ID1
    birads_ids: Sequence[int] = field(default_factory=lambda: tuple(BIRADS_IDS))

    def resolved(self) -> "SoftKDConfig":
        c = SoftKDConfig(**{**self.__dict__})
        if c.no_kd:
            c.lam = 0.0
        if c.no_soft:
            c.rho_max = 0.0
            c.lam = 0.0
            c.alpha = 0.0
        return c


def rho_at(step: int, total_steps: int, cfg: SoftKDConfig) -> float:
    """线性 warmup：前 rho_warmup_frac 的步数内从 0 升到 rho_max。"""
    if cfg.rho_max <= 0:
        return 0.0
    if cfg.rho_warmup_frac <= 0:      # 无 warmup：第 0 步即 rho_max
        return cfg.rho_max
    warm = max(1, int(math.ceil(total_steps * cfg.rho_warmup_frac)))
    return cfg.rho_max * min(1.0, step / warm)


def first_decoder_layer(model):
    """定位语言模型第一层 decoder（兼容 PEFT / 多模态包装）。

    扫描 named_modules 里名字含 language_model 的 ModuleList（名为 *.layers），否则退回第一个
    非 visual 的 *.layers / *.blocks。
    """
    fallback = None
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.ModuleList) or len(mod) == 0:
            continue
        if not (name.endswith(".layers") or name.endswith("layers")):
            continue
        if "language_model" in name:
            return mod[0]
        if fallback is None and "visual" not in name:
            fallback = mod[0]
    if fallback is not None:
        return fallback
    names = [n for n, m in model.named_modules() if isinstance(m, torch.nn.ModuleList)]
    raise RuntimeError(f"找不到 decoder layer，可用 ModuleList: {names[:10]}")


def find_text_model(model):
    """定位语言模型模块（我们只跑 LM，跳过视觉塔）。"""
    for name, mod in model.named_modules():
        if name.endswith("language_model") and not isinstance(mod, torch.nn.ModuleList):
            return mod
    for name, mod in model.named_modules():      # 回退：名为 text / *.text 的语言模型
        if (name == "text" or name.endswith(".text")) and not isinstance(mod, torch.nn.ModuleList):
            return mod
    return None


def capture_fused_embeds(lm):
    """在 text model 入口捕获第一次前向的全部实参（inputs_embeds / position_ids / attention_mask ...）。

    第 2 次前向会**原样复用**这些参数，只把 inputs_embeds 换掉，避免丢掉 3D rope position_ids 等信息。
    """
    box = {}

    def pre(module, args, kwargs):
        box["args"] = tuple(args)
        box["kwargs"] = dict(kwargs)
        e = kwargs.get("inputs_embeds")
        if e is None and args and torch.is_tensor(args[0]):
            e = args[0]
        if torch.is_tensor(e):
            box["e"] = e

    return box, lm.register_forward_pre_hook(pre, with_kwargs=True)


def finding_p(logits1: torch.Tensor, finding_pos: torch.Tensor, id0: int, id1: int,
              detach: bool = False) -> torch.Tensor:
    """p = P('1') / (P('0') + P('1'))，逐样本逐 finding。

    logits1: [B, K, V]；finding_pos: [B, n_find]（completion 内下标）；返回 [B, n_find]
    """
    b, n = finding_pos.shape
    l0 = logits1[torch.arange(b), finding_pos, id0]   # [B,n]
    l1 = logits1[torch.arange(b), finding_pos, id1]
    p = torch.sigmoid(l1 - l0)
    return p.detach() if detach else p


def birads_dist(logits: torch.Tensor, birads_pos: torch.Tensor, ids: Sequence[int]) -> torch.Tensor:
    """取 BI-RADS 位置在 '1'..'5' 上归一化的分布。logits [B,K,V]，birads_pos [B]。"""
    b = logits.shape[0]
    rows = logits[torch.arange(b), birads_pos][:, list(ids)].float()   # [B,5]
    return F.softmax(rows, dim=-1)


def _mix(emb_weight: torch.Tensor, p: torch.Tensor, id0: int, id1: int, dtype) -> torch.Tensor:
    """linear 模式：(1-p)E[0] + p E[1]。"""
    e0, e1 = emb_weight[id0], emb_weight[id1]
    return ((1.0 - p) * e0 + p * e1).to(dtype)


def make_vec_hook(layer, seq_positions, vectors):
    """在 decoder 第 0 层入口把指定位置的 hidden state 换成给定向量（= 替换输入嵌入）。"""

    def hook(module, args):
        hs = args[0].clone()
        for seq_pos, vec in zip(seq_positions, vectors):
            hs[0, seq_pos] = vec.to(hs.dtype)
        return (hs,) + tuple(args[1:])

    return layer.register_forward_pre_hook(hook)


def soft_kd_loss(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    finding_pos: torch.Tensor,
    birads_pos: torch.Tensor,
    mm_kwargs: Optional[Dict] = None,
    cfg: Optional[SoftKDConfig] = None,
    rho: float = 0.0,
    use_second: Optional[bool] = None,
    logits_to_keep: Optional[int] = None,
    mixer=None,
    reuse_embeds: bool = True,
    extra_metrics: Optional[Dict[str, float]] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """返回 (loss, parts)。

    parts: {"ce1", "ce2", "kl", "p"(B,n_find), "qT", "qS", "mask"(B,)}
    """
    cfg = cfg or SoftKDConfig()
    mm_kwargs = dict(mm_kwargs or {})
    B, L = input_ids.shape

    # completion 段：labels != -100 的连续后缀
    comp_mask = labels.ne(-100)
    comp_len = comp_mask.sum(dim=1)                       # [B]
    prompt_len = L - comp_len                             # [B]
    assert torch.all(comp_mask[:, 0] == False) or L == comp_len.max(), "labels 需为 prompt(-100) + completion 连续后缀"
    assert B == 1, "当前实现按 batch=1 设计（rho 逐样本、位置索引简单）；需要 batch>1 时再扩展"
    K = int(comp_len[0].item()) + 1
    if logits_to_keep is not None:
        K = logits_to_keep
        assert K == int(comp_len[0].item()) + 1, "logits_to_keep 必须等于 completion 长度 + 1"

    common = dict(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, **mm_kwargs)

    # ---------------- 第 1 次：标准 teacher forcing（顺带捕获融合嵌入） ----------------
    lm = find_text_model(model) if reuse_embeds else None
    box, cap = ({}, None)
    if lm is not None:
        box, cap = capture_fused_embeds(lm)
    try:
        logits1 = _forward(model, common, K)
    finally:
        if cap is not None:
            cap.remove()

    targets = labels[:, L - K + 1:]                        # 与 logits 行对齐（行 r 预测 completion 第 r 个）
    ce1 = _ce_at(logits1, targets)

    p = finding_p(logits1, finding_pos, cfg.id0, cfg.id1, cfg.detach_p)   # [B,n]
    qT = birads_dist(logits1, birads_pos, cfg.birads_ids).detach()

    if use_second is None:
        use_second = rho > 0 and not cfg.no_soft

    zero = input_ids.new_zeros((), dtype=torch.float32)
    if not use_second:
        parts = {"ce1": ce1.detach(), "ce2": zero, "kl": zero, "p": p, "qT": qT, "qS": None,
                 "ce1_raw": ce1}
        if extra_metrics is not None:
            extra_metrics.update({"ce1": float(ce1.detach()), "ce2": 0.0, "kl": 0.0,
                                  "mean_p": float(p.detach().mean())})
            return ce1, parts
        return ce1, parts

    # ---------------- 第 2 次：finding 位置软证据注入 ----------------
    from .soft_mixer import SoftMode, soft_vector

    layer0 = first_decoder_layer(model)
    emb_w = model.get_input_embeddings().weight
    seq_positions, vectors = [], []
    for b in range(B):
        for j in range(finding_pos.shape[1]):
            idx = int(finding_pos[b, j].item())
            if idx < 0 or idx >= int(comp_len[b].item()):
                continue
            seq_positions.append(int(prompt_len[b].item()) + idx)
            from .soft_mixer import base_id_from_p
            base_id = base_id_from_p(p[b, j], cfg.id0, cfg.id1)   # 模型自己的判断，不用 GT
            vectors.append(soft_vector(mixer, emb_w, p[b, j], j, base_id, cfg.id0, cfg.id1,
                                       emb_w.dtype, cfg.soft_mode))

    fused = box.get("e")
    used_reuse = False
    if fused is not None and lm is not None:
        hs = fused.clone()
        for seq_pos, vec in zip(seq_positions, vectors):
            hs[0, seq_pos] = vec.to(hs.dtype)
        # 复用第 1 次前向传给 language_model 的全部实参，只替换 inputs_embeds
        a2 = list(box.get("args", ()))
        k2 = dict(box.get("kwargs", {}))
        if a2 and torch.is_tensor(a2[0]):
            a2[0] = hs
            k2.pop("inputs_embeds", None)
        else:
            k2["inputs_embeds"] = hs
        out2 = lm(*a2, **k2)
        hidden = getattr(out2, "last_hidden_state", None)
        if hidden is None:
            hidden = out2[0]
        lm_head = getattr(model, "lm_head", None) or model.get_output_embeddings()
        logits2 = lm_head(hidden[:, -K:])
        used_reuse = True
    else:
        handle = make_vec_hook(layer0, seq_positions, vectors)
        try:
            logits2 = _forward(model, common, K)
        finally:
            handle.remove()

    qS = birads_dist(logits2, birads_pos, cfg.birads_ids)                 # [B,5]
    # targets 恰好覆盖整个 completion（长度 K-1 = comp_len），其下标即 completion 内下标
    gt_birads_tok = targets[torch.arange(B), birads_pos]
    ce2 = F.cross_entropy(logits2[torch.arange(B), birads_pos].float(), gt_birads_tok, reduction="mean")
    kl = torch.sum(qT * (torch.log(qT + 1e-8) - torch.log(qS + 1e-8)), dim=-1).mean()

    loss = ce1 + cfg.alpha * ce2 + cfg.lam * kl
    parts = {"ce1": ce1.detach(), "ce2": ce2.detach(), "kl": kl.detach(), "p": p, "qT": qT, "qS": qS,
             "ce1_raw": ce1, "ce2_raw": ce2, "kl_raw": kl,
             "birads_logits": logits2[0, birads_pos[0]][list(cfg.birads_ids)].detach(),
             "reuse": used_reuse}
    if extra_metrics is not None:
        extra_metrics.update({"ce1": float(ce1.detach()), "ce2": float(ce2.detach()),
                              "kl": float(kl.detach()), "mean_p": float(p.detach().mean())})
    return loss, parts


def _forward(model, common, K):
    try:
        out = model(logits_to_keep=K, **common)
    except TypeError:
        out = model(num_logits_to_keep=K, **common)
    return out.logits if hasattr(out, "logits") else out[0]


def _ce_at(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """logits [B,K,V] 与 targets [B,k] 对齐求平均 CE（只用前 k 行）。"""
    b, k = targets.shape
    rows = logits[:, :k].float()
    valid = targets.ne(-100)
    if not valid.any():
        return torch.zeros((), device=logits.device, dtype=torch.float32)
    per_tok = F.cross_entropy(rows.reshape(-1, rows.shape[-1]), targets.reshape(-1), reduction="none")
    return per_tok[valid.reshape(-1)].mean()
