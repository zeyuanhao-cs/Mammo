#!/usr/bin/env python3
"""软证据注入的两种模式。

linear    : 输入 = (1-p)·E["0"] + p·E["1"]（对照；对 p 的响应是"阈值式"的）
logit_mlp : 输入 = E[argmax token] + MLP([log p, log(1-p)], finding_type)（主方案）
            MLP: 2 -> 64 -> hidden_size，中间 GELU；最后一层 zero-init → 训练起点与硬 token 完全一致
            4 类 finding 共用 MLP，另加 4×64 的可学习类型嵌入（加到第一层输出上）
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class SoftMode:
    LINEAR = "linear"
    LOGIT_MLP = "logit_mlp"


def base_id_from_p(p, id0=15, id1=16) -> int:
    """模型自己的判断：p>0.5 取 "1"，否则取 "0"（与 soft-decode 推理一致，不看 GT）。"""
    pv = float(p) if not torch.is_tensor(p) else float(p.detach())
    return int(id1) if pv > 0.5 else int(id0)


class SoftMixer(nn.Module):
    def __init__(self, hidden_size: int, d_hidden: int = 64, n_types: int = 4, init: str = "small"):
        super().__init__()
        self.gate = "none"     # none | uncert（随 mixer 一起保存/加载，推理自动生效）
        self.fc1 = nn.Linear(2, d_hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_hidden, hidden_size)
        self.type_emb = nn.Parameter(torch.zeros(n_types, d_hidden))
        if init == "zero":
            nn.init.zeros_(self.fc2.weight)
        else:                      # small: N(0, 1e-3)
            nn.init.normal_(self.fc2.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, logp: torch.Tensor, type_idx: int) -> torch.Tensor:
        """logp: [2] = [log p, log(1-p)]；返回 [hidden_size] 的增量。"""
        h = self.fc1(logp) + self.type_emb[type_idx]
        return self.fc2(self.act(h))


def soft_vector(mixer, emb_w, p, type_idx, base_id, id0=15, id1=16, dtype=torch.float32,
                mode=SoftMode.LOGIT_MLP, gate=None):
    """构造该位置要写入的 hidden state（= 输入嵌入）。

    p       : tensor（允许梯度）或 float，P("1")
    base_id : 该位置的硬 token id。训练时 = GT 取值 token；soft-decode 时 = 刚生成的 argmax token。
              零初始化 MLP 时输出 = E[base_id]，与硬 token 完全一致。
    """
    if not torch.is_tensor(p):
        p = torch.tensor(float(p), dtype=torch.float32)
    p = p.clamp(1e-6, 1.0 - 1e-6)
    if mode == SoftMode.LINEAR:
        return (1.0 - p) * emb_w[id0] + p * emb_w[id1]

    if mixer is None:
        raise ValueError("soft_mode=logit_mlp 需要传入 mixer")
    fw = mixer.fc1.weight
    # 修 bug：feat 原先只转 dtype 不转 device——mixer 在 GPU 而 p 是 CPU 标量张量时
    # （soft-decode 推理路径 p 由 float 构造）会 device 不匹配报错。
    feat = torch.stack([torch.log(p), torch.log1p(-p)]).to(device=fw.device, dtype=fw.dtype)
    delta = mixer(feat, int(type_idx))
    # 不确定性门控：soft = E[base] + u·MLP(...)，u = 4p(1-p)（p=0.5 时 u=1，与不加开关一致）
    g = mixer.gate if gate is None else gate
    if g == "uncert":
        u = (4.0 * p * (1.0 - p)).to(device=delta.device, dtype=delta.dtype)
        delta = delta * u
    return emb_w[int(base_id)].to(delta.dtype) + delta


def save_soft_mixer(mixer, out_dir: str, name: str = "soft_mixer.pt"):
    import os
    if mixer is None:
        return None
    path = os.path.join(out_dir, name)
    torch.save({"state_dict": mixer.state_dict(),
                "hidden_size": mixer.fc2.out_features,
                "d_hidden": mixer.fc1.out_features,
                "n_types": mixer.type_emb.shape[0],
                "soft_gate": getattr(mixer, "gate", "none")}, path)
    return path


def load_soft_mixer(path: str, device="cpu", dtype=torch.float32):
    blob = torch.load(path, map_location="cpu")
    mixer = SoftMixer(blob["hidden_size"], blob.get("d_hidden", 64), blob.get("n_types", 4))
    mixer.gate = blob.get("soft_gate", "none")   # 推理自动继承训练时的门控设置
    mixer.load_state_dict(blob["state_dict"])
    return mixer.to(device=device, dtype=dtype).eval()
