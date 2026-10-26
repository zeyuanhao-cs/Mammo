#!/usr/bin/env python3
"""axis 模式单测 + linear/logit_mlp 回归（CPU）。

axis: v = E0 + s·(E1−E0),  s = clamp(p + u·r, 0, 1),  u = 4p(1−p),  r = axis_head(h)
回归: 与冻结副本 chain_v1/frozen_stage1/soft_mixer.py 逐元素比对 linear / logit_mlp(/uncert)
"""
import importlib.util
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, "/home/z/zhao/Mammography/Mammo/2-view")
from chain_v1.train.soft_mixer import (SoftMixer, SoftMode, load_soft_mixer,  # noqa: E402
                                      save_soft_mixer, soft_vector)

BASE = Path("/home/z/zhao/Mammography/model/Qwen3.5-4B")
idx = json.loads((BASE / "model.safetensors.index.json").read_text())["weight_map"]
key = next(k for k in idx if k.endswith("embed_tokens.weight"))
with safe_open(str(BASE / idx[key]), framework="pt", device="cpu") as f:
    emb = f.get_tensor(key)
E0, E1 = emb[15].float(), emb[16].float()
axis = E1 - E0


def s_of(v):
    d = v.float() - E0
    return float((d * axis).sum() / (axis * axis).sum())


torch.manual_seed(42)
m = SoftMixer(emb.shape[1], 64, 4, init="small")
m.soft_mode = SoftMode.AXIS

print("[1] 零初始化端点：p→0 得 E0、p→1 得 E1")
v0 = soft_vector(m, emb, torch.tensor(1e-6), 0, 15, mode=SoftMode.AXIS).float()
v1 = soft_vector(m, emb, torch.tensor(1 - 1e-6), 0, 15, mode=SoftMode.AXIS).float()
e0 = (v0 - E0).norm() / E0.norm()
e1 = (v1 - E1).norm() / E1.norm()
print(f"   rel_err(p=1e-6)={e0:.2e}  rel_err(p=1-1e-6)={e1:.2e}")
assert e0 < 1e-4 and e1 < 1e-4

print("\n[2] axis_head 随机非零（r ~ ±0.5）后的端点与中间修正")
with torch.no_grad():
    # 随机权重 + 偏置，使 r 的量级约 0.3–0.6（避免全被 clamp 压到 0）
    pm = 0.3
    feat = torch.stack([torch.log(torch.tensor(pm)), torch.log1p(torch.tensor(-pm))])
    h = m.fc1(feat) + m.type_emb[0]
    m.axis_head.weight.normal_(0.0, 1.0)
    m.axis_head.weight.mul_(0.05 / h.norm())
    m.axis_head.bias.fill_(0.5)
for _p in (0.1, 0.3, 0.5, 0.7, 0.9):
    _f = torch.stack([torch.log(torch.tensor(_p)), torch.log1p(torch.tensor(-_p))])
    _h = m.fc1(_f) + m.type_emb[0]
    print(f"   r(p={_p}) = {float(m.axis_head(m.act(_h)).squeeze()):+.4f}")
v0 = soft_vector(m, emb, torch.tensor(1e-6), 0, 15, mode=SoftMode.AXIS).float()
v1 = soft_vector(m, emb, torch.tensor(1 - 1e-6), 0, 15, mode=SoftMode.AXIS).float()
e0 = (v0 - E0).norm() / E0.norm()
e1 = (v1 - E1).norm() / E1.norm()
print(f"   rel_err(p=1e-6)={e0:.2e}  rel_err(p=1-1e-6)={e1:.2e}  (均需 <1e-4)")
assert e0 < 1e-4 and e1 < 1e-4
print("   p / s / s-p:")
for p in (0.1, 0.3, 0.5, 0.7, 0.9):
    s = s_of(soft_vector(m, emb, torch.tensor(p), 0, 15, mode=SoftMode.AXIS))
    print(f"     p={p:<4} s={s:+.4f}  s-p={s - p:+.4f}")
s03 = s_of(soft_vector(m, emb, torch.tensor(0.3), 0, 15, mode=SoftMode.AXIS))
assert abs(s03 - 0.3) > 1e-3, "p=0.3 时 s 应偏离 p（说明修正生效）"

print("\n[3] 梯度：能传到 p 与 mixer（用不饱和的 axis_head）")
mg = SoftMixer(emb.shape[1], 64, 4, init="small")
with torch.no_grad():
    mg.axis_head.bias.fill_(0.2)
p = torch.tensor(0.3, requires_grad=True)
soft_vector(mg, emb, p, 0, 15, mode=SoftMode.AXIS).sum().backward()
gsum = sum(float(x.grad.abs().sum()) for x in mg.parameters() if x.grad is not None)
print(f"   dL/dp={float(p.grad):.6e}  mixer |grad| 合计={gsum:.4f}")
assert p.grad is not None and abs(float(p.grad)) > 0 and gsum > 0

print("\n[4] 保存/加载：soft_mode 自动继承")
import tempfile  # noqa: E402
with tempfile.TemporaryDirectory() as td:
    save_soft_mixer(m, td)
    m2 = load_soft_mixer(str(Path(td) / "soft_mixer.pt"))
    print(f"   loaded soft_mode={m2.soft_mode}")
    assert m2.soft_mode == SoftMode.AXIS
    d = (soft_vector(m2, emb, torch.tensor(0.4), 0, 15, mode=m2.soft_mode).float()
         - soft_vector(m, emb, torch.tensor(0.4), 0, 15, mode=m.soft_mode).float()).norm()
    print(f"   reload 后向量差={d:.2e}")
    assert d < 1e-6

print("\n[5] 回归：与冻结副本逐元素一致（linear / logit_mlp / logit_mlp+uncert）")
spec = importlib.util.spec_from_file_location(
    "frozen_soft_mixer",
    "/home/z/zhao/Mammography/Mammo/2-view/chain_v1/frozen_stage1/soft_mixer.py")
frozen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frozen)

torch.manual_seed(7)
mreg = SoftMixer(emb.shape[1], 64, 4, init="small")
fm = frozen.SoftMixer(emb.shape[1], 64, 4, init="small")
fm.load_state_dict({k: v for k, v in mreg.state_dict().items() if k in fm.state_dict()})

grid = [1e-4, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1 - 1e-4]
for mode, gate in (("linear", "none"), ("logit_mlp", "none"), ("logit_mlp", "uncert")):
    worst = 0.0
    for p in grid:
        for t in range(4):
            for bid in (15, 16):
                v_new = soft_vector(mreg, emb, torch.tensor(p), t, bid, mode=mode, gate=gate).float()
                v_ref = frozen.soft_vector(fm, emb, torch.tensor(p), t, bid, mode=mode, gate=gate).float()
                worst = max(worst, float((v_new - v_ref).abs().max()))
    print(f"   {mode:10s} gate={gate:6s}  max|Δ|={worst:.2e}")
    assert worst < 1e-6, f"{mode}/{gate} 回归失败"

print("AXIS_MODE_TESTS_OK")
