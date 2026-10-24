#!/usr/bin/env python3
"""--soft-gate uncert 单元测试（CPU）。

口径: soft = E[base] + u·MLP(...), u = 4p(1-p)
 1) p=0.5 → u=1，与 gate=none 完全一致；
 2) p=0.01 → 偏移范数是不加开关时的 4·0.01·0.99 = 0.0396 倍；
 3) 梯度能传到 p 和 mixer 参数。
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from chain_v1.train.soft_mixer import SoftMixer, load_soft_mixer, save_soft_mixer, soft_vector  # noqa: E402

H = 32
torch.manual_seed(0)
mixer = SoftMixer(H, 16, 4, init="small").eval()
emb = torch.randn(20, H) * 0.02     # 与真实嵌入尺度接近，仅用于范数比较


def delta(p, gate):
    vec = soft_vector(mixer, emb, p, 0, 15, dtype=torch.float32, gate=gate)
    return vec - emb[15]


# 1) p=0.5 完全一致
p5 = torch.tensor(0.5)
d_none = delta(p5, "none")
d_unc = delta(p5, "uncert")
assert torch.allclose(d_none, d_unc, atol=0, rtol=0), (d_none - d_unc).abs().max()
print("[1] p=0.5: 与 gate=none 完全一致  max|Δ|=%.3e" % (d_none - d_unc).abs().max().item())

# 2) p=0.01 范数比 = 0.0396
p1 = torch.tensor(0.01)
r = (delta(p1, "uncert").norm() / delta(p1, "none").norm()).item()
assert abs(r - 0.0396) < 1e-6, r
print("[2] p=0.01: 偏移范数比 = %.6f（期望 0.0396）" % r)

# 3) 梯度到 p 与 mixer
pg = torch.tensor(0.3, requires_grad=True)
vec = soft_vector(mixer, emb, pg, 0, 15, dtype=torch.float32, gate="uncert")
vec.sum().backward()
gp = pg.grad.item()
assert abs(gp) > 0, gp
assert all(p.grad is not None for p in mixer.parameters()), "mixer 参数未拿到梯度"
gsum = sum(float(p.grad.abs().sum()) for p in mixer.parameters())
assert gsum > 0, gsum
print("[3] 梯度: dL/dp=%.4f, mixer |grad| 总和=%.4f" % (gp, gsum))

# 4) gate 随 mixer 存取（推理自动继承）
mixer.gate = "uncert"
tmp = Path("/tmp/_gate_test")
tmp.mkdir(exist_ok=True)
save_soft_mixer(mixer, str(tmp))
m2 = load_soft_mixer(str(tmp / "soft_mixer.pt"))
assert m2.gate == "uncert", m2.gate
assert torch.allclose(soft_vector(m2, emb, p1, 0, 15, gate=None),
                      soft_vector(mixer, emb, p1, 0, 15, gate="uncert")), "加载后未自动生效"
print("[4] gate=uncert 随 soft_mixer.pt 保存/加载，推理端自动生效")

print("SOFT_GATE_TESTS_OK")
