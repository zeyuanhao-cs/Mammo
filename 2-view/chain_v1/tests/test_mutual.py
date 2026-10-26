#!/usr/bin/env python3
"""--mutual（双视角互相验证）单元测试（CPU）。

[1] 关闭开关：视角 M 分支不触发（unknown 传了也不参与），损失口径 = ce1 + α·ce2 + λ·kl
[2] 选老师逻辑：A 更近 / M 更近 / 差值小 三种情况
[3] 梯度：只回到学生一侧；老师一侧（detach）梯度为 0
[4] 计数：n_am / n_ma / n_none 与构造一致
"""
import sys

import torch

sys.path.insert(0, "/home/z/zhao/Mammography/Mammo/2-view")
from chain_v1.train.core import SoftKDConfig, choose_teacher_kl, compose_loss  # noqa: E402

EPS = 1e-8
ok = True


def check(name, cond, detail=""):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
    ok = ok and bool(cond)


print("[1] 关闭 --mutual 时不做视角 M 分支，损失口径不变")
cfg_off = SoftKDConfig(alpha=0.05, lam=0.05, mutual=False).resolved()
cfg_on = SoftKDConfig(alpha=0.05, lam=0.05, mutual=True).resolved()
ce1 = torch.tensor(0.5)
ce2 = torch.tensor(0.3)
kl = torch.tensor(0.1)
loss_off = ce1 + cfg_off.alpha * ce2 + cfg_off.lam * kl
loss_on_ref = ce1 + cfg_on.alpha * ce2 + cfg_on.lam * kl
check("mutual=False 的 alpha/lam 未变", cfg_off.alpha == 0.05 and cfg_off.lam == 0.05)
check("mutual=False 损失 = ce1+α·ce2+λ·kl", torch.allclose(loss_off, loss_on_ref),
      f"(={float(loss_off):.4f})")

print("\n[2] 选老师：A 更近 / M 更近 / 差值 <0.05")
# qA/qM: 每行 5 类，GT 下标=0
qA = torch.tensor([[0.60, 0.1, 0.1, 0.1, 0.1],
                   [0.10, 0.1, 0.1, 0.1, 0.1],   # M=0.50 更近
                   [0.30, 0.1, 0.1, 0.1, 0.1]])  # 与 M 差 0.03
qM = torch.tensor([[0.10, 0.1, 0.1, 0.1, 0.1],
                   [0.50, 0.1, 0.1, 0.1, 0.1],
                   [0.33, 0.1, 0.1, 0.1, 0.1]])
gt_idx = torch.tensor([0, 0, 0])
kl, n_am, n_ma, n_none = choose_teacher_kl(qA, qM, gt_idx)
check("A 更近 → 老师 A（学生 M）", n_am == 1, f"n_am={n_am}")
check("M 更近 → 老师 M（学生 A）", n_ma == 1, f"n_ma={n_ma}")
check("差值 <0.05 → 不蒸馏", n_none == 1, f"n_none={n_none}")
check("KL 为有限正值", torch.isfinite(kl) and float(kl) > 0, f"kl={float(kl):.4f}")
kl_none, a2, m2, none2 = choose_teacher_kl(torch.tensor([[0.31] + [0.1725] * 4]),
                                          torch.tensor([[0.30] + [0.175] * 4]), torch.tensor([0]))
check("全不满足阈值时 KL=0 且计数正确", float(kl_none) == 0.0 and none2 == 1,
      f"(n_none={none2})")

print("\n[3] 梯度只回到学生一侧（老师 detach）")
qa = torch.tensor([[0.60, 0.1, 0.1, 0.1, 0.1]], requires_grad=True)
qm = torch.tensor([[0.10, 0.1, 0.1, 0.1, 0.1]], requires_grad=True)
kl3, _, _, _ = choose_teacher_kl(qa, qm, torch.tensor([0]))
kl3.backward()
g_a = float(qa.grad.abs().sum())
g_m = float(qm.grad.abs().sum())
check("老师侧（qA）梯度为 0", g_a == 0.0, f"|gradA|={g_a:.2e}")
check("学生侧（qM）梯度非 0", g_m > 0, f"|gradM|={g_m:.2e}")

qa2 = torch.tensor([[0.10, 0.1, 0.1, 0.1, 0.1]], requires_grad=True)   # M(0.60) 为老师
qm2 = torch.tensor([[0.60, 0.1, 0.1, 0.1, 0.1]], requires_grad=True)
kl4, _, _, _ = choose_teacher_kl(qa2, qm2, torch.tensor([0]))
kl4.backward()
check("M 为老师时 qA（学生）有梯度", float(qa2.grad.abs().sum()) > 0,
      f"|gradA|={float(qa2.grad.abs().sum()):.2e}")

print("\n[4] 计数一致性（4 样本：2 A-teacher、1 M-teacher、1 none）")
qA4 = torch.tensor([[0.6, 0.1, 0.1, 0.1, 0.1],
                    [0.6, 0.1, 0.1, 0.1, 0.1],
                    [0.1, 0.1, 0.1, 0.1, 0.1],
                    [0.3, 0.1, 0.1, 0.1, 0.1]])
qM4 = torch.tensor([[0.1, 0.1, 0.1, 0.1, 0.6],
                    [0.1, 0.1, 0.1, 0.1, 0.6],
                    [0.1, 0.1, 0.1, 0.1, 0.6],
                    [0.31, 0.1, 0.1, 0.17, 0.1]])
gt4 = torch.tensor([0, 0, 4, 0])
_, a4, m4, n4 = choose_teacher_kl(qA4, qM4, gt4)
check("n_am=2 / n_ma=1 / n_none=1", (a4, m4, n4) == (2, 1, 1), f"({a4},{m4},{n4})")

print("\n[5] --mutual 时损失不含 GT 老师项 kl_T（qT 任意值都不改变损失）")
c_on = SoftKDConfig(alpha=0.05, lam=0.05, mutual=True).resolved()
c_off = SoftKDConfig(alpha=0.05, lam=0.05, mutual=False).resolved()
ce1, ce2, ce_m = torch.tensor(0.50), torch.tensor(0.30), torch.tensor(0.40)
kl_m = torch.tensor(0.20)
l_a = compose_loss(ce1, ce2, ce_m, torch.tensor(0.10), kl_m, c_on)
l_b = compose_loss(ce1, ce2, ce_m, torch.tensor(999.0), kl_m, c_on)
l_c = compose_loss(ce1, ce2, ce_m, torch.tensor(-5.0), kl_m, c_on)
check("mutual: kl_T=0.10 / 999 / -5 损失完全相同",
      torch.allclose(l_a, l_b) and torch.allclose(l_a, l_c), f"(loss={float(l_a):.4f})")
expect = ce1 + 0.05 * (ce2 + ce_m) + 0.05 * kl_m
check("mutual 损失 = ce1+α(ce2+ce_m)+λ·kl_mutual", torch.allclose(l_a, expect),
      f"(={float(l_a):.4f})")
l_off = compose_loss(ce1, ce2, ce_m, torch.tensor(0.10), kl_m, c_off)
check("非 mutual 仍含 kl_T", torch.allclose(l_off, ce1 + 0.05 * ce2 + 0.05 * torch.tensor(0.10)),
      f"(={float(l_off):.4f})")

print("\n[6] --keep-gt-teacher（C3m+T）时 kl_T 重新进入损失")
c_kt = SoftKDConfig(alpha=0.05, lam=0.05, mutual=True, keep_gt_teacher=True).resolved()
kl_t = torch.tensor(0.10)
l_kt = compose_loss(ce1, ce2, ce_m, kl_t, kl_m, c_kt)
check("C3m+T 损失 = ce1+α(ce2+ce_m)+λ(kl_T+kl_mutual)",
      torch.allclose(l_kt, ce1 + 0.05 * (ce2 + ce_m) + 0.05 * (kl_t + kl_m)),
      f"(={float(l_kt):.4f})")
l_kt2 = compose_loss(ce1, ce2, ce_m, torch.tensor(0.90), kl_m, c_kt)
check("C3m+T 损失随 kl_T 变化", not torch.allclose(l_kt, l_kt2),
      f"(kl_T=0.9 → {float(l_kt2):.4f})")

print("\nMUTUAL_TESTS_" + ("OK" if ok else "FAILED"))
sys.exit(0 if ok else 1)
