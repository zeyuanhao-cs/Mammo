#!/usr/bin/env python3
"""SoftKDTrainer：把 core.soft_kd_loss 接到 transformers.Trainer 上。

日志（每 logging_steps 步）：ce1 / ce2 / kl / mean_p / rho / low_p_ratio
    low_p_ratio = 当步样本里「GT 有 finding 但 p<0.5」的比例（只统计 GT 阳性 finding 的位置）。
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
from pathlib import Path
from transformers import Trainer

from .core import SoftKDConfig, rho_at, soft_kd_loss


class SoftKDTrainer(Trainer):
    def __init__(self, *args, soft_cfg: Optional[SoftKDConfig] = None, soft_mixer=None,
                 unknown=None, reuse_embeds: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.soft_mixer = soft_mixer
        self.unknown = unknown          # [4,hidden] 视角 M 的可学习"未知"向量
        self.reuse_embeds = reuse_embeds
        self.soft_cfg = (soft_cfg or SoftKDConfig()).resolved()
        self._mb = 0   # micro-batch 计数（用于每优化步只记一次日志）
        self._acc = None  # 梯度累积窗口内的日志累加器（仅日志用）

    # ---- 辅助 ----
    def _total_steps(self) -> int:
        ms = getattr(self.state, "max_steps", 0)
        if ms and ms > 0:
            return int(ms)
        return max(1, int(self.args.max_steps) if self.args.max_steps and self.args.max_steps > 0 else 1000)

    def _low_p_ratio(self, p: torch.Tensor, gt_findings: Dict[str, int]) -> float:
        """GT 有阳性 finding 的样本中，p<0.5 的比例。p: [1, n_find] 按 FIND 顺序。"""
        from .data import FIND
        vals = [float(p[0, j]) for j, k in enumerate(FIND) if gt_findings.get(k, 0) == 1]
        if not vals:
            return float("nan")
        return 1.0 if min(vals) < 0.5 else 0.0

    # ---- optimizer：把 mixer 参数一起训练 ----
    def create_optimizer(self):
        opt = super().create_optimizer()
        if self.soft_mixer is not None or self.unknown is not None:
            params = [p for p in self.soft_mixer.parameters() if p.requires_grad] \
                if self.soft_mixer is not None else []
            if self.unknown is not None:
                params = params + [self.unknown]
            if params:
                opt.add_param_group({"params": params, "lr": self.args.learning_rate,
                                     "weight_decay": 0.0})
                if self.unknown is not None:
                    in_opt = any(any(q is self.unknown for q in g["params"])
                                 for g in opt.param_groups)
                    print(f"[mutual] unknown 已进优化器: {in_opt}；"
                          f"参数组数={len(opt.param_groups)}", flush=True)
        return opt

    # ---- 保存 mixer 到每个 checkpoint ----
    def _save(self, *args, **kwargs):
        out = super()._save(*args, **kwargs)
        try:
            from .soft_mixer import save_soft_mixer
            outdir = kwargs.get("output_dir") or (args[0] if args else None) or self.args.output_dir
            if self.soft_mixer is not None and outdir:
                save_soft_mixer(self.soft_mixer, outdir)
            if self.unknown is not None and outdir:
                # 视角 M 的可学习"未知"向量（此前未落盘，导致训练后无法复现视角 M）
                torch.save({"unknown": self.unknown.detach().cpu()},
                           Path(outdir) / "soft_unknown.pt")
        except Exception as e:  # noqa: BLE001
            print(f"[warn] 保存 soft_mixer 失败: {e!r}")
        return out

    def _move_mixer(self, device):
        if self.soft_mixer is not None and next(self.soft_mixer.parameters()).device != device:
            self.soft_mixer.to(device=device)
        if self.unknown is not None and self.unknown.device != device:
            self.unknown.data = self.unknown.data.to(device=device)

    # ---- loss ----
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        self._move_mixer(inputs["input_ids"].device)
        mm = inputs.pop("mm_kwargs")
        finding_pos = inputs.pop("finding_pos")
        birads_pos = inputs.pop("birads_pos")
        gt_findings = inputs.pop("gt_findings")
        inputs.pop("gt_birads", None)

        step = int(self.state.global_step)
        rho = rho_at(step, self._total_steps(), self.soft_cfg)
        use_second = (not self.soft_cfg.no_soft) and (rho > 0) and (torch.rand(1).item() < rho)

        metrics: Dict[str, float] = {}
        loss, parts = soft_kd_loss(
            model,
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
            finding_pos=finding_pos,
            birads_pos=birads_pos,
            mm_kwargs=mm,
            cfg=self.soft_cfg,
            rho=rho,
            use_second=use_second,
            mixer=self.soft_mixer,
            reuse_embeds=self.reuse_embeds,
            extra_metrics=metrics,
            unknown=self.unknown,
        )

        # ---- 训练日志（只影响日志，不改训练逻辑）----
        # ce1/ce2/kl/mean_p 取梯度累积窗口内的平均；n_second = 窗口内实际跑了第二遍
        # （软注入）的 micro-batch 数（use_second 由 rho 逐 micro-batch 伯努利采样）。
        self._mb += 1
        accum = max(1, int(self.args.gradient_accumulation_steps))
        if self._acc is None:
            self._acc = {"n": 0, "n2": 0, "ce1": 0.0, "ce2": 0.0, "kl": 0.0, "mean_p": 0.0,
                         "ce_m": 0.0, "kl_mutual": 0.0, "loss": 0.0,
                         "n_am": 0, "n_ma": 0, "n_none": 0}
        a = self._acc
        a["n"] += 1
        a["n2"] += int(bool(use_second))
        a["loss"] += float(loss.detach())
        for k in ("ce1", "ce2", "kl", "mean_p", "ce_m", "kl_mutual"):
            if k in metrics:
                a[k] += float(metrics[k])
        a["n_am"] += int(parts.get("n_am", 0))
        a["n_ma"] += int(parts.get("n_ma", 0))
        a["n_none"] += int(parts.get("n_none", 0))
        if self._mb % accum == 0:
            if step % max(1, int(self.args.logging_steps)) == 0:
                n = max(1, a["n"])
                log = {"rho": rho, "loss": a["loss"] / n,
                       "low_p_ratio": self._low_p_ratio(parts["p"].detach(), gt_findings),
                       "ce1": a["ce1"] / n, "ce2": a["ce2"] / n, "kl": a["kl"] / n,
                       "mean_p": a["mean_p"] / n,
                       "n_second": a["n2"], "n_micro": a["n"],
                       "ce_m": a["ce_m"] / n, "kl_mutual": a["kl_mutual"] / n,
                       "kl_T": a["kl"] / n,
                       "n_am": int(a["n_am"]), "n_ma": int(a["n_ma"]),
                       "n_none": int(a["n_none"]),
                       "n_mutual": int(a["n_am"] + a["n_ma"] + a["n_none"])}
                self.log(log)
            self._acc = None

        return (loss, None) if return_outputs else loss
