#!/usr/bin/env python3
"""SoftKDTrainer：把 core.soft_kd_loss 接到 transformers.Trainer 上。

日志（每 logging_steps 步）：ce1 / ce2 / kl / mean_p / rho / low_p_ratio
    low_p_ratio = 当步样本里「GT 有 finding 但 p<0.5」的比例（只统计 GT 阳性 finding 的位置）。
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
from transformers import Trainer

from .core import SoftKDConfig, rho_at, soft_kd_loss


class SoftKDTrainer(Trainer):
    def __init__(self, *args, soft_cfg: Optional[SoftKDConfig] = None, soft_mixer=None,
                 reuse_embeds: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.soft_mixer = soft_mixer
        self.reuse_embeds = reuse_embeds
        self.soft_cfg = (soft_cfg or SoftKDConfig()).resolved()
        self._mb = 0   # micro-batch 计数（用于每优化步只记一次日志）

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
        if self.soft_mixer is not None:
            params = [p for p in self.soft_mixer.parameters() if p.requires_grad]
            if params:
                opt.add_param_group({"params": params, "lr": self.args.learning_rate,
                                     "weight_decay": 0.0})
        return opt

    # ---- 保存 mixer 到每个 checkpoint ----
    def _save(self, *args, **kwargs):
        out = super()._save(*args, **kwargs)
        try:
            from .soft_mixer import save_soft_mixer
            outdir = kwargs.get("output_dir") or (args[0] if args else None) or self.args.output_dir
            if self.soft_mixer is not None and outdir:
                save_soft_mixer(self.soft_mixer, outdir)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] 保存 soft_mixer 失败: {e!r}")
        return out

    def _move_mixer(self, device):
        if self.soft_mixer is not None and next(self.soft_mixer.parameters()).device != device:
            self.soft_mixer.to(device=device)

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
        )

        # 训练日志（grad accumulation 内只在最后一个 micro-batch 记一次）
        self._mb += 1
        accum = max(1, int(self.args.gradient_accumulation_steps))
        if step % max(1, int(self.args.logging_steps)) == 0 and self._mb % accum == 0:
            p = parts["p"].detach()
            log = {"rho": rho, "low_p_ratio": self._low_p_ratio(p, gt_findings)}
            log.update({k: v for k, v in metrics.items() if isinstance(v, float)})
            self.log(log)

        return (loss, None) if return_outputs else loss
