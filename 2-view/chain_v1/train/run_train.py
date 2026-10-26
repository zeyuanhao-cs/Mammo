#!/usr/bin/env python3
"""chain_v1 训练入口（顺序 A：两次前向 + 软证据 KD）。

示例（先别开，确认参数后再说）：
    python chain_v1/train/run_train.py \
      --data data/llamafactory/train_llamafactory_os_sub_local.json \
      --output-dir /home/z/zhao/Mammography/model/chain_v1_softkd \
      --epochs 2 --rho-max 0.5 --alpha 1.0 --lam 1.0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments

HERE = Path(__file__).resolve().parent
CV1 = HERE.parent
PROJ = CV1.parent
sys.path.insert(0, str(PROJ / "scripts" / "vlm"))
sys.path.insert(0, str(CV1.parent))          # 使 chain_v1 可作为包导入

from llamafactory.chat import ChatModel  # noqa: E402
from chain_v1.train.core import SoftKDConfig  # noqa: E402
from chain_v1.train.data import SoftKDDataset, build_inputs, to_tensors  # noqa: E402
from chain_v1.train.trainer import SoftKDTrainer  # noqa: E402
from chain_v1.train.soft_mixer import SoftMixer, load_soft_mixer, save_soft_mixer, soft_vector  # noqa: E402


class SoftCollator:
    """batch=1 的 collator：拼 prompt+GT、算位置、转视觉张量。"""

    def __init__(self, model, tokenizer, template, processor):
        self.model = model
        self.tokenizer = tokenizer
        self.template = template
        self.processor = processor

    def __call__(self, batch):
        assert len(batch) == 1, "当前实现要求 per_device_train_batch_size=1"
        built = build_inputs(batch[0], self.tokenizer, self.template, self.processor)
        return to_tensors(built, self.processor, self.template, None, torch.bfloat16)


def lora_targets(model, freeze_vision_tower=False):
    """与 LLaMA-Factory 完全一致的目标模块集合（lora_target: all + freeze_vision_tower）。

    修 bug：原先只返回 find_all_linear_modules 的裸后缀名，PEFT 按后缀匹配会把
    visual.patch_embed.proj（Conv3D）与 visual.merger.linear_fc1/fc2 也包上 LoRA
    （LLaMA-Factory 训练的 checkpoint 里没有它们，导致 --init-adapter 键不匹配、
    且与原训练结构不一致）。现补上 LLaMA-Factory 同款 patch_target_modules：
    展开为完整模块路径并排除 lora_conflict_keys。
    """
    from llamafactory.model.model_utils.misc import find_all_linear_modules
    from llamafactory.model.model_utils.visual import COMPOSITE_MODELS, patch_target_modules
    names = find_all_linear_modules(model, freeze_vision_tower)
    fake_ft = SimpleNamespace(freeze_vision_tower=freeze_vision_tower,
                              freeze_multi_modal_projector=True,   # LLaMA-Factory 默认 True（merger 不挂 LoRA）
                              freeze_language_model=False)
    return patch_target_modules(model, fake_ft, names)


def latest_checkpoint(outdir: str):
    """output_dir 下最新的 checkpoint-<n>（自动续训用）。"""
    import re
    if not os.path.isdir(outdir):
        return None
    cks = [d for d in os.listdir(outdir) if re.fullmatch(r"checkpoint-\d+", d)]
    if not cks:
        return None
    cks.sort(key=lambda d: int(d.split("-")[-1]))
    return os.path.join(outdir, cks[-1])


def build_lora(model, rank=8, alpha=16, dropout=0.05, targets=None):
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout,
                     target_modules=targets or "all-linear", bias="none", task_type="CAUSAL_LM")
    return get_peft_model(model, cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/z/zhao/Mammography/model/Qwen3.5-4B")
    ap.add_argument("--data", default=str(PROJ / "data/llamafactory/train_llamafactory_os_sub_local.json"))
    ap.add_argument("--prompt-file", default=str(PROJ / "dataset_build/prompt.txt"))
    ap.add_argument("--output-dir", default="/home/z/zhao/Mammography/model/chain_v1_softkd")
    ap.add_argument("--image-max-pixels", type=int, default=786432)
    ap.add_argument("--image-min-pixels", type=int, default=512 * 512)
    ap.add_argument("--epochs", type=float, default=1.0, help="试验先用 1")
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--logging-steps", type=int, default=1)  # 逐步日志（B0 起，2026-10-24 改）
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--rho-max", type=float, default=0.5)
    ap.add_argument("--rho-warmup-frac", type=float, default=0.2)
    ap.add_argument("--soft-mode", choices=["linear", "logit_mlp", "axis"], default="logit_mlp",
                    help="软证据注入方式：logit_mlp(主方案) / linear(对照)")
    ap.add_argument("--soft-mixer-init", default="", help="可选：从 soft_mixer.pt 恢复 mixer 权重")
    ap.add_argument("--mutual", action="store_true",
                    help="双视角互相验证：视角 A=axis 软证据，视角 M=可学习未知向量；按 GT 概率差选老师")
    ap.add_argument("--keep-gt-teacher", action="store_true",
                    help="配合 --mutual：损失额外保留 GT 老师项 kl_T（C3m+T）")
    ap.add_argument("--soft-gate", choices=["none", "uncert"], default="none",
                    help="软注入门控：none=现有行为；uncert=delta×4p(1-p)（随 mixer 保存，推理自动生效）")
    ap.add_argument("--init-adapter", default="",
                    help="只加载已有 LoRA adapter 权重到新建的 LoRA（不改 optimizer/step 计数），"
                         "用于从已训模型接着训新 run（区别于 --resume）")
    ap.add_argument("--no-reuse-embeds", action="store_true",
                    help="关闭第 2 次前向复用融合嵌入（会重跑视觉塔，仅用于对照）")
    ap.add_argument("--mixer-init", choices=["zero", "small"], default="small",
                    help="MLP 末层初始化：small=N(0,1e-3)（默认）/ zero")
    ap.add_argument("--detach-p", action="store_true", help="p 不回传梯度（默认回传）")
    ap.add_argument("--no-kd", action="store_true", help="lam=0")
    ap.add_argument("--no-soft", action="store_true", help="关闭第 2 次前向（纯 SFT 基线）")
    ap.add_argument("--grad-ckpt", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--freeze-vision-tower", action="store_true", default=False,
                    help="与原配置一致应为 False（视觉塔纳入 LoRA）")
    ap.add_argument("--max-steps", type=int, default=-1, help=">0 时覆盖 epochs（20 步显存测试用）")
    ap.add_argument("--peak-mem", action="store_true", help="打印 torch 显存峰值（配合 20 步测试）")
    ap.add_argument("--grad-parity-terms", action="store_true",
                    help="梯度比对时按 CE1 / CE2 / KL 分项 backward 定位差异")
    ap.add_argument("--grad-parity-short", action="store_true",
                    help="梯度比对时把图像缩到 384x384（满长度在 24G 上放不下两份图）")
    ap.add_argument("--grad-parity", action="store_true",
                    help="跑一次复用 vs 不复用的梯度一致性比对后退出（不训练）")
    ap.add_argument("--resume", default="auto",
                    help="auto=自动从 output_dir 里最新 checkpoint 续训；off=从头；或显式路径")
    args = ap.parse_args()

    assert args.bs == 1, "batch>1 尚未支持（rho/位置按样本单条处理）"

    cm = ChatModel(args=dict(
        model_name_or_path=args.model, template="qwen3_5_nothink", trust_remote_code=True,
        infer_backend="huggingface", infer_dtype="bfloat16",
        image_max_pixels=args.image_max_pixels, image_min_pixels=args.image_min_pixels,
        enable_thinking=False, flash_attn="sdpa"))
    eng = cm.engine
    model, tokenizer, processor, template = eng.model, eng.tokenizer, eng.processor, eng.template
    model.train()
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    targets = lora_targets(model, freeze_vision_tower=args.freeze_vision_tower)
    print(f"[lora] {len(targets)} 个目标模块名（与 LLaMA-Factory lora_target=all 同源）: "
          f"{','.join(sorted(targets))}")
    model = build_lora(model, args.lora_rank, args.lora_alpha, args.lora_dropout, targets)

    if args.init_adapter:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        sd = load_file(os.path.join(args.init_adapter, "adapter_model.safetensors"))
        res = set_peft_model_state_dict(model, sd, adapter_name="default")
        # missing_keys 里绝大多数是主干权重（checkpoint 本就不含），只有含 lora_ 的才算没加载上
        lora_missing = [k for k in (res.missing_keys or []) if "lora_" in k] if res is not None else []
        n_unexpected = len(res.unexpected_keys) if res is not None else -1
        print(f"[init-adapter] 从 {args.init_adapter} 加载 {len(sd)} 个张量 "
              f"(lora_missing={len(lora_missing)}, unexpected={n_unexpected})", flush=True)
        if lora_missing:
            raise SystemExit(f"[init-adapter] LoRA 键未覆盖: {lora_missing[:5]} … 共 {len(lora_missing)}")
        if n_unexpected > 0:
            raise SystemExit(f"[init-adapter] checkpoint 有 {n_unexpected} 个未匹配键")

    ds = SoftKDDataset(args.data, tokenizer, template, limit=args.limit)
    collator = SoftCollator(model, tokenizer, template, processor)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to=[],
        gradient_checkpointing=bool(args.grad_ckpt),
        dataloader_num_workers=0,
        remove_unused_columns=False,
        seed=args.seed,
        max_steps=args.max_steps,
        data_seed=args.seed,
    )
    mixer = None
    if args.soft_mode in ("logit_mlp", "axis") and not args.no_soft:
        hidden = getattr(model.config, "hidden_size", None)
        if hidden is None:
            tc = getattr(model.config, "text_config", None)
            hidden = getattr(tc, "hidden_size", None) if tc is not None else None
        assert hidden, "无法从 config 读取 hidden_size"
        # 修 bug：mixer 初始化原先未固定随机种子，同 seed 42 的两次 run mixer 不同；
        # 现在创建前固定 seed（在 TrainingArguments 之后、Trainer set_seed 之前）。
        torch.manual_seed(args.seed)
        mixer = SoftMixer(int(hidden), 64, 4, init=args.mixer_init)
        if args.soft_mixer_init:
            mixer.load_state_dict(load_soft_mixer(args.soft_mixer_init).state_dict())
        mixer.gate = args.soft_gate     # 随 mixer 保存；推理端自动读取
        mixer.soft_mode = args.soft_mode
        print(f"[mixer] {args.soft_mode}: hidden={hidden}, d_hidden=64, n_types=4, init={args.mixer_init}, "
              f"gate={mixer.gate}, params={sum(p.numel() for p in mixer.parameters())}")
        # 前置检查：软向量偏移范数 / 对应 token 嵌入范数（计划 ≈1%）。
        # 修 bug：N(0,1e-3) 的 small init 实测偏移 ≈21%（远超计划的 1%）——初始化未按计划实现。
        # 现按 p=0.5 中性点实测比值校准 fc2.weight，使偏移 ≈1%，并打印校准前后比值。
        mixer.to(model.device)
        emb = model.get_input_embeddings().weight

        def _ratios():
            rs = []
            with torch.no_grad():
                for pv in (0.1, 0.3, 0.5, 0.7, 0.9):
                    base_e = emb[16].float() if pv > 0.5 else emb[15].float()
                    vec = soft_vector(mixer, emb, pv, 0, 16 if pv > 0.5 else 15,
                                      dtype=torch.float32, mode="logit_mlp")
                    rs.append(((vec - base_e).norm() / base_e.norm()).item())
            return sorted(rs)

        rs0 = _ratios()
        scale = 0.01 / max(rs0[2], 1e-9)
        with torch.no_grad():
            mixer.fc2.weight.mul_(scale)
        rs1 = _ratios()
        print(f"[mixer-scale] 校准前 ||delta||/||E[base]|| = {['%.4f' % r for r in rs0]} "
              f"median={rs0[2]:.4f}；按计划 1% 校准（×{scale:.4f}）后 = "
              f"{['%.4f' % r for r in rs1]} median={rs1[2]:.4f}", flush=True)

    soft_cfg = SoftKDConfig(alpha=args.alpha, lam=args.lam, rho_max=args.rho_max,
                            mutual=args.mutual, keep_gt_teacher=args.keep_gt_teacher,
                            rho_warmup_frac=args.rho_warmup_frac, detach_p=args.detach_p,
                            no_kd=args.no_kd, no_soft=args.no_soft, soft_mode=args.soft_mode,
                            soft_gate=args.soft_gate)
    unknown = None
    if args.mutual:
        emb = model.get_input_embeddings().weight
        init = (emb[15].detach().float() + emb[16].detach().float()) / 2.0
        unknown = torch.nn.Parameter(init.unsqueeze(0).repeat(4, 1).contiguous())
        print(f"[mutual] 未知向量 {tuple(unknown.shape)} 初始化=E0/E1 均值（fp32），"
              f"与 mixer 同优化器组", flush=True)
    trainer = SoftKDTrainer(model=model, args=targs, train_dataset=ds, data_collator=collator,
                            soft_cfg=soft_cfg, soft_mixer=mixer, unknown=unknown,
                            reuse_embeds=not args.no_reuse_embeds, processing_class=tokenizer)
    print(f"[cfg] data={args.data} n={len(ds)} alpha={args.alpha} lam={args.lam} "
          f"rho_max={args.rho_max} detach_p={args.detach_p} no_kd={args.no_kd} no_soft={args.no_soft} "
          f"mutual={args.mutual} keep_gt_teacher={args.keep_gt_teacher}")
    if args.grad_parity or args.grad_parity_terms or args.grad_parity_short:
        import math
        from chain_v1.train.core import soft_kd_loss as _loss

        if args.grad_parity_short:
            from PIL import Image
            ds.items[0]["images"] = [Image.open(x).convert("RGB").resize((384, 384))
                                     for x in ds.items[0]["images"]]
            print("[grad-parity] 短序列模式：图像 384x384", flush=True)

        # 保持 training 模式（HF 的 gradient checkpointing 只在 training 下生效），只把 dropout 置 0
        _dp = []
        for _m in trainer.model.modules():
            if isinstance(_m, torch.nn.Dropout):
                _dp.append((_m, _m.p))
                _m.p = 0.0
        print(f"[grad-parity] dropout 置 0：{len(_dp)} 个模块", flush=True)

        _dev = next(trainer.model.parameters()).device
        if trainer.soft_mixer is not None:
            trainer.soft_mixer.to(_dev)

        def _collect(use_reuse, seed=42):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            trainer.model.zero_grad(set_to_none=True)
            if trainer.soft_mixer is not None:
                trainer.soft_mixer.zero_grad(set_to_none=True)
            batch = trainer._prepare_inputs(collator([ds[0]]))
            mm = batch.pop("mm_kwargs")
            fpos = batch.pop("finding_pos")
            bpos = batch.pop("birads_pos")
            batch.pop("gt_findings", None)
            batch.pop("gt_birads", None)
            loss, parts = _loss(trainer.model, input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"], labels=batch["labels"],
                                finding_pos=fpos, birads_pos=bpos, mm_kwargs=mm,
                                cfg=trainer.soft_cfg, rho=1.0, use_second=True,
                                mixer=trainer.soft_mixer, reuse_embeds=use_reuse)
            ps = [(f"lora:{n}", q) for n, q in trainer.model.named_parameters() if q.requires_grad]
            ps += [(f"mixer:{n}", q) for n, q in trainer.soft_mixer.named_parameters()] if trainer.soft_mixer is not None else []
            if trainer.soft_mixer is not None:
                ps += [(f"mixer:{n}", q) for n, q in trainer.soft_mixer.named_parameters()]
            if args.grad_parity_terms:
                terms = {}
                for name, key, scale in (("ce1", "ce1_raw", 1.0),
                                         ("ce2", "ce2_raw", trainer.soft_cfg.alpha),
                                         ("kl", "kl_raw", trainer.soft_cfg.lam)):
                    for _, q in ps:
                        q.grad = None
                    parts[key].mul(scale).backward(retain_graph=(name != "kl"))
                    terms[name] = {n: (q.grad.detach().float().clone() if q.grad is not None else None)
                                   for n, q in ps}
                g = terms
            else:
                loss.backward()
                g = {n: (q.grad.detach().float().clone() if q.grad is not None else None) for n, q in ps}
            del loss, parts, batch, mm
            torch.cuda.empty_cache()
            return g

        def _stats(gX, gY):
            num = den = 0.0
            rows = []
            for n in gY:
                a, b = gX.get(n), gY.get(n)
                if a is None or b is None:
                    continue
                na = a.norm().item()
                d = (a - b).norm().item()
                num += d * d
                den += na * na
                rows.append([n, na, d, d / na if na > 0 else 0.0])
            tot = math.sqrt(den) if den > 0 else 0.0
            for r in rows:
                r.append(r[2] / tot if tot > 0 else 0.0)     # 差值 ÷ 全部梯度总范数
            rows.sort(key=lambda r: -r[3])
            return rows, (math.sqrt(num) / tot if tot > 0 else 0.0), tot

        gA1 = _collect(True)
        gA2 = _collect(True)
        gB1 = _collect(False)
        gB2 = _collect(False)

        if args.grad_parity_terms:
            out = ["# 梯度分项比对（含噪声底线）", "",
                   "| term | rms A/A | rms B/B | rms A/B | ×噪声 | 判定 |", "|---|---|---|---|---|---|"]
            for t in ("ce1", "ce2", "kl"):
                _, rAA_t, _ = _stats(gA1[t], gA2[t])
                _, rBB_t, _ = _stats(gB1[t], gB2[t])
                _, rAB_t, _ = _stats(gA1[t], gB1[t])
                b = max(rAA_t, rBB_t)
                v = rAB_t <= max(1.5 * b, 1e-6)
                out.append(f"| {t} | {rAA_t:.3e} | {rBB_t:.3e} | {rAB_t:.3e} | "
                           f"{(rAB_t / b if b > 0 else 0):.1f}× | {'PASS' if v else 'FAIL'} |")
                print(f"[term {t}] AA={rAA_t:.3e} BB={rBB_t:.3e} AB={rAB_t:.3e} -> "
                      f"{'PASS' if v else 'FAIL'}", flush=True)
            (CV1 / "outputs" / "grad_parity_terms.md").write_text("\n".join(out) + "\n", encoding="utf-8")
            print("\n".join(out))
            for _m, _p0 in _dp:
                _m.p = _p0
            return

        rAA, rmsAA, totA = _stats(gA1, gA2)
        rBB, rmsBB, totB = _stats(gB1, gB2)
        rAB, rmsAB, _ = _stats(gA1, gB1)
        base = max(rmsAA, rmsBB)
        ok = rmsAB <= max(1.5 * base, 1e-6)

        lines = ["# 梯度一致性：A(复用) vs B(不复用)，含噪声底线", "",
                 f"- 序列：{'短(384x384)' if args.grad_parity_short else '满长度'}；dropout 全置 0；training 模式（gc 生效）",
                 "",
                 "| 比较 | RMS 相对差（整体向量） | 说明 |", "|---|---|---|",
                 f"| A vs A（噪声） | {rmsAA:.3e} | 同一路径重复两次，种子固定 |",
                 f"| B vs B（噪声） | {rmsBB:.3e} | 同上 |",
                 f"| A vs B（信号） | {rmsAB:.3e} | 与噪声底线比较 |", "",
                 f"**判定**：信号 {rmsAB:.3e} vs 噪声底线 {base:.3e}（×1.5 = {1.5*base:.3e}）→ "
                 f"{'PASS（信号未明显高于噪声）' if ok else 'FAIL（信号显著高于噪声）'}", "",
                 "## A vs B 逐张量（按相对差排序，前 15）", "",
                 "| param | ‖gA‖ | ‖gA−gB‖ | 相对差 | 差值/总范数 |", "|---|---|---|---|---|"]
        for n, na, d, rel, rtot in rAB[:15]:
            lines.append(f"| {n} | {na:.4e} | {d:.4e} | {rel:.3e} | {rtot:.3e} |")
        lines += ["", "## B vs B 噪声底线（同序前 15 项对照）", "",
                  "| param | ‖gB1‖ | ‖gB1−gB2‖ | 相对差 | 差值/总范数 |", "|---|---|---|---|---|"]
        for n, na, d, rel, rtot in rBB[:15]:
            lines.append(f"| {n} | {na:.4e} | {d:.4e} | {rel:.3e} | {rtot:.3e} |")
        outp = CV1 / "outputs" / "grad_parity_train.md"
        outp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))
        for _m, _p0 in _dp:
            _m.p = _p0
        return

    if args.peak_mem:
        import transformers as _tf
        _orig = _tf.Trainer.train

        def _patched(self, *a, **k):
            torch.cuda.reset_peak_memory_stats()
            out = _orig(self, *a, **k)
            print(f"[mem] PEAK_ALLOCATED={torch.cuda.max_memory_allocated()/1024**3:.3f} GiB "
                  f"PEAK_RESERVED={torch.cuda.max_memory_reserved()/1024**3:.3f} GiB", flush=True)
            return out

        _tf.Trainer.train = _patched

    resume = args.resume
    if resume == "auto":
        resume = latest_checkpoint(args.output_dir)
    elif resume == "off":
        resume = None
    if resume:
        print(f"[resume] 从 {resume} 续训")
    trainer.train(resume_from_checkpoint=resume)

    # 训练结束：把最终产物放到 output_dir 根（与 LLaMA-Factory 行为一致），便于评测直接引用
    trainer.save_model(args.output_dir)
    if mixer is not None:
        save_soft_mixer(mixer, args.output_dir)
    (Path(args.output_dir) / "trainer_state.json").write_text(
        json.dumps({"global_step": int(trainer.state.global_step),
                    "max_steps": int(trainer.state.max_steps)}),
        encoding="utf-8")
    print(f"[done] global_step={trainer.state.global_step} -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
