#!/usr/bin/env python3
"""贴近训练的梯度一致性测试（GPU，eval 关 dropout，LoRA 不 merge、可训练，mixer 可训练）。

对每条路径：全新 forward 前 zero_grad（LoRA + mixer），只 backward 一次。
报告每个可训练参数的 ‖gA‖、‖gA−gB‖、相对差 ‖gA−gB‖/‖gA‖；gradient checkpointing 开/关各一次。
通过阈值：最大相对差 ≤ 1e-2。
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

CV1 = Path(__file__).resolve().parent.parent
PROJ = CV1.parent
sys.path.insert(0, str(PROJ / "scripts" / "vlm"))
sys.path.insert(0, str(CV1.parent))

from llamafactory.data import get_template_and_fix_tokenizer  # noqa: E402
from llamafactory.hparams import DataArguments  # noqa: E402
from llamafactory.model.model_utils.misc import find_all_linear_modules  # noqa: E402
from chain_v1.train.core import SoftKDConfig, soft_kd_loss  # noqa: E402
from chain_v1.train.data import SoftKDDataset, build_inputs  # noqa: E402
from chain_v1.train.soft_mixer import SoftMixer  # noqa: E402

BASE = "/home/z/zhao/Mammography/model/Qwen3.5-4B"
OUT = CV1 / "outputs" / "grad_parity_train.md"


def load():
    from llamafactory.chat import ChatModel
    cm = ChatModel(args=dict(model_name_or_path=BASE, template="qwen3_5_nothink",
                             trust_remote_code=True, infer_backend="huggingface",
                             infer_dtype="bfloat16", image_max_pixels=786432,
                             image_min_pixels=512 * 512, enable_thinking=False, flash_attn="sdpa"))
    eng = cm.engine
    model, tok, proc, template = eng.model, eng.tokenizer, eng.processor, eng.template
    model.config.use_cache = False
    model.config.use_cache = False
    if not getattr(model, "is_gradient_checkpointing", False):
        model.gradient_checkpointing_enable({"use_reentrant": False})
        model.enable_input_require_grads()
    print("gc after enable:", getattr(model, "is_gradient_checkpointing", None), flush=True)

    from peft import LoraConfig, get_peft_model
    targets = find_all_linear_modules(model, False)
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                                             target_modules=targets, bias="none",
                                             task_type="CAUSAL_LM"))
    model.eval()
    print("gc after peft :", getattr(model, "is_gradient_checkpointing", None), flush=True)
    return model, tok, proc, template


def mm_kwargs(template, proc, model, images, ids):
    mm = template.mm_plugin.get_mm_inputs(images=images, videos=[], audios=[], imglens=[len(images)],
                                         vidlens=[0], audlens=[0], batch_ids=[ids], processor=proc)
    kw = {}
    for k, v in mm.items():
        if isinstance(v, list) and isinstance(v[0], torch.Tensor):
            v = torch.stack(v)
        elif not isinstance(v, torch.Tensor):
            v = torch.tensor(v)
        if torch.is_floating_point(v):
            v = v.to(model.dtype)
        kw[k] = v.tolist() if k == "second_per_grid_ts" else v.to(model.device)
    if hasattr(proc, "create_mm_token_type_ids"):
        mtt = proc.create_mm_token_type_ids([ids])
        kw["mm_token_type_ids"] = (mtt if torch.is_tensor(mtt) else torch.tensor(mtt)).to(model.device)
    return kw


def trainable_params(model, mixer):
    ps = [(f"lora:{n}", p) for n, p in model.named_parameters() if p.requires_grad]
    ps += [(f"mixer:{n}", p) for n, p in mixer.named_parameters()]
    return ps


def run_once(model, mixer, built, template, proc, cfg, reuse, mode="full"):
    ps = trainable_params(model, mixer)
    for _, p in ps:
        p.grad = None
    ids = torch.tensor([built["input_ids"]], device=model.device)
    att = torch.ones_like(ids)
    mkw = mm_kwargs(template, proc, model, built["images"], built["input_ids"])
    loss, parts = soft_kd_loss(
        model, ids, att, torch.tensor([built["labels"]], device=model.device),
        torch.tensor([built["finding_pos"]]), torch.tensor([built["birads_pos"]]),
        mm_kwargs=mkw, cfg=cfg, rho=1.0, use_second=True, mixer=mixer, reuse_embeds=reuse)
    if mode == "full":
        loss.backward()
    else:  # ce1 / ce2 / kl 分项
        parts["ce1_raw"].backward(retain_graph=True)
        parts["ce2_raw"].backward(retain_graph=True)
        parts["kl_raw"].backward()
    return {n: (p.grad.detach().float().clone() if p.grad is not None else None) for n, p in ps}, parts


def compare(gA, gB):
    rows = []
    for n in gB:
        a, b = gA.get(n), gB.get(n)
        if a is None or b is None:
            continue
        na = a.norm().item()
        d = (a - b).norm().item()
        rows.append((n, na, d, d / na if na > 0 else 0.0))
    return rows


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gc", choices=["on", "off", "both"], default="both")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--short", action="store_true",
                    help="把两张图缩到 384x384（序列更短，仅用于梯度路径比对；全长度前向已在 real_parity 验证）")
    a = ap.parse_args()
    model, tok, proc, template = load()
    mixer = SoftMixer(2560, 64, 4, init="small").to(model.device)
    ds = SoftKDDataset(str(PROJ / "data/llamafactory/train_llamafactory_os_sub_local.json"), tok,
                       template, limit=1)
    if a.short:
        from PIL import Image
        ds.items[0]["images"] = [Image.open(x).convert("RGB").resize((384, 384)) for x in ds.items[0]["images"]]
    built = build_inputs(ds[0], tok, template, proc)
    print("seq_len:", len(built["input_ids"]), flush=True)
    cfg = SoftKDConfig(soft_mode="logit_mlp", alpha=1.0, lam=1.0)

    outp = Path(a.out)
    lines = []
    if not outp.exists():
        lines.append("# 梯度一致性（贴近训练：LoRA 不 merge、eval、mixer 可训练）\n")
    ok_all = True
    modes = {"on": [True], "off": [False], "both": [False, True]}[a.gc]
    for gc in modes:
        if gc:
            if not getattr(model, "is_gradient_checkpointing", False):
                model.gradient_checkpointing_enable({"use_reentrant": False})
                model.enable_input_require_grads()
        else:
            model.gradient_checkpointing_disable()
        try:
            gA, _ = run_once(model, mixer, built, template, proc, cfg, True)
            gB, _ = run_once(model, mixer, built, template, proc, cfg, False)
        except torch.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            msg = str(e).split("\n")[0][:160]
            print(f"[gc={gc}] OOM: {msg}", flush=True)
            ok_all = False
            lines += [f"## gradient checkpointing = {gc}", "",
                      f"- **OOM**：两份计算图 + LoRA 反传在该配置下放不进 24 GB（`{msg}`）", ""]
            continue
        rows = compare(gA, gB)
        worst = max(rows, key=lambda r: r[3])
        mean_rel = sum(r[3] for r in rows) / len(rows)
        passed = worst[3] <= 1e-2
        ok_all = ok_all and passed
        print(f"[gc={gc}] n_params={len(rows)} mean_rel={mean_rel:.3e} "
              f"max_rel={worst[3]:.3e} ({worst[0]}) {'PASS' if passed else 'FAIL'}", flush=True)
        lines += [f"## gradient checkpointing = {gc}", "",
                  f"- 参与比较的可训练张量：{len(rows)}（LoRA + mixer）",
                  f"- 平均相对差：{mean_rel:.3e}；最大相对差：**{worst[3]:.3e}**（`{worst[0]}`）",
                  f"- 判定：{'PASS (≤1e-2)' if passed else 'FAIL (>1e-2)'}", "",
                  "| param | ‖gA‖ | ‖gA−gB‖ | 相对差 |", "|---|---|---|---|"]
        for n, na, d, rel in sorted(rows, key=lambda r: -r[3])[:12]:
            lines.append(f"| {n} | {na:.4e} | {d:.4e} | {rel:.3e} |")
        lines.append("")

    lines += ["## 口径", "",
              "- A = 第 2 次前向复用第 1 次语言模型实参（只换 inputs_embeds）；B = 整模型重跑 + hook 注入。",
              "- 每条路径 backward 前对 LoRA 与 mixer 全部 zero_grad，只 backward 一次。",
              "- 训练模式下复用会让两次前向共享同一 dropout 掩码，这是唯一预期的行为差异；本测试 eval 关 dropout。"]
    with open(outp, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[-12:]))
    print("ALL_PASS" if ok_all else "NOT_PASS")


if __name__ == "__main__":
    main()
