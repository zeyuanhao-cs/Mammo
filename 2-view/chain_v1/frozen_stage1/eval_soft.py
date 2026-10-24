#!/usr/bin/env python3
"""chain_v1 批量软/硬解码评测（batch=1，单卡）。

模式
  hard      : 普通贪心（不注入）
  soft      : 软注入（mixer / linear），输出文本仍 argmax
  swap_hard : 软管线但注入向量 = E[argmax token]（软换硬对照，需要 adapter 但 delta=0）
  teacher   : 4 个征象强制为 GT 值注入（老师模式，soft=True + force_p=GT）
  scan      : 概率扫描，对 --force-field 的 p 取 --scan-values，其余字段按模型自身 p 注入；
              prefill 每样本只算一次，多个 p 共享

输出 jsonl（每样本一行）：
  普通: {id, mode, text, probs_norm, argmax_match, pred_birads, max_sign_p, p_birads_ge2/ge3,
         p_findings, p_birads, soft_events, n_gen, elapsed}
  scan: {id, mode, force_field, scan_p, gt_birads, gt_ge3, p_birads_ge3, p_findings}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent.parent                     # 2-view
sys.path.insert(0, str(PROJ / "scripts" / "vlm"))
sys.path.insert(0, str(PROJ))                 # chain_v1 包

from chain_v1.infer.soft_decode import soft_decode, decode_batch, rope_prefill  # noqa: E402
from chain_v1.train.soft_mixer import load_soft_mixer  # noqa: E402
from infer_2view_batched import field_probs_from_scores  # noqa: E402
from score_2view import load_instruction  # noqa: E402
from llamafactory.chat import ChatModel  # noqa: E402

FIND = ["mass", "suspicious_calcification", "asymmetry", "architectural_distortion"]
BIRADS_IDS = [16, 17, 18, 19, 20]             # '1'..'5'


def build_prompt_one(template, tokenizer, processor, instr, images, dtype=None, device=None):
    # 默认回退到 main() 设置的全局（eval_soft 主流程）；外部脚本需显式传入
    dtype = dtype if dtype is not None else globals().get("model_dtype", torch.bfloat16)
    device = device if device is not None else globals().get("model_device", "cuda")
    messages = template.mm_plugin.process_messages(
        [{"role": "user", "content": instr}], images, [], [], processor)
    pids, _ = template.encode_oneturn(
        tokenizer, messages + [{"role": "assistant", "content": ""}], None, None)
    pids, _ = template.mm_plugin.process_token_ids(pids, None, images, [], [], tokenizer, processor)
    mm = template.mm_plugin.get_mm_inputs(images=images, videos=[], audios=[],
                                          imglens=[len(images)], vidlens=[0], audlens=[0],
                                          batch_ids=[pids], processor=processor)
    kw = {}
    for k, v in mm.items():
        if isinstance(v, list) and isinstance(v[0], torch.Tensor):
            v = torch.stack(v)
        elif isinstance(v, list) and isinstance(v[0], list) and isinstance(v[0][0], torch.Tensor):
            v = torch.stack([torch.stack(x) for x in v])
        elif not isinstance(v, torch.Tensor):
            v = torch.tensor(v)
        if torch.is_floating_point(v):
            v = v.to(dtype)
        kw[k] = v.tolist() if k == "second_per_grid_ts" else v.to(device)
    if hasattr(processor, "create_mm_token_type_ids"):
        mtt = processor.create_mm_token_type_ids([pids])
        kw["mm_token_type_ids"] = (mtt if isinstance(mtt, torch.Tensor)
                                   else torch.tensor(mtt)).to(device)
    return torch.tensor([pids], device=device), kw


def stop_ids_of(template, tokenizer):
    ids = template.get_stop_token_ids(tokenizer)
    return set(int(x) for x in ids) if ids else set()


def summarize(rec_id, mode, r, probs_norm, elapsed, gt=None):
    bi = {k: float(v) for k, v in (probs_norm.get("breast_birads") or {}).items()}
    p_find = {k: float((probs_norm.get(f"findings.{k}") or {}).get("1", 0.0)) for k in FIND}
    row = {
        "id": rec_id, "mode": mode,
        "text": r["text"],
        "probs_norm": probs_norm,
        "pred_birads": int(max(bi.items(), key=lambda kv: kv[1])[0]) if len(bi) == 5 else None,
        "p_findings": p_find,
        "max_sign_p": max(p_find.values()),
        "p_birads": bi,
        "p_birads_ge2": 1.0 - bi.get("1", 0.0),
        "p_birads_ge3": 1.0 - bi.get("1", 0.0) - bi.get("2", 0.0),
        "soft_events": [(int(s), f, round(float(p), 4)) for s, f, p in r["soft_events"]],
        "n_gen": len(r["tokens"]),
        "elapsed": round(elapsed, 3),
    }
    if gt is not None:
        row["gt_birads"] = int(gt["breast_birads"])
        row["gt_density"] = gt["breast_density"]
        row["gt_findings"] = {k: int(gt["findings"][k]) for k in FIND}
    return row


def main():
    global model_dtype, model_device
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--mixer", default="", help="soft_mixer.pt（soft/swap 模式 logit_mlp 需要）")
    ap.add_argument("--soft-mode", choices=["linear", "logit_mlp"], default="logit_mlp")
    ap.add_argument("--data", default=str(PROJ / "data/splits/val_raw_local.json"))
    ap.add_argument("--ids-file", default="", help="json 列表文件：只跑这些 id（保持给定顺序）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--mode", required=True,
                    choices=["hard", "soft", "swap_hard", "teacher", "scan"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--force-field", default="mass", help="scan 模式扫描的字段")
    ap.add_argument("--batch-size", type=int, default=1,
                    help=">1 时 hard/soft/swap_hard/teacher 用 decode_batch 批推理")
    ap.add_argument("--scan-values", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    ap.add_argument("--hard-inject", action="store_true",
                    help="scan 模式改用硬 token 注入：--scan-values 0→\"0\"(id0)、1→\"1\"(id1)，不需 --mixer")
    ap.add_argument("--image-max-pixels", type=int, default=786432)
    args = ap.parse_args()

    assert not os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""), \
        "PYTORCH_CUDA_ALLOC_CONF 必须为空（本卡不支持 expandable_segments）"
    assert torch.cuda.is_available()
    torch.cuda.set_device(0)

    cm = ChatModel(args=dict(
        model_name_or_path="/home/z/zhao/Mammography/model/Qwen3.5-4B", template="qwen3_5",
        trust_remote_code=True, infer_backend="huggingface", infer_dtype="bfloat16",
        image_max_pixels=args.image_max_pixels, image_min_pixels=512 * 512,
        enable_thinking=False, flash_attn="sdpa",
        adapter_name_or_path=args.adapter, finetuning_type="lora"))
    eng = cm.engine
    tokenizer, processor, template, model = eng.tokenizer, eng.processor, eng.template, eng.model
    model.eval()
    model_dtype = model.dtype
    model_device = next(model.parameters()).device
    print(f"[model] {args.adapter} loaded, dtype={model_dtype}", flush=True)

    mixer = None
    soft = args.mode in ("soft", "swap_hard", "teacher", "scan")
    if args.mode == "swap_hard":
        force_hard = True          # 注入 E[argmax]，p 照常记录
        mixer = None
        if args.soft_mode == "logit_mlp":
            mixer = None           # delta=0，不需要 mixer
    else:
        force_hard = False
    if args.mode == "scan" and args.hard_inject:
        force_hard = True          # scan + 硬注入：token 跟随 --scan-values（0→"0", 1→"1"）
    if args.mode in ("soft", "scan", "teacher") and args.soft_mode == "logit_mlp" \
            and not (args.mode == "scan" and args.hard_inject):
        assert args.mixer, "--mode soft/scan/teacher + logit_mlp 需要 --mixer"
        mixer = load_soft_mixer(args.mixer, device=model_device, dtype=torch.float32)
        print(f"[mixer] gate={getattr(mixer, 'gate', 'none')}（随 soft_mixer.pt 自动加载）", flush=True)
        print(f"[mixer] {args.mixer} loaded", flush=True)
    teacher = args.mode == "teacher"
    scan = args.mode == "scan"
    scan_values = [float(x) for x in args.scan_values.split(",")] if scan else []

    instr = load_instruction(PROJ / "dataset_build" / "prompt.txt")
    stops = stop_ids_of(template, tokenizer)

    data = json.load(open(args.data, encoding="utf-8"))
    if args.ids_file:
        want = json.load(open(args.ids_file, encoding="utf-8"))
        dmap = {r["id"]: r for r in data}
        data = [dmap[i] for i in want]
    if args.limit:
        data = data[: args.limit]
    print(f"[data] {len(data)} 条, mode={args.mode}, soft={soft}, force_hard={force_hard}",
          flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in open(out, encoding="utf-8"):
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    fout = open(out, "a", encoding="utf-8")
    t_all = time.perf_counter()
    n_done = 0

    def build_case(rec):
        gt = rec["output"] if isinstance(rec["output"], dict) else json.loads(rec["output"])
        prompt_ids, kw = build_prompt_one(template, tokenizer, processor, instr,
                                          list(rec["images"]))
        force_p = None
        if teacher:
            force_p = {k: float(gt["findings"][k]) for k in FIND}
        return {"rec": rec, "gt": gt, "prompt_ids": prompt_ids, "mm_kwargs": kw,
                "force_p": force_p}

    todo = [rec for rec in data if rec["id"] not in done]
    use_batch = (args.batch_size > 1 and not scan)
    i = 0
    while i < len(todo):
        t0 = time.perf_counter()
        if scan:
            rec = todo[i]
            i += 1
            rid = rec["id"]
            c = build_case(rec)
            o, deltas = rope_prefill(model, c["prompt_ids"], c["mm_kwargs"])
            prefill = {"past": o.past_key_values, "logits": o.logits[0, -1].float(),
                       "seq_len": int(c["prompt_ids"].shape[-1]), "rope_deltas": deltas}
            for sv in scan_values:
                r = soft_decode(model, tokenizer, prompt_ids=c["prompt_ids"],
                                max_new_tokens=args.max_new_tokens, stop_ids=stops,
                                soft=True, soft_mode=args.soft_mode, mixer=mixer,
                                force_p={args.force_field: sv}, prefill=prefill,
                                force_hard=force_hard, return_rows=True)
                probs, probs_norm, _ = field_probs_from_scores(
                    tokenizer, r["tokens"],
                    [tokenizer.decode([t], skip_special_tokens=True) for t in r["tokens"]],
                    r["rows"])
                bi = {k: float(v) for k, v in (probs_norm.get("breast_birads") or {}).items()}
                p_find = {k: float((probs_norm.get(f"findings.{k}") or {}).get("1", 0.0))
                          for k in FIND}
                fout.write(json.dumps({
                    "id": rid, "mode": "scan", "force_field": args.force_field,
                    "scan_p": sv, "gt_birads": int(c["gt"]["breast_birads"]),
                    "gt_ge3": int(c["gt"]["breast_birads"]) >= 3,
                    "p_birads_ge3": 1.0 - bi.get("1", 0.0) - bi.get("2", 0.0),
                    "p_birads": bi, "p_findings": p_find,
                    "n_gen": len(r["tokens"]),
                }, ensure_ascii=False) + "\n")
            fout.flush()
            n_done += 1
            print(f"  [scan] {n_done}/{len(todo)} {rid[:10]} "
                  f"{time.perf_counter()-t0:.1f}s", flush=True)
            del prefill
            torch.cuda.empty_cache()
            continue

        if not use_batch:
            rec = todo[i]
            i += 1
            rid = rec["id"]
            c = build_case(rec)
            r = soft_decode(model, tokenizer, prompt_ids=c["prompt_ids"],
                            mm_kwargs=c["mm_kwargs"], max_new_tokens=args.max_new_tokens,
                            stop_ids=stops, soft=soft, soft_mode=args.soft_mode,
                            mixer=mixer, force_p=c["force_p"], force_hard=force_hard,
                            return_rows=True)
            pieces = [tokenizer.decode([t], skip_special_tokens=True) for t in r["tokens"]]
            _, probs_norm, argmax_match = field_probs_from_scores(
                tokenizer, r["tokens"], pieces, r["rows"])
            row = summarize(rid, args.mode, r, probs_norm,
                            time.perf_counter() - t0, c["gt"])
            row["argmax_match"] = argmax_match
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            n_done += 1
            el = time.perf_counter() - t_all
            print(f"  [{args.mode}] {n_done}/{len(todo)} {rid[:10]} {row['pred_birads']} "
                  f"{row['elapsed']:.1f}s  avg {el/max(n_done,1):.2f}s/case", flush=True)
            continue

        # ---- 批模式（hard / soft / swap_hard / teacher）：decode_batch ----
        chunk = todo[i: i + args.batch_size]
        i += len(chunk)
        try:
            cases = [build_case(rec) for rec in chunk]
            lens = {len(c["prompt_ids"]) for c in cases}
            assert len(lens) == 1, f"批内 prompt 长度不一致 {lens}"
            rs = decode_batch(model, tokenizer, cases, stop_ids=stops,
                              max_new_tokens=args.max_new_tokens, soft=soft,
                              force_hard=force_hard, mixer=mixer,
                              soft_mode=args.soft_mode, return_rows=True)
        except AssertionError as e:
            print(f"[batch] 回退逐条: {e}", flush=True)
            use_batch = False
            continue
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            args.batch_size = max(1, args.batch_size // 2)
            print(f"[oom] batch 降为 {args.batch_size} 重试", flush=True)
            i -= len(chunk)
            continue
        for rec, c, r in zip(chunk, cases, rs):
            pieces = [tokenizer.decode([t], skip_special_tokens=True) for t in r["tokens"]]
            _, probs_norm, argmax_match = field_probs_from_scores(
                tokenizer, r["tokens"], pieces, r["rows"])
            row = summarize(rec["id"], args.mode, r, probs_norm,
                            (time.perf_counter() - t0) / len(chunk), c["gt"])
            row["argmax_match"] = argmax_match
            row["batch"] = len(chunk)
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        fout.flush()
        n_done += len(chunk)
        el = time.perf_counter() - t_all
        print(f"  [{args.mode}] {n_done}/{len(todo)}  {el/max(n_done,1):.2f}s/case "
              f"(batch={len(chunk)})", flush=True)

    fout.close()
    print(f"[done] mode={args.mode} n={n_done} -> {out} "
          f"({(time.perf_counter()-t_all)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
