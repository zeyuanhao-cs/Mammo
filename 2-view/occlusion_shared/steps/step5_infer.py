#!/usr/bin/env python3
"""第 5 步 (GPU): 192 个乳房 x 4 种情况 (orig / lesion / ctrl1 / ctrl2) 两视角推理.

推理方式与第 4 步所用结果**完全一致**: 同一 adapter、同一 prompt、同一分辨率预算、
同一贪心解码、同一从 generate(output_scores=True) 取字段概率的实现
(直接复用仓库的 build_prompt / to_tensors / field_probs_from_scores).

每个 case 记录:
  max_sign_p    4 个征象 P(=1) 的最大值 (probs_norm)
  gt_sign_p     GT 病灶类别对应的 P(=1) 的最大值 (GT 无类别则 null)
  p_birads_ge2  1 - P(1)
  p_birads_ge3  1 - P(1) - P(2)
  pred_birads   argmax P(1..5)
输出: occlusion_probs/{condition}.jsonl
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import GenerationConfig

VLM_DIR = Path("/home/zhao/Mammography/Mammo/2-view/scripts/vlm")
sys.path.insert(0, str(VLM_DIR))
from infer_2view_batched import build_prompt, field_probs_from_scores, to_tensors  # noqa: E402
from infer_2view_local import load_instruction  # noqa: E402
from llamafactory.chat import ChatModel  # noqa: E402

FINDINGS = ["mass", "suspicious_calcification", "asymmetry", "architectural_distortion"]
BASE = Path("/nesi/project/massey04900/Mammography")
OCC = BASE / "occlusion"
MODEL_PATH = str(BASE / "model" / "Qwen3.5-4B")
ADAPTER = str(BASE / "model" / "Qwen3.5-4B-lora-2view-os-sub")
PROMPT_FILE = "/home/zhao/Mammography/prompt.txt"
IMAGE_MAX_PIXELS = 786432
IMAGE_MIN_PIXELS = 512 * 512
CONDITIONS = ["orig", "lesion", "ctrl1", "ctrl2"]


def build_cases(man, cond, occl_dir, ids=None):
    cases = []
    for bid in sorted(man["breasts"]):
        if ids and bid not in ids:
            continue
        e = man["breasts"][bid]
        imgs = []
        for view in ("CC", "MLO"):
            v = e["views"].get(view)
            if v is None:
                raise SystemExit(f"{bid} 缺 {view}")
            imgs.append(v["path"] if cond == "orig"
                        else str(Path(occl_dir) / cond / f"{bid}__{view}.png"))
        cases.append({"id": bid, "images": imgs, "entry": e})
    return cases


def derive(pn, e):
    p1 = {k: float(pn[f"findings.{k}"]["1"]) for k in FINDINGS}
    bi = {k: float(pn["breast_birads"][k]) for k in ("1", "2", "3", "4", "5")}
    gt_classes = e.get("gt_classes") or []
    return {
        "max_sign_p": max(p1.values()),
        "gt_sign_p": max((p1[k] for k in gt_classes if k in p1), default=None),
        "p_birads_ge2": 1.0 - bi["1"],
        "p_birads_ge3": 1.0 - bi["1"] - bi["2"],
        "pred_birads": int(max(bi.items(), key=lambda kv: kv[1])[0]),
        "p_findings": p1,
        "p_birads": bi,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(OCC / "occlusion_manifest.json"))
    ap.add_argument("--occluded-dir", default=str(OCC / "occluded"))
    ap.add_argument("--out-dir", default=str(OCC / "occlusion_probs"))
    ap.add_argument("--conditions", nargs="*", default=CONDITIONS)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", nargs="*", default=None)
    args = ap.parse_args()

    man = json.load(open(args.manifest, encoding="utf-8"))
    ids = set(args.ids) if args.ids else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    assert torch.cuda.is_available(), "CUDA unavailable"
    torch.cuda.set_device(0)
    print(f"[cuda] {torch.cuda.get_device_name(0)}  arch={torch.cuda.get_arch_list()}")

    instruction = load_instruction(Path(PROMPT_FILE))
    print(f"[prompt] {PROMPT_FILE}  <image> x{instruction.count('<image>')}")

    model_args = dict(
        model_name_or_path=MODEL_PATH,
        template="qwen3_5",
        trust_remote_code=True,
        infer_backend="huggingface",
        infer_dtype="bfloat16",
        image_max_pixels=IMAGE_MAX_PIXELS,
        image_min_pixels=IMAGE_MIN_PIXELS,
        enable_thinking=False,
        flash_attn="sdpa",
        adapter_name_or_path=ADAPTER,
        finetuning_type="lora",
    )
    t0 = time.perf_counter()
    cm = ChatModel(args=model_args)
    engine = cm.engine
    model, tokenizer, processor, template = engine.model, engine.tokenizer, engine.processor, engine.template
    torch.cuda.synchronize()
    print(f"[model] {ADAPTER} loaded in {time.perf_counter() - t0:.1f}s")

    eos_ids = template.get_stop_token_ids(tokenizer)
    gen_cfg = GenerationConfig(
        do_sample=False, num_beams=1, repetition_penalty=1.0,
        max_new_tokens=args.max_new_tokens, eos_token_id=eos_ids,
        pad_token_id=tokenizer.pad_token_id,
    )
    pad_id = tokenizer.pad_token_id

    for cond in args.conditions:
        cases = build_cases(man, cond, args.occluded_dir, ids)
        if args.limit:
            cases = cases[: args.limit]
        out = out_dir / f"{cond}.jsonl"
        done = set()
        if out.exists():
            for line in open(out, encoding="utf-8"):
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    pass
        todo = [c for c in cases if c["id"] not in done]
        print(f"\n[cond] {cond}: total={len(cases)} done={len(done)} todo={len(todo)} "
              f"batch={args.batch_size}")
        fout = open(out, "a", encoding="utf-8")
        t_all = time.perf_counter()
        bs = args.batch_size
        i = 0
        while i < len(todo):
            chunk = todo[i : i + bs]
            try:
                ids_list, imgs_list = [], []
                for c in chunk:
                    ids_, imgs = build_prompt(template, tokenizer, processor, instruction, list(c["images"]))
                    ids_list.append(ids_)
                    imgs_list.append(imgs)
                maxlen = max(len(x) for x in ids_list)
                input_ids = torch.full((len(chunk), maxlen), pad_id, dtype=torch.long)
                attn = torch.zeros((len(chunk), maxlen), dtype=torch.long)
                for j, one in enumerate(ids_list):
                    input_ids[j, maxlen - len(one):] = torch.tensor(one, dtype=torch.long)
                    attn[j, maxlen - len(one):] = 1
                input_ids, attn = input_ids.to(model.device), attn.to(model.device)

                flat = [im for imgs in imgs_list for im in imgs]
                mm_inputs = template.mm_plugin.get_mm_inputs(
                    images=flat, videos=[], audios=[],
                    imglens=[len(x) for x in imgs_list], vidlens=[0] * len(chunk),
                    audlens=[0] * len(chunk), batch_ids=ids_list, processor=processor,
                )
                mm_kwargs = to_tensors(mm_inputs, model)
                with torch.inference_mode():
                    gen = model.generate(
                        input_ids=input_ids, attention_mask=attn, generation_config=gen_cfg,
                        return_dict_in_generate=True, output_scores=True, **mm_kwargs,
                    )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                bs = max(1, bs // 2)
                print(f"[oom] batch 降为 {bs} 重试")
                continue

            seqs = gen.sequences[:, maxlen:]
            scores = gen.scores
            for j, c in enumerate(chunk):
                gids = seqs[j].tolist()
                if eos_ids:
                    for k, t in enumerate(gids):
                        if t in eos_ids:
                            gids = gids[:k]
                            break
                pieces = [tokenizer.decode([t], skip_special_tokens=True) for t in gids]
                raw = "".join(pieces).strip()
                rows = [s[j] for s in scores[: len(gids)]] if scores else []
                _, probs_norm, argmax_match = field_probs_from_scores(
                    tokenizer, gids, pieces, rows)
                d = derive(probs_norm, c["entry"])
                fout.write(json.dumps({
                    "id": c["id"], "condition": cond,
                    "gt_birads": c["entry"]["gt_birads"],
                    "gt_classes": c["entry"].get("gt_classes") or [],
                    "prediction_raw": raw,
                    "argmax_match": argmax_match,
                    "probs_norm": probs_norm,
                    **d,
                }, ensure_ascii=False) + "\n")
            fout.flush()
            i += len(chunk)
            n_done = i
            el = time.perf_counter() - t_all
            print(f"  [{cond}] {n_done}/{len(todo)}  {el / max(n_done, 1):.2f}s/case  "
                  f"eta {el / max(n_done, 1) * (len(todo) - n_done) / 60:.1f} min", flush=True)
        fout.close()
        n_lines = sum(1 for _ in open(out, encoding="utf-8"))
        print(f"[cond] {cond} done: {n_lines} lines -> {out}")

    print("\n[done] all conditions")


if __name__ == "__main__":
    main()
