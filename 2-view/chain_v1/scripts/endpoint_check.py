#!/usr/bin/env python3
"""端点核对：只把 mass 换成硬 token（E["0"]/E["1"]），其余 3 个征象仍走 axis 软注入。

与扫描完全同管线（同 adapter / 同 mixer / 同 prefill / 同 40 例 / 只改 mass 位置的向量），
用来验证 axis 在 p=0、p=1 两端是否与硬 token 等价。

用法:
  python endpoint_check.py run     --adapter ... --mixer ... --ids-file ... --out ...
  python endpoint_check.py compare --axis ... --mass-hard ... --global-hard ...
"""
import argparse
import json
import sys
from pathlib import Path

import torch

PROJ = Path("/home/z/zhao/Mammography/Mammo/2-view")
sys.path.insert(0, str(PROJ / "scripts" / "vlm"))
sys.path.insert(0, str(PROJ))

FIND = ["mass", "suspicious_calcification", "asymmetry", "architectural_distortion"]


def run(args):
    import chain_v1.infer.soft_decode as sdm
    from chain_v1.infer.eval_soft import (build_prompt_one, field_probs_from_scores,
                                          load_instruction, stop_ids_of)
    from chain_v1.train.soft_mixer import load_soft_mixer
    from llamafactory.chat import ChatModel

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
    print(f"[model] {args.adapter} loaded", flush=True)
    mixer = load_soft_mixer(args.mixer, device=next(model.parameters()).device,
                            dtype=torch.float32)
    print(f"[mixer] soft_mode={getattr(mixer, 'soft_mode', '?')} gate={getattr(mixer, 'gate', '?')}",
          flush=True)

    # ---- 关键：只在 mass(type_idx=0) 位置用硬 token，其余字段仍走原 soft_vector ----
    orig_soft_vector = sdm.soft_vector
    hits = {"mass": 0}      # 补丁生效计数（每例每点应命中 1 次）

    def patched(mixer_, emb_w, p, type_idx, base_id, id0=15, id1=16, dtype=torch.float32,
                mode="axis", gate=None):
        if int(type_idx) == 0:                      # mass
            hits["mass"] += 1
            tid = int(id1) if float(p) > 0.5 else int(id0)
            return emb_w[tid].to(torch.float32)
        return orig_soft_vector(mixer_, emb_w, p, type_idx, base_id, id0, id1, dtype, mode,
                                gate=gate)

    if args.axis_mass:
        # --axis-mass：关闭补丁，mass 也走 axis 软路径（用于判断偏差来自脚本还是 eval_soft 扫描路径）
        print("[patch] 已关闭：mass 走 axis 软路径（与 eval_soft --mode scan 同一条注入逻辑）",
              flush=True)
    else:
        sdm.soft_vector = patched

    instr = load_instruction(PROJ / "dataset_build" / "prompt.txt")
    stops = stop_ids_of(template, tokenizer)
    data = json.load(open(args.data, encoding="utf-8"))
    want = json.load(open(args.ids_file, encoding="utf-8"))
    dmap = {r["id"]: r for r in data}
    recs = [dmap[i] for i in want]
    print(f"[data] {len(recs)} 例 × {len(args.points)} 点（mass 硬 token，其余 axis 软）", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fout = open(out, "w", encoding="utf-8")
    for k, rec in enumerate(recs, 1):
        gt = rec["output"] if isinstance(rec["output"], dict) else json.loads(rec["output"])
        prompt_ids, kw = build_prompt_one(template, tokenizer, processor, instr, list(rec["images"]))
        o, deltas = sdm.rope_prefill(model, prompt_ids, kw)
        prefill = {"past": o.past_key_values, "logits": o.logits[0, -1].float(),
                   "seq_len": int(prompt_ids.shape[-1]), "rope_deltas": deltas}
        for p in args.points:
            r = sdm.soft_decode(model, tokenizer, prompt_ids=prompt_ids,
                                max_new_tokens=args.max_new_tokens, stop_ids=stops,
                                soft=True, soft_mode="axis", mixer=mixer,
                                force_p={"mass": p}, prefill=prefill,
                                force_base_token=True, return_rows=True)
            _, pn, _ = field_probs_from_scores(
                tokenizer, r["tokens"],
                [tokenizer.decode([t], skip_special_tokens=True) for t in r["tokens"]], r["rows"])
            bi = {kk: float(vv) for kk, vv in (pn.get("breast_birads") or {}).items()}
            fout.write(json.dumps({
                "id": rec["id"],
                "mode": "endpoint_axis_mass" if args.axis_mass else "endpoint_mass_hard",
                "scan_p": p,
                "gt_birads": int(gt["breast_birads"]),
                "gt_ge3": int(gt["breast_birads"]) >= 3,
                "p_birads_ge3": 1.0 - bi.get("1", 0.0) - bi.get("2", 0.0),
            }, ensure_ascii=False) + "\n")
        fout.flush()
        print(f"  [{k}/{len(recs)}] {rec['id'][:10]}", flush=True)
    fout.close()

    expect = len(recs) * len(args.points)
    print(f"[patch] mass 硬 token 命中 {hits['mass']} 次（期望 {expect}）", flush=True)
    if args.axis_mass:
        if hits["mass"] != 0:
            print("[ERROR] --axis-mass 下不应命中补丁", flush=True); raise SystemExit(4)
    elif hits["mass"] != expect:
        print(f"[ERROR] 补丁未生效：命中 {hits['mass']} ≠ {expect}（mass 位置没有被硬 token 替换）",
              flush=True)
        raise SystemExit(4)
    print(f"[done] {out}", flush=True)


def compare(args):
    def curve(fn):
        rows = [json.loads(l) for l in open(fn, encoding="utf-8") if l.strip()]
        by = {}
        for r in rows:
            by.setdefault(round(r["scan_p"], 2), {})[r["id"]] = r
        return by

    axis = curve(args.axis)
    mass = curve(args.mass_hard)
    glob = curve(args.global_hard) if args.global_hard and Path(args.global_hard).exists() else {}

    def mean(d, p):
        v = [r["p_birads_ge3"] for r in d[p].values() if r["gt_ge3"]]
        return sum(v) / len(v) if v else None

    print("## 端点核对（GT>=3 均值）\n")
    print("| p | axis 扫描（全软） | mass 硬 token（本脚本） | 差 | 全局 --hard-inject（参考） |")
    print("|---|---|---|---|---|")
    for p in (0.0, 1.0):
        a, m = mean(axis, p), mean(mass, p)
        g = mean(glob, p) if p in glob else None
        print(f"| {p:.1f} | {a:.4f} | {m:.4f} | {m - a:+.4f} | "
              f"{'—' if g is None else f'{g:.4f}'} |")

    print("\n## 差值最大的样本（mass 硬 vs axis 软）")
    for p in (0.0, 1.0):
        diffs = []
        for i in axis[p]:
            if i in mass[p]:
                diffs.append((abs(mass[p][i]["p_birads_ge3"] - axis[p][i]["p_birads_ge3"]), i,
                              axis[p][i]["p_birads_ge3"], mass[p][i]["p_birads_ge3"]))
        diffs.sort(reverse=True)
        print(f"\n p={p:.1f}：")
        for d, i, a, m in diffs[:3]:
            print(f"   {i}  axis={a:.4f}  mass-hard={m:.4f}  |Δ|={d:.4f}")
        if diffs and diffs[0][0] > 0.01:
            print(f"   ⚠ 最大差 {diffs[0][0]:.4f} > 0.01，需在报告中标出")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "compare"])
    ap.add_argument("--adapter", default="")
    ap.add_argument("--mixer", default="")
    ap.add_argument("--ids-file", default=str(PROJ / "chain_v1/outputs/pilot/ids2_scan20.json"))
    ap.add_argument("--data", default=str(PROJ / "data/splits/val_raw_local.json"))
    ap.add_argument("--out", default=str(PROJ / "chain_v1/outputs/pilot/c3a_endpoint.jsonl"))
    ap.add_argument("--points", default="0,1")
    ap.add_argument("--image-max-pixels", type=int, default=786432)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--axis-mass", action="store_true",
                    help="关闭补丁：mass 也走 axis 软路径（对照实验用）")
    ap.add_argument("--axis", default=str(PROJ / "chain_v1/outputs/pilot/c3a_scan.jsonl"))
    ap.add_argument("--mass-hard", default=str(PROJ / "chain_v1/outputs/pilot/c3a_endpoint.jsonl"))
    ap.add_argument("--global-hard", default=str(PROJ / "chain_v1/outputs/pilot/c3a_hardtok40.jsonl"))
    a = ap.parse_args()
    a.points = [float(x) for x in a.points.split(",")]
    (run if a.cmd == "run" else compare)(a)
