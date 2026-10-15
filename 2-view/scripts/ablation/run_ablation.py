#!/usr/bin/env python3
"""
Run A (Repeat-OS Qwen3.5-4B LoRA) 图像消融 —— 只推理(teacher-forcing 打分), 不训练。

设计 (与 score_2view.py 完全一致, 直接 import 复用其实现):
  - 条件序列 = 已有 orig 贪心预测文件 predictions_qwen3.5-4b_lora_os_test_all_2000.jsonl
    中模型自己的 JSON 输出 (prediction_raw), 所有条件共用同一条件序列,
    概率差异只来自图像替换 —— 受控消融。
  - prompt / 图像张数(2) / 预处理 (qwen3_5 模板, max_pixels=786432, min=512*512,
    bf16, sdpa, 同一 adapter) / 概率输出格式 与 score_2view.py 逐 token 一致。

6 个条件 (只替换输入图像, 标签不变):
  orig            原图 (smoke 阶段用于与已有 repeat_os_test_with_probs.jsonl 对拍;
                  full 阶段直接复制该文件, 不重跑)
  blank           CC+MLO 都换成训练集均值灰图 (与被替换图同尺寸)
  cc_only         保留 CC, MLO 换灰图
  mlo_only        保留 MLO, CC 换灰图
  swap_side       同 study 对侧乳房的 [CC, MLO]
  shuffle_patient 另一随机 study 的同侧 [CC, MLO] (study 级错排, seed=42)

输出: 2-view/outputs/ablation/<condition>/{smoke_,}test_with_probs.jsonl
      每行: {id, ground_truth, greedy_prediction, argmax_match, probs, probs_norm, seconds}
用法:
  python run_ablation.py --mode smoke   # 每条件前 5 个乳房 + orig 对拍 + 耗时预估
  python run_ablation.py --mode full    # 5 个消融条件全量 + 复制 orig (断点续跑)
"""

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent          # 2-view/scripts/ablation
SCRIPTS_DIR = THIS_DIR.parent                       # 2-view/scripts
PROJ_DIR = SCRIPTS_DIR.parent                       # 2-view
sys.path.insert(0, str(SCRIPTS_DIR))

import score_2view as S2V   # noqa: E402  (复用 load_instruction/extract_json_text/score_one)

DEFAULT_ADAPTER = "/nesi/project/massey04900/Mammography/model/Qwen3.5-4B-lora-2view-os"
DEFAULT_PRED_FILE = str(PROJ_DIR / "outputs/predictions/predictions_qwen3.5-4b_lora_os_test_all_2000.jsonl")
DEFAULT_DATA_FILE = "/nesi/project/massey04900/Mammography/test.json"
DEFAULT_TRAIN_FILE = "/nesi/project/massey04900/Mammography/train_repeat_os_birads_findings_cap4.json"
DEFAULT_ORIG_SCORED = str(PROJ_DIR / "outputs/scoring/repeat_os_test_with_probs.jsonl")
DEFAULT_OUT_ROOT = PROJ_DIR / "outputs" / "ablation"

SMOKE_N = 5
SEED = 42
ABLATION_CONDS = ["blank", "cc_only", "mlo_only", "swap_side", "shuffle_patient"]
ALL_CONDS = ["orig"] + ABLATION_CONDS
# orig 对拍容忍度 (bf16 前向可能有微小数值差异)
ORIG_GATE_TOL = 0.02


# ---------------- data ----------------

def load_test(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    recs = {r["id"]: r for r in raw}
    order = [r["id"] for r in raw]
    by_study = {}
    for r in raw:
        sid, lat = r["id"].rsplit("_", 1)
        by_study.setdefault(sid, {})[lat] = r
    for sid, m in by_study.items():
        assert "L" in m and "R" in m, f"study {sid} 缺少一侧乳房"
    return raw, recs, order, by_study


def load_preds(path):
    preds = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                preds[d["id"]] = d
    assert preds, f"empty pred file: {path}"
    return preds


# ---------------- 训练集均值灰图 ----------------

def _image_sum_size(p):
    """单图 (RGB 化后) 的像素和与像素数; 模块顶层函数, 供 multiprocessing.Pool 序列化."""
    import numpy as np
    from PIL import Image
    im = Image.open(p).convert("RGB")
    a = np.asarray(im, dtype=np.float64)
    return float(a.sum()), int(a.size)


def compute_train_mean(train_file, workers):
    """训练集所有唯一图像 (RGB 化后) 的像素均值; 多进程并行, 返回 (mean, n_images, elapsed)."""
    from multiprocessing import Pool

    with open(train_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    paths = sorted({p for r in data for p in r["images"]})

    t0 = time.perf_counter()
    with Pool(workers) as pool:
        res = pool.map(_image_sum_size, paths, chunksize=64)
    tot = sum(s for s, _ in res)
    n = sum(c for _, c in res)
    return tot / n, len(paths), time.perf_counter() - t0


def get_mean_gray(out_root, train_file, workers):
    """读取缓存或计算训练集均值, 存 out_root/mean_gray.json."""
    cache = out_root / "mean_gray.json"
    if cache.exists():
        d = json.loads(cache.read_text(encoding="utf-8"))
        print(f"[gray] 使用缓存均值 {d['mean']:.4f} (n_images={d['n_images']})")
        return d["mean"]
    mean, n_img, sec = compute_train_mean(train_file, workers)
    d = {"mean": mean, "n_images": n_img, "train_file": train_file,
         "elapsed_sec": round(sec, 1), "computed_at": time.strftime("%F %T")}
    cache.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[gray] 训练集均值 = {mean:.4f} (n_images={n_img}, {sec/60:.1f} min) -> {cache}")
    return mean


_gray_size_cache = {}


def gray_image(path, value):
    """与被替换图像同尺寸的均值灰图 (PIL RGB, 直接传给 llamafactory mm_plugin)."""
    from PIL import Image
    if path not in _gray_size_cache:
        with Image.open(path) as im:
            _gray_size_cache[path] = im.size
    w, h = _gray_size_cache[path]
    v = int(round(value))
    return Image.new("RGB", (w, h), (v, v, v))


# ---------------- 条件图像构造 ----------------

def build_shuffle_map(by_study, seed):
    """study 级随机错排 (derangement), seed 固定保证可复现."""
    studies = sorted(by_study.keys())
    rng = random.Random(seed)
    n = len(studies)
    while True:
        perm = list(range(n))
        rng.shuffle(perm)
        if all(perm[i] != i for i in range(n)):
            break
    return {studies[i]: studies[perm[i]] for i in range(n)}


def cond_images(cond, rec, by_study, shuffle_map, gray_value):
    """返回该条件下的 [CC, MLO] 图像列表 (路径 str 或 PIL Image)."""
    imgs = list(rec["images"])  # [CC, MLO]
    if cond == "orig":
        return imgs
    if cond in ("blank", "cc_only", "mlo_only"):
        out = []
        for i, p in enumerate(imgs):
            gray = (cond == "blank"
                    or (cond == "cc_only" and i == 1)   # cc_only: MLO 换灰
                    or (cond == "mlo_only" and i == 0))  # mlo_only: CC 换灰
            out.append(gray_image(p, gray_value) if gray else p)
        return out
    if cond == "swap_side":
        sid, lat = rec["id"].rsplit("_", 1)
        opp = by_study[sid]["L" if lat == "R" else "R"]
        return list(opp["images"])
    if cond == "shuffle_patient":
        sid, lat = rec["id"].rsplit("_", 1)
        src = by_study[shuffle_map[sid]][lat]
        return list(src["images"])
    raise ValueError(f"unknown condition: {cond}")


# ---------------- model ----------------

def load_model(adapter):
    import torch
    from llamafactory.chat import ChatModel
    assert torch.cuda.is_available(), "CUDA unavailable"
    torch.cuda.set_device(0)
    print(f"[cuda] device: {torch.cuda.get_device_name(0)}")
    model_args = dict(
        model_name_or_path=S2V.MODEL_PATH,
        template=S2V.TEMPLATE,
        trust_remote_code=True,
        infer_backend="huggingface",
        infer_dtype=S2V.INFER_DTYPE,
        image_max_pixels=S2V.IMAGE_MAX_PIXELS,
        image_min_pixels=S2V.IMAGE_MIN_PIXELS,
        enable_thinking=False,
        flash_attn="sdpa",
        adapter_name_or_path=adapter,
        finetuning_type="lora",
    )
    t0 = time.perf_counter()
    cm = ChatModel(args=model_args)
    engine = cm.engine
    model, tokenizer, processor, template = (
        engine.model, engine.tokenizer, engine.processor, engine.template)
    torch.cuda.synchronize()
    print(f"[model] loaded in {time.perf_counter() - t0:.1f}s (adapter={adapter})")
    return model, tokenizer, processor, template


# ---------------- 打分 ----------------

def load_done_ids(out_file):
    done = set()
    if out_file.exists():
        with open(out_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    pass
    return done


def run_condition(cond, ids, recs, preds, bundle, instruction,
                  by_study, shuffle_map, gray_value, out_file, tag):
    """单条件打分 (断点续跑), 返回 (n_new, n_none, elapsed_sec, sec_per_case)."""
    from tqdm.auto import tqdm
    model, tokenizer, processor, template = bundle
    done = load_done_ids(out_file)
    pending = [i for i in ids if i not in done and i in preds]
    print(f"[{cond}] total={len(ids)} done={len(done)} pending={len(pending)} -> {out_file}")
    if not pending:
        return 0, 0, 0.0, 0.0

    n_none = 0
    t_all = time.perf_counter()
    with open(out_file, "a", encoding="utf-8") as fout:
        pbar = tqdm(pending, desc=f"ablate_{tag}_{cond}", unit="case", dynamic_ncols=True)
        for sid in pbar:
            t0 = time.perf_counter()
            rec = recs[sid]
            images = cond_images(cond, rec, by_study, shuffle_map, gray_value)
            completion = S2V.extract_json_text(preds[sid].get("prediction_raw") or "")
            probs, probs_norm, argmax_match = None, None, None
            if completion:
                try:
                    probs, probs_norm, argmax_match, _flags = S2V.score_one(
                        model, tokenizer, processor, template, instruction,
                        {"id": sid, "images": images}, completion)
                except Exception as e:
                    print(f"\n[warn] {cond}/{sid}: {e!r}")
            else:
                print(f"\n[warn] {cond}/{sid}: pred file 无可解析 JSON")
            if probs is None:
                n_none += 1
            row = {
                "id": sid,
                "ground_truth": preds[sid].get("ground_truth"),
                "greedy_prediction": preds[sid].get("prediction_json"),
                "argmax_match": argmax_match,
                "probs": probs,
                "probs_norm": probs_norm,
                "seconds": round(time.perf_counter() - t0, 2),
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            pbar.set_postfix(avg=f"{(time.perf_counter() - t_all) / max(pbar.n, 1):.2f}s",
                             none=n_none)
    elapsed = time.perf_counter() - t_all
    print(f"[{cond}] new={len(pending)} none={n_none} elapsed={elapsed/60:.1f} min "
          f"({elapsed / len(pending):.2f}s/case)")
    return len(pending), n_none, elapsed, elapsed / len(pending)


# ---------------- orig 对拍 (smoke gate) ----------------

def flatten_probs(p):
    """把 probs/probs_norm 字典展平成 {field.cand: value} 便于逐值对比."""
    out = {}
    if not p:
        return out
    for field, d in p.items():
        if isinstance(d, dict):
            for c, v in d.items():
                out[f"{field}.{c}"] = v
    return out


def gate_orig_vs_scored(smoke_file, scored_file, ids):
    """对比 smoke 阶段 orig 条件与已有打分文件的同 id 概率, 返回 max_abs_diff."""
    scored = {}
    with open(scored_file, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["id"] in ids:
                scored[d["id"]] = d
    max_diff, n_cmp = 0.0, 0
    rows = {}
    with open(smoke_file, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            rows[d["id"]] = d
    for sid in ids:
        assert sid in rows, f"smoke 缺少 {sid}"
        assert sid in scored, f"已有打分文件缺少 {sid}"
        a, b = flatten_probs(rows[sid]["probs"]), flatten_probs(scored[sid]["probs"])
        assert set(a) == set(b), f"{sid} 字段不一致"
        for k in a:
            if a[k] is None or b[k] is None:
                continue
            max_diff = max(max_diff, abs(a[k] - b[k]))
            n_cmp += 1
    print(f"[gate] orig 对拍: {n_cmp} 个概率值, max_abs_diff={max_diff:.6f} (tol={ORIG_GATE_TOL})")
    return max_diff


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "full"], required=True)
    ap.add_argument("--adapter", default=DEFAULT_ADAPTER)
    ap.add_argument("--pred-file", default=DEFAULT_PRED_FILE,
                    help="orig 贪心预测 jsonl (teacher-forcing 条件序列)")
    ap.add_argument("--data", default=DEFAULT_DATA_FILE, help="官方 test.json")
    ap.add_argument("--train-file", default=DEFAULT_TRAIN_FILE, help="计算均值灰图用训练集")
    ap.add_argument("--orig-scored", default=DEFAULT_ORIG_SCORED,
                    help="已有 orig 全量打分文件 (full 阶段复制为 orig 条件)")
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--mean-workers", type=int, default=7)
    ap.add_argument("--budget-min", type=float, default=150.0,
                    help="full 阶段预计耗时超过此分钟数则中止 (防止超作业时限)")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    raw, recs, order, by_study = load_test(args.data)
    preds = load_preds(args.pred_file)
    missing = [i for i in order if i not in preds]
    assert not missing, f"pred 文件缺少 {len(missing)} 个 test id, 如 {missing[:3]}"
    print(f"[data] test={len(order)} studies={len(by_study)} preds={len(preds)}")

    gray_value = get_mean_gray(out_root, args.train_file, args.mean_workers)
    shuffle_map = build_shuffle_map(by_study, SEED)
    # 错排自检
    assert all(k != v for k, v in shuffle_map.items()) and len(shuffle_map) == len(by_study)

    instruction = S2V.load_instruction()
    bundle = load_model(args.adapter)

    if args.mode == "smoke":
        smoke_ids = order[:SMOKE_N]
        report = {"n_smoke": SMOKE_N, "seed": SEED, "mean_gray": gray_value, "conditions": {}}
        for cond in ALL_CONDS:
            out_file = out_root / cond / "smoke_with_probs.jsonl"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            if out_file.exists():
                out_file.unlink()   # smoke 每次重跑
            n, n_none, sec, spc = run_condition(
                cond, smoke_ids, recs, preds, bundle, instruction,
                by_study, shuffle_map, gray_value, out_file, "smoke")
            report["conditions"][cond] = {
                "n": n, "probs_none": n_none, "elapsed_sec": round(sec, 1),
                "sec_per_case": round(spc, 3)}
            assert n_none == 0, f"smoke {cond} 出现 probs=None, 终止"
        # orig 对拍
        max_diff = gate_orig_vs_scored(
            out_root / "orig" / "smoke_with_probs.jsonl", args.orig_scored, smoke_ids)
        report["orig_gate_max_abs_diff"] = round(max_diff, 6)
        report["orig_gate_pass"] = bool(max_diff <= ORIG_GATE_TOL)

        # 全量耗时预估: 5 个消融条件 x 2000
        rates = [report["conditions"][c]["sec_per_case"] for c in ABLATION_CONDS]
        est_min = sum(r * len(order) for r in rates) / 60 + 5  # +5min 模型加载/余量
        report["estimated_full_min"] = round(est_min, 1)
        (out_root / "smoke_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print()
        print("==================== SMOKE SUMMARY ====================")
        for cond in ALL_CONDS:
            c = report["conditions"][cond]
            print(f"  {cond:16s} {c['sec_per_case']:.2f}s/case  none={c['probs_none']}")
        print(f"  orig gate max_abs_diff={max_diff:.6f} pass={report['orig_gate_pass']}")
        print(f"  全量预估 (5 条件 x {len(order)}): {est_min:.0f} min "
              f"(budget={args.budget_min:.0f} min)")
        if not report["orig_gate_pass"]:
            print("  [ABORT] orig 对拍超差, 不进入全量")
            sys.exit(2)
        if est_min > args.budget_min:
            print(f"  [ABORT] 预估 {est_min:.0f} min 超预算 {args.budget_min:.0f} min, 不进入全量")
            sys.exit(3)

    elif args.mode == "full":
        # 复制 orig (不重跑, 与既往 Run A 结果完全一致)
        orig_out = out_root / "orig" / "test_with_probs.jsonl"
        orig_out.parent.mkdir(parents=True, exist_ok=True)
        if not orig_out.exists():
            n_lines = sum(1 for _ in open(args.orig_scored, encoding="utf-8"))
            assert n_lines == len(order), f"orig scored 文件行数 {n_lines} != {len(order)}"
            shutil.copyfile(args.orig_scored, orig_out)
            print(f"[orig] 复制 {args.orig_scored} -> {orig_out} ({n_lines} 行)")
        else:
            print(f"[orig] 已存在 {orig_out}")

        report = {"mean_gray": gray_value, "seed": SEED, "conditions": {}}
        t_full = time.perf_counter()
        for cond in ABLATION_CONDS:
            out_file = out_root / cond / "test_with_probs.jsonl"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            n, n_none, sec, spc = run_condition(
                cond, order, recs, preds, bundle, instruction,
                by_study, shuffle_map, gray_value, out_file, "full")
            report["conditions"][cond] = {
                "n": n, "probs_none": n_none, "elapsed_sec": round(sec, 1),
                "sec_per_case": round(spc, 3)}
        report["total_min"] = round((time.perf_counter() - t_full) / 60, 1)
        (out_root / "full_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        # 每个 meta.json 记录条件定义
        for cond in ALL_CONDS:
            meta = {
                "condition": cond,
                "n_test": len(order),
                "seed": SEED if cond == "shuffle_patient" else None,
                "mean_gray": gray_value if cond in ("blank", "cc_only", "mlo_only") else None,
                "adapter": args.adapter,
                "pred_file": args.pred_file,
                "data": args.data,
                "note": ("orig=复用已有打分文件" if cond == "orig" else
                         "图像替换, 标签与 prompt 不变"),
            }
            (out_root / cond / "meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        print()
        print("==================== FULL SUMMARY ====================")
        for cond in ABLATION_CONDS:
            c = report["conditions"][cond]
            print(f"  {cond:16s} new={c['n']} none={c['probs_none']} "
                  f"{c['elapsed_sec']/60:.1f} min ({c['sec_per_case']:.2f}s/case)")
        print(f"  TOTAL: {report['total_min']} min")
        print(f"Output root: {out_root}")


if __name__ == "__main__":
    main()
