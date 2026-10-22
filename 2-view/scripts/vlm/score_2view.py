#!/usr/bin/env python3
"""
2-view 字段级概率 scoring：对 LoRA 模型在 test 集上计算每个字段取值的 softmax 概率。

与 infer_2view.py 完全一致的设置：
  - ChatModel 以相同 model_args 加载（template=qwen3_5, enable_thinking=False, sdpa,
    bfloat16, image_max_pixels=786432, image_min_pixels=512*512, 同一 adapter）
  - 输入构建复用 HuggingfaceEngine._process_args 的同一路径
    (process_messages → encode_oneturn → process_token_ids → get_mm_inputs)，
    保证 prompt / 图像 token 与原推理逐 token 一致

方法：
  - 条件序列 = 已有贪心预测文件中的 prediction_raw（模型自己的 JSON 输出）
  - teacher-forcing 单次 forward（只保留 completion 段 logits），
    在每个字段取值 token 位置取 softmax，得到候选取值概率：
      breast_birads: P(1..5); breast_density: P(A..D); findings.*: P(0)/P(1)
  - 不重新训练，不用 test 集调阈值

输出 jsonl 每行: {id, ground_truth, greedy_prediction, argmax_match, probs, probs_norm}
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import torch
from tqdm.auto import tqdm
from llamafactory.chat import ChatModel

MODEL_PATH = "/nesi/project/massey04900/Mammography/model/Qwen3.5-4B"
TEMPLATE = "qwen3_5"
DATA_FILE = Path("/nesi/project/massey04900/Mammography/test.json")
PROMPT_FILE = Path("/home/zhao/Mammography/prompt.txt")
# 本地复跑可用 --model-path / --prompt-file 覆盖上面两个 NeSI 路径;
# 打分逻辑 (score_one / FIELD_SPECS / 输入构建) 与 NeSI 版逐字相同。
IMAGE_MAX_PIXELS = 786432
IMAGE_MIN_PIXELS = 512 * 512
INFER_DTYPE = "bfloat16"

FINDINGS = ["mass", "suspicious_calcification", "asymmetry", "architectural_distortion"]

THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", flags=re.DOTALL)

# 字段 -> (值正则, 候选列表)
FIELD_SPECS = {
    "breast_density": (re.compile(r'"breast_density"\s*:\s*"([ABCD])"'), ["A", "B", "C", "D"]),
    "breast_birads": (re.compile(r'"breast_birads"\s*:\s*([1-5])'), ["1", "2", "3", "4", "5"]),
}
for _k in FINDINGS:
    FIELD_SPECS[f"findings.{_k}"] = (re.compile(rf'"{_k}"\s*:\s*([01])'), ["0", "1"])

THIS_DIR = Path(__file__).resolve().parent          # 2-view/scripts/vlm
OUT_DIR = THIS_DIR.parent.parent / "outputs" / "scoring"  # 默认输出目录 (--out 可覆盖)


def extract_json_text(raw: str):
    """从模型原始输出提取 JSON 子串（与 infer_2view.parse_prediction 同逻辑, 但返回文本）."""
    if not raw:
        return None
    text = THINK_RE.sub("", raw).strip()
    m = FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def load_instruction(prompt_file: Optional[Path] = None) -> str:
    prompt_file = Path(prompt_file) if prompt_file is not None else PROMPT_FILE
    text = prompt_file.read_text(encoding="utf-8").strip()
    n = text.count("<image>")
    if n == 1:
        return text.replace("<image>", "<image>\n<image>", 1)
    if n == 2:
        return text
    raise AssertionError(f"prompt 需要 1 或 2 个 <image> 占位符, 实际 {n}")


def find_token_index(offsets, char_start: int):
    """定位覆盖 char_start 的 token 下标."""
    for i, (s, e) in enumerate(offsets):
        if s <= char_start < e:
            return i
    for i, (s, e) in enumerate(offsets):
        if s >= char_start:
            return i
    return None


def common_prefix_len(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, required=True, help="LoRA adapter 路径")
    parser.add_argument("--pred-file", type=str, required=True, help="已有贪心预测 jsonl（提供条件序列）")
    parser.add_argument("--out", type=str, required=True, help="输出 jsonl 路径")
    parser.add_argument("--limit", type=int, default=0, help="仅前 N 条（0=全部, smoke test 用）")
    parser.add_argument("--tag", type=str, default="", help="日志标识")
    parser.add_argument("--data", type=str, default=str(DATA_FILE), help="数据 json (id/images/output), 默认官方 test.json")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH, help="基座模型路径 (本地复跑时覆盖)")
    parser.add_argument("--prompt-file", type=str, default=str(PROMPT_FILE), help="prompt.txt 路径 (本地复跑时覆盖)")
    parser.add_argument("--image-max-pixels", type=int, default=IMAGE_MAX_PIXELS,
                        help=f"视觉最长边像素预算, 默认 {IMAGE_MAX_PIXELS}; 1386240=原生")
    parser.add_argument("--image-min-pixels", type=int, default=IMAGE_MIN_PIXELS,
                        help=f"视觉最短边像素预算, 默认 {IMAGE_MIN_PIXELS}")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA unavailable"
    torch.cuda.set_device(0)
    print(f"[cuda] device: {torch.cuda.get_device_name(0)}")

    # ---------------- data ----------------
    instruction = load_instruction(Path(args.prompt_file))
    with open(args.data, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    id2rec = {r["id"]: r for r in raw_data}

    preds = {}
    with open(args.pred_file, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            preds[d["id"]] = d
    assert preds, f"empty pred file: {args.pred_file}"

    ids = [r["id"] for r in raw_data]
    if args.limit > 0:
        ids = ids[: args.limit]

    out_file = Path(args.out)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_file.exists():
        with open(out_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    pass
    pending = [i for i in ids if i not in done and i in preds]
    print(f"[data] total={len(ids)} pred_file={len(preds)} done={len(done)} pending={len(pending)}")

    # ---------------- model（与 infer_2view.py 完全一致） ----------------
    model_args = dict(
        model_name_or_path=args.model_path,
        template=TEMPLATE,
        trust_remote_code=True,
        infer_backend="huggingface",
        infer_dtype=INFER_DTYPE,
        image_max_pixels=args.image_max_pixels,
        image_min_pixels=args.image_min_pixels,
        enable_thinking=False,
        flash_attn="sdpa",
        adapter_name_or_path=args.adapter,
        finetuning_type="lora",
    )
    t0 = time.perf_counter()
    cm = ChatModel(args=model_args)
    engine = cm.engine
    model, tokenizer, processor, template = engine.model, engine.tokenizer, engine.processor, engine.template
    torch.cuda.synchronize()
    print(f"[model] loaded in {time.perf_counter() - t0:.1f}s (adapter={args.adapter})")

    # ---------------- scoring ----------------
    n_multi_token = 0
    n_argmax_mismatch = 0
    n_field_missing = 0
    elapsed = 0.0
    t_all = time.perf_counter()

    with open(out_file, "a", encoding="utf-8") as fout:
        pbar = tqdm(pending, desc=f"score_{args.tag}", unit="case", dynamic_ncols=True)
        for sid in pbar:
            t0 = time.perf_counter()
            rec = id2rec[sid]
            pred = preds[sid]

            completion = extract_json_text(pred.get("prediction_raw") or "")
            probs, probs_norm, argmax_match = None, None, None
            if completion:
                try:
                    probs, probs_norm, argmax_match, flags = score_one(
                        model, tokenizer, processor, template, instruction, rec, completion
                    )
                    n_multi_token += flags["multi_token"]
                    n_argmax_mismatch += flags["argmax_mismatch"]
                    n_field_missing += flags["field_missing"]
                except Exception as e:
                    probs, probs_norm, argmax_match = None, None, None
                    print(f"\n[warn] {sid}: {e!r}")

            row = {
                "id": sid,
                "ground_truth": pred.get("ground_truth"),
                "greedy_prediction": pred.get("prediction_json"),
                "argmax_match": argmax_match,
                "probs": probs,
                "probs_norm": probs_norm,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()

            elapsed += time.perf_counter() - t0
            pbar.set_postfix(avg=f"{elapsed / (pbar.n + 1):.2f}s", mt=n_multi_token, mm=n_argmax_mismatch)

    total = time.perf_counter() - t_all
    print()
    print("==================== SCORING SUMMARY ====================")
    print(f"tag={args.tag} n={len(pending)}")
    print(f"multi-token candidates: {n_multi_token} (用首 token 概率近似)")
    print(f"argmax != greedy token: {n_argmax_mismatch} (应接近 0, 检验 teacher-forcing 一致性)")
    print(f"missing fields: {n_field_missing}")
    print(f"TOTAL: {total / 60:.1f} min ({total / 3600:.2f} h)")
    print(f"Output: {out_file}")


def score_one(model, tokenizer, processor, template, instruction, rec, completion):
    """单样本: 构建 prompt(与原推理一致) + teacher-force completion, 返回字段概率."""
    flags = {"multi_token": 0, "argmax_mismatch": 0, "field_missing": 0}
    images = list(rec["images"])

    # ---- 复刻 HuggingfaceEngine._process_args 的输入构建 ----
    messages = [{"role": "user", "content": instruction}]
    messages = template.mm_plugin.process_messages(messages, images, [], [], processor)
    paired_messages = messages + [{"role": "assistant", "content": ""}]
    prompt_ids, _ = template.encode_oneturn(tokenizer, paired_messages, None, None)
    prompt_ids, _ = template.mm_plugin.process_token_ids(prompt_ids, None, images, [], [], tokenizer, processor)
    prompt_len = len(prompt_ids)

    # ---- completion token化 (含 offset 映射) ----
    enc = tokenizer(completion, add_special_tokens=False, return_offsets_mapping=True)
    completion_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    assert len(completion_ids) == len(offsets)

    input_ids = prompt_ids + completion_ids
    ids_t = torch.tensor([input_ids], device=model.device)
    attn_t = torch.ones_like(ids_t)

    # ---- mm inputs (与 engine 相同的 tensor 转换) ----
    mm_inputs = template.mm_plugin.get_mm_inputs(
        images=images, videos=[], audios=[], imglens=[len(images)], vidlens=[0], audlens=[0],
        batch_ids=[prompt_ids], processor=processor,
    )
    mm_kwargs = {}
    for key, value in mm_inputs.items():
        if isinstance(value, list) and isinstance(value[0], torch.Tensor):
            value = torch.stack(value)
        elif isinstance(value, list) and isinstance(value[0], list) and isinstance(value[0][0], torch.Tensor):
            value = torch.stack([torch.stack(v) for v in value])
        elif not isinstance(value, torch.Tensor):
            value = torch.tensor(value)
        if torch.is_floating_point(value):
            value = value.to(model.dtype)
        if key == "second_per_grid_ts":
            mm_kwargs[key] = value.tolist()
        else:
            mm_kwargs[key] = value.to(model.device)

    # Qwen3.5 M-RoPE 需要 mm_token_type_ids (generate 内部自动处理, 直接 forward 需显式传入)
    if hasattr(processor, "create_mm_token_type_ids"):
        mtt = processor.create_mm_token_type_ids([input_ids])
        if not isinstance(mtt, torch.Tensor):
            mtt = torch.tensor(mtt)
        mm_kwargs["mm_token_type_ids"] = mtt.to(model.device)

    # ---- 单次 forward, 只保留 completion 段 logits ----
    K = len(completion_ids) + 1
    with torch.inference_mode():
        try:
            logits = model(
                input_ids=ids_t, attention_mask=attn_t, use_cache=False,
                logits_to_keep=K, **mm_kwargs,
            ).logits[0]
        except TypeError:
            try:
                logits = model(
                    input_ids=ids_t, attention_mask=attn_t, use_cache=False,
                    num_logits_to_keep=K, **mm_kwargs,
                ).logits[0]
            except TypeError:
                logits = model(input_ids=ids_t, attention_mask=attn_t, use_cache=False, **mm_kwargs).logits[0]
    # logits 可能是 [L,V] 或 [K,V]; 统一: 取预测 completion token j 的行 = seq位置 prompt_len+j-1
    logits = logits.float()
    if logits.shape[0] != len(input_ids):  # 已裁剪为最后 K 行
        assert logits.shape[0] == K
        # 行 r 对应 seq 位置 L-K+r = prompt_len-1+r, 预测 completion token r
        row_of = lambda j: j
    else:
        row_of = lambda j: prompt_len + j - 1

    probs, probs_norm, argmax_match = {}, {}, {}
    for field, (regex, cands) in FIELD_SPECS.items():
        m = regex.search(completion)
        if not m:
            flags["field_missing"] += 1
            probs[field] = None
            probs_norm[field] = None
            argmax_match[field] = None
            continue
        value_start = m.start(1)
        tok_idx = find_token_index(offsets, value_start)
        if tok_idx is None:
            flags["field_missing"] += 1
            probs[field] = None
            probs_norm[field] = None
            argmax_match[field] = None
            continue

        prefix_ids = completion_ids[:tok_idx]
        context_str = completion[: offsets[tok_idx][0]]
        # 该位置整行 softmax
        row = logits[row_of(tok_idx)]
        sm = torch.softmax(row, dim=-1)
        # 贪心 token 是否为全词表 argmax (teacher-forcing 一致性检验)
        if int(row.argmax()) != completion_ids[tok_idx]:
            flags["argmax_mismatch"] += 1
            argmax_match[field] = False
        else:
            argmax_match[field] = True

        p = {}
        for c in cands:
            full = tokenizer(context_str + c, add_special_tokens=False)["input_ids"]
            common = common_prefix_len(full, prefix_ids)
            cand_toks = full[common:]
            if len(cand_toks) != 1:
                flags["multi_token"] += 1
            p[c] = float(sm[cand_toks[0]])
        probs[field] = p
        z = sum(p.values())
        probs_norm[field] = {c: (v / z if z > 0 else 0.0) for c, v in p.items()} if z > 0 else None

    return probs, probs_norm, argmax_match, flags


if __name__ == "__main__":
    main()
