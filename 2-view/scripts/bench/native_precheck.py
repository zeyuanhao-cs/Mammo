#!/usr/bin/env python3
"""原生分辨率(1386240)内存测试前的确认：
 1) 一张测试图经 processor 后的实际尺寸 + 每视角视觉 token 数（对比 786432）
 2) 训练集最长样本总 token 数（确认 < cutoff_len 4096，不会截断）
 3) smart_resize 公式校验（对照 processor 真实 grid），用于快速扫描全训练集
"""
import json
import math
import random
import sys
from pathlib import Path

import torch
from PIL import Image
from llamafactory.chat import ChatModel

PROJ = Path(__file__).resolve().parent.parent.parent
BASE = "/home/z/zhao/Mammography/model/Qwen3.5-4B"
TRAIN = PROJ / "data" / "llamafactory" / "train_llamafactory_os_sub_local.json"
TEST = PROJ / "data" / "splits" / "test_local.json"
MIN_PX = 262144
FINDINGS = ["mass", "suspicious_calcification", "asymmetry", "architectural_distortion"]
CUTOFF = 4096


def gt_text(out):
    out = json.loads(out) if isinstance(out, str) else out
    return json.dumps({"breast_density": out["breast_density"],
                       "findings": {k: int(out["findings"][k]) for k in FINDINGS},
                       "breast_birads": int(out["breast_birads"])}, ensure_ascii=False)


def smart_resize(w, h, max_px, min_px, factor):
    h_bar = max(factor, round(h / factor) * factor)
    w_bar = max(factor, round(w / factor) * factor)
    if h_bar * w_bar > max_px:
        beta = math.sqrt((h * w) / max_px)
        h_bar = max(factor, math.floor(h / beta / factor) * factor)
        w_bar = max(factor, math.floor(w / beta / factor) * factor)
    elif h_bar * w_bar < min_px:
        beta = math.sqrt(min_px / (h * w))
        h_bar = math.ceil(h * beta / factor) * factor
        w_bar = math.ceil(w * beta / factor) * factor
    return w_bar, h_bar


def main():
    cm = ChatModel(args=dict(
        model_name_or_path=BASE, template="qwen3_5_nothink", trust_remote_code=True,
        infer_backend="huggingface", infer_dtype="bfloat16",
        image_max_pixels=1386240, image_min_pixels=MIN_PX,
        enable_thinking=False, flash_attn="sdpa",
    ))
    eng = cm.engine
    tokenizer, processor, template = eng.tokenizer, eng.processor, eng.template
    ip = processor.image_processor
    patch = int(getattr(ip, "patch_size", 16))
    ms = int(getattr(ip, "merge_size", 2))
    factor = patch * ms
    print(f"[proc] patch_size={patch} merge_size={ms} factor={factor} "
          f"min_pixels={getattr(ip,'min_pixels',None)}")

    train = json.load(open(TRAIN, encoding="utf-8"))
    instr = train[0]["instruction"]

    def build(images, max_px, response=""):
        processor.image_max_pixels = max_px
        msgs = [{"role": "user", "content": instr}]
        msgs = template.mm_plugin.process_messages(msgs, images, [], [], processor)
        paired = msgs + [{"role": "assistant", "content": response}]
        pids, rids = template.encode_oneturn(tokenizer, paired, None, None)
        pids, _ = template.mm_plugin.process_token_ids(pids, None, images, [], [], tokenizer, processor)
        mm = template.mm_plugin.get_mm_inputs(images=images, videos=[], audios=[],
                                             imglens=[len(images)], vidlens=[0], audlens=[0],
                                             batch_ids=[pids], processor=processor)
        return pids, rids, mm

    def grid_of(mm):
        g = mm["image_grid_thw"]
        return g.tolist() if torch.is_tensor(g) else g

    # ---------- 1) 单图 ----------
    rec = json.load(open(TEST, encoding="utf-8"))[0]
    print("\n============ 1) 一张测试图的处理结果 ============")
    print(f"  原图 (WxH): {[Image.open(p).size for p in rec['images']]}")
    for max_px in (786432, 1386240):
        pids, _rids, mm = build(list(rec["images"]), max_px)
        g = grid_of(mm)
        sizes = [(int(x[2]) * patch, int(x[1]) * patch) for x in g]
        toks = [int(x[1] * x[2] // (ms * ms)) for x in g]
        print(f"  max_pixels={max_px}: 处理后(WxH)={sizes}  视觉token/视角={toks}  "
              f"prompt token={len(pids)}  (2 视角视觉合计={sum(toks)})")

    # ---------- 3) 公式校验 ----------
    print("\n============ 3) smart_resize 公式校验（20 张随机训练图）============")
    rng = random.Random(42)
    idx = rng.sample(range(len(train)), 20)
    ok = 0
    for i in idx:
        imgs = list(train[i]["images"])
        _p, _r, mm = build(imgs, 1386240)
        real = [int(x[1] * x[2] // (ms * ms)) for x in grid_of(mm)]
        pred = []
        for p in imgs:
            w, h = Image.open(p).size
            wb, hb = smart_resize(w, h, 1386240, MIN_PX, factor)
            pred.append(int(wb * hb // (factor * factor)))
        ok += int(pred == real)
    print(f"  公式命中: {ok}/20")

    # ---------- 2) 全训练集最长样本 ----------
    print("\n============ 2) 训练集最长样本 token 数 ============")
    # overhead = 总 token - 视觉 token - response token（对固定 instruction 是常数）
    i0 = 0
    imgs0 = list(train[i0]["images"])
    pids, rids, mm = build(imgs0, 1386240, gt_text(train[i0]["output"]))
    img0 = sum(int(x[1] * x[2] // (ms * ms)) for x in grid_of(mm))
    overhead = len(pids) + len(rids) - img0 - len(rids)
    print(f"  overhead(文本+图像占位符+模版) = {overhead} tokens；示例 response token={len(rids)}")

    for max_px in (1386240, 786432):
        rows = []
        for r in train:
            img_tok = 0
            for p in r["images"]:
                w, h = Image.open(p).size
                wb, hb = smart_resize(w, h, max_px, MIN_PX, factor)
                img_tok += int(wb * hb // (factor * factor))
            resp = len(tokenizer(gt_text(r["output"]), add_special_tokens=False)["input_ids"])
            rows.append((overhead + img_tok + resp, Path(r["images"][0]).name, img_tok, resp))
        rows.sort(reverse=True)
        over = sum(1 for x in rows if x[0] >= CUTOFF)
        print(f"  max_pixels={max_px}: 最长={rows[0][0]} tokens (id={rows[0][1]}, 视觉={rows[0][2]}, 回复={rows[0][3]})"
              f" | top5={[x[0] for x in rows[:5]]} | ≥{CUTOFF} 的样本数={over}")
        if max_px == 1386240:
            print(f"    次长 {rows[1][0]}, 第3 {rows[2][0]}")
            print(f"    mean={sum(x[0] for x in rows)/len(rows):.1f}")


if __name__ == "__main__":
    main()
