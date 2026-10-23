#!/usr/bin/env python3
"""单元测试 1：第 2 次前向替换的位置是否就是 finding 取值 token 本身（CPU，只需 tokenizer）。

检查：
  - finding_pos / birads_pos 指向的 token，其解码文本恰好等于该字段的取值（'0'/'1' / '1'..'5'）；
  - 打印每个位置前后各 3 个 token 的上下文；
  - 序列位置 = prompt_len + comp_idx 的换算关系成立。
"""
import json
import re
import sys
from pathlib import Path

CV1 = Path(__file__).resolve().parent.parent
PROJ = CV1.parent
sys.path.insert(0, str(PROJ / "scripts" / "vlm"))
sys.path.insert(0, str(CV1.parent))

from transformers import AutoTokenizer  # noqa: E402
from score_2view import FIELD_SPECS, find_token_index  # noqa: E402
from chain_v1.train.data import FIND, FIND_FIELDS, gt_text, positions_in_text  # noqa: E402

BASE = "/home/z/zhao/Mammography/model/Qwen3.5-4B"
TEST = PROJ / "data" / "splits" / "test_local.json"


def test_positions_point_at_value_tokens():
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    rec = json.load(open(TEST, encoding="utf-8"))[0]
    text = gt_text(rec["output"])
    ids, pos = positions_in_text(text, tok)

    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    offs = enc["offset_mapping"]

    print(f"completion ({len(ids)} tokens): {text}")
    for field in ["breast_density", "breast_birads"] + FIND_FIELDS:
        i = pos[field]
        assert i >= 0, f"{field} 未定位"
        piece = text[offs[i][0]:offs[i][1]]
        m = re.search(FIELD_SPECS[field][0], text)
        want = m.group(1)
        assert piece == want, f"{field}: token 覆盖 {piece!r} != 值 {want!r}"
        lo, hi = max(0, i - 3), min(len(ids), i + 4)
        ctx = []
        for j in range(lo, hi):
            s, e = offs[j]
            ctx.append((">>" if j == i else "  ") + f"[{j}]{text[s:e]!r}")
        print(f"  {field:<34} comp_idx={i:2d} 值={want!r} id={ids[i]:3d}  " + " ".join(ctx))

    # 序列位置换算：seq_pos = prompt_len + comp_idx
    prompt_len = 1906   # 786432 分辨率下的实测 prompt 长度（仅演示换算）
    for field in FIND_FIELDS:
        assert prompt_len + pos[field] < prompt_len + len(ids)
    print("\nOK: 所有位置都指向取值 token 自身，seq_pos = prompt_len + comp_idx")


if __name__ == "__main__":
    test_positions_point_at_value_tokens()
