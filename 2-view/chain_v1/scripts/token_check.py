#!/usr/bin/env python3
"""0. 分词检查（纯 CPU）：确认 finding 的 0/1 与 BI-RADS 的 1..5 是独立单 token。"""
import json
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJ / "scripts" / "eval"))
from transformers import AutoTokenizer  # noqa: E402

BASE = "/home/z/zhao/Mammography/model/Qwen3.5-4B"
PROBS = PROJ / "outputs" / "scoring" / "os_sub_vt_test_b8_with_probs.jsonl"
FINDINGS = ["mass", "suspicious_calcification", "asymmetry", "architectural_distortion"]


def main():
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    print("=== 候选值的单 token 检查 ===")
    for v in ["0", "1", "2", "3", "4", "5", "A", "B", "C", "D"]:
        ids = tok.encode(v, add_special_tokens=False)
        print(f"  encode('{v}') = {ids} (len={len(ids)}), decode={tok.decode(ids)!r}")
    for v in ["0", "1"]:
        for pre in ['"', ', ', ': ', '{"findings": {"mass": ', '(', ' 0', '1']:
            ids = tok.encode(pre + v, add_special_tokens=False)
            print(f"  encode({pre!r}+'{v}') 末 token={ids[-1]} -> {tok.decode([ids[-1]])!r}")

    rec = json.loads(open(PROBS, encoding="utf-8").readline())
    raw = rec["greedy_prediction"]
    text = json.dumps(raw, ensure_ascii=False) if not isinstance(raw, str) else raw
    print(f"\n=== 一条样本的贪心输出 ===\n{text}")

    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    print(f"\n=== 完整 token 切分（共 {len(ids)} 个）===")
    for i, (t, (s, e)) in enumerate(zip(ids, offs)):
        print(f"  [{i:3d}] id={t:6d} span=({s:3d},{e:3d}) text={text[s:e]!r} dec={tok.decode([t])!r}")

    print("\n=== 各字段取值位置检查 ===")
    import re
    specs = {"breast_density": (r'"breast_density"\s*:\s*"([ABCD])"', 1),
             "breast_birads": (r'"breast_birads"\s*:\s*([1-5])', 1)}
    for k in FINDINGS:
        specs[f"findings.{k}"] = (rf'"{k}"\s*:\s*([01])', 1)
    for field, (pat, grp) in specs.items():
        m = re.search(pat, text)
        if not m:
            print(f"  {field}: 未匹配")
            continue
        pos = m.start(grp)
        idx = next((i for i, (s, e) in enumerate(offs) if s <= pos < e), None)
        piece = text[offs[idx][0]:offs[idx][1]] if idx is not None else None
        clean = piece == m.group(grp)
        print(f"  {field}: 值={m.group(grp)!r} 覆盖 token={piece!r} standalone={clean} "
              f"tok_id={ids[idx] if idx is not None else None} "
              f"enc(值)={tok.encode(m.group(grp), add_special_tokens=False)}")


if __name__ == "__main__":
    main()
