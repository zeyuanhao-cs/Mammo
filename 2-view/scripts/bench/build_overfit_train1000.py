#!/usr/bin/env python3
"""从 train_sub 抽 1000 个乳房（seed=42，按有病灶/无病灶分层比例抽样，保证含病灶样本）。
输出与 test_local.json 同 schema，供 infer_2view_batched.py 直接使用。
"""
import json
import random
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent.parent
SRC = PROJ / "data" / "splits" / "train_sub_raw_local.json"
DST = PROJ / "data" / "splits" / "overfit_train1000_seed42.json"
FINDINGS = ["mass", "suspicious_calcification", "asymmetry", "architectural_distortion"]
N, SEED = 1000, 42


def has_lesion(r):
    return any(int(r["output"]["findings"][k]) for k in FINDINGS)


def main():
    data = json.load(open(SRC, encoding="utf-8"))
    pos = [r for r in data if has_lesion(r)]
    neg = [r for r in data if not has_lesion(r)]
    n_pos = round(N * len(pos) / len(data))
    rng = random.Random(SEED)
    pick = rng.sample(pos, n_pos) + rng.sample(neg, N - n_pos)
    rng.shuffle(pick)
    out = [{"id": r["id"], "images": r["images"], "output": r["output"]} for r in pick]
    DST.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(DST, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[src] {SRC.name} n={len(data)} lesion+={len(pos)} ({len(pos)/len(data):.4f})")
    print(f"[out] {DST} n={len(out)} lesion+={sum(1 for r in out if has_lesion(r))}")


if __name__ == "__main__":
    main()
