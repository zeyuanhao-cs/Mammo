#!/usr/bin/env python3
"""确定样本上软注入的扰动：QC1 soft vs swap_hard（同批 100 例）。

报告并写入 quick_report.md 单独一节：
  - P(BI-RADS=1) 的中位数（两种模式）与平均绝对差
  - 预测 BI-RADS 不一致的个数
"""
import json
import statistics as st
from pathlib import Path

P = Path("/home/z/zhao/Mammography/Mammo/2-view/chain_v1/outputs/pilot")


def load(fn):
    f = P / fn
    if not f.exists():
        return {}
    out = {}
    for line in f.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["id"]] = r
    return out


soft = load("qc1_cert_soft.jsonl")
swap = load("qc1_cert_swap.jsonl")
ids = sorted(set(soft) & set(swap))
if not ids:
    print("[cert] 缺 qc1_cert_soft/swap.jsonl，跳过")
    raise SystemExit(0)


def p1(r):
    return float((r.get("probs_norm") or {}).get("breast_birads", {}).get("1", float("nan")))


def pred(r):
    bi = (r.get("probs_norm") or {}).get("breast_birads") or {}
    return int(max(bi.items(), key=lambda kv: kv[1])[0]) if bi else None


ps = [p1(soft[i]) for i in ids]
pw = [p1(swap[i]) for i in ids]
d = [abs(a - b) for a, b in zip(ps, pw)]
bad = [i for i in ids if pred(soft[i]) != pred(swap[i])]

L = ["## 确定样本上软注入的扰动", "",
     f"- 样本：GT BI-RADS=1 且起点模型 4 征象 p<0.05，val 取 {len(ids)} 例（QC1，batch=1）",
     f"- P(BI-RADS=1) 中位：软 {st.median(ps):.4f} / 软换硬 {st.median(pw):.4f}",
     f"- |ΔP(BI-RADS=1)| 平均：{st.mean(d):.4f}（中位 {st.median(d):.4f}，最大 {max(d):.4f}）",
     f"- 预测 BI-RADS 不一致：{len(bad)}/{len(ids)}"
     + (f"（{', '.join(bad[:5])}{' …' if len(bad) > 5 else ''}）" if bad else ""),
     ""]
sec = "\n".join(L)
print(sec)

rep = P / "quick_report.md"
rep.write_text((rep.read_text(encoding="utf-8") if rep.exists() else "# 阶段一报告\n") + "\n" + sec,
               encoding="utf-8")
print("[cert] 已追加到 quick_report.md")
