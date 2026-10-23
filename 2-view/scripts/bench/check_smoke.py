#!/usr/bin/env python3
"""冒烟三项检查（供 native_master.sh 调用）。

1) 日志里没有 expandable_segments / CUDA driver error
2) 续跑：第二次 training 块存在，且首个 step 编号 > 5；续跑首/末 loss 连续（同量级、有限）
3) 16 条推理：probs 16 行、每行有 6 个字段的 probs、pred 16 行且 JSON 非空

退出码：0 = 全通过；非 0 = 失败（原因写到 stdout，master 会转存到 native_status.md）
"""
import json
import re
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent.parent
LOG = PROJ / "logs" / "vlm" / "native_ep2.log"
PROBS = PROJ / "outputs" / "scoring" / "smoke_native_16_with_probs.jsonl"
PRED = PROJ / "outputs" / "predictions" / "predictions_qwen3.5-4b_smoke_native_16.jsonl"
FIND = ["mass", "suspicious_calcification", "asymmetry", "architectural_distortion"]


def fail(msg):
    print(f"FAIL: {msg}")
    sys.exit(1)


def main():
    if not LOG.exists():
        fail(f"日志不存在 {LOG}")
    txt = LOG.read_text(errors="ignore").replace("\r", "\n")

    # 1) expandable
    bad = [ln for ln in txt.splitlines()
           if "expandable_segments" in ln or "CUDA driver error: operation not supported" in ln]
    if bad:
        fail(f"出现 expandable/driver 错误：{bad[-1][:160]}")

    # 2) 续跑
    blocks = txt.split("***** Running training *****")
    if len(blocks) < 3:
        fail(f"只检测到 {len(blocks)-1} 次 training 块（期望 2 次：首跑 + 续跑）")
    second = blocks[-1]
    steps = [int(m) for m in re.findall(r"(\d+)/10 \[", second)]
    if not steps:
        fail("续跑后没有 step 进度行")
    if steps[0] <= 5:
        fail(f"续跑首步是 {steps[0]}，说明没从 checkpoint-5 接着跑")
    losses = [float(m) for m in re.findall(r"'loss': '([0-9.]+)'", txt)]
    if len(losses) < 4:
        fail(f"loss 记录过少（{len(losses)} 条）")
    if not all(0.0 <= x < 20 for x in losses):
        fail(f"loss 异常：{losses[:6]}")

    # 3) 推理
    if not PROBS.exists() or sum(1 for _ in open(PROBS, encoding="utf-8")) < 16:
        fail("probs 不足 16 行")
    n_ok = 0
    for line in open(PROBS, encoding="utf-8"):
        d = json.loads(line)
        pn = d.get("probs_norm") or {}
        if all(k in pn for k in ["breast_density", "breast_birads"] + [f"findings.{k}" for k in FIND]):
            n_ok += 1
    if n_ok < 16:
        fail(f"只有 {n_ok}/16 条 probs 完整")
    n_json = 0
    if PRED.exists():
        for line in open(PRED, encoding="utf-8"):
            if json.loads(line).get("prediction_json"):
                n_json += 1
    if n_json < 16:
        fail(f"只有 {n_json}/16 条 prediction_json 非空")

    print(f"PASS: expandable=0, 续跑首步={steps[0]}, loss 条数={len(losses)}, "
          f"probs={n_ok}/16, json={n_json}/16")
    sys.exit(0)


if __name__ == "__main__":
    main()
