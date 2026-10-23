#!/usr/bin/env python3
"""单元测试 4：soft-decode 在 p=0 / p=1 时与普通贪心解码逐 token 一致（CPU + 桩模型）。

桩模型按脚本吐 token，并记录每步被喂进去的输入（token id 或嵌入）。
令牌文本拼成 `"mass": 0` 的形式，用来触发 soft 替换。
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

CV1 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CV1.parent))

from chain_v1.infer.soft_decode import completed_finding_field, soft_decode  # noqa: E402

ID0, ID1 = 15, 16
PIECES = {1: '"', 2: "mass", 3: '":', 4: " ", 5: "0", 6: "1", 7: ",", 8: "}", 9: "x"}
SCRIPT = [1, 2, 3, 4, 5, 8]      # -> '"mass": 0}' ，第 5 个 token 是取值 '0'


def fake_decode(ids):
    return "".join(PIECES[i] for i in ids)


class ScriptedModel(nn.Module):
    """忽略输入内容，按脚本输出 logits；记录被喂入的输入。"""

    def __init__(self, script, mode):
        super().__init__()
        self.emb = nn.Embedding(32, 8)
        self.script = script
        self.mode = mode          # 'p0' | 'p1' | 'plain'
        self.calls = 0
        self.fed = []             # [('ids', tok) | ('emb', tensor)]

    def get_input_embeddings(self):
        return self.emb

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None,
                past_key_values=None, use_cache=False, **kw):
        if input_ids is not None:
            self.fed.append(("ids", int(input_ids[0, 0])))
        else:
            self.fed.append(("emb", inputs_embeds[0, 0].detach().clone()))
        tok = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        logits = torch.full((1, 1, 32), -20.0)
        logits[0, 0, tok] = 20.0
        if self.mode == "p0":
            logits[0, 0, ID0] = 15.0
            logits[0, 0, ID1] = -15.0
        elif self.mode == "p1":
            logits[0, 0, ID0] = -15.0
            logits[0, 0, ID1] = 15.0
        return SimpleNamespace(logits=logits, past_key_values="PAST")


def run(mode, soft):
    m = ScriptedModel(SCRIPT, mode)
    out = soft_decode(m, tokenizer=None, text_of=fake_decode,
                      prompt_ids=torch.zeros((1, 3), dtype=torch.long),
                      mm_kwargs={}, max_new_tokens=len(SCRIPT), soft=soft, soft_mode="linear")
    return m, out


def test_soft_decode_p01_equals_greedy():
    assert completed_finding_field('"mass": 0', ["mass"]) == "mass"
    assert completed_finding_field('"mass": 0,', ["mass"]) is None  # 已不是结尾

    # 普通贪心
    m_g, out_g = run("plain", soft=False)
    base = out_g["tokens"]
    assert base == SCRIPT, base
    assert all(k == "ids" for k, _ in m_g.fed), "无 soft 时应始终喂 token id"

    for mode in ("p0", "p1"):
        m, out = run(mode, soft=True)
        assert out["tokens"] == base, f"{mode}: 输出 token 与贪心不一致 {out['tokens']}"
        # 第 5 个 token（取值 '0'）之后那次喂入应是软嵌入
        k = 5
        assert m.fed[k][0] == "emb", f"{mode}: 第 {k} 次喂入应为嵌入，实际 {m.fed[k][0]}"
        emb = m.fed[k][1]
        target = m.emb.weight[ID0].detach() if mode == "p0" else m.emb.weight[ID1].detach()
        diff = float((emb - target).abs().max())
        assert diff < 1e-4, f"{mode}: 软嵌入与 E[{ID0 if mode=='p0' else ID1}] 偏差 {diff}"
        print(f"  {mode}: tokens 与贪心一致, 替换处 |emb-E|={diff:.2e}, events={out['soft_events']}")

    print("test_soft_decode_p01_equals_greedy: OK")


if __name__ == "__main__":
    test_soft_decode_p01_equals_greedy()
