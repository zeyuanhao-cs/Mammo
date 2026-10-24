"""对比 position_ids 修复前后 soft_evidence_probe 的新旧结果，写 pilot/probe_compare.md。

旧: chain_v1/outputs/soft_evidence_{test,val}.jsonl + soft_evidence_sensitivity.json
新: 同名 *_fixed.jsonl + soft_evidence_sensitivity_fixed.json
"""
import json
import statistics as st
from pathlib import Path

P = Path("/home/z/zhao/Mammography/Mammo/2-view/chain_v1/outputs")
OUT = P / "pilot" / "probe_compare.md"


def load_jsonl(p):
    p = Path(p)
    if not p.exists():
        return {}
    d = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        d[r["id"]] = r
    return d


def selfcheck(p):
    fp = Path(str(p).replace(".jsonl", "_selfcheck.json"))
    if fp.exists():
        return json.loads(fp.read_text(encoding="utf-8"))
    return {}


L = ["# soft_evidence_probe：position_ids 修复前(旧) vs 修复后(新)",
     "",
     "修复内容: forward_birads prefill 显式传 text arange + get_rope_index 视觉位置，",
     "与 b8(generate) 管线 rope 一致（同 soft_decode.rope_prefill）。"]

for split in ["test", "val"]:
    old = load_jsonl(P / f"soft_evidence_{split}.jsonl")
    new = load_jsonl(P / f"soft_evidence_{split}_fixed.jsonl")
    L += ["", f"## split={split}（旧 {len(old)} 条，新 {len(new)} 条）"]
    common = sorted(old.keys() & new.keys())
    if common:
        l1, ch_o, ch_n, ac_o, ac_n = [], 0, 0, 0, 0
        for i in common:
            po, pn = old[i].get("p_findings") or {}, new[i].get("p_findings") or {}
            ks = set(po) | set(pn)
            if ks:
                l1.append(sum(abs(float(po.get(k, 0)) - float(pn.get(k, 0))) for k in ks))
            bo, bn = str(old[i].get("b8_birads")), str(old[i].get("gt_birads"))
            ch_o += int(str(old[i].get("soft_birads")) == bo)
            ch_n += int(str(new[i].get("soft_birads")) == bo)
            ac_o += int(str(old[i].get("soft_birads")) == bn)
            ac_n += int(str(new[i].get("soft_birads")) == bn)
        n = len(common)
        L += [
            f"- 共同 id: {n}",
            (f"- p_findings 新旧 L1 差: mean={st.mean(l1):.5f} max={max(l1):.5f}" if l1
             else "- p_findings: 无可比字段"),
            f"- soft vs b8 argmax 一致率: 旧 {ch_o / n:.4f} -> 新 {ch_n / n:.4f}",
            f"- soft vs GT 一致率:        旧 {ac_o / n:.4f} -> 新 {ac_n / n:.4f}",
        ]
    else:
        L.append("- 无共同 id（新结果缺失或 id 集不同）")
    so, sn = selfcheck(P / f"soft_evidence_{split}.jsonl"), selfcheck(P / f"soft_evidence_{split}_fixed.jsonl")
    if so and sn:
        L.append(f"- 自检(hard vs b8): 旧 argmax一致 {so['argmax_agree']:.4f} / MAD {so['mean_abs_diff']:.6f}"
                 f" -> 新 {sn['argmax_agree']:.4f} / {sn['mean_abs_diff']:.6f}")

so_p, sn_p = P / "soft_evidence_sensitivity.json", P / "soft_evidence_sensitivity_fixed.json"
if so_p.exists() and sn_p.exists():
    so = json.loads(so_p.read_text(encoding="utf-8"))
    sn = json.loads(sn_p.read_text(encoding="utf-8"))
    ro = {r["id"]: r["p_ge3"] for r in so["rows"]}
    rn = {r["id"]: r["p_ge3"] for r in sn["rows"]}
    common = sorted(ro.keys() & rn.keys())
    L += ["", "## 敏感性扫描（mass p 0→1，p_ge3 曲线）"]
    if common:
        diffs = [max(abs(a - b) for a, b in zip(ro[i], rn[i])) for i in common]
        L.append(f"- 共同 id: {len(common)}; 曲线最大差 mean={st.mean(diffs):.5f} max={max(diffs):.5f}")
    else:
        L.append("- 无共同 id")
else:
    L += ["", "## 敏感性扫描：新旧文件不全，跳过"]

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
print("[ok]", OUT)
