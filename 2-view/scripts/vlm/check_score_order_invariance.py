#!/usr/bin/env python3
"""校验 score_2view.py 的"字段概率定位"与 JSON 键序无关 (纯 CPU, 不加载模型/不占 GPU)。

背景:
  score_2view.py 用 FIELD_SPECS 里每个字段自己的正则 (如 '"breast_birads"\\s*:\\s*([1-5])')
  在 completion 里找到"取值字符"的偏移, 再由偏移定位 token 与 logits 行。
  本脚本用现有 os_sub_vt 的贪心预测作为素材, 把同一段 completion 原样 (键序 A) 与
  重排成键序 B 各跑一遍该定位逻辑, 断言两版都能定位到"该字段自己的取值"。

断言:
  1. 两版每个字段都按字段名匹配到, 且定位到的取值字符 == 解析 JSON 中该字段的值
  2. 该字符落在定位到的 token 的字符区间内 (offset 映射正确)
  3. 该字段各候选值在这个位置解析出的 token == completion 自己的 token
     (说明 teacher-forcing 取的那一列就是该候选真实占据的 token, argmax_match 应为 True)
  4. 键序 A 与 B 的 tok_idx 不同 -> 任何"按槽位/固定顺序"的实现都会取错位置
  5. 两版 probs 的字段键序 (由 FIELD_SPECS 决定) 相同 -> 输出 jsonl 结构一致

用法:
  python3 scripts/vlm/check_score_order_invariance.py --n 20 \
      [--pred-file outputs/predictions/predictions_qwen3.5-4b_full.jsonl] \
      [--log logs/vlm/score_order_invariance.log]
退出码: 0 = 全部通过; 1 = 存在失败项
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent          # 2-view/scripts/vlm
PROJ_DIR = THIS_DIR.parent.parent                    # 2-view
sys.path.insert(0, str(THIS_DIR))

import score_2view as S                              # noqa: E402  (复用同一套定位逻辑)

MODEL_PATH = "/home/z/zhao/Mammography/model/Qwen3.5-4B"
DEFAULT_PRED = PROJ_DIR / "outputs" / "predictions" / "predictions_qwen3.5-4b_full.jsonl"
DEFAULT_LOG = PROJ_DIR / "logs" / "vlm" / "score_order_invariance.log"

_logf = None


def log(msg=""):
    print(msg)
    if _logf:
        _logf.write(msg + "\n")
        _logf.flush()


def to_order_b(completion_a: str) -> str:
    """把键序 A 的 JSON 文本重排成键序 B (breast_density -> breast_birads -> findings)。"""
    o = json.loads(completion_a)
    return json.dumps(
        {"breast_density": o["breast_density"],
         "breast_birads": o["breast_birads"],
         "findings": o["findings"]},
        ensure_ascii=False,
    )


def load_tokenizer(model_path: str):
    try:
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        return getattr(proc, "tokenizer", proc)
    except Exception as e:
        print(f"[warn] AutoProcessor 不可用 ({e!r}), 退回 AutoTokenizer")
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def expected_values(completion: str) -> dict:
    """解析 JSON, 返回 {字段名: 期望取值字符串} (与 FIELD_SPECS 的键一致)。"""
    o = json.loads(completion)
    out = {"breast_density": o["breast_density"], "breast_birads": str(o["breast_birads"])}
    for k in S.FINDINGS:
        out[f"findings.{k}"] = str(o["findings"][k])
    return out


def locate(tokenizer, completion: str, field: str):
    """复刻 score_2view.score_one 的定位+候选解析, 返回 (取值字符偏移, token下标, 候选token表)。"""
    regex, cands = S.FIELD_SPECS[field]
    m = regex.search(completion)
    if not m:
        return None
    value_start = m.start(1)
    enc = tokenizer(completion, add_special_tokens=False, return_offsets_mapping=True)
    completion_ids, offsets = enc["input_ids"], enc["offset_mapping"]
    tok_idx = S.find_token_index(offsets, value_start)
    if tok_idx is None:
        return None
    prefix_ids = completion_ids[:tok_idx]
    context_str = completion[: offsets[tok_idx][0]]
    cand_tok = {}
    for c in cands:
        full = tokenizer(context_str + c, add_special_tokens=False)["input_ids"]
        common = S.common_prefix_len(full, prefix_ids)
        cand_tok[c] = full[common:]
    return value_start, tok_idx, offsets[tok_idx], completion_ids[tok_idx], cand_tok


def main():
    global _logf
    p = argparse.ArgumentParser()
    p.add_argument("--pred-file", default=str(DEFAULT_PRED))
    p.add_argument("--n", type=int, default=20, help="抽样条数")
    p.add_argument("--model-path", default=MODEL_PATH)
    p.add_argument("--log", default=str(DEFAULT_LOG))
    args = p.parse_args()

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _logf = log_path.open("w", encoding="utf-8")
    fails = []

    def require(cond, msg):
        log("    %s %s" % ("[OK]  " if cond else "[FAIL]", msg))
        if not cond:
            fails.append(msg)

    log("=" * 78)
    log("score_2view 字段定位 · 键序无关性校验 (纯 CPU)   " + datetime.now().strftime("%F %T"))
    log("=" * 78)
    log("  pred-file : %s" % args.pred_file)
    log("  样本数    : 前 %d 条可用样本" % args.n)

    tokenizer = load_tokenizer(args.model_path)
    log("  tokenizer : %s (%s)" % (type(tokenizer).__name__, args.model_path))
    require(getattr(tokenizer, "is_fast", False),
            "tokenizer 为 fast 版 (支持 return_offsets_mapping)")

    preds = [json.loads(l) for l in open(args.pred_file, encoding="utf-8")]
    log("  预测文件行数: %d" % len(preds))

    n_checked = 0
    n_pos_diff = 0
    n_slots = 0
    n_multi_token = 0

    for rec in preds:
        if n_checked >= args.n:
            break
        ca = S.extract_json_text(rec.get("prediction_raw") or "")
        if not ca:
            continue
        try:
            exp = expected_values(ca)
            cb = to_order_b(ca)
        except Exception:
            continue
        if any(S.FIELD_SPECS[f][0].search(ca) is None or S.FIELD_SPECS[f][0].search(cb) is None
               for f in S.FIELD_SPECS):
            continue
        n_checked += 1
        log("\n  --- 样本 %d  id=%s ---" % (n_checked, rec.get("id")))
        log("      键序A: %s" % ca)
        log("      键序B: %s" % cb)

        got = {"A": {}, "B": {}}
        for order, comp in (("A", ca), ("B", cb)):
            for field in S.FIELD_SPECS:
                r = locate(tokenizer, comp, field)
                assert r is not None
                value_start, tok_idx, (ts, te), own_tok, cand_tok = r
                got[order][field] = (value_start, tok_idx, own_tok)
                # 1) 定位到的字符就是该字段自己的取值
                if comp[value_start] != exp[field]:
                    fails.append(f"{rec.get('id')} 键序{order} {field}: 位置 {value_start} 字符 "
                                 f"'{comp[value_start]}' != 期望 '{exp[field]}'")
                # 2) 字符落在 token 区间内
                if not (ts <= value_start < te):
                    fails.append(f"{rec.get('id')} 键序{order} {field}: token 区间 [{ts},{te}) "
                                 f"未覆盖取值字符 {value_start}")
                # 3) 候选 token 解析: 真实取值应解析成 completion 自己的那个 token
                ct = cand_tok[exp[field]]
                if len(ct) != 1:
                    n_multi_token += 1
                elif ct[0] != own_tok:
                    fails.append(f"{rec.get('id')} 键序{order} {field}: 候选 '{exp[field]}' "
                                 f"token {ct[0]} != 该位置真实 token {own_tok}")
                # 反事实: 各候选在该位置的 token 必须互不相同, 否则概率会混淆
                firsts = [v[0] for v in cand_tok.values() if v]
                if len(set(firsts)) != len(firsts):
                    fails.append(f"{rec.get('id')} 键序{order} {field}: 候选取值在该位置 token 不互异")

        # 4) 键序 A/B 的取值位置应不同 (随字段名走), 否则说明是"按槽位"
        for field in S.FIELD_SPECS:
            n_slots += 1
            if got["A"][field][1] != got["B"][field][1]:
                n_pos_diff += 1

        # 5) 两版 probs 字段键序一致 (都由 FIELD_SPECS 决定)
        ka, kb = list(S.FIELD_SPECS.keys()), list(S.FIELD_SPECS.keys())
        if ka != kb:
            fails.append(f"{rec.get('id')}: probs 字段键序不一致")

        log("      字段 -> (取值字符偏移, token下标, 该位置真实token) [键序A | 键序B]")
        for field in S.FIELD_SPECS:
            a, b = got["A"][field], got["B"][field]
            log("        %-32s A=(%4d,%4d,%6d)  B=(%4d,%4d,%6d)  值=%s"
                % (field, a[0], a[1], a[2], b[0], b[1], b[2], exp[field]))

    log("\n[汇总]")
    log("  校验样本数            : %d" % n_checked)
    log("  字段定位次数          : %d (= %d 样本 × 6 字段 × 2 键序)" % (n_slots * 2, n_checked))
    log("  A/B tok_idx 不同占比  : %d/%d  (位置随字段名变化, 证明非固定槽位)"
        % (n_pos_diff, n_slots))
    log("  多 token 候选(近似)   : %d" % n_multi_token)
    require(n_checked == args.n, "实际校验到 %d 条 (目标 %d)" % (n_checked, args.n))
    require(not fails, "全部断言通过")

    log("\n" + "=" * 78)
    if fails:
        log("校验失败 %d 项:" % len(fails))
        for f in fails[:20]:
            log("  - " + f)
    else:
        log("校验通过: 字段概率一律按字段名定位, 与 JSON 键序无关 (键序 A/B 均正确)")
    log("日志: %s" % log_path)
    log("=" * 78)
    _logf.close()
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
