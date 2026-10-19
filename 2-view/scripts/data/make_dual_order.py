#!/usr/bin/env python3
"""基于 os_sub 训练集 (7,019 行) 构建"双顺序"数据集。

设计:
  - 每行样本按 seed 42 独立随机分到顺序 A 或顺序 B
      顺序 A: breast_density -> findings -> breast_birads
      顺序 B: breast_density -> breast_birads -> findings
  - 标签值一律不变; 只改 JSON 字段顺序 (训练目标的键序决定模型生成顺序)
  - 每行用对应顺序的提示词 dataset_build/prompt_order_{a,b}.txt
    两个提示词相对原 prompt.txt 只有三处差异:
      1) "Consider ... then findings, then BI-RADS" 在 B 中改为与输出顺序一致
      2) 删掉两条 Consistency 约束
      3) "Use the full range of options..." 移到单独的 Guidance 段下
      另外新增一行输出顺序说明 + JSON 键序随顺序调整

自查 (全部写入 logs/dualorder_build.log):
  1. A / B 比例接近 50/50
  2. 两个提示词的差异位置与预期逐条吻合
  3. 写盘后回读全量校验: JSON 合法 / 标签值与原数据一致 / 键序与提示词顺序一致
  4. 随机抽 3 条 A + 3 条 B 逐条展示比对

输出:
  - data/llamafactory/train_llamafactory_dualorder_local.json
  - data/dataset_info.json += vindr_2view_train_dualorder_local
"""
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

SEED = 42
FINDINGS = ["mass", "suspicious_calcification", "asymmetry", "architectural_distortion"]
ORDER_A = ["breast_density", "findings", "breast_birads"]
ORDER_B = ["breast_density", "breast_birads", "findings"]

THIS_DIR = Path(__file__).resolve().parent
PROJ_DIR = THIS_DIR.parent.parent                      # 2-view
DATA_DIR = PROJ_DIR / "data"
RAW = DATA_DIR / "splits" / "train_sub_repeat_os_local.json"
ALPACA_OLD = DATA_DIR / "llamafactory" / "train_llamafactory_os_sub_local.json"
DST = DATA_DIR / "llamafactory" / "train_llamafactory_dualorder_local.json"
INFO = DATA_DIR / "dataset_info.json"
PROMPT_ORIG = PROJ_DIR / "dataset_build" / "prompt.txt"
PROMPT_A = PROJ_DIR / "dataset_build" / "prompt_order_a.txt"
PROMPT_B = PROJ_DIR / "dataset_build" / "prompt_order_b.txt"
LOG = PROJ_DIR / "logs" / "dualorder_build.log"

_logf = None


def log(msg=""):
    print(msg)
    if _logf:
        _logf.write(msg + "\n")
        _logf.flush()


def load_instruction(path: Path) -> str:
    """与 make_repeat_os_sub.py / infer_2view_local.py 同一套 <image> 规范化。"""
    text = path.read_text(encoding="utf-8").strip()
    n = text.count("<image>")
    if n == 1:
        return text.replace("<image>", "<image>\n<image>", 1)
    assert n == 2, f"{path}: 需要 1 或 2 个 <image>, 实际 {n}"
    return text


def render(gt: dict, order: str) -> str:
    """按指定顺序渲染训练目标 JSON (键序即生成顺序)。"""
    f = {k: int(gt["findings"][k]) for k in FINDINGS}
    if order == "A":
        obj = {"breast_density": gt["breast_density"],
               "findings": f,
               "breast_birads": int(gt["breast_birads"])}
    else:
        obj = {"breast_density": gt["breast_density"],
               "breast_birads": int(gt["breast_birads"]),
               "findings": f}
    return json.dumps(obj, ensure_ascii=False)


def keys_in_order(s: str, order: list) -> bool:
    pos = [s.index(k) for k in order]
    return pos == sorted(pos)


def main():
    global _logf
    LOG.parent.mkdir(parents=True, exist_ok=True)
    _logf = LOG.open("w", encoding="utf-8")
    fails = []

    def require(cond, msg):
        log("    %s %s" % ("[OK]  " if cond else "[FAIL]", msg))
        if not cond:
            fails.append(msg)

    log("=" * 80)
    log("双顺序数据集构建 + 自查      seed=%d      时间 %s"
        % (SEED, __import__("datetime").datetime.now().strftime("%F %T")))
    log("=" * 80)

    raw = json.loads(RAW.read_text(encoding="utf-8"))
    old = json.loads(ALPACA_OLD.read_text(encoding="utf-8"))
    log("\n[输入]")
    log("  %s  %d 行" % (RAW.relative_to(PROJ_DIR), len(raw)))
    log("  %s  %d 行 (交叉校验用)" % (ALPACA_OLD.relative_to(PROJ_DIR), len(old)))
    assert len(raw) == len(old) == 7019, "输入行数不是 7019"

    # ---------- 提示词自查 ----------
    log("\n[自查 0] 两个提示词相对原 prompt.txt 的差异")
    o_inst = PROMPT_ORIG.read_text(encoding="utf-8").strip()
    inst_a = load_instruction(PROMPT_A)
    inst_b = load_instruction(PROMPT_B)
    log("  prompt.txt         %d 字符" % len(o_inst))
    log("  prompt_order_a.txt %d 字符" % len(inst_a))
    log("  prompt_order_b.txt %d 字符" % len(inst_b))
    require("Consistency" not in inst_a and "Consistency" not in inst_b,
            "两个提示词都不再含 'Consistency' 段")
    require("If any finding is 1" not in inst_a and "If any finding is 1" not in inst_b,
            "两条一致性约束已删除")
    require("Use the full range of options" in inst_a and
            "Use the full range of options" in inst_b,
            "'Use the full range of options...' 保留")
    require("Guidance" in inst_a and "Guidance" in inst_b, "该句已置于 Guidance 段下")
    require(inst_a.count("<image>") == 2 and inst_b.count("<image>") == 2,
            "两个提示词都恰好 2 个 <image> 占位符")
    require(inst_a != inst_b, "两个提示词不相同")
    require("breast_density, findings, breast_birads" in inst_a and
            "breast_density, breast_birads, findings" in inst_b,
            "输出顺序说明分别为 A/B 顺序")
    # 原提示词里的 "Consider ..." 句
    CONSIDER_ORIG = "then findings, then the overall BI-RADS category."
    CONSIDER_B = "then the overall BI-RADS category, then findings."
    require(CONSIDER_ORIG in o_inst, "原 prompt.txt 含 'then findings, then ... BI-RADS' 句")
    require(CONSIDER_ORIG in inst_a,
            "'Consider ...' 句在 A 中与原提示词逐字相同")
    require(CONSIDER_B in inst_b and CONSIDER_ORIG not in inst_b,
            "'Consider ...' 句在 B 中已改为与输出顺序一致")
    require(o_inst.count("Consistency") == 1 and
            "If any finding is 1" in o_inst and "If all findings are 0" in o_inst,
            "原 prompt.txt 确有 Consistency 段与两条约束")
    require("Guidance" not in o_inst, "原 prompt.txt 无 Guidance 段 (该段为新增)")
    require("Report the fields in exactly this order" not in o_inst,
            "原 prompt.txt 无 'Report the fields ...' 行 (该行为新增)")
    # 除以下几处外,两版必须逐字相同: ①Consider 句 ②'Report the fields ...' 行 ③JSON 模板
    la, lb = inst_a.splitlines(), inst_b.splitlines()
    require(len(la) == len(lb), "两版行数相同 (%d vs %d)" % (len(la), len(lb)))
    diff = [i for i, (x, y) in enumerate(zip(la, lb)) if x != y]
    i_consider = next(i for i, ln in enumerate(la) if CONSIDER_ORIG in ln)
    i_report = next(i for i, ln in enumerate(la)
                    if ln.startswith("Report the fields in exactly this order"))
    tmpl0 = next(i for i, ln in enumerate(la) if ln.startswith('{"breast_density"'))
    allowed = {i_consider, i_report} | set(range(tmpl0, len(la)))
    must = {i_consider, i_report, len(la) - 1}
    require(set(diff) <= allowed,
            "两版差异未超出允许范围(仅 Consider 句/输出顺序行/JSON 模板), 实际差异行 %s"
            % [i + 1 for i in diff])
    require(must <= set(diff),
            "必改之处确实不同: Consider 句 / 输出顺序行 / 模板末行 (差异行 %s)"
            % [i + 1 for i in diff])
    # JSON 模板: 字段与取值占位符集合相同, 只是排列顺序不同
    TOK = r'"(?:breast_density|breast_birads|findings|mass|suspicious_calcification|asymmetry|architectural_distortion)"|<A\|B\|C\|D>\|<1\|2\|3\|4\|5>\|<0\|1>'
    ta = re.findall(TOK, inst_a[inst_a.index('{"breast_density"'):])
    tb = re.findall(TOK, inst_b[inst_b.index('{"breast_density"'):])
    require(sorted(ta) == sorted(tb) and ta != tb,
            "JSON 模板字段集合相同、仅顺序不同 (A=%d tokens, B=%d tokens)" % (len(ta), len(tb)))
    # Definitions 段必须逐字一致 (且与原提示词一致)
    def seg(s, a, b):
        return s[s.index(a):s.index(b)]
    require(seg(inst_a, "Definitions", "Guidance") == seg(inst_b, "Definitions", "Guidance"),
            "Definitions 段两版逐字相同")
    require(seg(inst_a, "Definitions", "Guidance") == seg(o_inst, "Definitions", "Consistency"),
            "Definitions 段与原提示词逐字相同")
    g_a = seg(inst_a, "Guidance", "Return ONLY")
    g_b = seg(inst_b, "Guidance", "Return ONLY")
    USE_RANGE = ("- Use the full range of options. In a screening population most breasts\n"
                 "  are BI-RADS 1; do not default to the middle category.")
    require(g_a == g_b, "Guidance 段两版逐字相同")
    require(USE_RANGE in o_inst, "原提示词含 'Use the full range ...' 两行")
    require(g_a.strip() == "Guidance\n" + USE_RANGE,
            "Guidance 段 == 新增标题 + 原文那两行 (内容整体搬移, 未改写)")
    # A 版 JSON 模板必须与原提示词完全一致 (这是 A 行能逐字节复现的前提)
    require(inst_a[inst_a.index('{"breast_density"'):] ==
            o_inst[o_inst.index('{"breast_density"'):],
            "A 版 JSON 模板与原提示词逐字相同 (键序 A)")

    # ---------- 分配 ----------
    rng = random.Random(SEED)
    orders = ["A" if rng.random() < 0.5 else "B" for _ in raw]
    cnt = Counter(orders)
    n = len(orders)
    log("\n[自查 1] A/B 分配比例 (逐行独立随机, seed=%d)" % SEED)
    log("    A = %d  (%.2f%%)" % (cnt["A"], 100 * cnt["A"] / n))
    log("    B = %d  (%.2f%%)" % (cnt["B"], 100 * cnt["B"] / n))
    log("    偏离 50/50 = %.2f 个百分点" % abs(100 * cnt["A"] / n - 50))
    require(abs(cnt["A"] / n - 0.5) <= 0.03, "A/B 比例接近 50/50 (容差 3%)")
    by_group = defaultdict(set)
    for r, od in zip(raw, orders):
        by_group[r["id"]].add(od)
    mixed = sum(1 for v in by_group.values() if len(v) > 1)
    log("    (参考) repeat-OS 重复组 %d 个, 其中 %d 个组内被分到不同顺序"
        % (len(by_group), mixed))

    # ---------- 写盘 ----------
    dual = [{"instruction": (inst_a if od == "A" else inst_b),
             "input": "",
             "output": render(r["output"], od),
             "images": list(r["images"])}
            for r, od in zip(raw, orders)]
    DST.write_text(json.dumps(dual, ensure_ascii=False), encoding="utf-8")
    log("\n[输出] %s  (%d 行, %.1f MB)"
        % (DST.relative_to(PROJ_DIR), len(dual), DST.stat().st_size / 1e6))

    # ---------- 回读全量校验 ----------
    log("\n[自查 2] 写盘后回读全量校验 (%d 行)" % len(dual))
    back = json.loads(DST.read_text(encoding="utf-8"))
    require(len(back) == n, "回读行数 == %d" % n)
    n_json = n_val = n_key = n_inst = n_byteA = n_imgs = 0
    for i, (b, r, o, od) in enumerate(zip(back, raw, old, orders)):
        try:
            parsed = json.loads(b["output"])
        except Exception:
            n_json += 1
            continue
        if parsed != json.loads(o["output"]):
            n_val += 1
        if not keys_in_order(b["output"], ORDER_A if od == "A" else ORDER_B):
            n_key += 1
        if b["instruction"] != (inst_a if od == "A" else inst_b):
            n_inst += 1
        if od == "A" and b["output"] != o["output"]:
            n_byteA += 1
        if list(b["images"]) != list(r["images"]) or list(b["images"]) != list(o["images"]):
            n_imgs += 1
    require(n_json == 0, "全部行 JSON 合法 (非法 %d)" % n_json)
    require(n_val == 0, "全部行标签值与原数据一致 (不一致 %d)" % n_val)
    require(n_key == 0, "全部行键序与提示词顺序一致 (不符 %d)" % n_key)
    require(n_inst == 0, "全部行 instruction 与提示词文件逐字一致 (异常 %d)" % n_inst)
    require(n_byteA == 0, "A 行的 output 与原 os_sub 训练目标逐字节相同 (不符 %d)" % n_byteA)
    require(n_imgs == 0, "全部行 images 与原始数据一致 (不符 %d)" % n_imgs)

    # ---------- 抽样展示 ----------
    log("\n[自查 3] 随机抽 3 条 A + 3 条 B 逐条比对")
    rng2 = random.Random(SEED)
    idx_a = [i for i, x in enumerate(orders) if x == "A"]
    idx_b = [i for i, x in enumerate(orders) if x == "B"]
    picks = [(i, "A") for i in rng2.sample(idx_a, 3)] + \
            [(i, "B") for i in rng2.sample(idx_b, 3)]
    for i, od in picks:
        r, o, b = raw[i], old[i], back[i]
        same_vals = json.loads(b["output"]) == json.loads(o["output"])
        same_canon = json.dumps(json.loads(b["output"]), sort_keys=True) == \
                     json.dumps(json.loads(o["output"]), sort_keys=True)
        ok = (same_vals and same_canon
              and keys_in_order(b["output"], ORDER_A if od == "A" else ORDER_B)
              and b["instruction"] == (inst_a if od == "A" else inst_b))
        log("  --- 行 %d   顺序 %s   id=%s ---" % (i, od, r["id"]))
        log("      原 output (顺序A) : %s" % o["output"])
        log("      新 output (顺序%s) : %s" % (od, b["output"]))
        log("      标签值与原数据一致 : %s" % same_vals)
        log("      规范化后完全相同   : %s   (即仅字段顺序不同)" % same_canon)
        log("      字符串与顺序A版不同 : %s" % (b["output"] != o["output"]))
        log("      JSON 合法          : True")
        log("      键序 == 提示词顺序 : %s" % keys_in_order(
            b["output"], ORDER_A if od == "A" else ORDER_B))
        log("      提示词             : prompt_order_%s.txt" % od.lower())
        if not ok:
            fails.append("抽样行 %d (顺序 %s) 校验失败" % (i, od))

    # ---------- 注册 ----------
    info = json.loads(INFO.read_text(encoding="utf-8"))
    info["vindr_2view_train_dualorder_local"] = {
        "file_name": "llamafactory/%s" % DST.name,
        "formatting": "alpaca",
        "columns": {"prompt": "instruction", "query": "input",
                    "response": "output", "images": "images"},
    }
    INFO.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    log("\n[注册] dataset_info.json += vindr_2view_train_dualorder_local")
    require("vindr_2view_train_dualorder_local" in
            json.loads(INFO.read_text(encoding="utf-8")), "注册成功")

    # ---------- 标签分布不变 ----------
    log("\n[参考] 标签分布应与 os_sub 完全一致 (只是重排键序)")
    b_old = Counter(json.loads(x["output"])["breast_birads"] for x in old)
    b_new = Counter(json.loads(x["output"])["breast_birads"] for x in back)
    require(b_old == b_new, "BI-RADS 分布未变")
    for k in FINDINGS:
        o_ = sum(1 for x in old if json.loads(x["output"])["findings"][k] == 1)
        n_ = sum(1 for x in back if json.loads(x["output"])["findings"][k] == 1)
        require(o_ == n_, "findings.%-26s 阳性 %d -> %d" % (k, o_, n_))

    log("\n" + "=" * 80)
    if fails:
        log("自查失败 %d 项, 不允许开训:" % len(fails))
        for f in fails:
            log("  - " + f)
    else:
        log("全部自查通过 -> 可以开训")
    log("=" * 80)
    _logf.close()
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
