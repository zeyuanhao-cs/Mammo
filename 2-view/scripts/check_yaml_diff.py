#!/usr/bin/env python3
"""比对两份训练 yaml, 确认差异"只"发生在指定的少数几个键上, 并把完整 diff 写入日志。

用途: 构造单变量对照实验 (如双顺序数据集) 时, 保证新配置除了预期的 key 之外逐字节
      与原配置一致, 避免引入意料外的差异。

用法:
  python3 scripts/check_yaml_diff.py --old 旧.yaml --new 新.yaml \
      --allow dataset,output_dir \
      --expect dataset=vindr_2view_train_dualorder_local \
      --expect output_dir=/home/z/zhao/Mammography/model/Qwen3.5-4B-lora-2view-dualorder \
      --log logs/vlm/dualorder_yaml_diff.log

退出码: 0 = 差异完全在允许范围内且取值符合预期; 1 = 存在额外差异 / 取值不符 / 预期差异缺失
"""
import argparse
import difflib
import sys
from datetime import datetime
from pathlib import Path

_logf = None


def log(msg=""):
    print(msg)
    if _logf:
        _logf.write(msg + "\n")
        _logf.flush()


def key_of(line: str):
    """取 `key: value` 行的 key; 注释/非键值行返回 None。"""
    if not line or line.lstrip().startswith("#") or ":" not in line:
        return None
    k, v = line.split(":", 1)
    k = k.strip()
    if not k or " " in k or "\t" in k:
        return None
    return k, v.split("  #")[0].split("#")[0].strip()


def main():
    global _logf
    p = argparse.ArgumentParser()
    p.add_argument("--old", required=True, help="原 yaml")
    p.add_argument("--new", required=True, help="新 yaml")
    p.add_argument("--allow", required=True, help="允许不同的 key, 逗号分隔")
    p.add_argument("--expect", action="append", default=[],
                   help="key=value, 期望新文件中该 key 的取值 (可重复)")
    p.add_argument("--log", required=True, help="日志路径")
    args = p.parse_args()

    allow = [k.strip() for k in args.allow.split(",") if k.strip()]
    expect = {}
    for e in args.expect:
        k, v = e.split("=", 1)
        expect[k.strip()] = v.strip()

    old_path, new_path, log_path = Path(args.old), Path(args.new), Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _logf = log_path.open("w", encoding="utf-8")

    old_lines = old_path.read_text(encoding="utf-8").splitlines()
    new_lines = new_path.read_text(encoding="utf-8").splitlines()
    fails = []

    def require(cond, msg):
        log("  %s %s" % ("[OK]  " if cond else "[FAIL]", msg))
        if not cond:
            fails.append(msg)

    log("=" * 78)
    log("yaml 差异校验   " + datetime.now().strftime("%F %T"))
    log("=" * 78)
    log("  old: %s" % old_path)
    log("  new: %s" % new_path)
    log("  允许不同的 key: %s" % ", ".join(allow))
    log("  行数: old=%d  new=%d" % (len(old_lines), len(new_lines)))
    require(len(old_lines) == len(new_lines), "两份文件行数相同 (结构未被增删)")

    # ---------------- 完整 diff ----------------
    log("\n[完整 diff]")
    diff = list(difflib.unified_diff(
        old_lines, new_lines,
        fromfile=str(old_path), tofile=str(new_path), lineterm="", n=2))
    if diff:
        for ln in diff:
            log("  " + ln)
    else:
        log("  (无差异)")

    # ---------------- 定位改动 ----------------
    log("\n[逐处改动]")
    changed_keys = []
    sm = difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False)
    n_ops = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        n_ops += 1
        o_seg, n_seg = old_lines[i1:i2], new_lines[j1:j2]
        if tag == "replace" and len(o_seg) == len(n_seg) == 1:
            ok, nv = key_of(o_seg[0]), key_of(n_seg[0])
            if ok and nv and ok[0] == nv[0]:
                k = ok[0]
                changed_keys.append(k)
                line_no = i1 + 1
                log("  行 %d  %s" % (line_no, k))
                log("      旧: %s" % o_seg[0])
                log("      新: %s" % n_seg[0])
                require(k in allow, "键 '%s' 在允许清单内" % k)
                if k in expect:
                    require(nv[1] == expect[k],
                            "键 '%s' 取值 == 期望 '%s' (实际 '%s')" % (k, expect[k], nv[1]))
                continue
        # 结构变化 / 多行替换 / 注释行变化: 一律不允许
        n_ops_display = "行 %d..%d" % (i1 + 1, i2)
        log("  %s  %s" % (n_ops_display, tag))
        for ln in o_seg:
            log("      - %s" % ln)
        for ln in n_seg:
            log("      + %s" % ln)
        require(False, "存在超出允许范围的改动 (%s, 行 %d..%d)" % (tag, i1 + 1, i2))

    # ---------------- 结论 ----------------
    log("\n[结论]")
    require(sorted(set(changed_keys)) == sorted(set(allow)),
            "实际改动集合 == 允许集合 %s (实际 %s)" % (sorted(set(allow)), sorted(set(changed_keys))))
    require(len(changed_keys) == len(set(changed_keys)),
            "每个键只改动一次 (改动 %d 处)" % len(changed_keys))
    require(all(expect.get(k) for k in allow) or not expect,
            "所有允许的键都给出了期望取值并已核对")
    log("  改动处数: %d, 影响键: %s" % (n_ops, ", ".join(sorted(set(changed_keys))) or "(无)"))

    log("\n" + "=" * 78)
    if fails:
        log("yaml 校验失败 %d 项, 不允许开训:" % len(fails))
        for f in fails:
            log("  - " + f)
    else:
        log("yaml 校验通过: 差异仅限 [%s] 共 %d 处, 其余逐字节相同"
            % (", ".join(allow), len(changed_keys)))
    log("日志: %s" % log_path)
    log("=" * 78)
    _logf.close()
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
