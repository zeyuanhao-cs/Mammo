#!/usr/bin/env bash
# C3m 训练结束后：诊断（视角 A/M）→ hard 评测 → soft 评测 → 报告
set -u
PY=/home/z/zhao/Mammography/miniconda3/envs/mammo/bin/python
PROJ=/home/z/zhao/Mammography/Mammo/2-view
CV1=$PROJ/chain_v1
OUTD=$CV1/outputs/pilot
LOGD=$CV1/logs
MDIR=/home/z/zhao/Mammography/model
M=$MDIR/chain_v1_C3m
STATUS=$CV1/outputs/pilot_status.md
say() { echo "[$(date '+%F %T')] $*" >> "$STATUS"; }
die() { say "STOP: $*"; exit 1; }

# 1) 等训练进程退出
while pgrep -f "output-dir $M" > /dev/null; do sleep 30; done
[ -f "$M/adapter_model.safetensors" ] || die "C3m 训练结束但缺 adapter"
say "C3m 训练结束，开始视角 A/M 诊断（只读）"

# 2) 诊断
UNK=""
if [ -f "$M/soft_unknown.pt" ]; then UNK="--unknown $M/soft_unknown.pt"; fi
"$PY" "$CV1/scripts/c3m_teacher_diag.py" --adapter "$M" $UNK \
  > "$LOGD/c3m_teacher_diag.log" 2>&1 || { say "诊断失败（跳过，继续评测）"; tail -5 "$LOGD/c3m_teacher_diag.log" | cut -c1-200 >> "$STATUS"; }
[ -f "$OUTD/c3m_teacher_diag.md" ] && say "诊断完成：outputs/pilot/c3m_teacher_diag.md"

# 3) hard 评测
if [ "$(wc -l < "$OUTD/c3m_test_hard.jsonl" 2>/dev/null || echo 0)" -lt 2000 ]; then
  say "[C3m] hard 评测 2000"
  "$PY" "$CV1/infer/eval_soft.py" --adapter "$M" --mode hard \
    --data "$PROJ/data/splits/test_local.json" --out "$OUTD/c3m_test_hard.jsonl" \
    > "$LOGD/eval_c3m-test-hard.log" 2>&1 || die "C3m hard 评测失败"
fi

# 4) soft 评测
if [ "$(wc -l < "$OUTD/c3m_test_soft.jsonl" 2>/dev/null || echo 0)" -lt 2000 ]; then
  say "[C3m] soft 评测 2000（linear）"
  "$PY" "$CV1/infer/eval_soft.py" --adapter "$M" --soft-mode linear --mode soft \
    --data "$PROJ/data/splits/test_local.json" --out "$OUTD/c3m_test_soft.jsonl" \
    > "$LOGD/eval_c3m-test-soft.log" 2>&1 || die "C3m soft 评测失败"
fi

# 5) 报告
"$PY" "$CV1/scripts/c3_branch_report.py" > "$LOGD/c3_report_c3m.log" 2>&1 \
  || die "报告生成失败"
say "C3m 全流程完成：outputs/c3_report.md"
