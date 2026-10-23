#!/bin/bash
# 软证据探针串行执行：等 val 的 b8 产物齐全 -> smoke -> test -> val(含敏感性) -> 报告
set -u
PROJ=/home/z/zhao/Mammography/Mammo/2-view
CV1=$PROJ/chain_v1
PY=/home/z/zhao/Mammography/miniconda3/envs/mammo/bin/python
AD=/home/z/zhao/Mammography/model/Qwen3.5-4B-lora-2view-os-sub-vt
LOG=$CV1/logs/soft_evidence.log
TEST_PROBS=$PROJ/outputs/scoring/os_sub_vt_test_b8_with_probs.jsonl
VAL_PROBS=$PROJ/outputs/scoring/os_sub_vt_val_b8_with_probs.jsonl

cd "$PROJ" || exit 1
mkdir -p "$CV1/outputs" "$CV1/logs"
ts() { date +"%H:%M:%S"; }
say() { echo "[$(ts)] $*" | tee -a "$LOG"; }

say "等待 val b8 产物 1600 条"
while [ "$(wc -l < "$VAL_PROBS" 2>/dev/null || echo 0)" -lt 1600 ]; do
  if ! tmux has-session -t oswt_val 2>/dev/null && pgrep -f "os_sub_vt_val_b8" >/dev/null; then :; fi
  sleep 60
done
say "val b8 就绪"

say "smoke（8 条，含 4 条 hard 自检）"
"$PY" "$CV1/scripts/soft_evidence_probe.py" --adapter "$AD" \
  --data "$PROJ/data/splits/test_local.json" --probs "$TEST_PROBS" \
  --out "$CV1/outputs/_smoke.jsonl" --split smoke --limit 8 --print-context 3 2>&1 | tail -60 | tee -a "$LOG"
SMOKE_N=$(wc -l < "$CV1/outputs/_smoke.jsonl" 2>/dev/null || echo 0)
if [ "$SMOKE_N" -lt 8 ]; then say "smoke 失败（只出 $SMOKE_N 条），停止"; exit 1; fi

say "test 全量（2000，含敏感性 20×11）"
"$PY" "$CV1/scripts/soft_evidence_probe.py" --adapter "$AD" \
  --data "$PROJ/data/splits/test_local.json" --probs "$TEST_PROBS" \
  --out "$CV1/outputs/soft_evidence_test.jsonl" --split test --limit 0 \
  --sens-out "$CV1/outputs/soft_evidence_sensitivity.json" --sens-n 20 2>&1 | tail -4 | tee -a "$LOG"

say "val 全量（1600）"
"$PY" "$CV1/scripts/soft_evidence_probe.py" --adapter "$AD" \
  --data "$PROJ/data/splits/val_raw_local.json" --probs "$VAL_PROBS" \
  --out "$CV1/outputs/soft_evidence_val.jsonl" --split val --limit 0 2>&1 | tail -4 | tee -a "$LOG"

say "生成报告"
"$PY" "$CV1/scripts/soft_evidence_report.py" 2>&1 | tail -5 | tee -a "$LOG"
say "SOFT_EVIDENCE_ALL_DONE"
