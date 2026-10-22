#!/bin/bash
# 等 e1/e2/e3 三个 b8 评测全部完成，自动生成 outputs/eval/ep3_b8_report.md
set -u
PROJ=/home/z/zhao/Mammography/Mammo/2-view
PY=/home/z/zhao/Mammography/miniconda3/envs/mammo/bin/python
LOG=$PROJ/logs/vlm/ep3_b8.log
RLOG=$PROJ/logs/vlm/ep3_b8_report.log

cd "$PROJ" || exit 1
ts() { date +"%H:%M:%S"; }

echo "[$(ts)] waiter start, watching $LOG" >> "$RLOG"
while ! grep -q "EP3_B8_ALL_DONE" "$LOG" 2>/dev/null; do
    if ! pgrep -f run_ep3_b8.sh >/dev/null; then
        echo "[$(ts)] runner exited without EP3_B8_ALL_DONE" >> "$RLOG"
        break
    fi
    sleep 30
done

echo "[$(ts)] generating report" >> "$RLOG"
"$PY" scripts/bench/ep3_b8_report.py >> "$RLOG" 2>&1
echo "[$(ts)] report rc=$?" >> "$RLOG"
