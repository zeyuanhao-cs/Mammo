#!/bin/bash
# 等待内存测试结束 -> 汇总 -> 成功则删除 _memtest_native 并自动启动 ep2 训练链
set -u
PROJ=/home/z/zhao/Mammography/Mammo/2-view
LOG=$PROJ/logs/vlm/memtest_native.log
GPU=$PROJ/logs/vlm/memtest_native_gpu.txt
SUM=$PROJ/logs/vlm/memtest_summary.txt
OUT=/home/z/zhao/Mammography/model/_memtest_native

cd "$PROJ" || exit 1
ts() { date +"%H:%M:%S"; }

while ! grep -q "TRAIN_RC=" "$LOG" 2>/dev/null; do
    if ! pgrep -f run_memtest.sh >/dev/null && ! pgrep -f train_memtest.py >/dev/null; then
        echo "[$(ts)] runner gone without TRAIN_RC" > "$SUM"
        exit 1
    fi
    sleep 20
done

RC=$(grep -o "TRAIN_RC=[0-9]*" "$LOG" | tail -1 | cut -d= -f2)
PEAK_SMI=$(grep -o "PEAK_GPU_SMI_MIB=[0-9]*" "$LOG" | tail -1 | cut -d= -f2)
PEAK_ALLOC=$(grep -o "PEAK_ALLOCATED=[0-9.]*" "$LOG" | tail -1 | cut -d= -f2)
PEAK_RES=$(grep -o "PEAK_RESERVED=[0-9.]*" "$LOG" | tail -1 | cut -d= -f2)
LAST_STEP=$(tr '\r' '\n' < "$LOG" | grep -oE "\{'loss':[^}]*\}" | tail -3)

{
  echo "time=$(ts)"
  echo "TRAIN_RC=$RC"
  echo "PEAK_GPU_SMI_MIB=$PEAK_SMI"
  echo "PEAK_ALLOCATED_GIB=$PEAK_ALLOC"
  echo "PEAK_RESERVED_GIB=$PEAK_RES"
  echo "last_train_logs:"
  echo "$LAST_STEP"
} > "$SUM"

if [ "$RC" = "0" ]; then
    echo "STATUS=PASS" >> "$SUM"
    rm -rf "$OUT"
    echo "removed $OUT: $([ -d "$OUT" ] && echo FAILED || echo ok)" >> "$SUM"
    tmux new-session -d -s ep2 "bash $PROJ/scripts/bench/ep2_chain.sh"
    echo "EP2_STARTED=$(ts)" >> "$SUM"
else
    echo "STATUS=FAIL (OOM or other)" >> "$SUM"
    tr '\r' '\n' < "$LOG" | grep -iE "out of memory|OutOfMemory|OOM" | tail -5 >> "$SUM"
fi
