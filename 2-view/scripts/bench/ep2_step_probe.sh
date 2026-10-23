#!/bin/bash
# 等 ep2 训练到第 20 步，抓取步速与前几个 loss，写入 logs/vlm/ep2_step20.txt
LOG=/home/z/zhao/Mammography/Mammo/2-view/logs/vlm/ep2_chain.log
OUT=/home/z/zhao/Mammography/Mammo/2-view/logs/vlm/ep2_step20.txt
while true; do
  STEP=$(tr '\r' '\n' < "$LOG" 2>/dev/null | grep -oE "[0-9]+/878" | tail -1 | cut -d/ -f1)
  if [ -n "${STEP:-}" ] && [ "$STEP" -ge 20 ] 2>/dev/null; then
    {
      echo "captured_at=$(date +'%F %T')"
      echo "step=$STEP"
      echo "--- tqdm ---"
      tr '\r' '\n' < "$LOG" | grep -oE "[0-9]+/878 \[[0-9:]+<[0-9:]+, *[0-9.]+s/it\]" | tail -4
      echo "--- losses ---"
      tr '\r' '\n' < "$LOG" | grep -oE "\{'loss': '[^']*'[^}]*\}" | head -4
    } > "$OUT"
    break
  fi
  sleep 20
done
