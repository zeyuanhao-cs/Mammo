#!/bin/bash
# 看护脚本: 等 gpu_hold_h100 (9814169) 变 RUNNING 后自动启动 relation 正式训练 (b)
# nohup 运行于登录节点, 不依赖任何会话。日志: 2-view/logs/watch_hold.log
HOLD_JOB=9814169
BASE=/home/zhao/Mammography/Mammo/2-view
LOG=$BASE/logs/watch_hold.log

echo "$(date '+%F %T') watcher started, waiting for job $HOLD_JOB RUNNING" >> "$LOG"
while true; do
  STATE=$(squeue -j $HOLD_JOB -h -o "%T" 2>/dev/null)
  echo "$(date '+%F %T') state=$STATE" >> "$LOG"
  if [ "$STATE" = "RUNNING" ]; then
    echo "$(date '+%F %T') job RUNNING, launching relation train (b) via srun --overlap" >> "$LOG"
    srun --jobid $HOLD_JOB --overlap bash $BASE/relation_loss/train_b_h100.sh \
      >> $BASE/logs/rel_train_b.log 2>&1
    RC=$?
    echo "$(date '+%F %T') srun finished rc=$RC" >> "$LOG"
    break
  fi
  # 作业消失(被取消/失败)则退出
  if [ -z "$STATE" ]; then
    echo "$(date '+%F %T') job $HOLD_JOB no longer in queue, watcher exits" >> "$LOG"
    break
  fi
  sleep 60
done
