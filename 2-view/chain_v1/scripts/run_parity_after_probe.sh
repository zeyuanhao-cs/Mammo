#!/bin/bash
# 等软证据探针跑完（GPU 释放）后，执行前向一致性测试
set -u
PROJ=/home/z/zhao/Mammography/Mammo/2-view
CV1=$PROJ/chain_v1
PY=/home/z/zhao/Mammography/miniconda3/envs/mammo/bin/python
LOG=$CV1/logs/soft_evidence.log
PLOG=$CV1/logs/forward_parity.log

cd "$PROJ" || exit 1
echo "[$(date +%H:%M:%S)] 等待探针结束" >> "$PLOG"
while ! grep -q "SOFT_EVIDENCE_ALL_DONE" "$LOG" 2>/dev/null; do
  sleep 60
done
echo "[$(date +%H:%M:%S)] 探针结束，开始前向一致性测试" >> "$PLOG"
"$PY" "$CV1/tests/test_forward_parity.py" >> "$PLOG" 2>&1
echo "[$(date +%H:%M:%S)] parity rc=$?" >> "$PLOG"
