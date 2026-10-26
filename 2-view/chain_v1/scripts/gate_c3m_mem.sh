#!/usr/bin/env bash
# 显存闸门：C3full 软推理结束后，先用最长 2 样本测 C3m 峰值显存；>21GB 停下不开训
set -u
PY=/home/z/zhao/Mammography/miniconda3/envs/mammo/bin/python
PROJ=/home/z/zhao/Mammography/Mammo/2-view
CV1=$PROJ/chain_v1
LOGD=$CV1/logs
STATUS=$CV1/outputs/pilot_status.md
say() { echo "[$(date '+%F %T')] $*" >> "$STATUS"; }

while ! grep -q ALL_DONE /tmp/c3soft_run.log 2>/dev/null; do sleep 20; done
say "显存闸门：C3full 软推理已结束，先测 C3m 峰值显存（不启动训练）"

# 停掉在等的 C3m 驱动（只是 sleep 等待，无损失），避免它抢先占卡
pkill -f "run_c3m.sh" 2>/dev/null
sleep 2
pkill -f "eval_soft.py" 2>/dev/null
sleep 3
for _ in $(seq 1 20); do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  [ "${free:-0}" -ge 20480 ] && break
  sleep 30
done
say "显存闸门：GPU 空闲 ${free:-?} MiB，开始 1 步压力测试"

rm -rf /home/z/zhao/Mammography/model/_c3m_memtest
"$PY" /tmp/memtest_mutual.py > "$LOGD/memtest_c3m.log" 2>&1
rc=$?
peak=$(grep -oE "PEAK_ALLOCATED=[0-9.]+" "$LOGD/memtest_c3m.log" | tail -1 | cut -d= -f2)
say "显存闸门：rc=$rc peak_allocated=${peak:-?} GiB（阈值 21）"

if [ $rc -ne 0 ] || [ -z "${peak:-}" ]; then
  say "STOP: 显存压力测试失败（见 logs/memtest_c3m.log），不开训"
  tail -6 "$LOGD/memtest_c3m.log" | cut -c1-300 >> "$STATUS"
  exit 1
fi
if "$PY" -c "import sys;sys.exit(0 if float('$peak')>21.0 else 1)"; then
  say "STOP: C3m 峰值显存 ${peak} GiB > 21 GiB，按约定不开训，等用户"
  exit 0
fi
say "显存闸门通过（${peak} GiB ≤ 21 GiB），重启 C3m 驱动"
nohup bash "$CV1/scripts/run_c3m.sh" > "$LOGD/c3m_master.out" 2>&1 &
