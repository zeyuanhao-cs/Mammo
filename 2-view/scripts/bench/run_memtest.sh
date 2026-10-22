#!/bin/bash
# 原生分辨率 20 步内存测试：跑训练 + 同步采样 nvidia-smi 显存峰值
set -u
PROJ=/home/z/zhao/Mammography/Mammo/2-view
PY=/home/z/zhao/Mammography/miniconda3/envs/mammo/bin/python
LOG=$PROJ/logs/vlm/memtest_native.log
GPU=$PROJ/logs/vlm/memtest_native_gpu.txt
CFG=$PROJ/configs/train_lora_2view_native_memtest_local.yaml

cd "$PROJ" || exit 1
mkdir -p logs/vlm
: > "$LOG"

bash scripts/bench/gpu_sampler.sh "$GPU" &
SPID=$!
trap 'kill $SPID 2>/dev/null' EXIT

"$PY" scripts/bench/train_memtest.py "$CFG" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}

kill $SPID 2>/dev/null
sleep 1
echo "PEAK_GPU_SMI_MIB=$(sort -n "$GPU" | tail -1)" | tee -a "$LOG"
echo "TRAIN_RC=$rc" | tee -a "$LOG"
