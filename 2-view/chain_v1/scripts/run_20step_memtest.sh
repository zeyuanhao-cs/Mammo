#!/bin/bash
# 完整方法（单次 backward、两图共存）在 A40 上的 20 步显存实测
set -u
PROJ=/home/z/zhao/Mammography/Mammo/2-view
PY=/home/z/zhao/Mammography/miniconda3/envs/mammo/bin/python
LOG=$PROJ/chain_v1/logs/memtest_20step.log
GPU=$PROJ/chain_v1/logs/memtest_20step_gpu.txt
OUT=/home/z/zhao/Mammography/model/_chain_v1_memtest

cd "$PROJ" || exit 1
: > "$LOG"
bash scripts/bench/gpu_sampler.sh "$GPU" &
SPID=$!
trap 'kill $SPID 2>/dev/null' EXIT

"$PY" chain_v1/train/run_train.py \
  --max-steps 20 --peak-mem --epochs 1 --rho-max 0.5 \
  --output-dir "$OUT" --save-steps 10000 --logging-steps 1 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
kill $SPID 2>/dev/null
sleep 1
echo "PEAK_GPU_SMI_MIB=$(sort -n "$GPU" 2>/dev/null | tail -1)" | tee -a "$LOG"
echo "MEMTEST_RC=$rc" | tee -a "$LOG"
