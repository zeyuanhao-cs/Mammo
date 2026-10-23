#!/bin/bash
# 20 步两段显存/步速实测：① 单次前向（--no-soft） ② 双次前向（rho=1，不复用）
set -u
PROJ=/home/z/zhao/Mammography/Mammo/2-view
PY=/home/z/zhao/Mammography/miniconda3/envs/mammo/bin/python
L=$PROJ/chain_v1/logs
cd "$PROJ" || exit 1
mkdir -p "$L"

echo "=== SEG1 single-forward $(date +%H:%M:%S) ===" | tee -a $L/memtest2.log
"$PY" chain_v1/train/run_train.py --max-steps 20 --no-soft --peak-mem \
  --output-dir /home/z/zhao/Mammography/model/_chain_v1_t1 --save-steps 10000 \
  --logging-steps 2 2>&1 | tee -a $L/memtest2_seg1.log | tail -3

echo "=== SEG2 double-forward $(date +%H:%M:%S) ===" | tee -a $L/memtest2.log
"$PY" chain_v1/train/run_train.py --max-steps 20 --no-reuse-embeds \
  --rho-max 1.0 --rho-warmup-frac 0 --peak-mem \
  --output-dir /home/z/zhao/Mammography/model/_chain_v1_t2 --save-steps 10000 \
  --logging-steps 2 2>&1 | tee -a $L/memtest2_seg2.log | tail -3

echo "=== DONE $(date +%H:%M:%S) ===" | tee -a $L/memtest2.log
