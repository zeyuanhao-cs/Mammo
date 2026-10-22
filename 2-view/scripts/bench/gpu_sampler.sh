#!/bin/bash
# 每 0.5s 采一次显存，记录到文件；结尾输出峰值
OUT=$1
: > "$OUT"
while true; do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$OUT"
    sleep 0.5
done
