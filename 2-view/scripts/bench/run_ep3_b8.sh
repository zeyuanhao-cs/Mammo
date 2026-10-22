#!/bin/bash
# vt-ep3 的 e1/e2/e3 (checkpoint-439/878/1317) 测试集，batch=8 新流水线
# 用法: tmux new-session -d -s b8_ep3 "bash scripts/bench/run_ep3_b8.sh"
set -u
PROJ=/home/z/zhao/Mammography/Mammo/2-view
PY=/home/z/zhao/Mammography/miniconda3/envs/mammo/bin/python
EP3=/home/z/zhao/Mammography/model/Qwen3.5-4B-lora-2view-os-sub-vt-ep3
DATA=$PROJ/data/splits/test_local.json
LOG=$PROJ/logs/vlm/ep3_b8.log

cd "$PROJ" || exit 1
mkdir -p outputs/predictions outputs/scoring outputs/eval logs/vlm
ts() { date +"%H:%M:%S"; }

for pair in 439:e1 878:e2 1317:e3; do
  CKPT=${pair%%:*}
  TAG=${pair##*:}
  ADAPTER=$EP3/checkpoint-$CKPT
  PRED=$PROJ/outputs/predictions/predictions_qwen3.5-4b_vt-ep3-${TAG}_test_b8_2000.jsonl
  PROBS=$PROJ/outputs/scoring/vt-ep3-${TAG}_test_b8_with_probs.jsonl
  METRICS=$PROJ/outputs/eval/metrics_probs_vt-ep3-${TAG}_test_b8.json

  echo "[$(ts)] === B8 $TAG (ckpt-$CKPT) infer+score ===" | tee -a "$LOG"
  if [ -f "$METRICS" ]; then
    echo "[$(ts)] skip $TAG (metrics exists)" | tee -a "$LOG"
    continue
  fi

  /usr/bin/time -f "B8_${TAG}_WALL=%e s" "$PY" scripts/vlm/infer_2view_batched.py \
      --model qwen3.5-4b --adapter "$ADAPTER" --data "$DATA" --limit 0 \
      --batch-size 8 --out-suffix "_vt-ep3-${TAG}_test_b8" \
      --out-pred "$PRED" --out-probs "$PROBS" --tag "vt-ep3-${TAG}-test-b8" 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  [ "$rc" -ne 0 ] && { echo "[$(ts)] $TAG infer FAILED rc=$rc, abort" | tee -a "$LOG"; exit 1; }

  echo "[$(ts)] === B8 $TAG summarize ===" | tee -a "$LOG"
  "$PY" scripts/eval/summarize_probs.py --probs "$PROBS" --out "$METRICS" \
      --tag "vt-ep3-${TAG}-test-b8" --n-boot 1000 2>&1 | tail -10 | tee -a "$LOG"
  echo "[$(ts)] === B8 $TAG DONE ===" | tee -a "$LOG"
done

echo "[$(ts)] EP3_B8_ALL_DONE" | tee -a "$LOG"
