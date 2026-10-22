#!/bin/bash
# 过拟合检查链: 3 个 checkpoint × (val 1600 / train1000) 的 b8 推理 + summarize + teacher-forcing 取值 CE
# 用法: tmux new-session -d -s overfit "bash scripts/bench/overfit_chain.sh"
set -u
PROJ=/home/z/zhao/Mammography/Mammo/2-view
PY=/home/z/zhao/Mammography/miniconda3/envs/mammo/bin/python
EP3=/home/z/zhao/Mammography/model/Qwen3.5-4B-lora-2view-os-sub-vt-ep3
LOG=$PROJ/logs/vlm/overfit_chain.log
VAL=$PROJ/data/splits/val_raw_local.json
TRAIN=$PROJ/data/splits/overfit_train1000_seed42.json

cd "$PROJ" || exit 1
mkdir -p outputs/scoring outputs/predictions outputs/eval logs/vlm
ts() { date +"%H:%M:%S"; }
nlines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }

echo "[$(ts)] === OVERFIT CHAIN START ===" | tee -a "$LOG"

for pair in 439:e1 878:e2 1317:e3; do
  CKPT=${pair%%:*}
  TAG=${pair##*:}
  AD=$EP3/checkpoint-$CKPT

  for sp in val:1600 train:1000; do
    SPLIT=${sp%%:*}
    N=${sp##*:}
    if [ "$SPLIT" = "val" ]; then DATA=$VAL; else DATA=$TRAIN; fi
    PROBS=$PROJ/outputs/scoring/overfit_${SPLIT}_${TAG}_b8_with_probs.jsonl
    PRED=$PROJ/outputs/predictions/overfit_${SPLIT}_${TAG}_b8_pred.jsonl
    METRICS=$PROJ/outputs/eval/metrics_overfit_${SPLIT}_${TAG}_b8.json
    CE=$PROJ/outputs/scoring/overfit_${SPLIT}_${TAG}_value_ce.jsonl

    if [ "$(nlines "$PROBS")" -lt "$N" ]; then
      echo "[$(ts)] infer $TAG/$SPLIT" | tee -a "$LOG"
      "$PY" scripts/vlm/infer_2view_batched.py --model qwen3.5-4b --adapter "$AD" \
        --data "$DATA" --limit 0 --batch-size 8 --out-suffix "_overfit_${SPLIT}_${TAG}_b8" \
        --out-pred "$PRED" --out-probs "$PROBS" --tag "overfit-${TAG}-${SPLIT}" 2>&1 | tail -3 | tee -a "$LOG"
    else
      echo "[$(ts)] skip infer $TAG/$SPLIT (probs complete)" | tee -a "$LOG"
    fi

    if [ ! -f "$METRICS" ]; then
      echo "[$(ts)] summarize $TAG/$SPLIT" | tee -a "$LOG"
      "$PY" scripts/eval/summarize_probs.py --probs "$PROBS" --out "$METRICS" \
        --tag "overfit-${TAG}-${SPLIT}" --n-boot 1000 2>&1 | tail -2 | tee -a "$LOG"
    fi

    # 取值 loss 改用 probs_norm 的 NLL（见 overfit_report.py），不再跑 teacher-forcing CE
    echo "[$(ts)] skip CE $TAG/$SPLIT (改用 probs_norm NLL)" | tee -a "$LOG"
  done
done

echo "[$(ts)] === BUILDING REPORT ===" | tee -a "$LOG"
"$PY" scripts/bench/overfit_report.py 2>&1 | tee -a "$LOG"
echo "[$(ts)] OVERFIT_ALL_DONE" | tee -a "$LOG"
