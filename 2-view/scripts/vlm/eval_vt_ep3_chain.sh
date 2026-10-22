#!/bin/bash
# ep3 三个 checkpoint (439/878/1317) × (val 1600 / test 2000) 的推理 -> 打分 -> 汇总
# 用法: tmux new-session -d -s vt_ep3_eval "bash scripts/vlm/eval_vt_ep3_chain.sh"
#      SPLITS="test" (默认) 或 SPLITS="val" 或 SPLITS="test val"
# 断点续跑: 已存在的 scoring 产物会跳过对应的 infer+score
set -u

PROJ=/home/z/zhao/Mammography/Mammo/2-view
PY=/home/z/zhao/Mammography/miniconda3/envs/mammo/bin/python
EP3=/home/z/zhao/Mammography/model/Qwen3.5-4B-lora-2view-os-sub-vt-ep3
BASE=/home/z/zhao/Mammography/model/Qwen3.5-4B
LOG=$PROJ/logs/vlm/eval_vt_ep3.log

cd "$PROJ" || exit 1
mkdir -p outputs/scoring outputs/eval

ts() { date +"%H:%M:%S"; }

for pair in 439:e1 878:e2 1317:e3; do
  CKPT=${pair%%:*}
  TAG=${pair##*:}
  ADAPTER=$EP3/checkpoint-$CKPT

  for split in ${SPLITS:-test}; do
    if [ "$split" = "val" ]; then
      DATA=data/splits/val_raw_local.json
    else
      DATA=data/splits/test_local.json
    fi

    SUFFIX=_vt-ep3-${TAG}_${split}
    PROBS=outputs/scoring/vt-ep3-${TAG}_${split}_with_probs.jsonl
    METRICS=outputs/eval/metrics_probs_vt-ep3-${TAG}_${split}.json

    if [ -f "$PROBS" ]; then
      echo "[$(ts)] skip $TAG/$split (probs exists)" | tee -a "$LOG"
    else
      echo "[$(ts)] === infer $TAG ($split) adapter=checkpoint-$CKPT ===" | tee -a "$LOG"
      "$PY" scripts/vlm/infer_2view_local.py \
        --model qwen3.5-4b --adapter "$ADAPTER" --data "$DATA" --limit 0 \
        --out-suffix "$SUFFIX" 2>&1 | tee -a "$LOG"
      rc=${PIPESTATUS[0]}
      if [ "$rc" -ne 0 ]; then echo "[$(ts)] infer FAILED rc=$rc, abort" | tee -a "$LOG"; exit 1; fi

      PRED=$(ls -t outputs/predictions/*"$SUFFIX"_*.jsonl 2>/dev/null | head -1)
      if [ -z "$PRED" ]; then echo "[$(ts)] pred file not found for $SUFFIX" | tee -a "$LOG"; exit 1; fi

      echo "[$(ts)] === score $TAG ($split) pred=$PRED ===" | tee -a "$LOG"
      "$PY" scripts/vlm/score_2view.py \
        --adapter "$ADAPTER" --pred-file "$PRED" --data "$DATA" \
        --model-path "$BASE" --prompt-file dataset_build/prompt.txt \
        --out "$PROBS" --tag "vt-ep3-${TAG}-${split}" 2>&1 | tee -a "$LOG"
      rc=${PIPESTATUS[0]}
      if [ "$rc" -ne 0 ]; then echo "[$(ts)] score FAILED rc=$rc, abort" | tee -a "$LOG"; exit 1; fi
    fi

    echo "[$(ts)] === summarize $TAG ($split) ===" | tee -a "$LOG"
    "$PY" scripts/eval/summarize_probs.py --probs "$PROBS" --out "$METRICS" \
      --tag "vt-ep3-${TAG}-${split}" --n-boot 1000 2>&1 | tail -5 | tee -a "$LOG"
  done
done

echo "[$(ts)] ALL DONE" | tee -a "$LOG"
