#!/usr/bin/env bash
# =============================================================================
# 双顺序 (dual-order) 全链: 训练 -> test 集 order A/B 推理 -> order A/B 概率打分
#
#   配置   2-view/configs/train_lora_2view_dualorder_local.yaml
#   数据   2-view/data/splits/test_local.json           (2000 条乳房级样本)
#   提示词 2-view/dataset_build/prompt_order_{a,b}.txt
#
#   产物:
#     /home/z/zhao/Mammography/model/Qwen3.5-4B-lora-2view-dualorder           (LoRA)
#     outputs/predictions/predictions_qwen3.5-4b_dualorder_{a,b}_2000.jsonl    (贪心预测)
#     outputs/scoring/dualorder_{a,b}_test_with_probs.jsonl                    (字段概率)
#
#   日志:  logs/vlm/dualorder_chain.log  +  每步各自的 dualorder_*.log
#
# 可续跑: 训练自动从 output_dir 内最近 checkpoint 续训; infer / score 按 id 续写跳过已完成。
# 用法:   bash 2-view/launch/run_dualorder_chain.sh
# =============================================================================
set -u
set -o pipefail

ROOT=/home/z/zhao/Mammography
MAMMO=$ROOT/Mammo
PROJ=$MAMMO/2-view
LOGDIR=$PROJ/logs/vlm
BIN=$ROOT/miniconda3/envs/mammo/bin
PY=$BIN/python
ADAPTER=$ROOT/model/Qwen3.5-4B-lora-2view-dualorder
MODEL=$ROOT/model/Qwen3.5-4B
DATA=$PROJ/data/splits/test_local.json
CONFIG=2-view/configs/train_lora_2view_dualorder_local.yaml

mkdir -p "$LOGDIR" "$PROJ/outputs/predictions" "$PROJ/outputs/scoring"

export HF_HOME=$ROOT/.hf_home
export TOKENIZERS_PARALLELISM=false
unset PYTHONPATH
cd "$MAMMO"

stamp() { date '+%F %T'; }
step()  { echo; echo "########## [$1] start $(stamp) ##########"; }
die()   { echo "!!! [$1] 失败 (exit=$2), 终止链条 $(stamp)"; exit "$2"; }

# ---------------------------------------------------------------- 1) 训练
step TRAIN
"$BIN/llamafactory-cli" train "$CONFIG" > "$LOGDIR/train_lora_dualorder_local.log" 2>&1
rc=$?
echo "[TRAIN] exit=$rc  $(stamp)"
[ $rc -ne 0 ] && die TRAIN $rc

# ---------------------------------------------------------------- 2/3) 推理 A / B
for od in a b; do
  step "INFER-$od"
  "$PY" 2-view/scripts/vlm/infer_2view_local.py \
      --model qwen3.5-4b \
      --adapter "$ADAPTER" \
      --prompt-file "$PROJ/dataset_build/prompt_order_$od.txt" \
      --data "$DATA" \
      --limit 0 \
      --out-suffix "_dualorder_$od" 2>&1 | tee "$LOGDIR/dualorder_infer_$od.log"
  rc=${PIPESTATUS[0]}
  echo "[INFER-$od] exit=$rc  $(stamp)"
  [ $rc -ne 0 ] && die "INFER-$od" $rc
done

# ---------------------------------------------------------------- 4/5) 打分 A / B
for od in a b; do
  step "SCORE-$od"
  "$PY" 2-view/scripts/vlm/score_2view.py \
      --adapter "$ADAPTER" \
      --model-path "$MODEL" \
      --data "$DATA" \
      --pred-file "$PROJ/outputs/predictions/predictions_qwen3.5-4b_dualorder_${od}_2000.jsonl" \
      --prompt-file "$PROJ/dataset_build/prompt_order_$od.txt" \
      --out "$PROJ/outputs/scoring/dualorder_${od}_test_with_probs.jsonl" \
      --tag "dualorder_$od" 2>&1 | tee "$LOGDIR/dualorder_score_$od.log"
  rc=${PIPESTATUS[0]}
  echo "[SCORE-$od] exit=$rc  $(stamp)"
  [ $rc -ne 0 ] && die "SCORE-$od" $rc
done

echo
echo "########## ALL DONE $(stamp) ##########"
