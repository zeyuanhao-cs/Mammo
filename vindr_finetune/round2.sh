#!/bin/bash
# =============================================================================
# 第二轮全自动链路:
#   1. 等待当前推理 (round1) 完成
#   2. 备份 round1 推理结果 + round1 adapter
#   3. 清空旧模型目录, 用 11.json 重新训练 (覆盖)
#   4. 训练成功后自动全量推理 direct_test.json -> outputs/direct_full
# 全程无交互; 任一关键步骤失败则中止并写明原因。
# =============================================================================
set -u
cd /hy-tmp/9_9/Mammo

FT=/hy-tmp/9_9/Mammo/vindr_finetune
OUT=$FT/outputs/direct_full
PRED=$OUT/predictions.jsonl
ADAPTER_DIR=/hy-tmp/9_9/Qwen/qwen-4b/lora_lesion_1to3
BACKUP_DIR=/hy-tmp/9_9/Qwen/qwen-4b/lora_lesion_1to3_round1

echo "[round2] ===== 步骤1: 等待当前推理完成 ====="
while true; do
  if ! pgrep -f "infer_full_test.py" > /dev/null 2>&1; then
    n=0
    [ -f "$PRED" ] && n=$(wc -l < "$PRED")
    echo "[round2] $(date '+%F %T') round1 推理进程已退出, 完成 $n/4000"
    if [ "$n" -lt 4000 ]; then
      echo "[round2] 警告: round1 推理可能未全部完成 ($n/4000), 继续执行 round2"
    fi
    break
  fi
  n=0
  [ -f "$PRED" ] && n=$(wc -l < "$PRED")
  echo "[round2] $(date '+%F %T') round1 推理进行中: $n/4000"
  sleep 60
done

echo "[round2] ===== 步骤2: 备份 round1 推理结果 ====="
if [ -f "$PRED" ]; then
  mv "$PRED" "$OUT/predictions_round1_lesion1to3.jsonl"
  echo "[round2] 已备份 -> predictions_round1_lesion1to3.jsonl"
fi
if [ -f "$OUT/run_config.json" ]; then
  mv "$OUT/run_config.json" "$OUT/run_config_round1.json"
fi

echo "[round2] ===== 步骤3: 备份旧 adapter 并清空模型目录 ====="
mkdir -p "$BACKUP_DIR"
if cp -f "$ADAPTER_DIR/adapter_config.json" "$ADAPTER_DIR/adapter_model.safetensors" "$BACKUP_DIR/" 2>/dev/null; then
  echo "[round2] round1 adapter 已备份到 $BACKUP_DIR"
else
  echo "[round2] 警告: round1 adapter 备份失败 (可能不存在), 继续清空"
fi
rm -rf "$ADAPTER_DIR"/*
echo "[round2] 模型目录已清空: $ADAPTER_DIR"

echo "[round2] ===== 步骤4: 训练 (11.json, 2792 条) ====="
llamafactory-cli train vindr_finetune/trial.yaml > "$FT/train_round2.log" 2>&1
RC=$?
echo "[round2] $(date '+%F %T') 训练退出码: $RC (日志: train_round2.log)"

if [ $RC -ne 0 ] || [ ! -f "$ADAPTER_DIR/adapter_model.safetensors" ]; then
  echo "[round2] 错误: 训练失败或最终 adapter 未保存, 中止后续推理"
  exit 1
fi
echo "[round2] 新 adapter 训练完成"

echo "[round2] ===== 步骤5: 全量推理 direct_test.json (4000 条) ====="
python3 -u vindr_finetune/infer_full_test.py --no-wait > "$FT/infer_round2.log" 2>&1
RC2=$?
echo "[round2] $(date '+%F %T') 推理退出码: $RC2 (日志: infer_round2.log)"

echo "[round2] ===== 全部完成, 结果: $OUT/predictions.jsonl ====="
